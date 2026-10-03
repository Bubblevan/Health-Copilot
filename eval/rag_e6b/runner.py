"""Run paired Vanilla/RSEL on runtime-visible reserved inputs only."""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.r2med_multiview import ensure_java_home
from eval.rag_e6.data import RUNTIME_EPISODE_FIELDS, E6Episode
from eval.rag_e6.llm import (
    COMPLETION_CEILING,
    MODEL_SHA256,
    REASONING_ENABLED,
    TEMPERATURE,
    TOP_P,
    CallJournal,
    LocalLlamaCppClient,
)
from eval.rag_e6.reader_executor import (
    DEFAULT_LLAMA_URL,
    _load_bge_cpu,
    _load_lamer_bridge,
    _load_runtime_checkpoint,
    _rank_ids_and_aliases,
    _retrieval_identity,
    _run_rsel,
    _run_vanilla,
    _verify_inherited_u3r_runtime_source,
    _verify_server_manifest,
    _write_immutable_bytes,
)
from eval.rag_e6.split import canonical_json_bytes, sha256_file
from eval.u3r_rag_transfer import (
    DEFAULT_BGE_PATH,
    DEFAULT_QWEN_PATH,
    DEFAULT_UPSTREAM_ROOT,
    DenseDocumentCache,
    U3RDocument,
    U3RRetriever,
    _text_sha256,
    load_runtime_corpora,
    load_upstream_lamer_prompt,
)

from .protocol import ARMS, POOL_SEEDS, RUN_ROOT_RELATIVE, read_json, sha256_bytes

DEFAULT_SERVER_MANIFEST = Path("runs/rag_e6/runtime/gpu_server_manifest.json")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _write_immutable_bytes(path, canonical_json_bytes(value) + b"\n")


def _require_committed_clean(repository_root: Path, relative: str) -> str:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", relative],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )
    if tracked.returncode != 0 or dirty.stdout.strip():
        raise ValueError(f"required input must be committed and clean: {relative}")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=repository_root,
        text=True,
    ).strip()
    if not commit:
        raise ValueError(f"could not identify committed input revision: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return commit


def _load_runtime_episodes(path: Path, pool: str) -> tuple[E6Episode, ...]:
    episodes: list[E6Episode] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != RUNTIME_EPISODE_FIELDS:
                raise ValueError(f"runtime contract mismatch at {path.name}:{line_number}")
            episode_id = row["episode_id"]
            query = row["query"]
            subject_id = row["subject_id"]
            observable = row["observable_state"]
            decision_time = row["decision_time"]
            if (
                not isinstance(episode_id, str)
                or not episode_id
                or episode_id in seen
                or not isinstance(query, str)
                or not query.strip()
                or not isinstance(subject_id, str)
                or not isinstance(observable, dict)
                or not isinstance(decision_time, str)
            ):
                raise ValueError(f"malformed runtime episode at {path.name}:{line_number}")
            families = observable.get("available_external_source_families")
            if not isinstance(families, list) or any(not isinstance(item, str) for item in families):
                raise ValueError("runtime source-family allowlist is malformed")
            seen.add(episode_id)
            episodes.append(E6Episode(
                episode_id=episode_id,
                subject_id=subject_id,
                partition=pool,
                split=pool,
                query=query,
                query_sha256=sha256_bytes(query.encode("utf-8")),
                decision_time=datetime.fromisoformat(decision_time),
                available_source_families=tuple(sorted(set(families))),
            ))
    return tuple(episodes)


def _paired_identity_pass(row: dict[str, Any]) -> bool:
    vanilla = row["arms"]["VANILLA_STRONG"]
    rsel = row["arms"]["RSEL_STRONG"]
    shared_fields = (
        "ranked_evidence_ids", "candidate_union_ids", "channel_ids",
        "evidence_identity_sha256", "evidence_aliases",
    )
    if any(vanilla.get(field) != rsel.get(field) for field in shared_fields):
        return False
    if rsel.get("generation_call_ids") != vanilla.get("generation_call_ids"):
        return False
    if rsel.get("rsel_action") == "FALLBACK_NO_MATCH":
        fields = (
            "answer", "answer_sha256", "cited_aliases", "used_evidence_ids",
            "output_contract_failure", "unknown_aliases",
        )
        return all(vanilla.get(field) == rsel.get(field) for field in fields)
    return True


def execute_reserved(
    *,
    repository_root: Path,
    bge_path: Path = DEFAULT_BGE_PATH,
    qwen_path: Path = DEFAULT_QWEN_PATH,
    upstream_root: Path = DEFAULT_UPSTREAM_ROOT,
    llama_url: str = DEFAULT_LLAMA_URL,
    server_manifest_path: Path = DEFAULT_SERVER_MANIFEST,
    cpu_threads: int = 8,
) -> dict[str, Any]:
    """Execute every materialized reserved runtime row with two paired arms."""
    repository_root = repository_root.resolve()
    run_root = repository_root / RUN_ROOT_RELATIVE
    materialized_root = run_root / "reserved"
    materialization_path = materialized_root / "reserved_materialization_manifest.json"
    materialization_relative = str(
        materialization_path.relative_to(repository_root)
    ).replace("\\", "/")
    materialization_commit = _require_committed_clean(
        repository_root, materialization_relative
    )
    materialization = read_json(materialization_path)
    method_freeze = read_json(run_root / "method_freeze.json")
    if (
        materialization.get("reserved_plan_sha256")
        != read_json(run_root / "method_freeze.json").get("reserved_plan_sha256")
        or materialization.get("evaluator_truth_opened") is not False
        or materialization.get("score_started") is not False
        or materialization.get("reserved_rows_materialized") is not True
    ):
        raise ValueError("reserved materialization manifest failed the gold-blind contract")
    frozen_sources = {
        **method_freeze.get("frozen_e6a_source_hashes", {}),
        **method_freeze.get("e6b_pipeline_source_hashes", {}),
        **method_freeze.get("retrieval_source_hashes", {}),
        **method_freeze.get("scoring_source_hashes", {}),
    }
    for relative, expected in frozen_sources.items():
        if sha256_file(repository_root / relative) != expected:
            raise ValueError(f"method-freeze source hash mismatch: {relative}")
    if sha256_file(qwen_path) != MODEL_SHA256:
        raise ValueError("pinned Qwen model SHA-256 mismatch")
    server_manifest = _verify_server_manifest(server_manifest_path, qwen_path=qwen_path)
    if sha256_file(bge_path / "model.safetensors") != _retrieval_identity()["standard"][
        "bge_weights_sha256"
    ]:
        raise ValueError("pinned BGE-large model SHA-256 mismatch")
    inherited_identity = _verify_inherited_u3r_runtime_source(repository_root)

    episodes_by_pool: dict[str, tuple[E6Episode, ...]] = {}
    corpora_by_pool: dict[str, dict[str, tuple[U3RDocument, ...]]] = {}
    for pool in POOL_SEEDS:
        pool_root = materialized_root / "runtime" / pool
        episodes = _load_runtime_episodes(pool_root / "episodes.jsonl", pool)
        corpus_manifest = read_json(pool_root / "runtime_corpus_manifest.json")
        corpus_path = pool_root / "runtime_corpus.jsonl"
        expected_count = int(materialization["actual_episodes_per_pool"][pool])
        if (
            len(episodes) != expected_count
            or sha256_file(pool_root / "episodes.jsonl")
            != materialization["runtime_corpus_hashes"][pool]["episodes_sha256"]
            or sha256_file(corpus_path)
            != materialization["runtime_corpus_hashes"][pool]["runtime_corpus_sha256"]
            or corpus_manifest.get("episode_count") != expected_count
            or corpus_manifest.get("runtime_corpus_sha256") != sha256_file(corpus_path)
            or corpus_manifest.get("evaluator_truth_opened") is not False
        ):
            raise ValueError(f"runtime pool identity/coverage failed for {pool}")
        corpora = load_runtime_corpora(corpus_path)
        if set(corpora) != {episode.episode_id for episode in episodes}:
            raise ValueError(f"visible runtime corpus IDs differ for {pool}")
        episodes_by_pool[pool] = episodes
        corpora_by_pool[pool] = corpora

    upstream_identity = load_upstream_lamer_prompt(upstream_root)
    client = LocalLlamaCppClient(
        llama_url,
        model_name=str(server_manifest["model_api_id"]),
        effective_context_size=int(server_manifest["effective_context_size"]),
    )
    output_root = run_root / "execution"
    output_root.mkdir(parents=True, exist_ok=True)
    journal_path = output_root / "reserved_generation_calls.jsonl"
    checkpoint_path = output_root / "reserved_episode_checkpoint.jsonl"
    journal = CallJournal(journal_path)
    checkpoint = _load_runtime_checkpoint(checkpoint_path)
    all_episodes = [item for pool in POOL_SEEDS for item in episodes_by_pool[pool]]
    episode_by_id = {item.episode_id: item for item in all_episodes}
    if len(episode_by_id) != len(all_episodes):
        raise ValueError("reserved runtime episode IDs collide across pools")
    for episode_id, row in checkpoint.items():
        episode = episode_by_id.get(episode_id)
        if episode is None or row.get("query_sha256") != episode.query_sha256:
            raise ValueError("reserved checkpoint does not match its frozen runtime query")
        if set(row.get("arms", {})) != set(ARMS) or not _paired_identity_pass(row):
            raise ValueError("reserved checkpoint paired-arm identity contract failed")
        call_ids = {row.get("retrieval_bridge", {}).get("call_id")}
        call_ids.update(row["arms"][arm].get("generation_call_ids", [])[0] for arm in ARMS)
        if None in call_ids or not call_ids.issubset(journal.completed):
            raise ValueError("reserved checkpoint references an absent generation result")

    missing = [item for item in all_episodes if item.episode_id not in checkpoint]
    retriever = None
    if missing:
        ensure_java_home()
        bge = _load_bge_cpu(
            bge_path,
            expected_sha256=_retrieval_identity()["standard"]["bge_weights_sha256"],
            threads=cpu_threads,
        )
        retriever = U3RRetriever(DenseDocumentCache(bge))

    started_at = time.monotonic()
    newly_completed = 0
    for pool in POOL_SEEDS:
        for episode in episodes_by_pool[pool]:
            if episode.episode_id in checkpoint:
                continue
            if retriever is None:
                raise RuntimeError("retrieval model was not initialized for unfinished rows")
            documents = corpora_by_pool[pool][episode.episode_id]
            original_bm25 = retriever._bm25(documents, episode.query)
            bridge = _load_lamer_bridge(
                episode=episode,
                documents=documents,
                original_bm25=original_bm25,
                journal=journal,
                client=client,
                upstream_root=upstream_root,
            )
            strong_view = retriever.strong(
                episode, documents, bridge["generated_text"], original_bm25
            )
            ranked_ids, candidate_ids, channels, evidence = _rank_ids_and_aliases(
                strong_view, documents
            )
            vanilla = _run_vanilla(
                arm="VANILLA_STRONG",
                episode=episode,
                evidence=evidence,
                retrieval_ids=ranked_ids,
                candidate_ids=candidate_ids,
                channels=channels,
                journal=journal,
                client=client,
            )
            rsel = _run_rsel(
                arm="RSEL_STRONG",
                episode=episode,
                evidence=evidence,
                baseline_row=vanilla,
            )
            if vanilla["evidence_identity_sha256"] != rsel["evidence_identity_sha256"]:
                raise ValueError("paired reader arms received non-identical evidence")
            row = {
                "episode_id": episode.episode_id,
                "subject_id": episode.subject_id,
                "pool": pool,
                "query_sha256": episode.query_sha256,
                "visible_source_families": list(episode.available_source_families),
                "visible_document_ids_sha256": sha256_bytes(canonical_json_bytes([
                    [item.doc_id, _text_sha256(item.text)] for item in documents
                ])),
                "retrieval_bridge": bridge,
                "retrieval_cost": {
                    "strong_search_invocations": strong_view.search_invocations,
                    "strong_lamer_bridge_call_id": bridge["call_id"],
                },
                "arms": {"VANILLA_STRONG": vanilla, "RSEL_STRONG": rsel},
            }
            if not _paired_identity_pass(row):
                raise ValueError("paired evidence or RSEL no-match parity failed")
            with checkpoint_path.open("ab") as handle:
                handle.write(canonical_json_bytes(row) + b"\n")
                handle.flush()
            checkpoint[episode.episode_id] = row
            newly_completed += 1
            if newly_completed % 8 == 0 or len(checkpoint) == len(all_episodes):
                print(json.dumps({
                    "stage": "reserved_execution",
                    "completed_this_process": newly_completed,
                    "completed_total": len(checkpoint),
                    "episodes_total": len(all_episodes),
                    "provider_calls_completed": len(journal.completed),
                    "elapsed_seconds": round(time.monotonic() - started_at, 1),
                    "last_episode": episode.episode_id,
                }, sort_keys=True), flush=True)

    if set(checkpoint) != set(episode_by_id):
        raise ValueError("not all reserved episodes were executed")
    ordered_rows = [checkpoint[item.episode_id] for item in all_episodes]
    reader_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in ordered_rows)
    reader_path = output_root / "reserved_reader_outputs.jsonl"
    _write_immutable_bytes(reader_path, reader_bytes)
    call_rows = list(journal.completed.values())
    execution_ids = set(episode_by_id)
    relevant_calls = [row for row in call_rows if row["call_id"].split("|", 1)[0] in execution_ids]
    failed = [row for row in relevant_calls if row.get("status") != "ok"]
    truncated = [row for row in relevant_calls if row.get("finish_reason") == "length"]
    provider_calls = sum(int(row.get("provider_calls", 0)) for row in relevant_calls)
    call_id_set = {row["call_id"] for row in relevant_calls}
    if len(relevant_calls) != len(all_episodes) * 2 or len(call_id_set) != len(all_episodes) * 2:
        raise ValueError("expected exactly one bridge and one Vanilla reader call record per episode")
    manifest = {
        "schema_version": "rag-e6b-gold-blind-execution-v1",
        "method": "RSEL-v1",
        "method_freeze_sha256": sha256_file(run_root / "method_freeze.json"),
        "materialization_manifest_sha256": sha256_file(
            materialized_root / "reserved_materialization_manifest.json"
        ),
        "materialization_commit": materialization_commit,
        "pool_counts": materialization["actual_episodes_per_pool"],
        "total_episode_count": len(all_episodes),
        "arms": list(ARMS),
        "all_reserved_episodes_executed": True,
        "all_required_arm_executions_complete": True,
        "primary_slice_applied_before_execution": False,
        "paired_ranked_evidence_identity_pass": True,
        "paired_evidence_bytes_identity_pass": True,
        "rsel_no_match_parity_pass": all(_paired_identity_pass(row) for row in ordered_rows),
        "generation_calls": len(relevant_calls),
        "provider_calls": provider_calls,
        "failed_generation_calls": len(failed),
        "truncated_generation_calls": len(truncated),
        "reader_output_sha256": sha256_bytes(reader_bytes),
        "call_journal_sha256": sha256_file(journal_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "generation_call_status_counts": dict(sorted(Counter(
            str(row.get("status")) for row in relevant_calls
        ).items())),
        "generation_call_stage_counts": dict(sorted(Counter(
            "|".join(row["call_id"].split("|")[1:]) for row in relevant_calls
        ).items())),
        "rsel_actions": dict(sorted(Counter(
            row["arms"]["RSEL_STRONG"].get("rsel_action") for row in ordered_rows
        ).items())),
        "extra_rsel_model_calls": 0,
        "retrieval_config_changed": False,
        "rsel_method_changed": False,
        "evaluator_truth_opened": False,
        "server_manifest_sha256": sha256_file(server_manifest_path),
        "model_sha256": MODEL_SHA256,
        "bge_sha256": _retrieval_identity()["standard"]["bge_weights_sha256"],
        "completion_ceiling": COMPLETION_CEILING,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "reasoning_enabled": REASONING_ENABLED,
        "upstream_lamer": upstream_identity,
        "inherited_u3r_runtime": inherited_identity,
        "runtime_corpus_hashes": materialization["runtime_corpus_hashes"],
    }
    _write_json(output_root / "reserved_execution_manifest.json", manifest)
    return manifest
