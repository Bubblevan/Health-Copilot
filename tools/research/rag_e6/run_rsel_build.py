"""Materialize RSEL-v1 over frozen, gold-blind Vanilla BUILD artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.rag_e6.data import load_partition_corpus, load_partition_episodes
from eval.rag_e6.llm import (
    COMPLETION_CEILING,
    CONTEXT_CEILING,
    MODEL_NAME,
    MODEL_SHA256,
    PROMPT_BYTE_SAFETY_MARGIN,
    REASONING_ENABLED,
    TEMPERATURE,
    TOP_P,
)
from eval.rag_e6.reader import evidence_identity_sha256, issue_evidence_aliases
from eval.rag_e6.reader_executor import (
    ARM_ORDER,
    DEFAULT_BGE_PATH,
    DEFAULT_QWEN_PATH,
    DEFAULT_RUNTIME_CORPUS_ROOT,
    DEFAULT_SERVER_MANIFEST,
    _code_manifest,
    _package_version,
    _retrieval_identity,
    _verify_inherited_u3r_runtime_source,
    _verify_server_manifest,
    _write_immutable_bytes,
)
from eval.rag_e6.rsel import rsel_output_row
from eval.rag_e6.split import canonical_json_bytes, sha256_file, write_immutable_json
from eval.u3r_rag_transfer import ANSWER_CONTEXT_K

U2F_ROOT = ROOT / "runs/integration/u2f-owned-v1-55955b2eff38"
SPLIT_MANIFEST = ROOT / "runs/rag_e6/split_manifest.json"
SOURCE_ROOT = ROOT / "runs/rag_e6/build_cav_v1"
OUTPUT_ROOT = ROOT / "runs/rag_e6/build_rsel_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected object at {path.name}:{line_number}")
            result.append(value)
    return result


def _verify_source_code(manifest: dict[str, Any]) -> None:
    commit = manifest.get("code_commit")
    hashes = manifest.get("code_sha256")
    if not isinstance(commit, str) or not isinstance(hashes, dict) or not hashes:
        raise ValueError("source BUILD has no frozen code identity")
    for relative_path, expected in hashes.items():
        source = subprocess.check_output(
            ["git", "show", f"{commit}:{relative_path}"], cwd=ROOT
        )
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"source BUILD commit code hash mismatch: {relative_path}")


def materialize_rsel_build() -> dict[str, Any]:
    source_manifest_path = SOURCE_ROOT / "build_manifest.json"
    source_output_path = SOURCE_ROOT / "build_reader_outputs.jsonl"
    source_journal_path = SOURCE_ROOT / "build_generation_calls.jsonl"
    source_manifest = _read_json(source_manifest_path)
    split_manifest = _read_json(SPLIT_MANIFEST)
    source_corpus_path = DEFAULT_RUNTIME_CORPUS_ROOT / "build_runtime_corpus.jsonl"
    source_corpus_manifest_path = DEFAULT_RUNTIME_CORPUS_ROOT / "build_corpus_manifest.json"
    corpus_manifest = _read_json(source_corpus_manifest_path)
    if (
        source_manifest.get("partition") != "BUILD"
        or source_manifest.get("episode_count") != 818
        or source_manifest.get("evaluator_truth_opened") is not False
        or source_manifest.get("future_train_outcomes_opened") is not False
        or source_manifest.get("reserved_test_ood_opened") is not False
        or source_manifest.get("reader_output_sha256") != sha256_file(source_output_path)
        or source_manifest.get("generation_call_journal_sha256")
        != sha256_file(source_journal_path)
        or source_manifest.get("subject_split_manifest_sha256") != sha256_file(SPLIT_MANIFEST)
        or source_manifest.get("runtime_corpus_sha256") != sha256_file(source_corpus_path)
        or source_manifest.get("runtime_corpus_manifest_sha256")
        != sha256_file(source_corpus_manifest_path)
        or split_manifest.get("evaluator_truth_opened") is not False
        or corpus_manifest.get("evaluator_truth_opened") is not False
    ):
        raise ValueError("frozen gold-blind source BUILD artifacts failed identity checks")
    _verify_source_code(source_manifest)
    if (
        source_manifest.get("retrieval") != _retrieval_identity()
        or source_manifest.get("generator", {}).get("model_sha256") != MODEL_SHA256
        or sha256_file(DEFAULT_QWEN_PATH) != MODEL_SHA256
        or sha256_file(DEFAULT_BGE_PATH / "model.safetensors")
        != _retrieval_identity()["standard"]["bge_weights_sha256"]
    ):
        raise ValueError("source BUILD model or retrieval identity differs from the frozen pins")
    inherited_identity = _verify_inherited_u3r_runtime_source(ROOT)
    if source_manifest.get("inherited_u3r_runtime") != inherited_identity:
        raise ValueError("source BUILD inherited U3-R runtime identity changed")
    server_manifest_path = DEFAULT_SERVER_MANIFEST
    server_manifest = _verify_server_manifest(server_manifest_path, qwen_path=DEFAULT_QWEN_PATH)
    if (
        source_manifest.get("server_manifest") != server_manifest
        or source_manifest.get("server_manifest_sha256") != sha256_file(server_manifest_path)
    ):
        raise ValueError("source BUILD server runtime differs from the recorded local GPU server")

    episodes = load_partition_episodes(
        u2f_root=U2F_ROOT,
        split_manifest_path=SPLIT_MANIFEST,
        partition="BUILD",
    )
    episode_by_id = {item.episode_id: item for item in episodes}
    corpora = load_partition_corpus(source_corpus_path, set(episode_by_id))
    source_rows = _read_jsonl(source_output_path)
    source_by_id = {row.get("episode_id"): row for row in source_rows}
    if (
        len(source_rows) != 818
        or len(source_by_id) != len(source_rows)
        or set(source_by_id) != set(episode_by_id)
    ):
        raise ValueError("source Vanilla outputs do not match the frozen BUILD episode set")
    expected_query_order = [[item.episode_id, item.query_sha256] for item in episodes]
    if source_manifest.get("query_order_sha256") != hashlib.sha256(
        canonical_json_bytes(expected_query_order)
    ).hexdigest():
        raise ValueError("source BUILD query order differs from the frozen episode input")

    source_events = _read_jsonl(source_journal_path)
    completed_by_id: dict[str, dict[str, Any]] = {}
    referenced_call_ids: set[str] = set()
    expected_record_hashes: dict[str, str] = {}
    for source in source_rows:
        bridge = source.get("retrieval_bridge", {})
        bridge_id = bridge.get("call_id")
        if not isinstance(bridge_id, str):
            raise TypeError("source BUILD row has no LameR bridge call identity")
        referenced_call_ids.add(bridge_id)
        expected_record_hashes[bridge_id] = bridge.get("call_record_sha256")
        for arm_name in ("VANILLA_OFF", "VANILLA_STANDARD", "VANILLA_STRONG"):
            arm = source["arms"].get(arm_name)
            if not isinstance(arm, dict) or len(arm.get("generation_call_ids", ())) != 1:
                raise ValueError("source Vanilla arm does not reference exactly one reader call")
            call_id, record_hash = (
                arm["generation_call_ids"][0],
                arm["generation_call_records_sha256"][0],
            )
            referenced_call_ids.add(call_id)
            expected_record_hashes[call_id] = record_hash
    for event in source_events:
        call_id = event.get("call_id")
        if event.get("event") == "completed" and call_id in referenced_call_ids:
            if call_id in completed_by_id:
                raise ValueError("duplicate completed source call record")
            completed_by_id[call_id] = event
    if set(completed_by_id) != referenced_call_ids:
        raise ValueError("source Vanilla outputs reference an absent completed model call")
    for call_id, expected_hash in expected_record_hashes.items():
        if hashlib.sha256(canonical_json_bytes(completed_by_id[call_id])).hexdigest() != expected_hash:
            raise ValueError("source Vanilla output call hash differs from its frozen journal")

    filtered_journal = b"".join(
        line
        for line in source_journal_path.read_bytes().splitlines(keepends=True)
        if json.loads(line).get("call_id") in referenced_call_ids
    )
    output_rows = []
    rsel_outcomes: Counter[str] = Counter()
    relation_count = 0
    for episode in episodes:
        source = source_by_id[episode.episode_id]
        if source.get("query_sha256") != episode.query_sha256:
            raise ValueError("source Vanilla row query hash differs from frozen runtime input")
        document_by_id = {item.doc_id: item for item in corpora[episode.episode_id]}
        arms: dict[str, dict[str, Any]] = {
            name: source["arms"][name]
            for name in ("VANILLA_OFF", "VANILLA_STANDARD", "VANILLA_STRONG")
        }
        for baseline_name, rsel_name in (
            ("VANILLA_STANDARD", "RSEL_STANDARD"),
            ("VANILLA_STRONG", "RSEL_STRONG"),
        ):
            baseline = source["arms"][baseline_name]
            ids = baseline.get("ranked_evidence_ids", [])
            if len(ids) > ANSWER_CONTEXT_K or any(doc_id not in document_by_id for doc_id in ids):
                raise ValueError("source Vanilla evidence is outside the frozen runtime corpus")
            evidence = issue_evidence_aliases([
                {"doc_id": doc_id, "text": document_by_id[doc_id].text}
                for doc_id in ids
            ])
            if evidence_identity_sha256(evidence) != baseline.get("evidence_identity_sha256"):
                raise ValueError("reconstructed source evidence identity does not match Vanilla")
            candidate = rsel_output_row(
                arm=rsel_name,
                question=episode.query,
                evidence=evidence,
                baseline_row=baseline,
            )
            for field in (
                "ranked_evidence_ids", "candidate_union_ids", "channel_ids",
                "evidence_identity_sha256",
            ):
                if candidate.get(field) != baseline.get(field):
                    raise ValueError(f"RSEL changed frozen evidence field: {field}")
            arms[rsel_name] = candidate
            rsel_outcomes[candidate["rsel_action"]] += 1
            relation_count += len(candidate["rsel_ledger"])
        if set(arms) != set(ARM_ORDER):
            raise ValueError("RSEL BUILD output has an incomplete arm set")
        output_rows.append({
            "episode_id": episode.episode_id,
            "subject_id": episode.subject_id,
            "partition": "BUILD",
            "split": episode.split,
            "query_sha256": episode.query_sha256,
            "visible_document_ids_sha256": source["visible_document_ids_sha256"],
            "retrieval_bridge": source["retrieval_bridge"],
            "retrieval_cost": source["retrieval_cost"],
            "arms": arms,
        })

    output_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in output_rows)
    output_path = OUTPUT_ROOT / "build_reader_outputs.jsonl"
    journal_path = OUTPUT_ROOT / "build_generation_calls.jsonl"
    _write_immutable_bytes(output_path, output_payload)
    _write_immutable_bytes(journal_path, filtered_journal)

    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "schema_version": "rag-e6a-runtime-freeze-v1",
        "method": "RSEL-v1",
        "partition": "BUILD",
        "arms": list(ARM_ORDER),
        "episode_count": len(episodes),
        "arm_execution_count": len(episodes) * len(ARM_ORDER),
        "all_partition_episodes_executed": True,
        "primary_slice_applied_before_execution": False,
        "dataset_root_sha256": source_manifest["dataset_root_sha256"],
        "source_u2f_manifest_sha256": source_manifest["source_u2f_manifest_sha256"],
        "source_train_runtime_episodes_sha256": source_manifest[
            "source_train_runtime_episodes_sha256"
        ],
        "subject_split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "runtime_corpus_manifest_sha256": sha256_file(source_corpus_manifest_path),
        "runtime_corpus_sha256": sha256_file(source_corpus_path),
        "query_order_sha256": hashlib.sha256(canonical_json_bytes(expected_query_order)).hexdigest(),
        "reader_output_sha256": hashlib.sha256(output_payload).hexdigest(),
        "generation_call_journal_sha256": hashlib.sha256(filtered_journal).hexdigest(),
        "generation_calls_by_stage": dict(sorted(Counter(
            "|".join(call_id.split("|")[1:3]) for call_id in referenced_call_ids
        ).items())),
        "code_commit": current_commit,
        "generator": {
            "model_name": MODEL_NAME,
            "model_api_id": source_manifest["generator"]["model_api_id"],
            "model_sha256": MODEL_SHA256,
            "context_ceiling": CONTEXT_CEILING,
            "completion_ceiling": COMPLETION_CEILING,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "reasoning_enabled": REASONING_ENABLED,
            "retry_count": 0,
            "prompt_byte_safety_margin": PROMPT_BYTE_SAFETY_MARGIN,
            "generation_calls_attempted": sum(
                int(event.get("provider_calls", 0)) for event in completed_by_id.values()
            ),
            "generation_calls_nonempty": sum(
                event.get("status") == "ok" for event in completed_by_id.values()
            ),
            "generation_calls_truncated": sum(
                event.get("finish_reason") == "length" for event in completed_by_id.values()
            ),
            "generation_calls_rejected_by_context_guard": sum(
                event.get("status") == "context_contract_violation"
                for event in completed_by_id.values()
            ),
            "actual_backend": server_manifest["backend"],
        },
        "method_added_generation_calls": 0,
        "rsel_outcomes": dict(sorted(rsel_outcomes.items())),
        "rsel_relation_count": relation_count,
        "retrieval": _retrieval_identity(),
        "upstream_lamer": source_manifest["upstream_lamer"],
        "inherited_u3r_runtime": inherited_identity,
        "server_manifest_sha256": sha256_file(server_manifest_path),
        "server_manifest": server_manifest,
        "bge_device": source_manifest["bge_device"],
        "cpu_threads_for_bge": source_manifest["cpu_threads_for_bge"],
        "packages": {
            name: _package_version(name)
            for name in ("torch", "sentence-transformers", "pyserini", "gensim")
        },
        "code_sha256": _code_manifest(),
        "source_vanilla_build": {
            "manifest_sha256": sha256_file(source_manifest_path),
            "reader_output_sha256": sha256_file(source_output_path),
            "generation_journal_sha256": sha256_file(source_journal_path),
            "code_commit": source_manifest["code_commit"],
            "reused_call_count": len(referenced_call_ids),
            "unreferenced_cav_calls_excluded": True,
        },
        "method_selection_truth_provenance": {
            "BUILD_truth_was_already_opened_before_RSEL_selection": True,
            "RSEL_transform_read_evaluator_truth": False,
            "confirmatory_status": "development_only_until_FROZEN_DEV_passes",
        },
        "evaluator_truth_opened": False,
        "future_train_outcomes_opened": False,
        "reserved_test_ood_materialized": False,
        "reserved_test_ood_opened": False,
        "llm_cannot_access_evaluator_truth": True,
        "llm_cannot_select_actions": True,
        "llm_cannot_change_retrieval_or_evidence_scope": True,
    }
    write_immutable_json(OUTPUT_ROOT / "build_manifest.json", manifest)
    return manifest


def main() -> None:
    if (OUTPUT_ROOT / "build_manifest.json").exists():
        raise FileExistsError("RSEL BUILD manifest already exists; refusing a second materialization")
    split_manifest = _read_json(SPLIT_MANIFEST)
    if split_manifest.get("dataset_root_sha256") != "e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134":
        raise ValueError("E6A subject split does not match frozen U2-F")
    manifest = materialize_rsel_build()
    print(json.dumps({
        "partition": manifest["partition"],
        "method": manifest["method"],
        "episode_count": manifest["episode_count"],
        "arm_execution_count": manifest["arm_execution_count"],
        "generation_calls_attempted": manifest["generator"]["generation_calls_attempted"],
        "method_added_generation_calls": manifest["method_added_generation_calls"],
        "rsel_outcomes": manifest["rsel_outcomes"],
        "reader_output_sha256": manifest["reader_output_sha256"],
        "evaluator_truth_opened": manifest["evaluator_truth_opened"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
