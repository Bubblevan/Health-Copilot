"""Run the six frozen R2MED public TEST arms once, then evaluate after ranking freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.r2med_candidate_union import fuse_dual_source_rrf
from eval.r2med_crb import BGE_QUERY_PREFIX, bm25_query_text, crb_dense_text, crb_lexical_query
from eval.r2med_crb_data import (
    SOURCE_MANIFEST_PATH,
    load_partition_inputs,
    load_source_manifest,
)
from eval.r2med_crb_evaluator import (
    evaluate_candidate_complementarity,
    evaluate_rankings,
    paired_stratified_bootstrap,
)
from eval.r2med_final_test import (
    ARTIFACT_ROOT,
    BOOTSTRAP,
    CANDIDATE_PATH,
    FROZEN_ARMS,
    GENERATION_CONFIG,
    LOCK_PATH,
    LOCKED_CODE_PATHS,
    QWEN_IDENTITY,
    REPORT_PATH,
    START_MARKER_PATH,
    TEST_COUNTS,
    TEST_TOTAL,
    append_jsonl_once,
    canonical_json_sha256,
    generation_statistics,
    read_jsonl_records,
    read_lock,
    resume_claim_gates,
    sha256_file,
    validate_complete_generation,
    validate_final_lock,
    validate_generation_budget,
    validate_rankings,
)
from eval.r2med_gar_generation import (
    LocalLlamaCppClient,
    R2MedGARGenerator,
    load_upstream_prompt_catalog,
    prompt_sha256,
)
from eval.r2med_multiview import (
    LuceneBM25Index,
    RankedDocument,
    dense_search_many,
    encode_bge,
    make_four_channels,
    weighted_rrf,
)
from tools.freeze_r2med_final_eval import DEFAULT_UPSTREAM
from tools.generate_r2med_crb_compact_repair import (
    COMPACT_JSON_SCHEMA,
    COMPACT_PROMPT_TEMPLATE,
    _sha256_text,
)
from tools.generate_r2med_crb_compact_repair import (
    _generation_record as compact_generation_record,
)
from tools.generate_r2med_gar import (
    DEFAULT_SERVER,
    PORT,
    _check_qwen,
    _feedback_for_query,
    _llama_version,
    _server_ready,
    _start_server,
)
from tools.verify_r2med_models import E_ROOT, verify_models

DEFAULT_SOURCE_ROOT = Path(r"E:\Health-Copilot-E1.2\sources")
DEFAULT_BGE_ROOT = E_ROOT / "models/bge-large-en-v1.5"
MODEL_CACHE = E_ROOT / "cache/huggingface"


class InfrastructureCallError(RuntimeError):
    """A local generation service failure; no response is checkpointed as completed."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _verify_committed_lock(lock: Mapping[str, Any]) -> str:
    if _git("status", "--porcelain", "--", LOCK_PATH.as_posix()):
        raise ValueError("final TEST lock is modified or untracked; commit it before execution")
    if _git("status", "--porcelain", "--", *LOCKED_CODE_PATHS):
        raise ValueError("execution-critical files changed after the final TEST lock")
    subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{LOCK_PATH.as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
    )
    if _git("status", "--porcelain", "--", LOCK_PATH.as_posix()):
        raise ValueError("HEAD does not contain the exact final TEST lock bytes")
    lock_commit = _git("log", "-1", "--format=%H", "--", LOCK_PATH.as_posix())
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", lock_commit, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    for relative, expected in lock["code"]["files_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"frozen execution file hash drift: {relative}")
    if _git("rev-parse", lock["code"]["commit"]) != lock["code"]["commit"]:
        raise ValueError("frozen implementation commit is not present in this repository")
    return lock_commit


def _open_or_resume_marker(lock_sha: str, lock_commit: str) -> dict[str, Any]:
    path = ROOT / START_MARKER_PATH
    if (ROOT / REPORT_PATH).exists():
        raise FileExistsError("final TEST report already exists; refusing a second public TEST execution")
    if path.exists():
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker.get("lock_sha256") != lock_sha or marker.get("status") != "IN_PROGRESS":
            raise ValueError("existing TEST execution marker does not permit this lock/resume")
        marker["resume_count"] = int(marker.get("resume_count", 0)) + 1
        marker["last_resumed_at_utc"] = datetime.now(UTC).isoformat()
        path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return marker
    marker = {
        "schema_version": "r2med-final-test-one-shot-marker-v1",
        "status": "IN_PROGRESS",
        "lock_sha256": lock_sha,
        "lock_commit": lock_commit,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "resume_count": 0,
        "test_accessed": True,
        "completed_query_rows_are_never_regenerated": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(marker, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return marker


def _write_json_atomic(path: Path, value: Any) -> str:
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_file(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return sha256_file(path)


def _load_checkpoint(
    final_path: Path,
    partial_path: Path,
    expected_ids: Sequence[str],
    *,
    method: str,
    prompt_sha: str,
) -> dict[str, dict[str, Any]]:
    selected = final_path if final_path.exists() else partial_path
    if not selected.exists():
        return {}
    rows = read_jsonl_records(selected)
    by_id: dict[str, dict[str, Any]] = {}
    expected = set(expected_ids)
    for row in rows:
        query_id = row.get("query_id")
        if (
            query_id not in expected
            or row.get("method") != method
            or row.get("prompt_sha256") != prompt_sha
            or row.get("error") not in (None, "invalid_compact_json_or_constraints")
        ):
            raise ValueError(f"incompatible generation checkpoint: {selected}")
        if query_id in by_id:
            raise ValueError(f"duplicate completed query in generation checkpoint: {query_id}")
        by_id[str(query_id)] = row
    return by_id


def _finalize_generation(
    final_path: Path,
    partial_path: Path,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    expected_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], str]:
    ordered = [dict(rows_by_id[query_id]) for query_id in expected_ids]
    if final_path.exists():
        existing = read_jsonl_records(final_path)
        if existing != ordered:
            raise ValueError(f"immutable completed generation differs from checkpoint: {final_path}")
    else:
        _write_jsonl_atomic(final_path, ordered)
        if partial_path.exists():
            partial_path.unlink()
    try:
        final_path.chmod(final_path.stat().st_mode & ~0o222)
    except OSError:
        pass
    return ordered, sha256_file(final_path)


def _write_generation_manifest(
    path: Path,
    *,
    method: str,
    prompt_sha_by_subset: Mapping[str, str],
    artifacts: Mapping[str, tuple[Path, Sequence[Mapping[str, Any]], str]],
    source_identity: Mapping[str, Mapping[str, Any]],
    query_order_sha: Mapping[str, str],
    llama_version: str,
) -> dict[str, Any]:
    subset_rows = []
    total_rows: list[Mapping[str, Any]] = []
    for subset, (artifact, rows, artifact_sha) in artifacts.items():
        total_rows.extend(rows)
        subset_rows.append({
            "subset": subset,
            "query_count": len(rows),
            "query_sha256": source_identity[subset]["files"]["query.jsonl"]["sha256"],
            "corpus_sha256": source_identity[subset]["files"]["corpus.jsonl"]["sha256"],
            "query_order_sha256": query_order_sha[subset],
            "prompt_sha256": prompt_sha_by_subset[subset],
            "artifact_path": str(artifact),
            "artifact_sha256": artifact_sha,
        })
    manifest = {
        "schema_version": "r2med-final-test-generation-v1",
        "partition": "TEST",
        "method": method,
        "test_status": "PUBLIC_BENCHMARK_REUSED",
        "generator": {"identity": QWEN_IDENTITY, "llama_cpp_version": llama_version, "endpoint_bind": "127.0.0.1"},
        "generation_config": GENERATION_CONFIG,
        "query_count": len(total_rows),
        "call_count": len(total_rows),
        "completed_count": sum(bool(row.get("completed")) for row in total_rows),
        "valid_count": sum(bool(row.get("valid")) for row in total_rows),
        "fallback_count": sum(bool(row.get("fallback_original")) for row in total_rows),
        "truncation_count": sum(bool(row.get("truncated")) for row in total_rows),
        "prompt_sha256_by_subset": dict(prompt_sha_by_subset),
        "dataset_source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "subsets": subset_rows,
        "immutable": True,
    }
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError(f"existing generation manifest differs from the frozen artifacts: {path}")
    else:
        _write_json_atomic(path, manifest)
    return manifest


def _source_entries() -> dict[str, dict[str, Any]]:
    manifest = load_source_manifest(SOURCE_MANIFEST_PATH)
    return {entry["name"]: entry for entry in manifest["datasets"]["TEST"]}


def _generation_artifacts(
    inputs,
    source_entries: Mapping[str, Mapping[str, Any]],
    prompt_catalog: Mapping[str, Mapping[str, str]],
    bm25_rankings: Mapping[str, Mapping[str, Sequence[RankedDocument]]],
    *,
    server_executable: Path,
    port: int,
    expected_lamer_prompt_sha: Mapping[str, str],
    expected_compact_prompt_sha: str,
    expected_compact_schema_sha: str,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    qwen_identity = _check_qwen()
    if qwen_identity["sha256"] != QWEN_IDENTITY["sha256"]:
        raise ValueError("local Qwen GGUF drifted from the TEST lock")
    if not server_executable.is_file():
        raise FileNotFoundError(f"llama.cpp server executable not found: {server_executable}")

    input_by_subset = {subset.name: subset for subset in inputs}
    prompt_shas = {
        subset.name: prompt_sha256(prompt_catalog["lamer"][subset.upstream_prompt_family])
        for subset in inputs
    }
    if prompt_shas != dict(expected_lamer_prompt_sha):
        raise ValueError("LameR upstream prompt family SHA drifted from the committed TEST lock")
    compact_sha = _sha256_text(COMPACT_PROMPT_TEMPLATE)
    if compact_sha != expected_compact_prompt_sha:
        raise ValueError("compact CRB prompt SHA drifted from the committed TEST lock")
    if canonical_json_sha256(COMPACT_JSON_SCHEMA) != expected_compact_schema_sha:
        raise ValueError("compact CRB JSON schema drifted from the committed TEST lock")
    lamer_rows_by_subset: dict[str, dict[str, dict[str, Any]]] = {}
    compact_rows_by_subset: dict[str, dict[str, dict[str, Any]]] = {}
    all_pending = False
    for subset in inputs:
        ids = [query.query_id for query in subset.queries]
        lamer_final = ARTIFACT_ROOT / "generation/lamer" / f"{subset.name}.jsonl"
        lamer_partial = lamer_final.with_suffix(".jsonl.partial")
        crb_final = ARTIFACT_ROOT / "generation/compact_crb_q" / f"{subset.name}.jsonl"
        crb_partial = crb_final.with_suffix(".jsonl.partial")
        lamer_rows_by_subset[subset.name] = _load_checkpoint(
            lamer_final, lamer_partial, ids, method="lamer", prompt_sha=prompt_shas[subset.name]
        )
        compact_rows_by_subset[subset.name] = _load_checkpoint(
            crb_final, crb_partial, ids, method="crb_q_compact_repair", prompt_sha=compact_sha
        )
        if any(row.get("subset") != subset.name for row in lamer_rows_by_subset[subset.name].values()):
            raise ValueError(f"LameR checkpoint has a cross-subset row: {subset.name}")
        if any(row.get("subset") != subset.name for row in compact_rows_by_subset[subset.name].values()):
            raise ValueError(f"compact CRB checkpoint has a cross-subset row: {subset.name}")
        all_pending = all_pending or len(lamer_rows_by_subset[subset.name]) < len(ids)
        all_pending = all_pending or len(compact_rows_by_subset[subset.name]) < len(ids)

    process = None
    log_handle = None
    llama_version = _llama_version(server_executable)
    try:
        if all_pending:
            log_path = ARTIFACT_ROOT / "logs" / f"final-test-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.log"
            process, log_handle = _start_server(server_executable, port, log_path)
            _server_ready(port)
            client = LocalLlamaCppClient(f"http://127.0.0.1:{port}/v1")
            lamer_generators = {
                subset.name: R2MedGARGenerator(subset.upstream_prompt_family, client, prompt_catalog)
                for subset in inputs
            }
            for subset in inputs:
                docs = {document.doc_id: document.text for document in subset.documents}
                top_ids = {query_id: [item.doc_id for item in ranking[:10]] for query_id, ranking in bm25_rankings[subset.name].items()}
                for query in subset.queries:
                    if query.query_id not in lamer_rows_by_subset[subset.name]:
                        feedback_ids = top_ids[query.query_id]
                        feedback = _feedback_for_query(query.query_id, top_ids, docs)
                        view = lamer_generators[subset.name].generate(query.query_id, query.text, "lamer", feedback)
                        if view.error:
                            raise InfrastructureCallError(
                                f"LameR local generation failed for {subset.name}/{query.query_id}: {view.error}"
                            )
                        record = view.to_dict()
                        record.update({
                            "subset": subset.name,
                            "prompt_sha256": prompt_shas[subset.name],
                            "feedback_depth": 10,
                            "feedback_doc_ids": feedback_ids,
                        })
                        append_jsonl_once(
                            ARTIFACT_ROOT / "generation/lamer" / f"{subset.name}.jsonl.partial",
                            record,
                            expected_method="lamer",
                            expected_prompt_sha=prompt_shas[subset.name],
                        )
                        lamer_rows_by_subset[subset.name][query.query_id] = record
            for subset in inputs:
                for query in subset.queries:
                    if query.query_id not in compact_rows_by_subset[subset.name]:
                        record = compact_generation_record(query.query_id, query.text, client)
                        if record.get("error") not in (None, "invalid_compact_json_or_constraints"):
                            raise InfrastructureCallError(
                                f"compact CRB local generation failed for {subset.name}/{query.query_id}: {record['error']}"
                            )
                        record.update({"subset": subset.name, "prompt_sha256": compact_sha})
                        append_jsonl_once(
                            ARTIFACT_ROOT / "generation/compact_crb_q" / f"{subset.name}.jsonl.partial",
                            record,
                            expected_method="crb_q_compact_repair",
                            expected_prompt_sha=compact_sha,
                        )
                        compact_rows_by_subset[subset.name][query.query_id] = record
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_handle is not None:
            log_handle.close()

    finalized: dict[str, dict[str, list[dict[str, Any]]]] = {"lamer": {}, "compact_crb_q": {}}
    manifests: dict[str, Any] = {}
    for method, table, method_name, prompt_by_subset in (
        ("lamer", lamer_rows_by_subset, "lamer", prompt_shas),
        ("compact_crb_q", compact_rows_by_subset, "crb_q_compact_repair", {name: compact_sha for name in input_by_subset}),
    ):
        artifact_entries = {}
        for subset in inputs:
            name = subset.name
            ids = [query.query_id for query in subset.queries]
            rows_map = table[name]
            if set(rows_map) != set(ids):
                missing = [query_id for query_id in ids if query_id not in rows_map]
                raise InfrastructureCallError(f"generation incomplete for {method}/{name}; pending {len(missing)} queries")
            ordered = [rows_map[query_id] for query_id in ids]
            validate_complete_generation(
                ordered,
                ids,
                expected_method=method_name,
                expected_prompt_sha=prompt_by_subset[name],
            )
            final_path = ARTIFACT_ROOT / "generation" / method / f"{name}.jsonl"
            partial_path = final_path.with_suffix(".jsonl.partial")
            frozen_rows, digest = _finalize_generation(final_path, partial_path, rows_map, ids)
            finalized[method][name] = frozen_rows
            artifact_entries[name] = (final_path, frozen_rows, digest)
        manifest_path = ROOT / "runs/rag_r2med_final_test/generation" / f"{method}_manifest.json"
        query_order_hashes = {
            subset.name: hashlib.sha256("\n".join(q.query_id for q in subset.queries).encode()).hexdigest()
            for subset in inputs
        }
        manifests[method] = _write_generation_manifest(
            manifest_path,
            method=method,
            prompt_sha_by_subset=prompt_by_subset,
            artifacts=artifact_entries,
            source_identity=source_entries,
            query_order_sha=query_order_hashes,
            llama_version=llama_version,
        )
    return finalized, manifests


def _load_or_encode_test_corpus(model, subset, model_sha: str) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    source = _source_entries()[subset.name]
    root = ARTIFACT_ROOT / "embeddings"
    vectors_path = root / f"{subset.name}.npy"
    identity_path = root / f"{subset.name}.json"
    expected = {
        "schema_version": "r2med-final-test-bge-cache-v1",
        "subset": subset.name,
        "corpus_sha256": source["files"]["corpus.jsonl"]["sha256"],
        "model_revision": FROZEN_ARMS["B1_BGE_large"]["revision"],
        "model_weights_sha256": model_sha,
        "document_count": len(subset.dense_documents),
        "dimensions": 1024,
        "max_sequence_length": 512,
        "normalized": False,
    }
    if vectors_path.exists() or identity_path.exists():
        if not vectors_path.is_file() or not identity_path.is_file():
            raise ValueError(f"incomplete frozen BGE cache for {subset.name}")
        if json.loads(identity_path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"BGE cache identity differs from final TEST lock: {subset.name}")
        vectors = np.load(vectors_path, mmap_mode="r")
        if vectors.shape != (len(subset.dense_documents), 1024) or vectors.dtype != np.float32:
            raise ValueError(f"BGE cache shape mismatch: {subset.name}")
        return vectors, expected
    vectors = encode_bge(model, [doc.text for doc in subset.dense_documents], batch_size=32, show_progress_bar=True)
    if vectors.shape != (len(subset.dense_documents), 1024):
        raise ValueError(f"unexpected BGE embedding shape for {subset.name}: {vectors.shape}")
    root.mkdir(parents=True, exist_ok=True)
    import uuid

    temporary_vectors = vectors_path.with_name(f".{vectors_path.name}.{uuid.uuid4().hex}.partial")
    with temporary_vectors.open("xb") as handle:
        np.save(handle, vectors, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_vectors.replace(vectors_path)
    _write_json_atomic(identity_path, expected)
    return vectors, expected


def _ranking_row(subset: str, query_id: str, method: str, items: Sequence[RankedDocument]) -> dict[str, Any]:
    return {
        "subset": subset,
        "query_id": query_id,
        "method": method,
        "ranking": [{"doc_id": item.doc_id, "score": float(item.score)} for item in items],
    }


def _save_arm_rankings(
    arm: str,
    subsets,
    rows_by_subset: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, str]]:
    all_rankings: dict[str, dict[str, list[str]]] = {}
    hashes: dict[str, str] = {}
    for subset in subsets:
        rows = list(rows_by_subset[subset.name])
        ids = [query.query_id for query in subset.queries]
        if [str(row["query_id"]) for row in rows] != ids:
            raise ValueError(f"ranking artifact query order differs from frozen TEST source: {arm}/{subset.name}")
        rankings = {str(row["query_id"]): [str(item["doc_id"]) for item in row["ranking"]] for row in rows}
        validate_rankings(rankings, ids, {doc.doc_id for doc in subset.documents})
        path = ARTIFACT_ROOT / "rankings" / arm / f"{subset.name}.jsonl"
        if path.exists():
            existing = read_jsonl_records(path)
            if [row.get("query_id") for row in existing] != ids or any(
                row.get("subset") != subset.name or row.get("method") != rows[index].get("method")
                for index, row in enumerate(existing)
            ):
                raise ValueError(f"existing frozen ranking has incompatible identity: {path}")
            existing_rankings = {
                str(row["query_id"]): [str(item["doc_id"]) for item in row["ranking"]]
                for row in existing
            }
            validate_rankings(existing_rankings, ids, {doc.doc_id for doc in subset.documents})
            if any(existing_rankings[qid] != rankings[qid] for qid in ids):
                raise ValueError(f"deterministic replay ranking differs from existing TEST artifact: {path}")
            all_rankings[subset.name] = existing_rankings
            digest = sha256_file(path)
        else:
            digest = _write_jsonl_atomic(path, rows)
            all_rankings[subset.name] = rankings
        hashes[subset.name] = digest
    return all_rankings, hashes


def _query_rows_to_maps(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    metrics: dict[str, dict[str, Any]] = {}
    subsets: dict[str, str] = {}
    for row in rows:
        key = f"{row['subset']}::{row['query_id']}"
        metrics[key] = dict(row)
        subsets[key] = str(row["subset"])
    return metrics, subsets


def _evaluate_and_report(
    *,
    lock: Mapping[str, Any],
    lock_sha: str,
    lock_commit: str,
    inputs,
    rankings: Mapping[str, Mapping[str, Mapping[str, Sequence[str]]]],
    ranking_hashes: Mapping[str, Mapping[str, str]],
    generations: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    generation_manifests: Mapping[str, Mapping[str, Any]],
    source_root: Path,
) -> dict[str, Any]:
    validate_generation_budget(
        [row for subset_rows in generations["lamer"].values() for row in subset_rows],
        [row for subset_rows in generations["compact_crb_q"].values() for row in subset_rows],
    )
    metrics_rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm, by_subset in rankings.items():
        rows, summary = evaluate_rankings("TEST", by_subset, source_root=source_root)
        metrics_rows_by_arm[arm] = rows
        summaries[arm] = summary

    def bootstrap(arm: str, baseline_arm: str) -> dict[str, Any]:
        candidate_map, subset_map = _query_rows_to_maps(metrics_rows_by_arm[arm])
        baseline_map, other_subset_map = _query_rows_to_maps(metrics_rows_by_arm[baseline_arm])
        if subset_map != other_subset_map:
            raise ValueError("paired TEST metric query/subset identity drift")
        return paired_stratified_bootstrap(
            candidate_map,
            baseline_map,
            subset_map,
            metric="ndcg@10",
            resamples=BOOTSTRAP["resamples"],
            seed=BOOTSTRAP["seed"],
        )

    comparisons = {
        "DualSource_vs_BM25": bootstrap("B5_DualSource_RRF_lam0.5", "B0_BM25"),
        "DualSource_vs_ordinary_RRF": bootstrap("B5_DualSource_RRF_lam0.5", "B2_BM25_BGE_RRF"),
        "DualSource_vs_LameR_MV": bootstrap("B5_DualSource_RRF_lam0.5", "B3_LameR_MV"),
        "LameR_MV_vs_ordinary_RRF": bootstrap("B3_LameR_MV", "B2_BM25_BGE_RRF"),
    }
    for comparison in comparisons.values():
        comparison["relative_improvement_percent"] = None
    for key, candidate_arm, baseline_arm in (
        ("DualSource_vs_BM25", "B5_DualSource_RRF_lam0.5", "B0_BM25"),
        ("DualSource_vs_ordinary_RRF", "B5_DualSource_RRF_lam0.5", "B2_BM25_BGE_RRF"),
        ("DualSource_vs_LameR_MV", "B5_DualSource_RRF_lam0.5", "B3_LameR_MV"),
        ("LameR_MV_vs_ordinary_RRF", "B3_LameR_MV", "B2_BM25_BGE_RRF"),
    ):
        baseline = summaries[baseline_arm]["macro_equal_subset_weight"]["ndcg@10"]
        delta = summaries[candidate_arm]["macro_equal_subset_weight"]["ndcg@10"] - baseline
        comparisons[key]["mean_delta"] = delta
        comparisons[key]["relative_improvement_percent"] = 100.0 * delta / baseline if baseline else None

    gates = resume_claim_gates(summaries, comparisons)
    candidate_by_subset: dict[str, dict[str, dict[str, list[str]]]] = {}
    for subset in TEST_COUNTS:
        candidate_by_subset[subset] = {
            "lamer": rankings["B3_LameR_MV"][subset],
            "crb": rankings["B4_Compact_CRB_Q"][subset],
        }
    complementarity = evaluate_candidate_complementarity(
        "TEST", candidate_by_subset, source_root=source_root
    )
    dual_r100 = summaries["B5_DualSource_RRF_lam0.5"]["macro_equal_subset_weight"]["recall@100"]
    candidate_report = {
        **complementarity,
        "dual_source_ranked_recall@100": dual_r100,
        "raw_union_is_candidate_pool_ceiling_not_final_ranked_recall": True,
    }
    candidate_path = ROOT / CANDIDATE_PATH
    if candidate_path.exists():
        if json.loads(candidate_path.read_text(encoding="utf-8")) != candidate_report:
            raise ValueError("existing candidate analysis differs from the frozen ranking/qrels replay")
    else:
        _write_json_atomic(candidate_path, candidate_report)

    generation_summary = {
        "LameR": generation_statistics([row for subset in generations["lamer"].values() for row in subset]),
        "Compact_CRB_Q": generation_statistics([row for subset in generations["compact_crb_q"].values() for row in subset]),
        "paid_api_calls": 0,
        "generation_manifests": {
            key: {
                "path": str(ROOT / "runs/rag_r2med_final_test/generation" / f"{key}_manifest.json"),
                "sha256": sha256_file(ROOT / "runs/rag_r2med_final_test/generation" / f"{key}_manifest.json"),
                "query_count": value["query_count"],
                "artifact_hashes": {row["subset"]: row["artifact_sha256"] for row in value["subsets"]},
            }
            for key, value in generation_manifests.items()
        },
    }
    report = {
        "schema_version": "r2med-final-public-test-report-v1",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "test_status": "PUBLIC_BENCHMARK_REUSED",
        "test_accessed": True,
        "test_executed_once": True,
        "test_config_drift": False,
        "lock": {
            "path": str(ROOT / LOCK_PATH),
            "sha256": lock_sha,
            "lock_commit": lock_commit,
            "implementation_commit": lock["code"]["commit"],
        },
        "source_manifest_sha256": lock["source"]["manifest_sha256"],
        "split_counts": {**TEST_COUNTS, "total": TEST_TOTAL},
        "arms": FROZEN_ARMS,
        "generation": generation_summary,
        "ranking_artifacts": {
            arm: {
                subset: {
                    "path": str(ARTIFACT_ROOT / "rankings" / arm / f"{subset}.jsonl"),
                    "sha256": digest,
                    "query_count": len(rankings[arm][subset]),
                }
                for subset, digest in by_subset.items()
            }
            for arm, by_subset in ranking_hashes.items()
        },
        "metrics": {
            "primary": "equal-subset macro nDCG@10",
            "arm_summaries": summaries,
            "secondary_metrics": ["mrr@10", "recall@5", "recall@10", "recall@50", "recall@100"],
        },
        "paired_bootstrap": {
            "protocol": BOOTSTRAP,
            "interpretation": "exploratory paired uncertainty intervals; no confirmatory significance or multiplicity claim",
            "comparisons": comparisons,
        },
        "candidate_complementarity": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "metrics": candidate_report,
        },
        "gates": gates,
        "claim_boundary": {
            "novel_method_dev_win": False,
            "public_test_can_support_basic_baseline_claim_only_if_gate_passes": True,
            "strongest_gar_claim_only_if_strong_method_gate_passes": True,
            "prohibited": ["SOTA", "clinical superiority", "clinical validation", "unseen test", "untouched holdout"],
        },
        "gold_boundary": "All six rankings were frozen and hashed before this report's first qrels access.",
    }
    report_path = ROOT / REPORT_PATH
    if report_path.exists():
        raise FileExistsError(f"final TEST report already exists: {report_path}")
    _write_json_atomic(report_path, report)
    marker_path = ROOT / START_MARKER_PATH
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["status"] = "COMPLETE"
    marker["completed_at_utc"] = datetime.now(UTC).isoformat()
    marker["report_sha256"] = sha256_file(report_path)
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_final_test(
    *,
    source_root: Path,
    bge_root: Path,
    upstream_root: Path,
    server_executable: Path,
    port: int = PORT,
) -> dict[str, Any]:
    lock = read_lock(ROOT / LOCK_PATH)
    validate_final_lock(lock)
    lock_commit = _verify_committed_lock(lock)
    lock_sha = sha256_file(ROOT / LOCK_PATH)
    load_source_manifest(SOURCE_MANIFEST_PATH)
    if sha256_file(SOURCE_MANIFEST_PATH) != lock["source"]["manifest_sha256"]:
        raise ValueError("source manifest changed after TEST lock")
    verified = verify_models(bge_root=bge_root)
    if verified["qwen"]["verified_sha256"] != lock["models"]["generator"]["sha256"]:
        raise ValueError("Qwen model differs from TEST lock")
    if verified["bge_large"]["weights_sha256"] != lock["models"]["dense_retriever"]["weights_sha256"]:
        raise ValueError("BGE model differs from TEST lock")
    if not server_executable.is_file():
        raise FileNotFoundError(f"llama.cpp server executable not found: {server_executable}")
    marker = _open_or_resume_marker(lock_sha, lock_commit)
    marker["test_accessed"] = True
    marker["test_inputs_started_at_utc"] = datetime.now(UTC).isoformat()
    (ROOT / START_MARKER_PATH).write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # The one-shot marker is durable before this first TEST query/corpus load.
    source_entries = _source_entries()
    inputs = load_partition_inputs("TEST", source_root=source_root, source_manifest_path=SOURCE_MANIFEST_PATH)
    if {subset.name: len(subset.queries) for subset in inputs} != TEST_COUNTS:
        raise ValueError("loaded TEST query counts differ from the committed final lock")
    for subset in inputs:
        entry = source_entries[subset.name]
        locked = lock["source"]["test_query_corpus_identity"][subset.name]
        if (
            len(subset.documents) != locked["corpus_document_count"]
            or len(subset.queries) != locked["query_count"]
            or entry["files"]["query.jsonl"]["sha256"] != locked["query_sha256"]
            or entry["files"]["corpus.jsonl"]["sha256"] != locked["corpus_sha256"]
        ):
            raise ValueError(f"loaded TEST source identity differs from final lock: {subset.name}")
    prompt_catalog = load_upstream_prompt_catalog(upstream_root)
    os.environ["HF_HOME"] = str(MODEL_CACHE)
    os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE / "transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(bge_root), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    model.eval()
    model_sha = verified["bge_large"]["weights_sha256"]

    base_rankings: dict[str, dict[str, dict[str, list[RankedDocument]]]] = {
        "B0_BM25": {},
        "B1_BGE_large": {},
        "B2_BM25_BGE_RRF": {},
    }
    indexes: dict[str, LuceneBM25Index] = {}
    corpus_vectors_by_subset = {}
    dense_doc_ids_by_subset = {}
    for subset in inputs:
        ids = [query.query_id for query in subset.queries]
        bm25 = LuceneBM25Index([(document.doc_id, document.text) for document in subset.documents])
        indexes[subset.name] = bm25
        bm25_rows = {query.query_id: bm25.search(query.text, top_k=100) for query in subset.queries}
        corpus_vectors, _ = _load_or_encode_test_corpus(model, subset, model_sha)
        corpus_vectors_by_subset[subset.name] = corpus_vectors
        dense_doc_ids = [document.doc_id for document in subset.dense_documents]
        dense_doc_ids_by_subset[subset.name] = dense_doc_ids
        original_query_vectors = encode_bge(
            model,
            [BGE_QUERY_PREFIX + query.text for query in subset.queries],
            batch_size=32,
            show_progress_bar=False,
        )
        bge_rows_list = dense_search_many(
            original_query_vectors,
            corpus_vectors,
            dense_doc_ids,
            top_k=100,
            device="cuda",
        )
        bge_rows = {query_id: rows for query_id, rows in zip(ids, bge_rows_list, strict=True)}
        base_rankings["B0_BM25"][subset.name] = bm25_rows
        base_rankings["B1_BGE_large"][subset.name] = bge_rows
        base_rankings["B2_BM25_BGE_RRF"][subset.name] = {
            query.query_id: [
                RankedDocument(item.doc_id, item.score)
                for item in fuse_dual_source_rrf(
                    [row.doc_id for row in bm25_rows[query.query_id]],
                    [row.doc_id for row in bge_rows[query.query_id]],
                    lamer_weight=1.0,
                    crb_weight=1.0,
                    k=60,
                )[:100]
            ]
            for query in subset.queries
        }
        print(f"{subset.name}: frozen BM25/BGE/RRF retrieval complete ({len(ids)} queries)", flush=True)

    actual_llama_version = _llama_version(server_executable)
    if actual_llama_version != lock["generator"].get("llama_cpp_version"):
        raise ValueError("llama.cpp version differs from the final TEST lock")
    generated, generation_manifests = _generation_artifacts(
        inputs,
        source_entries,
        prompt_catalog,
        base_rankings["B0_BM25"],
        server_executable=server_executable,
        port=port,
        expected_lamer_prompt_sha=lock["prompts"]["lamer_upstream_template_sha256_by_subset"],
        expected_compact_prompt_sha=lock["prompts"]["compact_crb_template_sha256"],
        expected_compact_schema_sha=lock["prompts"]["compact_schema_sha256"],
    )

    ranked_rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: {subset.name: [] for subset in inputs}
        for arm in FROZEN_ARMS
    }
    for subset in inputs:
        name = subset.name
        queries = list(subset.queries)
        ids = [query.query_id for query in queries]
        q_by_id = {query.query_id: query.text for query in queries}
        corpus_vectors = corpus_vectors_by_subset[name]
        doc_ids = dense_doc_ids_by_subset[name]
        bge_original = base_rankings["B1_BGE_large"][name]
        original_vectors = encode_bge(
            model,
            [BGE_QUERY_PREFIX + q_by_id[query_id] for query_id in ids],
            batch_size=32,
            show_progress_bar=False,
        )
        lamer_generated = [row["generated_text"] for row in generated["lamer"][name]]
        lamer_vectors = encode_bge(
            model,
            [BGE_QUERY_PREFIX + text for text in lamer_generated],
            batch_size=32,
            show_progress_bar=False,
        )
        lamer_avg_vectors = (original_vectors + lamer_vectors) / 2.0
        lamer_dense_rows = dense_search_many(
            lamer_avg_vectors, corpus_vectors, doc_ids, top_k=100, device="cuda"
        )
        compact_structured = [row["structured"] for row in generated["compact_crb_q"][name]]
        compact_pseudo_vectors = encode_bge(
            model,
            [BGE_QUERY_PREFIX + crb_dense_text(value) for value in compact_structured],
            batch_size=32,
            show_progress_bar=False,
        )
        compact_dense_rows = dense_search_many(
            compact_pseudo_vectors, corpus_vectors, doc_ids, top_k=100, device="cuda"
        )
        lamer_mv_by_id: dict[str, list[RankedDocument]] = {}
        compact_by_id: dict[str, list[RankedDocument]] = {}
        for i, query_id in enumerate(ids):
            lamer_text = bm25_query_text("lamer", q_by_id[query_id], lamer_generated[i])
            compact_text = crb_lexical_query(q_by_id[query_id], compact_structured[i])
            lamer_bridge = indexes[name].search(lamer_text, top_k=100)
            compact_bridge = indexes[name].search(compact_text, top_k=100)
            lamer_mv_by_id[query_id] = weighted_rrf(
                make_four_channels(
                    bm25_original=base_rankings["B0_BM25"][name][query_id],
                    bm25_bridge=lamer_bridge,
                    bge_original=bge_original[query_id],
                    bge_generated=lamer_dense_rows[i],
                ),
                rrf_k=20,
                weights=[1, 2, 1, 2],
                top_k=100,
            )
            compact_by_id[query_id] = weighted_rrf(
                make_four_channels(
                    bm25_original=base_rankings["B0_BM25"][name][query_id],
                    bm25_bridge=compact_bridge,
                    bge_original=bge_original[query_id],
                    bge_generated=compact_dense_rows[i],
                ),
                rrf_k=20,
                weights=[1, 1, 1, 1],
                top_k=100,
            )

        for query in queries:
            qid = query.query_id
            ranked_rows_by_arm["B0_BM25"][name].append(_ranking_row(name, qid, "bm25", base_rankings["B0_BM25"][name][qid]))
            ranked_rows_by_arm["B1_BGE_large"][name].append(_ranking_row(name, qid, "bge_large", bge_original[qid]))
            ranked_rows_by_arm["B2_BM25_BGE_RRF"][name].append(_ranking_row(name, qid, "bm25_bge_rrf", base_rankings["B2_BM25_BGE_RRF"][name][qid]))
            ranked_rows_by_arm["B3_LameR_MV"][name].append(_ranking_row(name, qid, "lamer_mv", lamer_mv_by_id[qid]))
            ranked_rows_by_arm["B4_Compact_CRB_Q"][name].append(_ranking_row(name, qid, "compact_crb_q_mv", compact_by_id[qid]))
            dual = fuse_dual_source_rrf(
                [row.doc_id for row in lamer_mv_by_id[qid]],
                [row.doc_id for row in compact_by_id[qid]],
                lamer_weight=1.0,
                crb_weight=0.5,
                k=60,
            )
            ranked_rows_by_arm["B5_DualSource_RRF_lam0.5"][name].append(
                _ranking_row(name, qid, "dual_source_rrf_lam0.5", [RankedDocument(row.doc_id, row.score) for row in dual])
            )
        print(f"{name}: all frozen ranking channels constructed", flush=True)

    all_rankings: dict[str, dict[str, dict[str, list[str]]]] = {}
    ranking_hashes: dict[str, dict[str, str]] = {}
    for arm in FROZEN_ARMS:
        all_rankings[arm], ranking_hashes[arm] = _save_arm_rankings(
            arm, inputs, ranked_rows_by_arm[arm]
        )
    total_ranked_queries = sum(len(subset) for arm in all_rankings.values() for subset in arm.values())
    if total_ranked_queries != TEST_TOTAL * len(FROZEN_ARMS):
        raise ValueError("not all six arms have exactly 303 frozen rankings")
    ranking_freeze = {
        "schema_version": "r2med-final-test-rankings-frozen-v1",
        "test_status": "PUBLIC_BENCHMARK_REUSED",
        "lock_sha256": lock_sha,
        "six_arms": list(FROZEN_ARMS),
        "query_count_per_arm": TEST_TOTAL,
        "ranking_hashes": ranking_hashes,
        "qrels_opened": False,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
    }
    ranking_freeze_path = ARTIFACT_ROOT / "rankings_frozen_manifest.json"
    if ranking_freeze_path.exists():
        existing_freeze = json.loads(ranking_freeze_path.read_text(encoding="utf-8"))
        stable_keys = set(ranking_freeze) - {"frozen_at_utc"}
        if any(existing_freeze.get(key) != ranking_freeze.get(key) for key in stable_keys):
            raise ValueError("existing ranking freeze manifest differs from current locked ranking identities")
        ranking_freeze = existing_freeze
    else:
        _write_json_atomic(ranking_freeze_path, ranking_freeze)

    # No relevance information is read before all six ranking files and their hashes are frozen.
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    report = _evaluate_and_report(
        lock=lock,
        lock_sha=lock_sha,
        lock_commit=lock_commit,
        inputs=inputs,
        rankings=all_rankings,
        ranking_hashes=ranking_hashes,
        generations=generated,
        generation_manifests=generation_manifests,
        source_root=source_root,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--bge-root", type=Path, default=DEFAULT_BGE_ROOT)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--llama-server", type=Path, default=Path(DEFAULT_SERVER))
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    result = run_final_test(
        source_root=args.source_root,
        bge_root=args.bge_root,
        upstream_root=args.upstream,
        server_executable=args.llama_server,
        port=args.port,
    )
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
