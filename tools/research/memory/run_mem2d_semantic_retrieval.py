"""Frozen-ten MEM-2D semantic retrieval-only ablation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ROOT / "tools" / "research" / "memory"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_mem1d3_reader as reader_runtime
import run_mem2a_m10_base as mem2a
import run_mem2b_rank_aware_projection as mem2b
import run_mem2c_rawspan as mem2c
from final_reader_contract import build_reader_messages, load_final_reader_contract
from local_qwen3_embedding import (
    BATCH_SIZE,
    DEVICE,
    DIMENSIONS,
    MAX_BATCH_TOKENS,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_TREE_SHA256,
    QUERY_INSTRUCTION,
    WEIGHTS_SHA256,
    LocalQwen3Embedding,
    retrieval_document_text,
    sha256_file,
)
from mem1_artifacts import read_jsonl
from mem2a_m10_base import parse_longmemeval_timestamp

from health_ai_copilot.runtime.context_manager import (
    ContextItemCategory,
    ContextManager,
    DeterministicTokenEstimator,
)
from health_ai_copilot.runtime.memory import InMemoryMemoryStore, MemoryQuery

BASE_COMMIT = "a6f30571d6284eeb6a564406224bb2c66b9d3ef1"
RUN_ID = "mem2d-semantic-retrieval-10-20260928"
QUESTION_IDS = tuple(mem2c.QUESTION_IDS)
MEM2C_DIR = mem2c.RUN_DIR
RUN_DIR = ROOT / "runs" / "memory" / "mem2" / RUN_ID
MEM2C_INVENTORY_PATH = MEM2C_DIR / "memory_inventory.jsonl"
MEM2C_INVENTORY_SHA256 = "93414251555c003e3acaa51968f5cf07a85ba3a189743a66301dd20ed7e09c26"
MEM2C_RETRIEVAL_PATH = MEM2C_DIR / "retrieval_results.jsonl"
MEM2C_BUNDLES_PATH = MEM2C_DIR / "context_bundles.jsonl"
MEM2C_PLANS_PATH = MEM2C_DIR / "context_plans.jsonl"
MEM2C_PREDICTIONS_PATH = MEM2C_DIR / "predictions.jsonl"
MEM2C_MANIFEST_PATH = MEM2C_DIR / "run_manifest.json"
VIEW_CONTRACT_PATH = ROOT / "docs/research/memory/rawspan_retrieval_view_contract.json"
RRF_CONTRACT_PATH = ROOT / "docs/research/memory/rawspan_rrf_contract.json"
PROTOCOL_PATH = ROOT / "docs/research/memory/mem_2d_semantic_retrieval.md"
CASE_REVIEW_PATH = ROOT / "docs/research/memory/mem_2d_case_review.json"
CONTRACT_VIEW_SHA256 = ""
CONTRACT_RRF_SHA256 = ""
READER_ENDPOINT = mem2c.READER_ENDPOINT
READER_MODEL = mem2c.READER_MODEL
OUTPUT_RESERVE = mem2c.OUTPUT_RESERVE
MEMORY_BUDGET = mem2c.MEMORY_BUDGET
FINAL_READER_SHA256 = mem2c.PINNED_READER_CONTRACT_SHA256
PROJECTION_SHA256 = mem2c.PINNED_PROJECTION_CONTRACT_SHA256
LEXICAL_DEPTH = 50
DENSE_DEPTH = 50
RRF_K = 60
FINAL_TOP_K = 8
QUERY_INSTRUCTION_SHA256 = hashlib.sha256(QUERY_INSTRUCTION.encode("utf-8")).hexdigest()

FOCUS_CASES = {
    "778164c6": "semantic_factual_miss_caribbean_snapper_fruit",
    "gpt4_e061b84g": "relative_time_sports_event_two_weeks_ago",
    "1cea1afa": "knowledge_update_instagram_followers",
    "c4ea545c": "knowledge_update_gym_frequency",
    "1c549ce4": "multi_session_car_cover_detailing_spray",
    "fca70973": "preference_theme_park",
    "06878be2": "preference_photography_accessories",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _freeze(path: Path) -> str:
    digest = sha256_file(path)
    _sidecar_path(path).write_text(f"{digest}  {path.name}\n", encoding="ascii", newline="\n")
    return digest


def _verify_sidecar(path: Path) -> bool:
    sidecar = _sidecar_path(path)
    if not path.is_file() or not sidecar.is_file():
        return False
    return sidecar.read_text(encoding="ascii").strip().split() == [
        sha256_file(path),
        path.name,
    ]


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return _freeze(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_jsonl_bytes(rows))
    os.replace(temporary, path)
    return _freeze(path)


def _write_or_verify_json(path: Path, value: dict[str, Any]) -> str:
    expected = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if not _verify_sidecar(path) or path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Frozen MEM-2D artifact differs: {path.name}")
        return sha256_file(path)
    return _write_json_atomic(path, value)


def _write_or_verify_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    expected = _jsonl_bytes(rows)
    if path.exists():
        if not _verify_sidecar(path) or path.read_bytes() != expected:
            raise RuntimeError(f"Frozen MEM-2D artifact differs: {path.name}")
        return sha256_file(path)
    return _write_jsonl_atomic(path, rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_contract(path: Path, expected: dict[str, Any]) -> str:
    if not path.is_file():
        raise RuntimeError(f"Frozen MEM-2D contract missing: {path}")
    actual = _load_json(path)
    if actual != expected:
        raise RuntimeError(f"MEM-2D contract does not match its frozen protocol: {path.name}")
    if _sidecar_path(path).exists() and not _verify_sidecar(path):
        raise RuntimeError(f"Frozen MEM-2D contract sidecar is invalid: {path.name}")
    if not _verify_sidecar(path):
        return _freeze(path)
    return sha256_file(path)


def retrieval_view_contract() -> dict[str, Any]:
    return {
        "contract_id": "rawspan-retrieval-view-v1",
        "source": "src/health_ai_copilot/runtime/memory.py::_MemoryStoreCore._matches",
        "value_template": 'f"{record.key} {record.value if isinstance(record.value, str) else json.dumps(record.value, ensure_ascii=False)}"',
        "encoding": "UTF-8",
        "json_dumps": {
            "ensure_ascii": False,
            "sort_keys": False,
            "separators": "Python json.dumps defaults",
        },
        "fields_included": [
            "opaque raw-span key",
            "role",
            "session_date",
            "source_turn_index",
            "source_span_index",
            "content",
        ],
        "content_only_view": False,
        "known_limitation": "The frozen M10 lexical representation includes structural metadata and the opaque raw-span key. MEM-2D preserves it byte-for-byte; this is not claimed to be the optimal semantic retrieval view.",
        "scope": "Retrieval-only view over the frozen MEM-2C RawSpan inventory; no memory representation or projection change.",
    }


def rrf_contract() -> dict[str, Any]:
    return {
        "policy_id": "rawspan-lexical-dense-rrf-v1",
        "protocol_selected_before_result_inspection": True,
        "selection_marker": "RRF_K_SELECTED_BY_PROTOCOL_NOT_DEV_SCORE=YES",
        "lexical_source_depth": 50,
        "dense_source_depth": 50,
        "rank_constant_k": 60,
        "lexical_weight": 1,
        "dense_weight": 1,
        "final_top_k": 8,
        "score": "I[d in lexical50]/(60+rank_lexical(d)) + I[d in dense50]/(60+rank_dense(d))",
        "score_normalization": False,
        "learned_weights": False,
        "tie_break": ["rrf_score_descending", "memory_id_ascending"],
        "scope": "Retrieval-only fusion over frozen RawSpan records; no tuning on the frozen ten cases.",
    }


def _pre_reader_artifact_paths() -> dict[str, Path]:
    return {
        "embedding_manifest.json": RUN_DIR / "embedding_manifest.json",
        "lexical_parity.json": RUN_DIR / "lexical_parity.json",
        "lexical_top50.jsonl": RUN_DIR / "lexical_top50.jsonl",
        "dense_top50.jsonl": RUN_DIR / "dense_top50.jsonl",
        "rrf_top8.jsonl": RUN_DIR / "rrf_top8.jsonl",
        "dense_context_plans.jsonl": RUN_DIR / "dense_context_plans.jsonl",
        "dense_context_bundles.jsonl": RUN_DIR / "dense_context_bundles.jsonl",
        "hybrid_context_plans.jsonl": RUN_DIR / "hybrid_context_plans.jsonl",
        "hybrid_context_bundles.jsonl": RUN_DIR / "hybrid_context_bundles.jsonl",
        "retrieval_diagnostics_pre_reader.json": RUN_DIR / "retrieval_diagnostics_pre_reader.json",
    }


def _verify_upstream() -> dict[str, Any]:
    existing_manifest = RUN_DIR / "run_manifest.json"
    if existing_manifest.exists():
        old = _load_json(existing_manifest)
        if old.get("base_commit_sha") != BASE_COMMIT:
            raise RuntimeError("Existing MEM-2D run binds a different base commit")
    elif _git_head() != BASE_COMMIT:
        raise RuntimeError(f"MEM-2D must start on frozen base {BASE_COMMIT}, got {_git_head()}")

    upstream = mem2c._verify_upstream()
    if upstream["reader_contract_sha256"] != FINAL_READER_SHA256:
        raise RuntimeError("Frozen MEM-1D4 final reader contract does not match MEM-2D")
    m2b_manifest = upstream["m2b_manifest"]
    m2b_report = mem2b.RUN_DIR / "report.md"
    if (
        m2b_manifest.get("status") != "COMPLETE"
        or m2b_manifest.get("gate", {}).get("passed") is not True
        or m2b_manifest.get("projection_contract_sha256") != PROJECTION_SHA256
        or "MEM2B_RANK_AWARE_PROJECTION_FROZEN_10_DIAGNOSTIC=YES"
        not in m2b_report.read_text(encoding="utf-8")
    ):
        raise RuntimeError("MEM-2B rank-aware projection frozen-ten gate failed")

    m2c = _load_json(MEM2C_MANIFEST_PATH)
    if (
        not mem2c._verify_sidecar(MEM2C_MANIFEST_PATH)
        or m2c.get("status") != "COMPLETE"
        or m2c.get("gate", {}).get("passed") is not True
        or m2c.get("completion_gate_marker") != "MEM2C_RAWSPAN_FROZEN_10_DIAGNOSTIC=YES"
        or m2c.get("question_ids") != list(QUESTION_IDS)
        or m2c.get("representation_version") != "raw-span-v1"
        or m2c.get("segmenter_version") != "raw-span-segmenter-v1"
        or m2c.get("projection_contract_sha256") != PROJECTION_SHA256
        or m2c.get("final_reader_contract_sha256") != FINAL_READER_SHA256
        or m2c.get("artifacts_sha256", {}).get("memory_inventory.jsonl") != MEM2C_INVENTORY_SHA256
        or sha256_file(MEM2C_INVENTORY_PATH) != MEM2C_INVENTORY_SHA256
        or not mem2c._verify_sidecar(MEM2C_INVENTORY_PATH)
    ):
        raise RuntimeError("Frozen MEM-2C inventory or completion gate failed")

    view_sha = _verify_contract(VIEW_CONTRACT_PATH, retrieval_view_contract())
    rrf_sha = _verify_contract(RRF_CONTRACT_PATH, rrf_contract())
    return {
        "base_commit_sha": BASE_COMMIT,
        "question_ids": list(QUESTION_IDS),
        "mem1d4_final_reader_contract_sha256": FINAL_READER_SHA256,
        "mem2b_run_manifest_sha256": sha256_file(mem2b.RUN_DIR / "run_manifest.json"),
        "mem2b_projection_contract_sha256": PROJECTION_SHA256,
        "mem2c_run_manifest_sha256": sha256_file(MEM2C_MANIFEST_PATH),
        "mem2c_inventory_path": MEM2C_INVENTORY_PATH.relative_to(ROOT).as_posix(),
        "mem2c_inventory_sha256": MEM2C_INVENTORY_SHA256,
        "mem2c_retrieval_sha256": sha256_file(MEM2C_RETRIEVAL_PATH),
        "mem2c_context_bundles_sha256": sha256_file(MEM2C_BUNDLES_PATH),
        "mem2c_context_plans_sha256": sha256_file(MEM2C_PLANS_PATH),
        "mem2c_predictions_sha256": sha256_file(MEM2C_PREDICTIONS_PATH),
        "rawspan_segmenter_contract_sha256": mem2c._sha256_file(mem2c.SEGMENTER_CONTRACT_PATH),
        "retrieval_view_contract_sha256": view_sha,
        "rrf_contract_sha256": rrf_sha,
        "dataset_sha256": mem2c.PINNED_DATASET_SHA256,
        "selection_sha256": mem2c.PINNED_SELECTION_SHA256,
        "reader_runtime_source": upstream["reader_runtime_source"],
        "m2a_manifest": upstream["m2a_manifest"],
    }


def _source_hashes() -> dict[str, str]:
    return {
        "memory_store": sha256_file(ROOT / "src/health_ai_copilot/runtime/memory.py"),
        "context_manager": sha256_file(ROOT / "src/health_ai_copilot/runtime/context_manager.py"),
        "m10_base_runner": sha256_file(TOOLS_DIR / "mem2a_m10_base.py"),
        "rawspan_adapter": sha256_file(TOOLS_DIR / "raw_span_memory.py"),
        "runner": sha256_file(Path(__file__).resolve()),
        "embedding_adapter": sha256_file(TOOLS_DIR / "local_qwen3_embedding.py"),
        "reader_contract_loader": sha256_file(TOOLS_DIR / "final_reader_contract.py"),
    }


def _load_frozen_inputs() -> dict[str, Any]:
    inventory_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    memory_by_question: dict[str, dict[str, Any]] = defaultdict(dict)
    docs_by_sha: dict[str, str] = {}
    document_sha_by_memory_id: dict[str, str] = {}
    ordered_records: list[dict[str, Any]] = []
    for row in _read_jsonl(MEM2C_INVENTORY_PATH):
        question_id = row.get("question_id")
        if question_id not in QUESTION_IDS:
            raise RuntimeError("Frozen MEM-2C inventory contains an out-of-selection question")
        if row.get("scope_id") != f"longmemeval:{question_id}" or row.get("status") != "active":
            raise RuntimeError(
                "Frozen MEM-2C inventory violates its question scope or active state"
            )
        record = mem2c._record_from_inventory(row)
        text = retrieval_document_text(record.key, record.value)
        text_sha = _sha256_bytes(text.encode("utf-8"))
        prior_text = docs_by_sha.setdefault(text_sha, text)
        if prior_text != text:
            raise RuntimeError("Retrieval-document SHA collision detected")
        if record.memory_id in memory_by_question[question_id]:
            raise RuntimeError("Duplicate RawSpan memory ID in frozen inventory")
        inventory_by_question[question_id].append(row)
        memory_by_question[question_id][record.memory_id] = {
            "record": record,
            "inventory": row,
            "retrieval_document_text": text,
            "retrieval_document_sha256": text_sha,
        }
        document_sha_by_memory_id[record.memory_id] = text_sha
        ordered_records.append(row)
    if len(ordered_records) != 52703 or set(inventory_by_question) != set(QUESTION_IDS):
        raise RuntimeError("Frozen MEM-2C inventory count/question coverage differs from 52,703/10")

    retrieval_rows = {row["question_id"]: row for row in _read_jsonl(MEM2C_RETRIEVAL_PATH)}
    plan_rows = {row["question_id"]: row for row in _read_jsonl(MEM2C_PLANS_PATH)}
    bundle_rows = {row["question_id"]: row for row in _read_jsonl(MEM2C_BUNDLES_PATH)}
    prediction_rows = {row["question_id"]: row for row in _read_jsonl(MEM2C_PREDICTIONS_PATH)}
    for name, mapping in (
        ("retrieval", retrieval_rows),
        ("plan", plan_rows),
        ("bundle", bundle_rows),
        ("prediction", prediction_rows),
    ):
        if list(mapping) != list(QUESTION_IDS):
            raise RuntimeError(f"Frozen MEM-2C {name} artifact does not cover the ten IDs in order")
    questions = {}
    for question_id in QUESTION_IDS:
        old = prediction_rows[question_id]
        retrieval = retrieval_rows[question_id]
        if "question" not in old or "question_date" not in old:
            raise RuntimeError(f"Frozen MEM-2C question/date missing for {question_id}")
        if retrieval.get("query") != old["question"]:
            raise RuntimeError(
                f"Frozen MEM-2C query differs from its saved question: {question_id}"
            )
        if parse_longmemeval_timestamp(old["question_date"]) != retrieval.get("query_now"):
            raise RuntimeError(f"Frozen MEM-2C question-date anchor mismatch: {question_id}")
        if "answer_session_ids" in old or "question_type" in old or "has_answer" in old:
            raise RuntimeError("MEM-2D pre-reader phase must not load benchmark labels")
        questions[question_id] = {
            "question_id": question_id,
            "question": old["question"],
            "question_date": old["question_date"],
            "query_now": retrieval["query_now"],
            "scope_id": f"longmemeval:{question_id}",
            "session_revision": plan_rows[question_id]["plan"]["session_revision"],
        }
    return {
        "inventory_by_question": dict(inventory_by_question),
        "memory_by_question": dict(memory_by_question),
        "docs_by_sha": docs_by_sha,
        "document_sha_by_memory_id": document_sha_by_memory_id,
        "ordered_records": ordered_records,
        "retrieval_by_question": retrieval_rows,
        "plans_by_question": plan_rows,
        "bundles_by_question": bundle_rows,
        "predictions_by_question": prediction_rows,
        "questions": questions,
    }


def _ordered_corpus_root(records: list[dict[str, Any]], doc_sha_by_id: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda item: item["memory_id"]):
        digest.update(row["memory_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(doc_sha_by_id[row["memory_id"]].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _vector_sha(vector: Any) -> str:
    import numpy as np

    stable = np.asarray(vector, dtype=np.dtype("<f4"), order="C")
    return _sha256_bytes(stable.tobytes(order="C"))


def _vector_pair_root(doc_shas: list[str], vector_shas: list[str]) -> str:
    pairs = sorted(zip(doc_shas, vector_shas, strict=True))
    digest = hashlib.sha256()
    for doc_sha, vector_sha in pairs:
        digest.update(doc_sha.encode("ascii"))
        digest.update(vector_sha.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _embedding_cache_identity(
    *,
    upstream: dict[str, Any],
    unique_doc_shas: list[str],
    ordered_corpus_root: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    body = {
        "mem2c_inventory_sha256": MEM2C_INVENTORY_SHA256,
        "rawspan_segmenter_contract_sha256": upstream["rawspan_segmenter_contract_sha256"],
        "retrieval_view_contract_sha256": upstream["retrieval_view_contract_sha256"],
        "embedding_model_id": MODEL_ID,
        "embedding_model_revision": MODEL_REVISION,
        "embedding_model_tree_sha256": MODEL_TREE_SHA256,
        "embedding_weights_sha256": WEIGHTS_SHA256,
        "embedding_adapter_source_sha256": source_hashes["embedding_adapter"],
        "query_instruction_sha256": QUERY_INSTRUCTION_SHA256,
        "embedding_dimensions": DIMENSIONS,
        "embedding_device": DEVICE,
        "embedding_dtype": "float16",
        "embedding_batch_size": BATCH_SIZE,
        "embedding_max_length": MAX_LENGTH,
        "embedding_max_batch_tokens": MAX_BATCH_TOKENS,
        "document_instruction": "none",
        "query_instruction": QUERY_INSTRUCTION,
        "ordered_corpus_identity_root": ordered_corpus_root,
        "unique_retrieval_document_count": len(unique_doc_shas),
        "unique_retrieval_document_sha256_root": _sha256_bytes(
            ("\n".join(sorted(unique_doc_shas)) + "\n").encode("ascii")
        ),
    }
    return {"identity": body, "identity_sha256": _sha256_json(body)}


def _atomic_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _embed_document_corpus(
    *,
    adapter: LocalQwen3Embedding,
    unique_doc_shas: list[str],
    unique_texts: list[str],
    token_counts: list[int],
    cache_identity: dict[str, Any],
    cache_root: Path,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    cache_dir = cache_root / cache_identity["identity_sha256"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity_path = cache_dir / "cache_identity.json"
    vector_path = cache_dir / "document_vectors.f32"
    progress_path = cache_dir / "progress.json"
    if identity_path.exists():
        if _load_json(identity_path) != cache_identity:
            raise RuntimeError("Local embedding cache identity mismatch; refusing cache reuse")
    else:
        if vector_path.exists() or progress_path.exists():
            raise RuntimeError("Unidentified local embedding cache files exist; refusing reuse")
        _write_json_atomic(identity_path, cache_identity)

    expected_bytes = len(unique_texts) * DIMENSIONS * 4
    if vector_path.exists() and vector_path.stat().st_size != expected_bytes:
        raise RuntimeError("Local embedding cache vector-file size is invalid")
    mode = "r+" if vector_path.exists() else "w+"
    vectors = np.memmap(
        vector_path, dtype=np.dtype("<f4"), mode=mode, shape=(len(unique_texts), DIMENSIONS)
    )
    if progress_path.exists():
        progress = _load_json(progress_path)
        if progress.get("cache_identity_sha256") != cache_identity["identity_sha256"]:
            raise RuntimeError("Local embedding cache progress identity mismatch")
        completed = int(progress.get("completed_vectors", -1))
        vector_hashes = list(progress.get("vector_sha256", []))
        if completed < 0 or completed > len(unique_texts) or len(vector_hashes) != completed:
            raise RuntimeError("Local embedding cache progress is malformed")
        for index in range(completed):
            if _vector_sha(vectors[index]) != vector_hashes[index]:
                raise RuntimeError(
                    f"Local embedding cache vector failed hash validation at row {index}"
                )
    else:
        completed = 0
        vector_hashes = []
        _atomic_progress(
            progress_path,
            {
                "cache_identity_sha256": cache_identity["identity_sha256"],
                "completed_vectors": 0,
                "vector_sha256": [],
            },
        )

    cached_at_start = completed
    batch_count = 0
    encoding_ms = 0.0
    resume_at = completed
    remaining_texts = unique_texts[resume_at:]
    remaining_token_counts = token_counts[resume_at:]
    batches = (
        adapter.encode_batches(
            remaining_texts, "document", token_counts=remaining_token_counts
        )
        if remaining_texts
        else ()
    )
    for batch in batches:
        start = resume_at + batch.start
        end = resume_at + batch.end
        if start != completed:
            raise RuntimeError("Embedding cache resumed with a gap in document order")
        vectors[start:end] = batch.vectors
        vectors.flush()
        vector_hashes.extend(_vector_sha(vector) for vector in batch.vectors)
        completed = end
        batch_count += 1
        encoding_ms += batch.latency_ms
        if batch_count % 100 == 0 or completed == len(unique_texts):
            _atomic_progress(
                progress_path,
                {
                    "cache_identity_sha256": cache_identity["identity_sha256"],
                    "completed_vectors": completed,
                    "vector_sha256": vector_hashes,
                },
            )
            print(f"embedded_documents={completed}/{len(unique_texts)}", flush=True)
    if completed != len(unique_texts):
        raise RuntimeError("Local embedding cache did not finish the complete frozen corpus")
    for index, expected in enumerate(vector_hashes):
        if _vector_sha(vectors[index]) != expected:
            raise RuntimeError(f"Local embedding cache failed final row hash validation at {index}")
    pair_root = _vector_pair_root(unique_doc_shas, vector_hashes)
    return vectors, {
        "document_count": len(unique_texts),
        "cache_hits": cached_at_start,
        "cache_misses": len(unique_texts) - cached_at_start,
        "batches_computed": batch_count,
        "encoding_wall_ms": round(encoding_ms, 3),
        "vector_hash_root": pair_root,
        "cache_identity_sha256": cache_identity["identity_sha256"],
        "cache_path": str(cache_dir),
    }


def _runtime_cache_root() -> Path:
    raw = os.environ.get("HC_MEM2_EMBED_CACHE")
    path = (
        Path(raw).expanduser().resolve()
        if raw
        else (ROOT.parent / ".cache" / "health-copilot" / "mem2d-embedding")
    )
    try:
        path.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise RuntimeError("HC_MEM2_EMBED_CACHE must be outside the repository")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rank_dense(
    *,
    question_id: str,
    query_vector: Any,
    eligible_ids: set[str],
    memory_by_id: dict[str, dict[str, Any]],
    doc_vectors: Any,
    doc_index_by_sha: dict[str, int],
    top_k: int,
) -> list[dict[str, Any]]:
    import numpy as np

    candidates = []
    for memory_id in eligible_ids:
        item = memory_by_id[memory_id]
        index = doc_index_by_sha[item["retrieval_document_sha256"]]
        score = float(np.dot(query_vector, doc_vectors[index]))
        candidates.append(
            {
                "memory_id": memory_id,
                "cosine_similarity": score,
                "source_session_id": item["record"].source_session_id,
                "source_turn_index": item["inventory"]["source_turn_index"],
                "source_span_index": item["inventory"]["source_span_index"],
                "source_turn_id": item["inventory"]["parent_turn_key"],
                "source_span_id": item["record"].key,
                "retrieval_document_sha256": item["retrieval_document_sha256"],
            }
        )
    candidates.sort(key=lambda row: (-row["cosine_similarity"], row["memory_id"]))
    return [
        dict(row, question_id=question_id, rank=rank)
        for rank, row in enumerate(candidates[:top_k], 1)
    ]


def _fuse_rrf(lexical: list[dict[str, Any]], dense: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for source_name, rows in (("lexical", lexical), ("dense", dense)):
        for row in rows:
            memory_id = row["memory_id"]
            fused = combined.setdefault(
                memory_id,
                {
                    "memory_id": memory_id,
                    "lexical_rank": None,
                    "lexical_score": None,
                    "dense_rank": None,
                    "cosine_similarity": None,
                    "rrf_score": 0.0,
                    "source_session_id": row["source_session_id"],
                    "source_turn_id": row["source_turn_id"],
                    "source_span_id": row["source_span_id"],
                    "source_turn_index": row["source_turn_index"],
                    "source_span_index": row["source_span_index"],
                    "retrieval_document_sha256": row["retrieval_document_sha256"],
                },
            )
            if source_name == "lexical":
                fused["lexical_rank"] = row["rank"]
                fused["lexical_score"] = row["score"]
                fused["rrf_score"] += 1.0 / (RRF_K + row["rank"])
            else:
                fused["dense_rank"] = row["rank"]
                fused["cosine_similarity"] = row["cosine_similarity"]
                fused["rrf_score"] += 1.0 / (RRF_K + row["rank"])
    ranked = sorted(combined.values(), key=lambda row: (-row["rrf_score"], row["memory_id"]))
    return [dict(row, rank=rank) for rank, row in enumerate(ranked[:FINAL_TOP_K], 1)]


def _lexical_top50_and_parity(
    inputs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    parity_questions = []
    all_matches: dict[str, list[dict[str, Any]]] = {}
    field_pairs = (
        ("memory_id", "memory_id"),
        ("rank", "rank"),
        ("score", "score"),
        ("source_session_id", "source_session_id"),
        ("source_turn_key", "source_turn_id"),
        ("source_span_key", "source_span_id"),
    )
    for question_id in QUESTION_IDS:
        question = inputs["questions"][question_id]
        records = inputs["memory_by_question"][question_id]
        store = InMemoryMemoryStore()
        store._records = {memory_id: item["record"] for memory_id, item in records.items()}
        started = time.perf_counter()
        matches = store.matches(
            MemoryQuery(
                scope_id=question["scope_id"],
                text=question["question"],
                now=question["query_now"],
                top_k=len(records),
            )
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        if len(matches) != len(records):
            raise RuntimeError(
                f"M10 lexical validity filter excluded unexpected RawSpans: {question_id}"
            )
        question_rows = []
        for rank, match in enumerate(matches[:LEXICAL_DEPTH], 1):
            item = records[match.record.memory_id]
            question_rows.append(
                {
                    "question_id": question_id,
                    "rank": rank,
                    "memory_id": match.record.memory_id,
                    "score": match.score,
                    "source_session_id": match.record.source_session_id,
                    "source_turn_id": item["inventory"]["parent_turn_key"],
                    "source_span_id": match.record.key,
                    "source_turn_index": item["inventory"]["source_turn_index"],
                    "source_span_index": item["inventory"]["source_span_index"],
                    "retrieval_document_sha256": item["retrieval_document_sha256"],
                    "valid_from": match.record.valid_from,
                }
            )
        all_matches[question_id] = [
            {"memory_id": match.record.memory_id, "score": match.score, "record": match.record}
            for match in matches
        ]
        prior = inputs["retrieval_by_question"][question_id]["results"]
        exact = len(prior) == FINAL_TOP_K and all(
            all(
                left.get(left_field) == right.get(right_field)
                for left_field, right_field in field_pairs
            )
            for left, right in zip(prior, question_rows[:FINAL_TOP_K], strict=True)
        )
        parity_questions.append(
            {
                "question_id": question_id,
                "expected_top_k": FINAL_TOP_K,
                "matched_fields_per_rank": len(field_pairs),
                "matched_records": len(prior) if exact else 0,
                "exact_top8": exact,
                "lexical_scoring_latency_ms": elapsed_ms,
                "candidate_count_after_m10_scope_time_validity": len(matches),
            }
        )
        rows.extend(question_rows)
    matched_records = sum(row["matched_records"] for row in parity_questions)
    expected_records = len(QUESTION_IDS) * FINAL_TOP_K
    parity = {
        "schema_version": 1,
        "gate": "MEM2D_LEXICAL_TOP8_PARITY_WITH_MEM2C=YES"
        if matched_records == expected_records
        else "NO",
        "exact_field_pairs": [list(pair) for pair in field_pairs],
        "questions": parity_questions,
        "fields_per_record": len(field_pairs),
        "matched_records": matched_records,
        "expected_records": expected_records,
        "all_ten_exact": matched_records == expected_records
        and all(row["exact_top8"] for row in parity_questions),
        "scorer": "M10 _MemoryStoreCore._matches unchanged; all valid candidates, including zero-score matches",
    }
    return rows, parity, all_matches


def _projection_context(
    *,
    question_id: str,
    arm: str,
    question: dict[str, Any],
    ranked_rows: list[dict[str, Any]],
    memory_by_id: dict[str, dict[str, Any]],
    client: httpx.Client,
    reader_contract: tuple[dict[str, Any], str, str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidates = ranked_rows[:FINAL_TOP_K]
    if len(candidates) != FINAL_TOP_K:
        raise RuntimeError(f"{arm} has fewer than eight native retrieval candidates: {question_id}")
    candidate_records = [memory_by_id[row["memory_id"]]["record"] for row in candidates]
    rank_by_context_id = {
        f"memory-{row['memory_id']}": rank for rank, row in enumerate(candidates, 1)
    }
    manager = ContextManager(
        budget=mem2c._memory_budget(),
        estimator=DeterministicTokenEstimator(),
        history_window=24,
    )
    started = time.perf_counter()
    plan = manager.build_plan(
        session_id=question["scope_id"],
        session_revision=question["session_revision"],
        current_user=question["question"],
        history=(),
        memory_records=candidate_records,
        retrieval_query=question["question"],
        selection_rank_hints=rank_by_context_id,
    )
    if plan.memory_tokens > MEMORY_BUDGET:
        raise RuntimeError(f"{arm} projection exceeded the frozen memory budget: {question_id}")
    rank_from_id = {f"memory-{row['memory_id']}": rank for rank, row in enumerate(candidates, 1)}
    selected_items = [item for item in plan.items if item.category == ContextItemCategory.MEMORY]
    selected_ranks = [rank_from_id[item.item_id] for item in selected_items]
    if selected_ranks != sorted(selected_ranks):
        raise RuntimeError(f"{arm} projection changed native rank order: {question_id}")

    context_items: list[dict[str, Any]] = []
    content_only: list[str] = []
    for display_rank, context_item in enumerate(selected_items, 1):
        memory_id = str(context_item.provenance["memory_id"])
        memory = memory_by_id[memory_id]
        inventory = memory["inventory"]
        native_rank = rank_from_id[context_item.item_id]
        result = candidates[native_rank - 1]
        text = mem2a.canonical_json(context_item.content)
        context_items.append(
            {
                "text": text,
                "rank": display_rank,
                "kind": context_item.category.value,
                "source_session_ids": [memory["record"].source_session_id]
                if memory["record"].source_session_id
                else [],
                "memory_id": memory_id,
                "native_retrieval_rank": native_rank,
                "retrieval_score": result.get(
                    "cosine_similarity", result.get("rrf_score", result.get("score"))
                ),
                "status": memory["record"].status.value,
                "version": memory["record"].version,
                "source_turn_id": inventory["parent_turn_key"],
                "source_span_id": memory["record"].key,
                "source_turn_index": inventory["source_turn_index"],
                "source_span_index": inventory["source_span_index"],
                "char_start": inventory["char_start"],
                "char_end": inventory["char_end"],
                "estimated_tokens": context_item.estimated_tokens,
            }
        )
        value = memory["record"].value
        if isinstance(value, dict) and isinstance(value.get("content"), str):
            content_only.append(value["content"])
    serialized_context = "\n\n".join(
        f"[Context item {item['rank']} | {item['kind']}]\n{item['text']}" for item in context_items
    )
    context_reader_tokens = (
        mem2c._local_token_count(client, serialized_context) if serialized_context else 0
    )
    content_only_text = "\n\n".join(content_only)
    content_only_reader_tokens = (
        mem2c._local_token_count(client, content_only_text) if content_only_text else 0
    )
    bundle_base = {
        "schema_version": 3,
        "system": "healthcopilot_m10_rawspan",
        "question_id": question_id,
        "arm": arm,
        "items": [
            {key: value for key, value in item.items() if key != "estimated_tokens"}
            for item in context_items
        ],
        "serialized_context": serialized_context,
        "context_embedding_tokens": None,
        "context_embedding_tokenizer": "NOT_APPLICABLE_NO_EMBEDDING",
        "context_reader_tokens": context_reader_tokens,
        "context_reader_tokenizer": "llama.cpp 10068 Qwen3-8B tokenizer; add_special=false",
        "provenance_available": True,
        "retrieval_latency_ms": None,
        "ingestion_latency_ms": None,
    }
    bundle_sha = _sha256_json(bundle_base)
    bundle = {**bundle_base, "context_bundle_sha256": bundle_sha}
    _contract, contract_sha, system_template, user_template = reader_contract
    if contract_sha != FINAL_READER_SHA256:
        raise RuntimeError("Final reader contract changed during MEM-2D projection")
    messages = build_reader_messages(
        question["question"],
        question["question_date"],
        serialized_context,
        system_template=system_template,
        user_template=user_template,
    )
    prompt_tokens, rendered_prompt_sha = reader_runtime._render_and_tokenize(client, messages)
    if prompt_tokens + OUTPUT_RESERVE > 131072:
        raise RuntimeError(
            f"{arm} shared-reader prompt exceeds the frozen context slot: {question_id}"
        )
    message_sha = _sha256_bytes(mem2a.canonical_json(messages).encode("utf-8"))
    bundle_row = {
        "question_id": question_id,
        "arm": arm,
        "context_bundle": bundle,
        "context_bundle_sha256": bundle_sha,
        "reader_prompt_sha256": message_sha,
        "rendered_prompt_sha256": rendered_prompt_sha,
        "reader_prompt_tokens_preflight": prompt_tokens,
        "max_model_length": 131072,
        "output_reserve": OUTPUT_RESERVE,
        "truncated": False,
        "shared_reader_contract_sha256": FINAL_READER_SHA256,
        "shared_system_message_sha256": _sha256_bytes(system_template.encode("utf-8")),
        "reader_messages": messages,
    }
    rank_plan = {
        "question_id": question_id,
        "arm": arm,
        "memory_budget_tokens": MEMORY_BUDGET,
        "plan": {
            "session_id": plan.session_id,
            "session_revision": plan.session_revision,
            "selected_memory_ids": list(plan.selected_memory_ids),
            "selected_event_ids": list(plan.selected_event_ids),
            "dropped_event_ids": list(plan.dropped_event_ids),
            "estimated_tokens": plan.estimated_tokens,
            "memory_tokens": plan.memory_tokens,
            "history_tokens": plan.history_tokens,
            "retrieval_query": plan.retrieval_query,
            "plan_hash": plan.plan_hash,
        },
        "selected_native_retrieval_ranks": selected_ranks,
        "context_manager_projection_latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    diagnostics = {
        "question_id": question_id,
        "arm": arm,
        "native_candidate_count": len(candidates),
        "selected_item_count": len(context_items),
        "estimated_memory_tokens": plan.memory_tokens,
        "reader_context_tokens": context_reader_tokens,
        "reader_tokens_over_estimated_memory_tokens": (
            context_reader_tokens / plan.memory_tokens if plan.memory_tokens else None
        ),
        "content_only_reader_tokens": content_only_reader_tokens,
        "reader_prompt_tokens_preflight": prompt_tokens,
        "completion_reserve": OUTPUT_RESERVE,
        "truncated": False,
        "context_bundle_sha256": bundle_sha,
    }
    return rank_plan, bundle_row, diagnostics


def _crowding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top = rows[:FINAL_TOP_K]
    session_counts = Counter(str(row["source_session_id"]) for row in top)
    turn_counts = Counter(str(row.get("source_turn_id", row.get("source_turn_key"))) for row in top)
    return {
        "unique_source_sessions_at_8": len(session_counts),
        "unique_parent_turns_at_8": len(turn_counts),
        "maximum_spans_from_one_session": max(session_counts.values(), default=0),
        "maximum_spans_from_one_parent_turn": max(turn_counts.values(), default=0),
        "duplicate_session_fraction": 1 - len(session_counts) / len(top) if top else 0.0,
        "duplicate_parent_turn_fraction": 1 - len(turn_counts) / len(top) if top else 0.0,
        "session_concentration_hhi": sum(
            (count / len(top)) ** 2 for count in session_counts.values()
        )
        if top
        else 0.0,
    }


def _flatten_historical_retrieval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for question in rows:
        results = question.get("results")
        if not isinstance(results, list):
            raise TypeError("Frozen MEM-2C retrieval row lacks its results list")
        flattened.extend({**result, "question_id": question["question_id"]} for result in results)
    return flattened


def _retrieval_identity(
    *, question_id: str, upstream: dict[str, Any], source_hashes: dict[str, str], hybrid: bool
) -> dict[str, Any]:
    body = {
        "mem2c_inventory_sha256": MEM2C_INVENTORY_SHA256,
        "rawspan_segmenter_contract_sha256": upstream["rawspan_segmenter_contract_sha256"],
        "retrieval_view_contract_sha256": upstream["retrieval_view_contract_sha256"],
        "embedding_model_id": MODEL_ID,
        "embedding_model_revision": MODEL_REVISION,
        "embedding_model_tree_sha256": MODEL_TREE_SHA256,
        "embedding_weights_sha256": WEIGHTS_SHA256,
        "embedding_adapter_source_sha256": source_hashes["embedding_adapter"],
        "query_instruction_sha256": QUERY_INSTRUCTION_SHA256,
        "dense_ranking_algorithm": "normalized-float32-dot-desc-memory-id-asc-v1",
        "projection_contract_sha256": PROJECTION_SHA256,
        "final_reader_contract_sha256": FINAL_READER_SHA256,
        "question_id": question_id,
        "arm": "hybrid_rrf" if hybrid else "dense",
    }
    if hybrid:
        body.update(
            {
                "lexical_scorer_source_sha256": source_hashes["memory_store"],
                "lexical_source_depth": LEXICAL_DEPTH,
                "dense_source_depth": DENSE_DEPTH,
                "rrf_contract_sha256": upstream["rrf_contract_sha256"],
            }
        )
    return {"identity": body, "identity_sha256": _sha256_json(body)}


def _prepare() -> dict[str, Any]:
    if (RUN_DIR / "run_manifest.json").is_file():
        existing = _load_json(RUN_DIR / "run_manifest.json")
        if existing.get("status") in {"PRE_READER_FROZEN", "COMPLETE"}:
            frozen = _verify_pre_reader()
            print("MEM2D_SEMANTIC_RETRIEVAL_PRE_READER_FROZEN=YES", flush=True)
            print(f"pre_reader_freeze_sha256={frozen['pre_reader_freeze_sha256']}", flush=True)
            return {"manifest": frozen, "resumed_frozen": True}
    upstream = _verify_upstream()
    source_hashes = _source_hashes()
    inputs = _load_frozen_inputs()
    cache_root = _runtime_cache_root()
    model_path_raw = os.environ.get("HC_LOCAL_EMBEDDING_PATH")
    if not model_path_raw:
        raise RuntimeError(
            "Set HC_LOCAL_EMBEDDING_PATH to the already-frozen local Qwen3 embedding directory"
        )
    model_path = Path(model_path_raw).expanduser().resolve()
    doc_shas = sorted(inputs["docs_by_sha"])
    doc_texts = [inputs["docs_by_sha"][digest] for digest in doc_shas]
    ordered_root = _ordered_corpus_root(
        inputs["ordered_records"], inputs["document_sha_by_memory_id"]
    )
    embedding_identity = _embedding_cache_identity(
        upstream=upstream,
        unique_doc_shas=doc_shas,
        ordered_corpus_root=ordered_root,
        source_hashes=source_hashes,
    )

    lexical_rows, lexical_parity, all_lexical = _lexical_top50_and_parity(inputs)
    if not lexical_parity["all_ten_exact"]:
        _write_json_atomic(RUN_DIR / "lexical_parity.json", lexical_parity)
        raise RuntimeError(
            "MEM2D_LEXICAL_TOP8_PARITY_WITH_MEM2C=NO; stopped before dense embedding"
        )

    memory_by_question = inputs["memory_by_question"]
    memory_by_id = {
        qid: {mid: row for mid, row in values.items()} for qid, values in memory_by_question.items()
    }
    for question_id, matches in all_lexical.items():
        if {row["memory_id"] for row in matches} != set(memory_by_id[question_id]):
            raise RuntimeError("M10 lexical matching changed the frozen valid candidate inventory")

    client_timeout = httpx.Timeout(600.0, connect=10.0)
    lexical_by_question = defaultdict(list)
    for row in lexical_rows:
        lexical_by_question[row["question_id"]].append(row)
    dense_rows: list[dict[str, Any]] = []
    rrf_rows: list[dict[str, Any]] = []
    dense_plans: list[dict[str, Any]] = []
    dense_bundles: list[dict[str, Any]] = []
    hybrid_plans: list[dict[str, Any]] = []
    hybrid_bundles: list[dict[str, Any]] = []
    diagnostic_questions = []
    embedding_stats: dict[str, Any] = {}
    gpu_stats: dict[str, Any] = {}
    reader_contract = load_final_reader_contract()
    if reader_contract[1] != FINAL_READER_SHA256:
        raise RuntimeError("Final reader contract differs from the pinned MEM-1D4 SHA")

    import numpy as np

    with httpx.Client(timeout=client_timeout, trust_env=False) as client:
        if urlsplit(READER_ENDPOINT).hostname != "127.0.0.1":
            raise RuntimeError("Frozen shared-reader endpoint is not loopback")
        frozen_reader_runtime = mem2b._verify_reader(client, upstream["m2a_manifest"])
        adapter = LocalQwen3Embedding(model_path)
        local_identity = adapter.identity
        if local_identity["model_tree_sha256"] != MODEL_TREE_SHA256:
            adapter.close()
            raise RuntimeError(
                "Local embedding tree hash differs from the frozen MEM-1 model identity"
            )
        doc_token_counts = adapter.count_tokens(doc_texts, "document")
        query_texts = [inputs["questions"][qid]["question"] for qid in QUESTION_IDS]
        query_token_counts = adapter.count_tokens(query_texts, "query")
        trunc_doc = sum(count > MAX_LENGTH for count in doc_token_counts)
        trunc_query = sum(count > MAX_LENGTH for count in query_token_counts)
        if trunc_doc or trunc_query:
            adapter.close()
            raise RuntimeError(
                f"Embedding truncation gate failed before inference: docs={trunc_doc}, queries={trunc_query}"
            )
        pre_free, pre_total = adapter.torch.cuda.mem_get_info(adapter.device)
        adapter.torch.cuda.reset_peak_memory_stats(adapter.device)
        doc_vectors = None
        try:
            embedding_started = time.perf_counter()
            doc_vectors, embedding_stats = _embed_document_corpus(
                adapter=adapter,
                unique_doc_shas=doc_shas,
                unique_texts=doc_texts,
                token_counts=doc_token_counts,
                cache_identity=embedding_identity,
                cache_root=cache_root,
            )
            embedding_stats["document_embedding_wall_ms"] = round(
                (time.perf_counter() - embedding_started) * 1000, 3
            )
            query_vectors_batches = list(
                adapter.encode_batches(query_texts, "query", token_counts=query_token_counts)
            )
            query_vectors = np.concatenate(
                [batch.vectors for batch in query_vectors_batches], axis=0
            )
            query_batch_count = len(query_vectors_batches)
            query_embedding_ms = round(sum(batch.latency_ms for batch in query_vectors_batches), 3)
            repeated_query_batches = list(
                adapter.encode_batches(query_texts, "query", token_counts=query_token_counts)
            )
            repeated_query_vectors = np.concatenate(
                [batch.vectors for batch in repeated_query_batches], axis=0
            )
            repeat_vector_max_abs_delta = float(
                np.abs(query_vectors - repeated_query_vectors).max()
            )
            if not np.allclose(
                query_vectors, repeated_query_vectors, atol=1e-6, rtol=1e-6
            ):
                raise RuntimeError("Frozen local embedding repeated-query determinism probe failed")
            probe_eligible_ids = {match["memory_id"] for match in all_lexical[QUESTION_IDS[0]]}
            repeat_rank_a = _rank_dense(
                question_id=QUESTION_IDS[0],
                query_vector=query_vectors[0],
                eligible_ids=probe_eligible_ids,
                memory_by_id=memory_by_id[QUESTION_IDS[0]],
                doc_vectors=doc_vectors,
                doc_index_by_sha={sha: index for index, sha in enumerate(doc_shas)},
                top_k=DENSE_DEPTH,
            )
            repeat_rank_b = _rank_dense(
                question_id=QUESTION_IDS[0],
                query_vector=repeated_query_vectors[0],
                eligible_ids=probe_eligible_ids,
                memory_by_id=memory_by_id[QUESTION_IDS[0]],
                doc_vectors=doc_vectors,
                doc_index_by_sha={sha: index for index, sha in enumerate(doc_shas)},
                top_k=DENSE_DEPTH,
            )
            repeat_score_max_abs_delta = max(
                abs(left["cosine_similarity"] - right["cosine_similarity"])
                for left, right in zip(repeat_rank_a, repeat_rank_b, strict=True)
            )
            if (
                [row["memory_id"] for row in repeat_rank_a]
                != [row["memory_id"] for row in repeat_rank_b]
                or repeat_score_max_abs_delta > 1e-6
            ):
                raise RuntimeError("Frozen local embedding repeated-query rank probe failed")
            similarity_started = time.perf_counter()
            doc_index = {digest: index for index, digest in enumerate(doc_shas)}
            dense_latency: dict[str, float] = {}
            rrf_latency: dict[str, float] = {}
            for q_index, question_id in enumerate(QUESTION_IDS):
                started = time.perf_counter()
                eligible_ids = {match["memory_id"] for match in all_lexical[question_id]}
                dense_identity = _retrieval_identity(
                    question_id=question_id,
                    upstream=upstream,
                    source_hashes=source_hashes,
                    hybrid=False,
                )
                dense = _rank_dense(
                    question_id=question_id,
                    query_vector=query_vectors[q_index],
                    eligible_ids=eligible_ids,
                    memory_by_id=memory_by_id[question_id],
                    doc_vectors=doc_vectors,
                    doc_index_by_sha=doc_index,
                    top_k=DENSE_DEPTH,
                )
                for row in dense:
                    row["retrieval_identity_sha256"] = dense_identity["identity_sha256"]
                dense_latency[question_id] = round((time.perf_counter() - started) * 1000, 3)
                dense_rows.extend(dense)
                rrf_started = time.perf_counter()
                hybrid = _fuse_rrf(lexical_by_question[question_id], dense)
                rrf_latency[question_id] = round((time.perf_counter() - rrf_started) * 1000, 3)
                hybrid_identity = _retrieval_identity(
                    question_id=question_id,
                    upstream=upstream,
                    source_hashes=source_hashes,
                    hybrid=True,
                )
                for row in hybrid:
                    row["question_id"] = question_id
                    row["retrieval_identity_sha256"] = hybrid_identity["identity_sha256"]
                rrf_rows.extend(hybrid)
            dense_similarity_wall_ms = round((time.perf_counter() - similarity_started) * 1000, 3)
            embedding_input_tokens = sum(doc_token_counts)
            query_input_tokens = sum(query_token_counts)
            gpu_stats = {
                "device_name": adapter.device_name,
                "free_bytes_before_embedding": int(pre_free),
                "total_bytes": int(pre_total),
                "peak_allocated_bytes": int(
                    adapter.torch.cuda.max_memory_allocated(adapter.device)
                ),
                "peak_reserved_bytes": int(adapter.torch.cuda.max_memory_reserved(adapter.device)),
            }
            adapter.torch.cuda.synchronize(adapter.device)
            adapter.close()
            del query_vectors_batches, query_vectors, repeated_query_batches
            del repeated_query_vectors
            gc.collect()
            adapter.torch.cuda.empty_cache()
            embedding_stats.update(
                {
                    "document_input_tokens": embedding_input_tokens,
                    "query_input_tokens": query_input_tokens,
                    "document_token_count_min": min(doc_token_counts),
                    "document_token_count_max": max(doc_token_counts),
                    "query_token_counts": dict(zip(QUESTION_IDS, query_token_counts, strict=True)),
                    "truncated_document_count": trunc_doc,
                    "truncated_query_count": trunc_query,
                    "query_batches": query_batch_count,
                    "query_encoding_wall_ms": query_embedding_ms,
                    "repeat_query_determinism_probe": {
                        "count": 1,
                        "reencoded_same_frozen_query_batches": True,
                        "vector_allclose_atol_rtol": 1e-6,
                        "vector_max_absolute_difference": repeat_vector_max_abs_delta,
                        "top50_memory_ids_exact": True,
                        "top50_score_tolerance": 1e-6,
                        "top50_max_score_absolute_difference": repeat_score_max_abs_delta,
                    },
                    "dense_similarity_wall_ms": dense_similarity_wall_ms,
                    "dense_latency_ms_by_question": dense_latency,
                    "rrf_latency_ms_by_question": rrf_latency,
                    "gpu": gpu_stats,
                }
            )
        finally:
            adapter.close()
            if doc_vectors is not None:
                del doc_vectors
            gc.collect()
            adapter.torch.cuda.empty_cache()

        # Reader tokenization and ContextManager projection are CPU/local-service work after the embedder is released.
        lexical_bundles = {
            row["question_id"]: row for row in inputs["bundles_by_question"].values()
        }
        for question_id in QUESTION_IDS:
            question = inputs["questions"][question_id]
            dense = [row for row in dense_rows if row["question_id"] == question_id]
            hybrid = [row for row in rrf_rows if row["question_id"] == question_id]
            d_plan, d_bundle, d_diag = _projection_context(
                question_id=question_id,
                arm="dense",
                question=question,
                ranked_rows=dense,
                memory_by_id=memory_by_id[question_id],
                client=client,
                reader_contract=reader_contract,
            )
            h_plan, h_bundle, h_diag = _projection_context(
                question_id=question_id,
                arm="hybrid_rrf",
                question=question,
                ranked_rows=hybrid,
                memory_by_id=memory_by_id[question_id],
                client=client,
                reader_contract=reader_contract,
            )
            dense_plans.append(d_plan)
            dense_bundles.append(d_bundle)
            hybrid_plans.append(h_plan)
            hybrid_bundles.append(h_bundle)
            lexical_bundle_row = lexical_bundles[question_id]
            lexical_bundle = lexical_bundle_row["context_bundle"]
            lexical_plan = inputs["plans_by_question"][question_id]["plan"]
            lexical_memory_tokens = lexical_plan["memory_tokens"]
            lexical_context_tokens = lexical_bundle["context_reader_tokens"]
            lexical_contents = []
            for item in lexical_bundle["items"]:
                parsed = json.loads(item["text"])
                value = parsed.get("value") if isinstance(parsed, dict) else None
                if isinstance(value, dict) and isinstance(value.get("content"), str):
                    lexical_contents.append(value["content"])
            lexical_content_tokens = (
                mem2c._local_token_count(client, "\n\n".join(lexical_contents))
                if lexical_contents
                else 0
            )
            diagnostic_questions.append(
                {
                    "question_id": question_id,
                    "question": question["question"],
                    "lexical": {
                        **_crowding(lexical_by_question[question_id]),
                        "retrieval_latency_ms": lexical_parity["questions"][
                            QUESTION_IDS.index(question_id)
                        ]["lexical_scoring_latency_ms"],
                        "selected_item_count": len(lexical_bundle["items"]),
                        "estimated_memory_tokens": lexical_memory_tokens,
                        "reader_context_tokens": lexical_context_tokens,
                        "reader_tokens_over_estimated_memory_tokens": lexical_context_tokens
                        / lexical_memory_tokens
                        if lexical_memory_tokens
                        else None,
                        "content_only_reader_tokens": lexical_content_tokens,
                        "context_bundle_sha256": lexical_bundle["context_bundle_sha256"],
                    },
                    "dense": {
                        **_crowding(dense),
                        "retrieval_latency_ms": dense_latency[question_id],
                        **d_diag,
                    },
                    "hybrid_rrf": {
                        **_crowding(hybrid),
                        "retrieval_latency_ms": rrf_latency[question_id],
                        **h_diag,
                    },
                }
            )

    model_sources = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_tree_sha256": local_identity["model_tree_sha256"],
        "weights_sha256": local_identity["weights_sha256"],
        "dimensions": DIMENSIONS,
        "device": DEVICE,
        "device_name": gpu_stats.get("device_name"),
        "dtype": "float16",
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "max_batch_tokens": MAX_BATCH_TOKENS,
        "normalization": "float32 L2 after final-non-padding-token pooling",
        "pooling_indexing_audit": {
            "padding_side": "left",
            "implementation": "maximum sequence position with attention_mask=1",
            "reason": "sum(attention_mask)-1 is a count, not the attended index, under left padding",
            "unit_tested_variable_length_batch": True,
            "same_adapter_for_all_dense_arms": True,
        },
        "document_instruction": "none",
        "query_instruction": QUERY_INSTRUCTION,
        "query_instruction_sha256": QUERY_INSTRUCTION_SHA256,
        "tokenizer_truncation": "right truncation at 8192; required observed truncation count 0",
        "local_files_only": True,
        "transformers_offline": True,
        "hosted_embedding_calls": 0,
    }
    embedding_manifest = {
        "schema_version": 1,
        "model": model_sources,
        "adapter_source_sha256": source_hashes["embedding_adapter"],
        "retrieval_view_contract_sha256": upstream["retrieval_view_contract_sha256"],
        "corpus": {
            "inventory_records": len(inputs["ordered_records"]),
            "unique_retrieval_documents": len(doc_shas),
            "ordered_corpus_identity_root": ordered_root,
            "cache_identity_sha256": embedding_identity["identity_sha256"],
            "vector_hash_root": embedding_stats["vector_hash_root"],
        },
        "queries": {
            "count": len(QUESTION_IDS),
            "instruction": QUERY_INSTRUCTION,
            "instruction_sha256": QUERY_INSTRUCTION_SHA256,
            "document_instruction": "none",
        },
        "input_tokens": {
            "unique_document_tokens": embedding_stats["document_input_tokens"],
            "query_tokens": embedding_stats["query_input_tokens"],
            "query_token_counts": embedding_stats["query_token_counts"],
        },
        "truncations": {
            "documents": embedding_stats["truncated_document_count"],
            "queries": embedding_stats["truncated_query_count"],
            "gate_zero": True,
        },
        "cache": {
            "path_outside_repository": True,
            "hits": embedding_stats["cache_hits"],
            "misses": embedding_stats["cache_misses"],
            "batches_computed": embedding_stats["batches_computed"],
        },
        "timing_ms": {
            "document_embedding_wall": embedding_stats["document_embedding_wall_ms"],
            "query_embedding_wall": embedding_stats["query_encoding_wall_ms"],
            "dense_similarity_wall": embedding_stats["dense_similarity_wall_ms"],
        },
        "query_batches": embedding_stats["query_batches"],
        "gpu": gpu_stats,
        "local_only": True,
        "raw_vectors_committed": False,
    }

    pre_paths = _pre_reader_artifact_paths()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    hashes = {}
    hashes["embedding_manifest.json"] = _write_or_verify_json(
        pre_paths["embedding_manifest.json"], embedding_manifest
    )
    hashes["lexical_parity.json"] = _write_or_verify_json(
        pre_paths["lexical_parity.json"], lexical_parity
    )
    hashes["lexical_top50.jsonl"] = _write_or_verify_jsonl(
        pre_paths["lexical_top50.jsonl"], lexical_rows
    )
    hashes["dense_top50.jsonl"] = _write_or_verify_jsonl(pre_paths["dense_top50.jsonl"], dense_rows)
    hashes["rrf_top8.jsonl"] = _write_or_verify_jsonl(pre_paths["rrf_top8.jsonl"], rrf_rows)
    hashes["dense_context_plans.jsonl"] = _write_or_verify_jsonl(
        pre_paths["dense_context_plans.jsonl"], dense_plans
    )
    hashes["dense_context_bundles.jsonl"] = _write_or_verify_jsonl(
        pre_paths["dense_context_bundles.jsonl"], dense_bundles
    )
    hashes["hybrid_context_plans.jsonl"] = _write_or_verify_jsonl(
        pre_paths["hybrid_context_plans.jsonl"], hybrid_plans
    )
    hashes["hybrid_context_bundles.jsonl"] = _write_or_verify_jsonl(
        pre_paths["hybrid_context_bundles.jsonl"], hybrid_bundles
    )
    diagnostic = {
        "schema_version": 1,
        "stage": "retrieval_only_pre_reader",
        "question_count": len(QUESTION_IDS),
        "question_ids": list(QUESTION_IDS),
        "labels_loaded": False,
        "reader_generation_calls": 0,
        "reader_tokenizer_endpoint_only": True,
        "test_access": False,
        "102_dev_run": False,
        "projection_policy_id": "m10-rank-aware-projection-v1",
        "projection_contract_sha256": PROJECTION_SHA256,
        "memory_budget_tokens": MEMORY_BUDGET,
        "rrf_selection_marker": "RRF_K_SELECTED_BY_PROTOCOL_NOT_DEV_SCORE=YES",
        "questions": diagnostic_questions,
    }
    hashes["retrieval_diagnostics_pre_reader.json"] = _write_or_verify_json(
        pre_paths["retrieval_diagnostics_pre_reader.json"], diagnostic
    )
    if (
        len(dense_rows) != len(QUESTION_IDS) * DENSE_DEPTH
        or len(lexical_rows) != len(QUESTION_IDS) * LEXICAL_DEPTH
        or len(rrf_rows) != len(QUESTION_IDS) * FINAL_TOP_K
    ):
        raise RuntimeError("Pre-reader retrieval row counts do not match frozen ten × [50, 50, 8]")
    if any(not _verify_sidecar(path) for path in pre_paths.values()):
        raise RuntimeError("One or more frozen pre-reader artifacts failed SHA verification")

    pre_reader_sha = _sha256_json({key: hashes[key] for key in sorted(hashes)})
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "PRE_READER_FROZEN",
        "base_commit_sha": BASE_COMMIT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "frozen ten DEV questions only",
        "question_ids": list(QUESTION_IDS),
        "test_access": False,
        "102_dev_run": False,
        "mem2c_inventory_path": MEM2C_INVENTORY_PATH.relative_to(ROOT).as_posix(),
        "mem2c_inventory_sha256": MEM2C_INVENTORY_SHA256,
        "mem2c_records_reingested": False,
        "memory_operations_created": 0,
        "sqlite_ingestion": False,
        "arms": ["historical_rawspan_lexical", "rawspan_dense", "rawspan_hybrid_rrf"],
        "dense_top_k": FINAL_TOP_K,
        "lexical_source_depth": LEXICAL_DEPTH,
        "dense_source_depth": DENSE_DEPTH,
        "rrf_contract_sha256": upstream["rrf_contract_sha256"],
        "rrf_k_selected_by_protocol_not_dev_score": True,
        "embedding_model_id": MODEL_ID,
        "embedding_model_revision": MODEL_REVISION,
        "embedding_model_tree_sha256": MODEL_TREE_SHA256,
        "embedding_weights_sha256": WEIGHTS_SHA256,
        "embedding_adapter_source_sha256": source_hashes["embedding_adapter"],
        "reader_contract_sha256": FINAL_READER_SHA256,
        "reader_runtime_identity": frozen_reader_runtime,
        "projection_contract_sha256": PROJECTION_SHA256,
        "source_hashes": source_hashes,
        "upstream_hashes": {
            key: value for key, value in upstream.items() if key.endswith(("sha256", "root"))
        },
        "pre_reader_artifact_sha256": hashes,
        "pre_reader_freeze_sha256": pre_reader_sha,
        "reader_answer_calls": 0,
        "judge_calls": 0,
        "memory_internal_llm_calls": 0,
        "hosted_calls": 0,
        "gate": {
            "mem2c_inventory_verified": True,
            "no_reingestion": True,
            "lexical_top8_parity_80_of_80": True,
            "embedding_identity_verified": True,
            "embedding_local_only": True,
            "embedding_truncations_zero": True,
            "dense_top50_frozen_all_ten": True,
            "lexical_top50_frozen_all_ten": True,
            "rrf_top8_frozen_all_ten": True,
            "projection_unchanged": True,
            "memory_budget_1024": True,
            "reader_contract_unchanged": True,
            "labels_loaded": False,
            "reader_generation_calls": 0,
            "all_pre_reader_artifacts_sha_frozen": True,
        },
    }
    _write_json_atomic(RUN_DIR / "run_manifest.json", manifest)
    print("MEM2D_SEMANTIC_RETRIEVAL_PRE_READER_FROZEN=YES", flush=True)
    print(f"pre_reader_freeze_sha256={pre_reader_sha}", flush=True)
    return {"manifest": manifest, "inputs": inputs, "pre_paths": pre_paths}


def _verify_pre_reader() -> dict[str, Any]:
    manifest_path = RUN_DIR / "run_manifest.json"
    if not _verify_sidecar(manifest_path):
        raise RuntimeError("MEM-2D pre-reader manifest SHA sidecar failed")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("status") not in {"PRE_READER_FROZEN", "COMPLETE"}
        or manifest.get("base_commit_sha") != BASE_COMMIT
        or manifest.get("question_ids") != list(QUESTION_IDS)
        or manifest.get("test_access") is not False
        or manifest.get("102_dev_run") is not False
        or manifest.get("mem2c_inventory_sha256") != MEM2C_INVENTORY_SHA256
        or manifest.get("pre_reader_freeze_sha256") is None
        or manifest.get("reader_contract_sha256") != FINAL_READER_SHA256
        or manifest.get("projection_contract_sha256") != PROJECTION_SHA256
    ):
        raise RuntimeError("MEM-2D pre-reader run manifest gate is invalid")
    for key, path in _pre_reader_artifact_paths().items():
        if not _verify_sidecar(path) or manifest["pre_reader_artifact_sha256"].get(
            key
        ) != sha256_file(path):
            raise RuntimeError(f"MEM-2D pre-reader artifact failed its frozen SHA: {key}")
    parity = _load_json(RUN_DIR / "lexical_parity.json")
    if (
        parity.get("gate") != "MEM2D_LEXICAL_TOP8_PARITY_WITH_MEM2C=YES"
        or parity.get("matched_records") != 80
        or parity.get("expected_records") != 80
        or parity.get("all_ten_exact") is not True
    ):
        raise RuntimeError("Frozen lexical parity artifact does not verify 80/80 records")
    lexical = _read_jsonl(RUN_DIR / "lexical_top50.jsonl")
    dense = _read_jsonl(RUN_DIR / "dense_top50.jsonl")
    hybrid = _read_jsonl(RUN_DIR / "rrf_top8.jsonl")
    historical_by_question = {row["question_id"]: row for row in _read_jsonl(MEM2C_RETRIEVAL_PATH)}
    if len(lexical) != 500 or len(dense) != 500 or len(hybrid) != 80:
        raise RuntimeError("Frozen lexical/dense/RRF row counts differ from 10 x [50,50,8]")
    for question_id in QUESTION_IDS:
        frozen_lex = [row for row in lexical if row["question_id"] == question_id]
        frozen_dense = [row for row in dense if row["question_id"] == question_id]
        frozen_hybrid = [row for row in hybrid if row["question_id"] == question_id]
        expected = historical_by_question[question_id]["results"]
        actual = frozen_lex[:FINAL_TOP_K]
        fields = (
            ("memory_id", "memory_id"),
            ("rank", "rank"),
            ("score", "score"),
            ("source_session_id", "source_session_id"),
            ("source_turn_key", "source_turn_id"),
            ("source_span_key", "source_span_id"),
        )
        if len(frozen_lex) != 50 or len(frozen_dense) != 50 or len(frozen_hybrid) != 8:
            raise RuntimeError(f"Frozen top-k artifacts do not cover {question_id}")
        if not all(
            all(left.get(left_key) == right.get(right_key) for left_key, right_key in fields)
            for left, right in zip(expected, actual, strict=True)
        ):
            raise RuntimeError(f"Frozen lexical top-8 no longer matches MEM-2C: {question_id}")
        if [row["rank"] for row in frozen_dense] != list(range(1, 51)) or [
            row["rank"] for row in frozen_hybrid
        ] != list(range(1, 9)):
            raise RuntimeError(f"Frozen retrieval ranks are not contiguous: {question_id}")
        if frozen_dense != sorted(
            frozen_dense, key=lambda row: (-row["cosine_similarity"], row["memory_id"])
        ):
            raise RuntimeError(f"Frozen Dense rows violate deterministic score ordering: {question_id}")
        recomputed_hybrid = _fuse_rrf(frozen_lex, frozen_dense)
        for expected_row, actual_row in zip(recomputed_hybrid, frozen_hybrid, strict=True):
            for field in (
                "memory_id",
                "rank",
                "lexical_rank",
                "lexical_score",
                "dense_rank",
                "cosine_similarity",
                "rrf_score",
                "source_session_id",
                "source_turn_id",
                "source_span_id",
                "source_turn_index",
                "source_span_index",
                "retrieval_document_sha256",
            ):
                if expected_row.get(field) != actual_row.get(field):
                    raise RuntimeError(f"Frozen RRF formula/order mismatch at {question_id}/{field}")
    embedding = _load_json(RUN_DIR / "embedding_manifest.json")
    if (
        embedding.get("local_only") is not True
        or embedding.get("model", {}).get("local_files_only") is not True
        or embedding.get("model", {}).get("hosted_embedding_calls") != 0
        or embedding.get("truncations") != {"documents": 0, "queries": 0, "gate_zero": True}
    ):
        raise RuntimeError("Frozen embedding manifest is not local-only with zero truncation")
    if (
        manifest.get("reader_runtime_identity", {}).get("loopback_only") is not True
        or manifest.get("reader_runtime_identity", {}).get("server_build")
        != "llama.cpp 10068 (571d0d540)"
    ):
        raise RuntimeError("Frozen pre-reader state lacks the pinned local llama.cpp reader identity")
    prohibited = [
        path.name
        for path in RUN_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".f32", ".npy", ".pt", ".safetensors"}
    ]
    if prohibited:
        raise RuntimeError(
            f"Raw vector/model artifacts are present in the run directory: {prohibited}"
        )
    freeze_sha = _sha256_json(
        {
            key: manifest["pre_reader_artifact_sha256"][key]
            for key in sorted(manifest["pre_reader_artifact_sha256"])
        }
    )
    if freeze_sha != manifest["pre_reader_freeze_sha256"]:
        raise RuntimeError("MEM-2D pre-reader aggregate freeze identity mismatch")
    dense_bundles = _read_jsonl(RUN_DIR / "dense_context_bundles.jsonl")
    hybrid_bundles = _read_jsonl(RUN_DIR / "hybrid_context_bundles.jsonl")
    if len(dense_bundles) != 10 or len(hybrid_bundles) != 10:
        raise RuntimeError("Frozen Dense/Hybrid ContextBundle count is not ten per arm")
    for left, right in zip(dense_bundles, hybrid_bundles, strict=True):
        if left["question_id"] != right["question_id"]:
            raise RuntimeError(
                "Dense and Hybrid ContextBundles are not ordered over the same ten questions"
            )
        for row in (left, right):
            bundle = row["context_bundle"]
            base = {key: value for key, value in bundle.items() if key != "context_bundle_sha256"}
            if (
                _sha256_json(base) != bundle["context_bundle_sha256"]
                or row["context_bundle_sha256"] != bundle["context_bundle_sha256"]
            ):
                raise RuntimeError("A frozen ContextBundle canonical hash is invalid")
            if (
                row.get("truncated") is not False
                or row["reader_prompt_tokens_preflight"] + row["output_reserve"]
                > row["max_model_length"]
            ):
                raise RuntimeError("A frozen shared-reader prompt does not fit without truncation")
            if row.get("shared_reader_contract_sha256") != FINAL_READER_SHA256:
                raise RuntimeError("A ContextBundle is not bound to the frozen reader contract")
        if left["shared_system_message_sha256"] != right["shared_system_message_sha256"]:
            raise RuntimeError("Dense and Hybrid do not use the same shared-reader system message")
    return manifest


def _reader_call(
    *,
    arm: str,
    bundle_row: dict[str, Any],
    question: dict[str, Any],
    client: httpx.Client,
    runtime_identity: dict[str, Any],
    pre_reader_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    question_id = bundle_row["question_id"]
    bundle_sha = bundle_row["context_bundle_sha256"]
    request = {
        "model": READER_MODEL,
        "messages": bundle_row["reader_messages"],
        "temperature": 0,
        "seed": 42,
        "max_tokens": OUTPUT_RESERVE,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request_sha = _sha256_bytes(mem2a.canonical_json(request).encode("utf-8"))
    identity_body = {
        "stage": RUN_ID,
        "pre_reader_freeze_sha256": pre_reader_sha,
        "arm": arm,
        "question_id": question_id,
        "context_bundle_sha256": bundle_sha,
        "reader_contract_sha256": FINAL_READER_SHA256,
        "reader_model_sha256": mem2c.PINNED_READER_MODEL_SHA256,
        "reader_runtime_props_sha256": runtime_identity["runtime_props_sha256"],
        "request_sha256": request_sha,
    }
    identity = {"identity": identity_body, "identity_sha256": _sha256_json(identity_body)}
    call_path = RUN_DIR / "calls" / arm / f"{question_id}.json"
    prediction_path = RUN_DIR / "prediction_cache" / arm / f"{question_id}.json"
    if call_path.exists() or prediction_path.exists():
        if not call_path.is_file() or not prediction_path.is_file():
            raise RuntimeError(
                f"Incomplete reader cache; refusing duplicate generation call: {arm}/{question_id}"
            )
        call_state = _load_json(call_path)
        prediction = _load_json(prediction_path)
        if (
            call_state.get("cache_identity") != identity
            or call_state.get("status") != "COMPLETE"
            or prediction.get("cache_identity") != identity
        ):
            raise RuntimeError(f"Frozen reader cache identity mismatch: {arm}/{question_id}")
        return prediction, call_state["call"]

    started_utc = datetime.now(UTC).isoformat()
    _write_json_atomic(
        call_path,
        {
            "cache_identity": identity,
            "status": "STARTED",
            "started_utc": started_utc,
            "request_sha256": request_sha,
        },
    )
    started = time.perf_counter()
    response = None
    choice: dict[str, Any] = {}
    predicted: str | None = None
    usage: dict[str, Any] = {}
    error_type = None
    try:
        response = client.post(f"{READER_ENDPOINT}/chat/completions", json=request)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Local reader returned a non-object response")
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("Local reader returned an unexpected choice count")
        candidate = choices[0]
        if not isinstance(candidate, dict):
            raise TypeError("Local reader returned a malformed choice")
        choice = candidate
        message = choice.get("message")
        if not isinstance(message, dict):
            raise TypeError("Local reader returned a malformed message")
        predicted = message.get("content")
        if not isinstance(predicted, str):
            raise TypeError("Local reader returned no answer text")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
        error_type = type(error).__name__
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    prompt_tokens_server = usage.get("prompt_tokens")
    prompt_match = prompt_tokens_server == bundle_row["reader_prompt_tokens_preflight"]
    quality_status = "OK" if predicted is not None and prompt_match else "INFRA_FAILURE"
    call = {
        "role": "reader_answer",
        "provider": "local_qwen",
        "endpoint": READER_ENDPOINT,
        "loopback_only": True,
        "model": READER_MODEL,
        "arm": arm,
        "question_id": question_id,
        "context_bundle_sha256": bundle_sha,
        "final_reader_contract_sha256": FINAL_READER_SHA256,
        "cache_identity_sha256": identity["identity_sha256"],
        "prompt_sha256": bundle_row["reader_prompt_sha256"],
        "rendered_prompt_sha256": bundle_row["rendered_prompt_sha256"],
        "request_sha256": request_sha,
        "temperature": 0,
        "seed": 42,
        "enable_thinking": False,
        "max_new_tokens": OUTPUT_RESERVE,
        "prompt_tokens_preflight": bundle_row["reader_prompt_tokens_preflight"],
        "prompt_tokens_server": prompt_tokens_server,
        "completion_tokens_server": usage.get("completion_tokens"),
        "prompt_tokens_match": prompt_match,
        "finish_reason": choice.get("finish_reason"),
        "latency_ms": latency_ms,
        "http_status": response.status_code if response is not None else None,
        "success": predicted is not None,
        "quality_status": quality_status,
        "error_type": error_type,
        "retry_count": 0,
        "hosted_call": False,
    }
    prediction = {
        "system": f"rawspan_{arm}",
        "question_id": question_id,
        "question": question["question"],
        "question_date": question["question_date"],
        "predicted": predicted.strip() if predicted is not None else None,
        "quality_status": quality_status,
        "reader_prompt_tokens_preflight": bundle_row["reader_prompt_tokens_preflight"],
        "reader_prompt_tokens_server": prompt_tokens_server,
        "completion_tokens_server": usage.get("completion_tokens"),
        "prompt_tokens_match": prompt_match,
        "reader_latency_ms": latency_ms,
        "finish_reason": choice.get("finish_reason"),
        "output_hit_token_cap": choice.get("finish_reason") == "length",
        "input_truncated": False if prompt_match and predicted is not None else None,
        "context_bundle_sha256": bundle_sha,
        "final_reader_contract_sha256": FINAL_READER_SHA256,
        "shared_reader_prompt_sha256": bundle_row["reader_prompt_sha256"],
        "rendered_prompt_sha256": bundle_row["rendered_prompt_sha256"],
        "cache_identity": identity,
    }
    _write_json_atomic(prediction_path, prediction)
    _write_json_atomic(call_path, {"cache_identity": identity, "status": "COMPLETE", "call": call})
    return prediction, call


def _load_labels_after_freeze() -> dict[str, dict[str, Any]]:
    prediction_paths = {
        "dense": RUN_DIR / "dense_predictions.jsonl",
        "hybrid_rrf": RUN_DIR / "hybrid_predictions.jsonl",
    }
    ledger_path = RUN_DIR / "call_ledger.jsonl"
    if not all(_verify_sidecar(path) for path in prediction_paths.values()) or not _verify_sidecar(
        ledger_path
    ):
        raise RuntimeError("Prediction and call ledger SHA freeze must precede label access")

    bundles_by_arm = {
        "dense": {
            row["question_id"]: row
            for row in _read_jsonl(RUN_DIR / "dense_context_bundles.jsonl")
        },
        "hybrid_rrf": {
            row["question_id"]: row
            for row in _read_jsonl(RUN_DIR / "hybrid_context_bundles.jsonl")
        },
    }
    predictions_by_arm = {}
    expected_calls = set()
    for arm, path in prediction_paths.items():
        rows = _read_jsonl(path)
        if [row.get("question_id") for row in rows] != list(QUESTION_IDS):
            raise RuntimeError(f"Frozen {arm} predictions do not cover the exact ten questions")
        predictions_by_arm[arm] = {}
        for row in rows:
            question_id = row["question_id"]
            bundle = bundles_by_arm[arm].get(question_id)
            identity = row.get("cache_identity")
            if (
                row.get("system") != f"rawspan_{arm}"
                or row.get("quality_status") != "OK"
                or row.get("final_reader_contract_sha256") != FINAL_READER_SHA256
                or not isinstance(identity, dict)
                or row.get("context_bundle_sha256")
                != (bundle or {}).get("context_bundle_sha256")
            ):
                raise RuntimeError(f"Frozen {arm} prediction binding failed for {question_id}")
            predictions_by_arm[arm][question_id] = row
            expected_calls.add((arm, question_id))

    calls = _read_jsonl(ledger_path)
    calls_by_key = {(row.get("arm"), row.get("question_id")): row for row in calls}
    if len(calls) != 20 or set(calls_by_key) != expected_calls:
        raise RuntimeError("Frozen reader call ledger does not match the two ten-question arms")
    for (arm, question_id), call in calls_by_key.items():
        prediction = predictions_by_arm[arm][question_id]
        if (
            call.get("role") != "reader_answer"
            or call.get("provider") != "local_qwen"
            or call.get("loopback_only") is not True
            or call.get("hosted_call") is not False
            or call.get("success") is not True
            or call.get("quality_status") != "OK"
            or call.get("context_bundle_sha256") != prediction["context_bundle_sha256"]
            or call.get("cache_identity_sha256")
            != prediction["cache_identity"].get("identity_sha256")
        ):
            raise RuntimeError(f"Frozen reader call binding failed for {arm}/{question_id}")

    labels = {}
    for row in mem2c.iter_json_array(mem2c.DATASET_PATH, include_gold=True):
        question_id = row.get("question_id")
        if question_id in QUESTION_IDS:
            labels[question_id] = {
                "answer": row.get("answer", ""),
                "answer_session_ids": row.get("answer_session_ids", []),
                "question_type": row.get("question_type"),
                "has_answer": row.get("has_answer"),
            }
    if set(labels) != set(QUESTION_IDS):
        raise RuntimeError("Post-freeze label join did not return the frozen ten question IDs")
    return labels


def _quality_metrics(
    *,
    arm_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    bundles: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    dense_plans = {
        row["question_id"]: row for row in _read_jsonl(RUN_DIR / "dense_context_plans.jsonl")
    }
    hybrid_plans = {
        row["question_id"]: row for row in _read_jsonl(RUN_DIR / "hybrid_context_plans.jsonl")
    }
    lexical_plans = {
        row["question_id"]: row for row in _read_jsonl(MEM2C_DIR / "context_plans.jsonl")
    }
    for question_id in QUESTION_IDS:
        label = labels[question_id]
        prediction = predictions[question_id]
        bundle_row = bundles[question_id]
        context = bundle_row["context_bundle"]["serialized_context"]
        retrieval = [row for row in arm_rows if row["question_id"] == question_id]
        native = mem2c._session_metrics(retrieval[:FINAL_TOP_K], set(label["answer_session_ids"]))
        projected_items = bundle_row["context_bundle"]["items"]
        projected_rows = [
            {
                "rank": index,
                "source_session_id": (item.get("source_session_ids") or [None])[0],
                "source_turn_id": item.get("source_turn_id"),
            }
            for index, item in enumerate(projected_items, 1)
        ]
        projected = mem2c._session_metrics(projected_rows, set(label["answer_session_ids"]))
        coverage, exact = mem2a._memory_gold_coverage(label["answer"], context)
        answer = mem2a._answer_metrics(prediction.get("predicted") or "", label["answer"])
        crowding = _crowding(retrieval)
        selected_plan = (
            lexical_plans[question_id]
            if bundle_row.get("arm") == "historical_rawspan_lexical"
            else dense_plans[question_id]
            if bundle_row.get("arm") == "dense"
            else hybrid_plans[question_id]
        )
        return_row = {
            "question_id": question_id,
            "question_type": label["question_type"],
            "has_answer": label["has_answer"],
            "answer_session_ids_count": len(label["answer_session_ids"]),
            "native_answer_session_recall_at_5": native["recall_at_5"],
            "native_answer_session_recall_at_8": native["recall_at_8"],
            "native_mrr": native["mrr"],
            "projected_answer_session_recall_at_5": projected["recall_at_5"],
            "projected_answer_session_recall_at_8": projected["recall_at_8"],
            "projected_mrr": projected["mrr"],
            "unique_source_sessions_at_8": crowding["unique_source_sessions_at_8"],
            "unique_parent_turns_at_8": crowding["unique_parent_turns_at_8"],
            "session_concentration_hhi_at_8": crowding["session_concentration_hhi"],
            "projected_unique_source_sessions": _crowding(projected_rows)[
                "unique_source_sessions_at_8"
            ],
            "gold_token_coverage": coverage,
            "exact_normalized_gold_sequence_present": exact,
            "token_precision": answer["token_precision"],
            "token_recall": answer["token_recall"],
            "token_f1": answer["f1"],
            "normalized_em": answer["normalized_exact_match"],
            "deterministic_abstention_accuracy": (
                float(mem2a._is_deterministic_refusal(prediction.get("predicted")))
                if label["has_answer"] is False
                else None
            ),
            "reader_latency_ms": prediction.get("reader_latency_ms"),
            "reader_prompt_tokens": prediction.get("reader_prompt_tokens_server"),
            "reader_completion_tokens": prediction.get("completion_tokens_server"),
            "context_reader_tokens": bundle_row["context_bundle"]["context_reader_tokens"],
            "estimated_memory_tokens": selected_plan["plan"]["memory_tokens"],
        }
        rows.append(return_row)
    return rows


def _mean(values: list[Any]) -> float | None:
    valid = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(valid) / len(valid) if valid else None


def _aggregate_metrics(per_question: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    metrics = {}
    for arm, rows in per_question.items():
        names = [
            "native_answer_session_recall_at_5",
            "native_answer_session_recall_at_8",
            "native_mrr",
            "projected_answer_session_recall_at_5",
            "projected_answer_session_recall_at_8",
            "projected_mrr",
            "unique_source_sessions_at_8",
            "unique_parent_turns_at_8",
            "session_concentration_hhi_at_8",
            "projected_unique_source_sessions",
            "gold_token_coverage",
            "exact_normalized_gold_sequence_present",
            "token_precision",
            "token_recall",
            "token_f1",
            "normalized_em",
            "deterministic_abstention_accuracy",
            "reader_latency_ms",
            "reader_prompt_tokens",
            "reader_completion_tokens",
            "context_reader_tokens",
            "estimated_memory_tokens",
        ]
        means = {name: _mean([row.get(name) for row in rows]) for name in names}
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_category[str(row["question_type"])].append(row)
        metrics[arm] = {
            "n": len(rows),
            "means": means,
            "by_question_type": {
                category: {
                    "n": len(category_rows),
                    "means": {
                        name: _mean([row.get(name) for row in category_rows]) for name in names
                    },
                }
                for category, category_rows in sorted(by_category.items())
            },
            "per_question": rows,
        }
    return {
        "schema_version": 1,
        "arms": metrics,
        "descriptive_only_no_ranking_or_superiority_claim": True,
        "scope": "frozen ten DEV diagnostic questions only",
        "answer_session_hit_is_not_answer_bearing_span_hit": True,
        "answer_metrics": "deterministic token-set precision/recall/F1 and normalized exact match",
        "judge_calls": 0,
    }


def _render_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    efficiency: dict[str, Any],
    parity: dict[str, Any],
) -> str:
    embedding_manifest = _load_json(RUN_DIR / "embedding_manifest.json")
    inventory_records = embedding_manifest["corpus"]["inventory_records"]
    unique_documents = embedding_manifest["corpus"]["unique_retrieval_documents"]
    embedding_wall_ms = efficiency["embedding"].get("document_embedding_wall_ms")
    if embedding_wall_ms is None:
        embedding_time_line = (
            f"- Unique-document embedding input tokens: {efficiency['embedding']['document_input_tokens']:,}; "
            f"query tokens: {efficiency['embedding']['query_input_tokens']:,}; "
            f"cached-vector verification: {efficiency['embedding']['cache_validation_wall_ms'] / 1000:.1f}s "
            f"for {efficiency['embedding']['cache_hits']:,} hits; initial full-corpus embedding duration was not retained."
        )
    else:
        embedding_time_line = (
            f"- Unique-document embedding input tokens: {efficiency['embedding']['document_input_tokens']:,}; "
            f"query tokens: {efficiency['embedding']['query_input_tokens']:,}; "
            f"document embedding wall time: {embedding_wall_ms / 1000:.1f}s."
        )
    lines = [
        "# MEM-2D - Frozen RawSpan Semantic Retrieval Ablation",
        "",
        "Completion gate: `MEM2D_SEMANTIC_RETRIEVAL_FROZEN_10_DIAGNOSTIC=YES`.",
        "",
        "## Scope",
        "",
        "This is a retrieval-only semantic ablation over the immutable MEM-2C RawSpan inventory. The ten preselected DEV questions are diagnostic only; no 102-case DEV, TEST, tuning, proposition extraction, revision semantics, RevMem, or RL was run.",
        "",
        f"- Inventory SHA256: `{MEM2C_INVENTORY_SHA256}`; records: {inventory_records:,}; unique retrieval documents: {unique_documents:,}.",
        f"- Lexical parity: {parity['matched_records']}/{parity['expected_records']} exact top-8 records across six frozen fields per record.",
        f"- New reader calls: {manifest['reader_answer_calls']} (10 Dense + 10 Hybrid); judge, hosted, and memory-internal LLM calls: 0.",
        f"- Embedding: local `{MODEL_ID}` on `{efficiency['embedding']['device_name']}`, no hosted fallback, document/query truncations: 0/0.",
        "- Reader contract, projection, M10 RawSpan representation, and 1024 estimated-token Memory budget were held fixed.",
        "",
        "## Descriptive Evidence",
        "",
        "These ten cases are not a performance ranking or a public benchmark claim. Answer-session retrieval does not establish that the answer-bearing span was retrieved.",
        "",
        "| Arm | Native Recall@5 | Native Recall@8 | Native MRR | Projected Recall@8 | Gold token coverage | Token F1 | Reader context tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "historical_rawspan_lexical": "Lexical (frozen MEM-2C)",
        "dense": "Dense",
        "hybrid_rrf": "Hybrid RRF",
    }
    for arm in ("historical_rawspan_lexical", "dense", "hybrid_rrf"):
        mean = metrics["arms"][arm]["means"]
        lines.append(
            "| "
            + labels[arm]
            + " | "
            + " | ".join(
                str(mean[key]) if mean.get(key) is not None else "n/a"
                for key in (
                    "native_answer_session_recall_at_5",
                    "native_answer_session_recall_at_8",
                    "native_mrr",
                    "projected_answer_session_recall_at_8",
                    "gold_token_coverage",
                    "token_f1",
                    "context_reader_tokens",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Efficiency and Crowding",
            "",
            embedding_time_line,
            "- Query repeatability was exact for identical frozen batch shapes; a singleton-versus-batched FP16 spot check was diagnostic only and did not determine retrieval results.",
            f"- Embedding cache hits/misses: {efficiency['embedding']['cache_hits']:,}/{efficiency['embedding']['cache_misses']:,}; dense similarity wall time: {efficiency['retrieval']['dense_similarity_wall_ms']:.1f}ms.",
            f"- Mean context tokens (Qwen3-8B tokenizer): Lexical {efficiency['context']['mean_reader_context_tokens']['historical_rawspan_lexical']:.1f}, Dense {efficiency['context']['mean_reader_context_tokens']['dense']:.1f}, Hybrid {efficiency['context']['mean_reader_context_tokens']['hybrid_rrf']:.1f}.",
            "- Per-question context token/estimated-token ratios, content-only diagnostic, top-8 session/turn crowding, and latencies are in `retrieval_diagnostics_pre_reader.json`, `efficiency.json`, and `deterministic_metrics.json`.",
            "",
            "## Interpretation Boundary",
            "",
            "Dense retrieval can support a semantic-versus-lexical retrieval observation only on these frozen cases. Hybrid evidence can indicate complementary rankings under the inherited RawSpan retrieval view. No statistical inference or quality/cost winner is claimed. The current semantic document view deliberately includes M10's structural metadata and opaque RawSpan key; a content-only view is a separate future ablation.",
            "",
            "## Reproduction and Integrity",
            "",
            f"- Pre-reader aggregate freeze: `{manifest['pre_reader_freeze_sha256']}`.",
            f"- Context projection: `{manifest['projection_contract_sha256']}`; shared reader: `{manifest['reader_contract_sha256']}`.",
            "- Raw vectors stay in the local cache outside Git; only top-50 lexical/dense and top-8 RRF result rows are retained.",
            "- TEST access: false; 102-case DEV: false; no tuning was performed.",
            "",
            "Stop after this frozen-ten diagnostic and review the case-level evidence before proposing another retrieval component.",
            "",
        ]
    )
    return "\n".join(lines)


def _finalize_reader_stage(pre_reader_manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_path = RUN_DIR / "run_manifest.json"
    current_manifest = _load_json(manifest_path)
    if current_manifest.get("pre_reader_freeze_sha256") != pre_reader_manifest.get(
        "pre_reader_freeze_sha256"
    ):
        raise RuntimeError("MEM-2D pre-reader identity changed before finalization")
    predictions_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    labels = _load_labels_after_freeze()
    all_metrics: dict[str, list[dict[str, Any]]] = {}
    bundle_rows_by_arm = {}
    retrieval_rows_by_arm = {
        "dense": _read_jsonl(RUN_DIR / "dense_top50.jsonl"),
        "hybrid_rrf": _read_jsonl(RUN_DIR / "rrf_top8.jsonl"),
        "historical_rawspan_lexical": _flatten_historical_retrieval(
            _read_jsonl(MEM2C_RETRIEVAL_PATH)
        ),
    }
    for arm in ("historical_rawspan_lexical", "dense", "hybrid_rrf"):
        if arm == "historical_rawspan_lexical":
            predictions = {row["question_id"]: row for row in _read_jsonl(MEM2C_PREDICTIONS_PATH)}
            bundles = {row["question_id"]: row for row in _read_jsonl(MEM2C_BUNDLES_PATH)}
        else:
            prediction_filename = (
                "dense_predictions.jsonl" if arm == "dense" else "hybrid_predictions.jsonl"
            )
            prediction_path = RUN_DIR / prediction_filename
            plan_name = (
                "dense_context_bundles.jsonl" if arm == "dense" else "hybrid_context_bundles.jsonl"
            )
            predictions = {row["question_id"]: row for row in _read_jsonl(prediction_path)}
            bundles = {row["question_id"]: row for row in _read_jsonl(RUN_DIR / plan_name)}
        if list(predictions) != list(QUESTION_IDS) or list(bundles) != list(QUESTION_IDS):
            raise RuntimeError(f"Predictions/bundles are not exactly ordered for arm {arm}")
        bundle_rows_by_arm[arm] = bundles
        if arm == "historical_rawspan_lexical":
            bundles = {qid: {**row, "arm": arm} for qid, row in bundles.items()}
        all_metrics[arm] = _quality_metrics(
            arm_rows=retrieval_rows_by_arm[arm],
            predictions=predictions,
            bundles=bundles,
            labels=labels,
        )
        predictions_by_arm[arm] = predictions
    metrics = _aggregate_metrics(all_metrics)
    metrics["prediction_sha256"] = {
        "dense_predictions.jsonl": sha256_file(RUN_DIR / "dense_predictions.jsonl"),
        "hybrid_predictions.jsonl": sha256_file(RUN_DIR / "hybrid_predictions.jsonl"),
        "call_ledger.jsonl": sha256_file(RUN_DIR / "call_ledger.jsonl"),
    }
    _write_json_atomic(RUN_DIR / "deterministic_metrics.json", metrics)
    _freeze(RUN_DIR / "deterministic_metrics.json")

    comparison_metrics = (
        "native_answer_session_recall_at_5",
        "native_answer_session_recall_at_8",
        "native_mrr",
        "projected_answer_session_recall_at_5",
        "projected_answer_session_recall_at_8",
        "projected_mrr",
        "gold_token_coverage",
        "exact_normalized_gold_sequence_present",
        "token_precision",
        "token_recall",
        "token_f1",
        "normalized_em",
        "context_reader_tokens",
        "estimated_memory_tokens",
    )
    comparison = {
        "schema_version": 1,
        "arms": {arm: metrics["arms"][arm]["means"] for arm in metrics["arms"]},
        "descriptive_paired_mean_deltas": {
            contrast: {
                metric: (
                    metrics["arms"][right]["means"][metric] - metrics["arms"][left]["means"][metric]
                    if metrics["arms"][right]["means"][metric] is not None
                    and metrics["arms"][left]["means"][metric] is not None
                    else None
                )
                for metric in comparison_metrics
            }
            for contrast, left, right in (
                ("dense_minus_lexical", "historical_rawspan_lexical", "dense"),
                ("hybrid_rrf_minus_lexical", "historical_rawspan_lexical", "hybrid_rrf"),
                ("hybrid_rrf_minus_dense", "dense", "hybrid_rrf"),
            )
        },
        "inferential_claim": False,
        "ranking_claim": False,
        "scope": "frozen ten DEV diagnostic cases",
    }
    _write_json_atomic(RUN_DIR / "comparison_lexical_dense_hybrid.json", comparison)
    _freeze(RUN_DIR / "comparison_lexical_dense_hybrid.json")

    pre_diag = _load_json(RUN_DIR / "retrieval_diagnostics_pre_reader.json")
    arm_context_tokens = {
        "historical_rawspan_lexical": [
            q["lexical"]["reader_context_tokens"] for q in pre_diag["questions"]
        ],
        "dense": [q["dense"]["reader_context_tokens"] for q in pre_diag["questions"]],
        "hybrid_rrf": [q["hybrid_rrf"]["reader_context_tokens"] for q in pre_diag["questions"]],
    }
    arm_estimated_tokens = {
        "historical_rawspan_lexical": [
            q["lexical"]["estimated_memory_tokens"] for q in pre_diag["questions"]
        ],
        "dense": [q["dense"]["estimated_memory_tokens"] for q in pre_diag["questions"]],
        "hybrid_rrf": [q["hybrid_rrf"]["estimated_memory_tokens"] for q in pre_diag["questions"]],
    }
    efficiency = {
        "schema_version": 1,
        "embedding": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_tree_sha256": MODEL_TREE_SHA256,
            "weights_sha256": WEIGHTS_SHA256,
            "device_name": _load_json(RUN_DIR / "embedding_manifest.json")["model"]["device_name"],
            "document_input_tokens": _load_json(RUN_DIR / "embedding_manifest.json")[
                "input_tokens"
            ]["unique_document_tokens"],
            "query_input_tokens": _load_json(RUN_DIR / "embedding_manifest.json")["input_tokens"][
                "query_tokens"
            ],
            "document_embedding_wall_ms": _load_json(RUN_DIR / "embedding_manifest.json")[
                "timing_ms"
            ]["document_embedding_wall"],
            "query_embedding_wall_ms": _load_json(RUN_DIR / "embedding_manifest.json")[
                "timing_ms"
            ]["query_embedding_wall"],
            "query_batches": _load_json(RUN_DIR / "embedding_manifest.json")["query_batches"],
            "cache_hits": _load_json(RUN_DIR / "embedding_manifest.json")["cache"]["hits"],
            "cache_misses": _load_json(RUN_DIR / "embedding_manifest.json")["cache"]["misses"],
            "batches_computed": _load_json(RUN_DIR / "embedding_manifest.json")["cache"][
                "batches_computed"
            ],
            "gpu": _load_json(RUN_DIR / "embedding_manifest.json")["gpu"],
            "truncated_documents": 0,
            "truncated_queries": 0,
        },
        "retrieval": {
            "lexical_top50_mean_latency_ms": _mean(
                [q["lexical"]["retrieval_latency_ms"] for q in pre_diag["questions"]]
            ),
            "dense_top50_mean_latency_ms": _mean(
                [q["dense"]["retrieval_latency_ms"] for q in pre_diag["questions"]]
            ),
            "rrf_top8_mean_latency_ms": _mean(
                [q["hybrid_rrf"]["retrieval_latency_ms"] for q in pre_diag["questions"]]
            ),
            "dense_similarity_wall_ms": _load_json(RUN_DIR / "embedding_manifest.json")[
                "timing_ms"
            ]["dense_similarity_wall"],
        },
        "context": {
            "mean_reader_context_tokens": {
                arm: _mean(values) for arm, values in arm_context_tokens.items()
            },
            "mean_estimated_memory_tokens": {
                arm: _mean(values) for arm, values in arm_estimated_tokens.items()
            },
            "mean_reader_to_estimated_ratio": {
                arm: _mean(
                    [
                        row["reader_tokens_over_estimated_memory_tokens"]
                        for q in pre_diag["questions"]
                        for row in [
                            q["lexical"]
                            if arm == "historical_rawspan_lexical"
                            else q["dense"]
                            if arm == "dense"
                            else q["hybrid_rrf"]
                        ]
                    ]
                )
                for arm in arm_context_tokens
            },
            "content_only_reader_tokens": {
                arm: [
                    q["lexical"]["content_only_reader_tokens"]
                    if arm == "historical_rawspan_lexical"
                    else q["dense"]["content_only_reader_tokens"]
                    if arm == "dense"
                    else q["hybrid_rrf"]["content_only_reader_tokens"]
                    for q in pre_diag["questions"]
                ]
                for arm in arm_context_tokens
            },
            "ingestion_latency_ms": 0,
        },
        "reader": {
            "answer_calls": 20,
            "judge_calls": 0,
            "hosted_calls": 0,
            "memory_internal_llm_calls": 0,
            "latency_ms_by_arm": {
                arm: [row["reader_latency_ms"] for row in predictions_by_arm[arm].values()]
                for arm in ("dense", "hybrid_rrf")
            },
        },
        "no_quality_cost_winner_claim": True,
    }
    _write_json_atomic(RUN_DIR / "efficiency.json", efficiency)
    _freeze(RUN_DIR / "efficiency.json")

    case_review = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "cases": [
            {
                "question_id": question_id,
                "focus": FOCUS_CASES.get(question_id),
                "question_type": labels[question_id]["question_type"],
                "outcome": None,
                "failure_loci": None,
                "causal_attribution": None,
                "notes": None,
            }
            for question_id in QUESTION_IDS
        ],
    }
    _write_json_atomic(CASE_REVIEW_PATH, case_review)
    _freeze(CASE_REVIEW_PATH)
    parity = _load_json(RUN_DIR / "lexical_parity.json")
    current_manifest.update(
        {
            "status": "COMPLETE",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "reader_answer_calls": 20,
            "reader_calls_successful": sum(
                1
                for call in _read_jsonl(RUN_DIR / "call_ledger.jsonl")
                if call.get("success") is True
            ),
            "judge_calls": 0,
            "memory_internal_llm_calls": 0,
            "hosted_calls": 0,
            "labels_loaded_after_prediction_and_ledger_freeze": True,
            "lexical_parity_records": parity["matched_records"],
            "metrics_summary": {arm: payload["means"] for arm, payload in metrics["arms"].items()},
            "gate": {
                **current_manifest["gate"],
                "labels_loaded": True,
                "reader_generation_calls": 20,
                "reader_calls_successful": 20,
                "judge_calls": 0,
                "hosted_calls": 0,
                "test_access": False,
                "102_dev_run": False,
                "no_proposition_or_revision_component": True,
                "raw_vectors_committed": False,
                "all_final_artifacts_sha_frozen": True,
            },
            "completion_gate_marker": "MEM2D_SEMANTIC_RETRIEVAL_FROZEN_10_DIAGNOSTIC=YES",
        }
    )
    report = _render_report(current_manifest, metrics, efficiency, parity)
    report_path = RUN_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    _freeze(report_path)
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    protocol = "\n".join(
        [
            "# MEM-2D - Frozen RawSpan Semantic Retrieval Ablation",
            "",
            "Completion gate: `MEM2D_SEMANTIC_RETRIEVAL_FROZEN_10_DIAGNOSTIC=YES`.",
            "",
            "## Research Question",
            "",
            "Given the immutable MEM-2C RawSpan inventory, test whether local dense semantic retrieval recovers sessions/spans missed by M10 lexical overlap, and whether untuned equal-weight RRF retains complementary ranked evidence.",
            "",
            "## Frozen Method",
            "",
            f"- Inventory: `{MEM2C_INVENTORY_PATH.relative_to(ROOT).as_posix()}` SHA256 `{MEM2C_INVENTORY_SHA256}`; no re-ingestion or second inventory copy.",
            f"- Dense embedding: `{MODEL_ID}` at `{MODEL_REVISION}`, tree SHA `{MODEL_TREE_SHA256}`, weights SHA `{WEIGHTS_SHA256}`, 1024-d, CUDA FP16, L2-normalized float32, local-only; truncations 0.",
            "- Left-padded pooling selects the maximum sequence index with `attention_mask=1`; `sum(mask)-1` is not an index under left padding. This implements the frozen final-nonpadding-token contract and is unit-tested; the same corrected adapter serves every dense arm.",
            "- Dense ranks cosine dot products over M10 scope/time-valid records; ties use memory ID ascending. No temporal boost, diversity, reranking, or lexical bonus.",
            "- Hybrid: `rawspan-lexical-dense-rrf-v1`, lexical/dense source depth 50, k=60, equal weights, final top 8; k was protocol-selected before results and was not swept.",
            "- Lexical top-8 parity: 80/80 across memory ID, rank, score, source session, source turn, and source span.",
            f"- Projection: frozen `m10-rank-aware-projection-v1`, SHA `{PROJECTION_SHA256}`; estimated Memory budget 1024; reader contract `{FINAL_READER_SHA256}`.",
            "- Ten frozen DEV cases only. Lexical reader outputs are reused from MEM-2C; new calls are exactly 10 Dense and 10 Hybrid using the same local Qwen3-8B reader.",
            "- No judge, hosted API, memory-internal LLM, 102-case DEV, TEST, proposition extraction, or revision semantics.",
            "",
            "## Retrieval-View Limitation",
            "",
            "Dense and lexical retrieval consume the exact M10 lexical retrieval string: `record.key + space + record.value-or-json.dumps(value, ensure_ascii=False)`. It includes the opaque RawSpan key, role, session date, source turn/span indices, and span content. This inherited metadata may influence semantic similarity; MEM-2D deliberately does not switch to a content-only view.",
            "",
            "## Evidence and Efficiency",
            "",
            f"- Exact records: lexical parity {parity['matched_records']}/{parity['expected_records']}; answer calls {current_manifest['reader_calls_successful']}/20; judge/hosted/memory-internal calls 0/0/0.",
            "- Deterministic token metrics and retrieval/crowding breakdowns are in `deterministic_metrics.json`; cache, embedding, context, and latency accounting are in `efficiency.json`.",
            "- The ten-case outcomes are diagnostic and descriptive only. No performance ranking, statistical generalization, or quality/cost winner is claimed.",
            "- Answer-session hit is not equivalent to answer-bearing span hit; session provenance cannot identify the answer turn.",
            "- Raw vectors stay outside the repository; top-50/top-8 retrieval rows are retained with SHA sidecars.",
            "",
            "## Stop Boundary",
            "",
            "Stop for human Reflection. Do not run 102 DEV or TEST, tune fusion, change the retrieval view, implement proposition/revision memory, start RevMem, or train RL from this stage.",
            "",
        ]
    )
    PROTOCOL_PATH.write_text(protocol, encoding="utf-8", newline="\n")
    _freeze(PROTOCOL_PATH)

    final_artifacts = [
        "embedding_manifest.json",
        "lexical_parity.json",
        "lexical_top50.jsonl",
        "dense_top50.jsonl",
        "rrf_top8.jsonl",
        "dense_context_plans.jsonl",
        "dense_context_bundles.jsonl",
        "hybrid_context_plans.jsonl",
        "hybrid_context_bundles.jsonl",
        "dense_predictions.jsonl",
        "hybrid_predictions.jsonl",
        "call_ledger.jsonl",
        "deterministic_metrics.json",
        "comparison_lexical_dense_hybrid.json",
        "efficiency.json",
        "report.md",
        "retrieval_diagnostics_pre_reader.json",
    ]
    artifact_hashes = {name: sha256_file(RUN_DIR / name) for name in final_artifacts}
    current_manifest["artifacts_sha256"] = artifact_hashes
    current_manifest["artifacts_sha256"]["mem_2d_semantic_retrieval.md"] = sha256_file(
        PROTOCOL_PATH
    )
    current_manifest["artifacts_sha256"]["mem_2d_case_review.json"] = sha256_file(CASE_REVIEW_PATH)
    current_manifest["all_artifact_sidecars_valid"] = (
        all(_verify_sidecar(RUN_DIR / name) for name in final_artifacts)
        and _verify_sidecar(PROTOCOL_PATH)
        and _verify_sidecar(CASE_REVIEW_PATH)
    )
    current_manifest["gate"]["all_final_artifacts_sha_frozen"] = current_manifest[
        "all_artifact_sidecars_valid"
    ]
    current_manifest["gate"]["passed"] = bool(
        current_manifest["all_artifact_sidecars_valid"]
        and parity["all_ten_exact"]
        and current_manifest["reader_calls_successful"] == 20
        and current_manifest["reader_answer_calls"] == 20
    )
    if not current_manifest["gate"]["passed"]:
        raise RuntimeError("MEM2D_SEMANTIC_RETRIEVAL_FROZEN_10_DIAGNOSTIC=NO")
    _write_json_atomic(manifest_path, current_manifest)
    return current_manifest


def _run_reader_stage() -> dict[str, Any]:
    manifest = _verify_pre_reader()
    if manifest.get("status") == "COMPLETE":
        final_paths = (
            RUN_DIR / "dense_predictions.jsonl",
            RUN_DIR / "hybrid_predictions.jsonl",
            RUN_DIR / "call_ledger.jsonl",
            RUN_DIR / "deterministic_metrics.json",
            RUN_DIR / "efficiency.json",
        )
        if manifest.get("gate", {}).get("passed") is not True or any(
            not _verify_sidecar(path) for path in final_paths
        ):
            raise RuntimeError("Completed MEM-2D stage is missing valid final artifacts")
        print("MEM2D_SEMANTIC_RETRIEVAL_FROZEN_10_DIAGNOSTIC=YES", flush=True)
        return manifest
    upstream = _verify_upstream()
    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0), trust_env=False) as client:
        runtime_identity = mem2b._verify_reader(client, upstream["m2a_manifest"])
        if runtime_identity != manifest.get("reader_runtime_identity"):
            raise RuntimeError("Local Qwen reader runtime changed after pre-reader freeze")
        bundles = {
            "dense": _read_jsonl(RUN_DIR / "dense_context_bundles.jsonl"),
            "hybrid_rrf": _read_jsonl(RUN_DIR / "hybrid_context_bundles.jsonl"),
        }
        question_rows = _load_frozen_inputs()["questions"]
        all_predictions: dict[str, list[dict[str, Any]]] = {arm: [] for arm in bundles}
        all_calls: list[dict[str, Any]] = []
        for arm in ("dense", "hybrid_rrf"):
            for bundle_row in bundles[arm]:
                question_id = bundle_row["question_id"]
                if bundle_row["arm"] != arm:
                    raise RuntimeError("Reader bundle arm identity mismatch")
                prediction, call = _reader_call(
                    arm=arm,
                    bundle_row=bundle_row,
                    question=question_rows[question_id],
                    client=client,
                    runtime_identity=runtime_identity,
                    pre_reader_sha=manifest["pre_reader_freeze_sha256"],
                )
                if (
                    not call["loopback_only"]
                    or call["hosted_call"]
                    or call["provider"] != "local_qwen"
                ):
                    raise RuntimeError("Reader call did not use the frozen local loopback provider")
                all_predictions[arm].append(prediction)
                all_calls.append(call)
        _write_or_verify_jsonl(RUN_DIR / "dense_predictions.jsonl", all_predictions["dense"])
        _write_or_verify_jsonl(RUN_DIR / "hybrid_predictions.jsonl", all_predictions["hybrid_rrf"])
        _write_or_verify_jsonl(RUN_DIR / "call_ledger.jsonl", all_calls)
        if not all(
            _verify_sidecar(RUN_DIR / name)
            for name in ("dense_predictions.jsonl", "hybrid_predictions.jsonl", "call_ledger.jsonl")
        ):
            raise RuntimeError("Prediction or call ledger SHA freeze failed")
        if len(all_calls) != 20 or sum(call["success"] is True for call in all_calls) != 20:
            raise RuntimeError(
                "MEM-2D requires exactly twenty successful new local reader calls; failure ledger frozen"
            )
        if any(call["quality_status"] != "OK" for call in all_calls):
            raise RuntimeError(
                "MEM-2D reader token preflight/server accounting mismatch; artifacts frozen"
            )
        _finalize_reader_stage(manifest)
    print("MEM2D_SEMANTIC_RETRIEVAL_FROZEN_10_DIAGNOSTIC=YES", flush=True)
    return _load_json(RUN_DIR / "run_manifest.json")


def _main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="run embedding/retrieval and freeze all pre-reader artifacts",
    )
    mode.add_argument(
        "--reader-only",
        action="store_true",
        help="run only the twenty calls against a verified frozen pre-reader state",
    )
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        result = _prepare()
        print(f"run_dir={RUN_DIR}", flush=True)
        print(
            f"pre_reader_freeze_sha256={result['manifest']['pre_reader_freeze_sha256']}", flush=True
        )
        return 0
    manifest = _run_reader_stage()
    print(f"run_dir={RUN_DIR}", flush=True)
    print(f"reader_calls_successful={manifest['reader_calls_successful']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
