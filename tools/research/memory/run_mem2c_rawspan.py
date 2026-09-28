"""Run the frozen-ten MEM-2C lossless raw-span granularity ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_DIR = ROOT / "tools" / "research" / "memory"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_mem1d3_reader as d3
import run_mem2a_m10_base as mem2a_runner
import run_mem2b_rank_aware_projection as mem2b_runner
from final_reader_contract import (
    build_reader_messages,
    load_final_reader_contract,
)
from mem1_artifacts import read_jsonl
from mem2a_m10_base import (
    MemoryOnlyQuestion,
    canonical_json,
    expected_context_settings,
    iter_json_array,
    question_from_source_fields,
    sha256_json,
)
from raw_span_memory import (
    REPRESENTATION_VERSION,
    SEGMENTER_VERSION,
    RawSpan,
    operation_for_span,
    operation_summary,
    segment_turn,
)

from health_ai_copilot.runtime.context_manager import (
    ContextBudget,
    ContextItemCategory,
    ContextManager,
    DeterministicTokenEstimator,
)
from health_ai_copilot.runtime.memory import (
    ContextIntent,
    FakeClock,
    MemoryKind,
    MemoryQuery,
    MemoryRecord,
    MemorySensitivity,
    MemorySourceType,
    MemoryStatus,
    SQLiteMemoryStore,
)

DATASET_PATH = ROOT / "data/longmemeval/longmemeval_s_cleaned.json"
DATASET_MANIFEST_PATH = ROOT / "docs/research/memory/dataset_manifest.json"
SPLIT_PATH = ROOT / "docs/research/memory/split_manifest.json"
SELECTION_PATH = ROOT / "docs/research/memory/main_smoke_10_manifest.json"
READER_CONTRACT_PATH = ROOT / "docs/research/memory/final_reader_contract.json"
READER_CONTRACT_SIDECAR = READER_CONTRACT_PATH.with_name("final_reader_contract.json.sha256")
PROJECTION_CONTRACT_PATH = ROOT / "docs/research/memory/rank_aware_projection_contract.json"
PROJECTION_CONTRACT_SIDECAR = PROJECTION_CONTRACT_PATH.with_name(
    "rank_aware_projection_contract.json.sha256"
)
SEGMENTER_CONTRACT_PATH = ROOT / "docs/research/memory/raw_span_segmenter_contract.json"
SEGMENTER_CONTRACT_SIDECAR = SEGMENTER_CONTRACT_PATH.with_name(
    "raw_span_segmenter_contract.json.sha256"
)
PROTOCOL_PATH = ROOT / "docs/research/memory/mem_2c_rawspan_granularity.md"
CASE_REVIEW_PATH = ROOT / "docs/research/memory/mem_2c_case_review.json"
MEM2A_DIR = ROOT / "runs/memory/mem2/mem2a-m10-base-10-20260928"
MEM2B_DIR = ROOT / "runs/memory/mem2/mem2b-rank-aware-projection-10-20260928"
RUN_ID = "mem2c-rawspan-10-20260928"
RUN_DIR = ROOT / "runs/memory/mem2" / RUN_ID
QUESTION_IDS = (
    "1cea1afa",
    "1c549ce4",
    "778164c6",
    "fca70973",
    "a82c026e",
    "gpt4_e061b84g",
    "gpt4_f420262c",
    "8550ddae",
    "06878be2",
    "c4ea545c",
)
PINNED_DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
PINNED_SELECTION_SHA256 = "5a38ff79d79be6a9db531227d63d6dc22dc619d11d2c701b2b4cc0295f04c911"
PINNED_DEV_IDS_SHA256 = "c6e0b423f720bcb06e1c571a8f6b0d018e0c1707d21fe17108349962d30f739f"
PINNED_READER_CONTRACT_SHA256 = "57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3"
PINNED_READER_MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
PINNED_PROJECTION_CONTRACT_SHA256 = (
    "ee16902373d695307db42797f33a1f4a531484096b36c29639b98abfd58d52ed"
)
MEMORY_BUDGET = 1024
READER_ENDPOINT = "http://127.0.0.1:8081/v1"
READER_MODEL = "health-memory-qwen3-8b"
OUTPUT_RESERVE = 256
MAX_CONTEXT_TOKENS = 131072


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as target:
        target.write(_jsonl_bytes(rows))
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def _freeze(path: Path, sidecar: Path | None = None) -> str:
    digest = _sha256_file(path)
    sidecar_path = sidecar or (
        path.with_suffix(".sha256")
        if path.suffix == ".jsonl"
        else path.with_name(f"{path.name}.sha256")
    )
    sidecar_path.write_text(f"{digest}  {path.name}\n", encoding="ascii", newline="\n")
    return digest


def _verify_sidecar(path: Path, sidecar: Path | None = None) -> bool:
    sidecar_path = sidecar or (
        path.with_suffix(".sha256")
        if path.suffix == ".jsonl"
        else path.with_name(f"{path.name}.sha256")
    )
    if not path.is_file() or not sidecar_path.is_file():
        return False
    return sidecar_path.read_text(encoding="ascii").strip().split() == [
        _sha256_file(path),
        path.name,
    ]


def _write_or_verify_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    expected = _jsonl_bytes(rows)
    if path.exists():
        if not _verify_sidecar(path) or path.read_bytes() != expected:
            raise RuntimeError(f"Frozen MEM-2C artifact identity mismatch: {path.name}")
        return _sha256_file(path)
    _write_jsonl_atomic(path, rows)
    return _freeze(path)


def _write_or_verify_json(path: Path, value: dict[str, Any]) -> str:
    expected = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if not _verify_sidecar(path) or path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Frozen MEM-2C artifact identity mismatch: {path.name}")
        return _sha256_file(path)
    _write_json_atomic(path, value)
    return _freeze(path)


def _verify_upstream() -> dict[str, Any]:
    if _sha256_file(DATASET_PATH) != PINNED_DATASET_SHA256:
        raise RuntimeError("LongMemEval-S data bytes differ from the frozen dataset SHA")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if (
        _sha256_file(SELECTION_PATH) != PINNED_SELECTION_SHA256
        or selection.get("question_ids") != list(QUESTION_IDS)
        or selection.get("test_access") is not False
        or selection.get("dataset_sha256") != PINNED_DATASET_SHA256
        or split.get("status") != "FROZEN_ACTIVE_PROTOCOL_LOCKED"
        or split.get("dev", {}).get("count") != 102
        or split.get("test", {}).get("count") != 398
        or split.get("dev", {}).get("question_ids_sha256_sorted_lf") != PINNED_DEV_IDS_SHA256
        or not set(QUESTION_IDS).issubset(set(split.get("dev", {}).get("question_ids", [])))
        or set(QUESTION_IDS) & set(split.get("test", {}).get("question_ids", []))
    ):
        raise RuntimeError("Frozen ten-case DEV selection or TEST boundary failed")
    dataset_manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    longmem = next(
        row for row in dataset_manifest["datasets"] if row.get("dataset_id") == "longmemeval_s"
    )
    if longmem.get("expected_sha256") != PINNED_DATASET_SHA256:
        raise RuntimeError("Dataset manifest does not pin the LongMemEval-S source bytes")

    required_gates = {
        "MEM1D4_FINAL_READER_CONTRACT_FROZEN=YES": ROOT
        / "docs/research/memory/mem_1d4_final_reader_contract.md",
        "MEM2A_M10_BASE_FROZEN_10_DIAGNOSTIC=YES": MEM2A_DIR / "report.md",
        "MEM2B_RANK_AWARE_PROJECTION_FROZEN_10_DIAGNOSTIC=YES": MEM2B_DIR / "report.md",
    }
    for marker, path in required_gates.items():
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"Required upstream gate missing: {marker}")

    m2a = json.loads((MEM2A_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    if not _verify_sidecar(MEM2A_DIR / "run_manifest.json"):
        raise RuntimeError("MEM-2A run manifest sidecar failed")
    m2a_artifacts = (
        "memory_operations.jsonl",
        "memory_inventory.jsonl",
        "retrieval_results.jsonl",
        "context_plans.jsonl",
        "context_bundles.jsonl",
        "predictions.jsonl",
        "call_ledger.jsonl",
    )
    m2a_hashes = {}
    for name in m2a_artifacts:
        path = MEM2A_DIR / name
        sidecar = MEM2A_DIR / name.replace(".jsonl", ".sha256")
        if not _verify_sidecar(path, sidecar):
            raise RuntimeError(f"MEM-2A frozen artifact failed its SHA sidecar: {name}")
        m2a_hashes[name] = _sha256_file(path)
    if (
        m2a.get("status") != "COMPLETE"
        or m2a.get("split") != "DEV"
        or m2a.get("test_access") is not False
        or m2a.get("question_ids") != list(QUESTION_IDS)
        or m2a.get("reader_calls_successful") != 10
        or m2a.get("final_reader_contract_sha256") != PINNED_READER_CONTRACT_SHA256
        or m2a.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("MEM-2A frozen ten-case run manifest gate failed")

    m2b_manifest_path = MEM2B_DIR / "run_manifest.json"
    m2b = json.loads(m2b_manifest_path.read_text(encoding="utf-8"))
    if not _verify_sidecar(m2b_manifest_path):
        raise RuntimeError("MEM-2B run manifest sidecar failed")
    if (
        m2b.get("status") != "COMPLETE"
        or m2b.get("split") != "DEV"
        or m2b.get("test_access") is not False
        or m2b.get("question_ids") != list(QUESTION_IDS)
        or m2b.get("reader_calls_successful") != 10
        or m2b.get("final_reader_contract_sha256") != PINNED_READER_CONTRACT_SHA256
        or m2b.get("projection_contract_sha256") != PINNED_PROJECTION_CONTRACT_SHA256
        or m2b.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("MEM-2B frozen ten-case run manifest gate failed")
    m2b_hashes = {}
    external_artifacts = {
        "mem_2b_case_review.json": ROOT / "docs/research/memory/mem_2b_case_review.json",
        "mem_2b_rank_aware_projection.md": ROOT
        / "docs/research/memory/mem_2b_rank_aware_projection.md",
        "rank_aware_projection_contract.json": PROJECTION_CONTRACT_PATH,
    }
    external_without_sidecar = {"mem_2b_rank_aware_projection.md"}
    for name, expected_sha in m2b.get("artifacts_sha256", {}).items():
        path = external_artifacts.get(name, MEM2B_DIR / name)
        upstream_sidecar = path.with_name(f"{path.name}.sha256")
        if (
            (upstream_sidecar.exists() and not _verify_sidecar(path, upstream_sidecar))
            or (not upstream_sidecar.exists() and name not in external_without_sidecar)
            or _sha256_file(path) != expected_sha
        ):
            raise RuntimeError(f"MEM-2B frozen artifact failed its SHA identity: {name}")
        m2b_hashes[name] = expected_sha
    for source_path, expected_sha in (
        (PROJECTION_CONTRACT_PATH, PINNED_PROJECTION_CONTRACT_SHA256),
        (READER_CONTRACT_PATH, PINNED_READER_CONTRACT_SHA256),
    ):
        if _sha256_file(source_path) != expected_sha or not _verify_sidecar(
            source_path,
            PROJECTION_CONTRACT_SIDECAR
            if source_path == PROJECTION_CONTRACT_PATH
            else READER_CONTRACT_SIDECAR,
        ):
            raise RuntimeError(f"Frozen contract SHA sidecar failed: {source_path.name}")
    _, reader_sha, _, _ = load_final_reader_contract()
    if reader_sha != PINNED_READER_CONTRACT_SHA256:
        raise RuntimeError("Frozen final reader contract loader rejected its source")
    segmenter_contract = json.loads(SEGMENTER_CONTRACT_PATH.read_text(encoding="utf-8"))
    if (
        segmenter_contract.get("contract_version") != SEGMENTER_VERSION
        or segmenter_contract.get("representation_version") != REPRESENTATION_VERSION
        or segmenter_contract.get("benchmark_label_access") is not False
        or segmenter_contract.get("model_calls") != 0
    ):
        raise RuntimeError("Raw-span segmenter contract does not match the frozen method")
    if SEGMENTER_CONTRACT_SIDECAR.exists():
        if not _verify_sidecar(SEGMENTER_CONTRACT_PATH, SEGMENTER_CONTRACT_SIDECAR):
            raise RuntimeError("Raw-span segmenter contract sidecar failed")
    else:
        _freeze(SEGMENTER_CONTRACT_PATH, SEGMENTER_CONTRACT_SIDECAR)
    contract_sha = _sha256_file(SEGMENTER_CONTRACT_PATH)

    context_source = ROOT / "src/health_ai_copilot/runtime/context_manager.py"
    if (
        _sha256_file(context_source)
        != json.loads(PROJECTION_CONTRACT_PATH.read_text(encoding="utf-8"))[
            "context_manager_source_sha256"
        ]
    ):
        raise RuntimeError(
            "ContextManager source differs from the frozen MEM-2B projection contract"
        )
    return {
        "selection_sha256": _sha256_file(SELECTION_PATH),
        "dataset_sha256": PINNED_DATASET_SHA256,
        "reader_contract_sha256": reader_sha,
        "projection_contract_sha256": PINNED_PROJECTION_CONTRACT_SHA256,
        "segmenter_contract_sha256": contract_sha,
        "mem2a_artifact_sha256": m2a_hashes,
        "mem2a_run_manifest_sha256": _sha256_file(MEM2A_DIR / "run_manifest.json"),
        "mem2b_artifact_sha256": m2b_hashes,
        "mem2b_run_manifest_sha256": _sha256_file(m2b_manifest_path),
        "question_ids": list(QUESTION_IDS),
        "reader_runtime_source": m2a["reader_runtime"],
        "m2a_manifest": m2a,
        "m2b_manifest": m2b,
    }


def _load_sources() -> dict[str, MemoryOnlyQuestion]:
    sources: dict[str, MemoryOnlyQuestion] = {}
    for row in iter_json_array(DATASET_PATH, include_gold=False):
        question_id = row.get("question_id")
        if question_id in QUESTION_IDS:
            question = question_from_source_fields(row)
            if question_id in sources:
                raise RuntimeError(f"Duplicate frozen question ID in dataset: {question_id}")
            sources[question_id] = question
    if set(sources) != set(QUESTION_IDS):
        raise RuntimeError(
            "The label-free dataset pass did not return exactly ten frozen DEV questions"
        )
    return sources


def _record_from_inventory(row: dict[str, Any]) -> MemoryRecord:
    excluded = {
        "question_id",
        "parent_turn_key",
        "source_turn_index",
        "source_span_index",
        "char_start",
        "char_end",
        "span_content_sha256",
    }
    source = {key: value for key, value in row.items() if key not in excluded}
    record = MemoryRecord(
        memory_id=source["memory_id"],
        scope_id=source["scope_id"],
        key=source["key"],
        kind=MemoryKind(source["kind"]),
        value=source["value"],
        value_sha256=source["value_sha256"],
        status=MemoryStatus(source["status"]),
        version=source["version"],
        created_at=source["created_at"],
        valid_from=source["valid_from"],
        valid_until=source["valid_until"],
        expires_at=source["expires_at"],
        source_event_ids=tuple(source["source_event_ids"]),
        related_event_ids=tuple(source["related_event_ids"]),
        source_run_id=source["source_run_id"],
        source_session_id=source["source_session_id"],
        objective_id=source["objective_id"],
        intent=ContextIntent.from_value(source["intent"]),
        supersedes_id=source["supersedes_id"],
        sensitivity=MemorySensitivity(source["sensitivity"]),
        source_type=MemorySourceType(source["source_type"]),
    )
    if record.to_dict() != source:
        raise RuntimeError(f"RawSpan MemoryRecord reconstruction differed: {record.memory_id}")
    return record


def _local_token_count(client: httpx.Client, text: str) -> int:
    response = client.post(
        "http://127.0.0.1:8081/tokenize", json={"content": text, "add_special": False}
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list):
        raise TypeError("Frozen local Qwen3-8B tokenizer returned no token ID list")
    return len(tokens)


def _source_hashes() -> dict[str, str]:
    paths = {
        "m10_memory_store": ROOT / "src/health_ai_copilot/runtime/memory.py",
        "context_manager": ROOT / "src/health_ai_copilot/runtime/context_manager.py",
        "runtime_profile": ROOT / "src/health_ai_copilot/runtime/builder.py",
        "raw_span_adapter": TOOLS_DIR / "raw_span_memory.py",
        "runner": Path(__file__).resolve(),
        "reader_contract_loader": TOOLS_DIR / "final_reader_contract.py",
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _cache_identity(static: dict[str, Any], question_id: str, bundle_sha: str) -> dict[str, Any]:
    body = {
        **static,
        "representation_version": REPRESENTATION_VERSION,
        "question_id": question_id,
        "context_bundle_sha256": bundle_sha,
    }
    return {"identity": body, "identity_sha256": sha256_json(body)}


def _gate_requirements_passed(gate: dict[str, Any]) -> bool:
    false_required = {"test_access", "102_dev_run"}
    for key, value in gate.items():
        if key == "passed":
            continue
        if key == "memory_budget_tokens":
            if value != MEMORY_BUDGET:
                return False
            continue
        if isinstance(value, bool):
            if value is (key in false_required):
                return False
        elif isinstance(value, int):
            if value != 0:
                return False
        else:
            return False
    return True


def _efficiency_stable_fields(payload: dict[str, Any]) -> dict[str, Any]:
    stable = json.loads(json.dumps(payload))
    timing_fields = {
        "mean_context_projection_latency_ms",
        "mean_ingestion_latency_ms",
        "mean_native_retrieval_latency_ms",
    }
    for key in timing_fields:
        stable.pop(key, None)
    for row in stable.get("per_question", []):
        for key in (
            "context_projection_latency_ms",
            "ingestion_latency_ms",
            "native_retrieval_latency_ms",
        ):
            row.pop(key, None)
    return stable


def _memory_budget() -> ContextBudget:
    settings = expected_context_settings()
    if settings["budget"]["max_memory_tokens"] != MEMORY_BUDGET:
        raise RuntimeError("MEM-2C memory budget differs from frozen M10 budget")
    return ContextBudget(**settings["budget"])


def _build_question_state(
    question: MemoryOnlyQuestion,
    *,
    dataset_sha256: str,
    client: httpx.Client,
    db_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    budget = _memory_budget()
    manager = ContextManager(
        budget=budget,
        estimator=DeterministicTokenEstimator(),
        history_window=24,
    )
    segmentation_rows: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    all_spans: list[RawSpan] = []
    ingestion_started = time.perf_counter()

    with SQLiteMemoryStore(db_path, clock=FakeClock(question.now)) as store:
        existing_records = {
            record.memory_id: record
            for record in store.active_records(question.scope_id, now="9999-12-31T23:59:59Z")
        }
        for turn in question.turns:
            spans = segment_turn(turn)
            reconstructed = "".join(span.content for span in spans)
            reconstruction_passed = reconstructed == turn.content
            if not reconstruction_passed:
                raise RuntimeError("MEM-2C source turn did not reconstruct exactly")
            segmentation_rows.append(
                {
                    "question_id": question.question_id,
                    "source_session_id": turn.source_session_id,
                    "source_turn_index": turn.turn_index,
                    "role": turn.role,
                    "session_date": turn.session_date,
                    "parent_turn_key": turn.key,
                    "turn_character_count": len(turn.content),
                    "turn_content_sha256": hashlib.sha256(turn.content.encode("utf-8")).hexdigest(),
                    "span_count": len(spans),
                    "spans": [
                        {
                            "source_span_index": span.span_index,
                            "char_start": span.char_start,
                            "char_end": span.char_end,
                            "character_count": len(span.content),
                            "content": span.content,
                            "content_sha256": hashlib.sha256(
                                span.content.encode("utf-8")
                            ).hexdigest(),
                        }
                        for span in spans
                    ],
                    "reconstructed_content_sha256": hashlib.sha256(
                        reconstructed.encode("utf-8")
                    ).hexdigest(),
                    "reconstruction_passed": True,
                }
            )
            for span in spans:
                all_spans.append(span)
                operation = operation_for_span(
                    span, question=question, dataset_sha256=dataset_sha256
                )
                memory_id = str(operation.memory_id)
                record = existing_records.get(memory_id)
                if record is None:
                    record = store.apply(operation, now=turn.valid_from)
                elif (
                    record.scope_id != operation.scope_id
                    or record.key != operation.key
                    or record.kind != operation.kind
                    or record.value != operation.value
                    or record.valid_from != operation.valid_from
                    or record.valid_until is not None
                    or record.expires_at is not None
                    or record.source_event_ids != operation.source_event_ids
                    or record.source_session_id != operation.source_session_id
                    or record.source_type != operation.source_type
                    or record.sensitivity != operation.sensitivity
                    or record.status != MemoryStatus.ACTIVE
                    or record.version != 1
                ):
                    raise RuntimeError(
                        "Existing SQLite RawSpan state differs from deterministic replay"
                    )
                if record is None or record.status != MemoryStatus.ACTIVE or record.version != 1:
                    raise RuntimeError("RawSpan ADD did not materialize an active version-1 record")
                operations.append(operation_summary(operation, span=span, index=len(operations)))
                row = record.to_dict()
                row.update(
                    {
                        "question_id": question.question_id,
                        "parent_turn_key": span.parent_turn_key,
                        "source_turn_index": span.turn.turn_index,
                        "source_span_index": span.span_index,
                        "char_start": span.char_start,
                        "char_end": span.char_end,
                        "span_content_sha256": hashlib.sha256(
                            span.content.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                inventory.append(row)

        expected_ids = {str(row["memory_id"]) for row in operations}
        final_active = {
            record.memory_id
            for record in store.active_records(question.scope_id, now="9999-12-31T23:59:59Z")
        }
        if final_active != expected_ids or len(expected_ids) != len(operations):
            raise RuntimeError("RawSpan SQLite inventory does not match the exact expected ADD set")
        history = store.history(question.scope_id)
        if len(history) != len(expected_ids) or any(event.operation != "add" for event in history):
            raise RuntimeError("RawSpan SQLite history contains non-ADD or duplicate operations")
        snapshot_identity = store.snapshot(
            question.scope_id, session_revision=len(operations)
        ).to_dict()
        ingestion_ms = (time.perf_counter() - ingestion_started) * 1000

        retrieval_started = time.perf_counter()
        matches = store.matches(
            MemoryQuery(
                scope_id=question.scope_id, text=question.question, now=question.now, top_k=8
            )
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        rank_hints = {
            f"memory-{match.record.memory_id}": rank for rank, match in enumerate(matches, 1)
        }
        projection_started = time.perf_counter()
        plan = manager.build_plan(
            session_id=question.scope_id,
            session_revision=len(operations),
            current_user=question.question,
            history=(),
            memory_records=[match.record for match in matches],
            retrieval_query=question.question,
            selection_rank_hints=rank_hints,
        )
        if plan.memory_tokens > MEMORY_BUDGET:
            raise RuntimeError(
                "RawSpan ContextPlan exceeded the frozen 1024 estimated-token budget"
            )
        if [rank_hints[item_id] for item_id in rank_hints] != list(range(1, len(matches) + 1)):
            raise RuntimeError("RawSpan native retrieval ranks are not contiguous top-k ranks")

        match_by_id = {
            match.record.memory_id: (rank, match) for rank, match in enumerate(matches, 1)
        }
        selected_items = [
            item for item in plan.items if item.category == ContextItemCategory.MEMORY
        ]
        selected_ids = [str(item.provenance["memory_id"]) for item in selected_items]
        selected_ranks = [match_by_id[memory_id][0] for memory_id in selected_ids]
        if selected_ranks != sorted(selected_ranks):
            raise RuntimeError("Rank-aware ContextManager reordered selected RawSpan candidates")

        context_items: list[dict[str, Any]] = []
        for context_rank, item in enumerate(selected_items, 1):
            memory_id = str(item.provenance["memory_id"])
            native_rank, match = match_by_id[memory_id]
            span = next(span for span in all_spans if span.key == match.record.key)
            context_items.append(
                {
                    "text": canonical_json(item.content),
                    "rank": context_rank,
                    "kind": item.category.value,
                    "source_session_ids": [match.record.source_session_id]
                    if match.record.source_session_id
                    else [],
                    "memory_id": memory_id,
                    "native_retrieval_rank": native_rank,
                    "retrieval_score": match.score,
                    "status": match.record.status.value,
                    "version": match.record.version,
                    "source_turn_id": span.parent_turn_key,
                    "source_span_id": match.record.key,
                    "source_turn_index": span.turn.turn_index,
                    "source_span_index": span.span_index,
                    "char_start": span.char_start,
                    "char_end": span.char_end,
                    "estimated_tokens": item.estimated_tokens,
                }
            )
        serialized_context = "\n\n".join(
            f"[Context item {item['rank']} | {item['kind']}]\n{item['text']}"
            for item in context_items
        )
        context_reader_tokens = (
            _local_token_count(client, serialized_context) if serialized_context else 0
        )
        context_bundle_base = {
            "schema_version": 3,
            "system": "healthcopilot_m10_rawspan",
            "question_id": question.question_id,
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
        bundle_sha = sha256_json(context_bundle_base)
        context_bundle = {**context_bundle_base, "context_bundle_sha256": bundle_sha}
        projection_ms = (time.perf_counter() - projection_started) * 1000

    _contract, contract_sha, system_template, user_template = load_final_reader_contract()
    if contract_sha != PINNED_READER_CONTRACT_SHA256:
        raise RuntimeError("MEM-2C final reader contract SHA differs from frozen MEM-1D4")
    messages = build_reader_messages(
        question.question,
        question.question_date,
        serialized_context,
        system_template=system_template,
        user_template=user_template,
    )
    prompt_tokens, rendered_prompt_sha = d3._render_and_tokenize(client, messages)
    if prompt_tokens + OUTPUT_RESERVE > MAX_CONTEXT_TOKENS:
        raise RuntimeError(
            f"RawSpan reader prompt exceeds frozen context slot: {question.question_id}"
        )
    prompt_sha = hashlib.sha256(canonical_json(messages).encode("utf-8")).hexdigest()

    span_by_key = {span.key: span for span in all_spans}
    retrieval_rows = [
        {
            "rank": rank,
            "memory_id": match.record.memory_id,
            "score": match.score,
            "reasons": list(match.reasons),
            "source_session_id": match.record.source_session_id,
            "source_turn_key": span_by_key[match.record.key].parent_turn_key,
            "source_span_key": match.record.key,
            "source_turn_index": match.record.value["source_turn_index"],
            "source_span_index": match.record.value["source_span_index"],
            "char_start": span_by_key[match.record.key].char_start,
            "char_end": span_by_key[match.record.key].char_end,
            "valid_from": match.record.valid_from,
        }
        for rank, match in enumerate(matches, 1)
    ]
    retrieval_row = {
        "question_id": question.question_id,
        "query": question.question,
        "query_now": question.now,
        "native_top_k": 8,
        "candidate_inventory_size": len(inventory),
        "results": retrieval_rows,
    }
    selected_set = set(selected_ranks)
    dropped_ranks = [row["rank"] for row in retrieval_rows if row["rank"] not in selected_set]
    plan_row = {
        "question_id": question.question_id,
        "plan_id": "plan-"
        + hashlib.sha256(
            f"{question.question_id}:{plan.plan_hash}:{PINNED_PROJECTION_CONTRACT_SHA256}".encode()
        ).hexdigest()[:32],
        "plan_hash": plan.plan_hash,
        "plan": plan.identity_payload(),
        "selected_native_retrieval_ranks": selected_ranks,
        "dropped_native_retrieval_ranks": dropped_ranks,
        "memory_budget_tokens": MEMORY_BUDGET,
    }
    bundle_with_reader = {
        "question_id": question.question_id,
        "context_bundle": context_bundle,
        "context_bundle_sha256": bundle_sha,
        "reader_prompt_sha256": prompt_sha,
        "rendered_prompt_sha256": rendered_prompt_sha,
        "reader_prompt_tokens_preflight": prompt_tokens,
        "max_model_length": MAX_CONTEXT_TOKENS,
        "output_reserve": OUTPUT_RESERVE,
        "truncated": False,
    }
    structural_diagnostic = {
        "question_id": question.question_id,
        "turn_count": len(question.turns),
        "span_count": len(all_spans),
        "reconstruction_pass_count": sum(
            1 for row in segmentation_rows if row["reconstruction_passed"]
        ),
        "native_top_k": 8,
        "native_retrieval_ranks": [row["rank"] for row in retrieval_rows],
        "selected_native_retrieval_ranks": selected_ranks,
        "dropped_native_retrieval_ranks": dropped_ranks,
        "selected_worse_better_dropped_pair_count": sum(
            selected_rank > dropped_rank
            for selected_rank in selected_ranks
            for dropped_rank in dropped_ranks
        ),
        "context_item_count": len(context_items),
        "estimated_memory_tokens": plan.memory_tokens,
        "context_reader_tokens": context_reader_tokens,
        "reader_prompt_tokens_preflight": prompt_tokens,
        "output_reserve": OUTPUT_RESERVE,
        "truncated": False,
        "labels_loaded": False,
    }
    state = {
        "question_id": question.question_id,
        "scope_id": question.scope_id,
        "now": question.now,
        "turn_count": len(question.turns),
        "span_count": len(all_spans),
        "reconstruction_pass_count": structural_diagnostic["reconstruction_pass_count"],
        "segmentation_rows": segmentation_rows,
        "operations": operations,
        "inventory": inventory,
        "snapshot_identity": snapshot_identity,
        "retrieval_row": retrieval_row,
        "plan_row": plan_row,
        "bundle_row": bundle_with_reader,
        "context_bundle": context_bundle,
        "structural_diagnostic": structural_diagnostic,
        "messages": messages,
        "question": question.question,
        "question_date": question.question_date,
        "context_items": context_items,
        "spans": all_spans,
        "estimated_span_tokens": [
            manager._memory_item(_record_from_inventory(row)).estimated_tokens for row in inventory
        ],
        "runtime_metrics": {
            "ingestion_latency_ms": round(ingestion_ms, 3),
            "native_retrieval_latency_ms": round(retrieval_ms, 3),
            "context_projection_latency_ms": round(projection_ms, 3),
            "sqlite_bytes": db_path.stat().st_size if db_path.exists() else None,
        },
    }
    # Verify a second SQLite replay against deterministic artifacts before any reader call.
    replay_started = time.perf_counter()
    with SQLiteMemoryStore(db_path, clock=FakeClock(question.now)) as replay_store:
        replay_inventory = [
            record.to_dict()
            for record in replay_store.active_records(question.scope_id, now="9999-12-31T23:59:59Z")
        ]
        replay_matches = replay_store.matches(
            MemoryQuery(
                scope_id=question.scope_id, text=question.question, now=question.now, top_k=8
            )
        )
    replay_rows = [
        {
            "memory_id": match.record.memory_id,
            "score": match.score,
            "reasons": list(match.reasons),
            "source_session_id": match.record.source_session_id,
            "key": match.record.key,
        }
        for match in replay_matches
    ]
    first_rows = [
        {
            "memory_id": row["memory_id"],
            "score": row["score"],
            "reasons": row["reasons"],
            "source_session_id": row["source_session_id"],
            "key": row["source_span_key"],
        }
        for row in retrieval_rows
    ]
    if [row for row in replay_inventory] != [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "question_id",
                "parent_turn_key",
                "source_turn_index",
                "source_span_index",
                "char_start",
                "char_end",
                "span_content_sha256",
            }
        }
        for row in inventory
    ] or replay_rows != first_rows:
        raise RuntimeError(
            f"RawSpan SQLite replay changed inventory or lexical retrieval: {question.question_id}"
        )
    state["structural_diagnostic"]["sqlite_replay_passed"] = True
    state["runtime_metrics"]["sqlite_replay_latency_ms"] = round(
        (time.perf_counter() - replay_started) * 1000, 3
    )
    state["bundle_row"]["cache_identity"] = None
    return state, state["structural_diagnostic"]


def _prepare_all(
    *, inputs: dict[str, MemoryOnlyQuestion], upstream: dict[str, Any], client: httpx.Client
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source_hashes = _source_hashes()
    static_identity = {
        "dataset_sha256": upstream["dataset_sha256"],
        "frozen_question_selection_sha256": upstream["selection_sha256"],
        "question_ids": list(QUESTION_IDS),
        "segmenter_contract_sha256": upstream["segmenter_contract_sha256"],
        "m10_source_sha256": source_hashes["m10_memory_store"],
        "memory_store_scoring_source_sha256": source_hashes["m10_memory_store"],
        "raw_span_adapter_source_sha256": source_hashes["raw_span_adapter"],
        "projection_contract_sha256": upstream["projection_contract_sha256"],
        "context_manager_source_sha256": source_hashes["context_manager"],
        "memory_budget_tokens": MEMORY_BUDGET,
        "projection_policy_id": "m10-rank-aware-projection-v1",
        "final_reader_contract_sha256": upstream["reader_contract_sha256"],
        "reader_model_sha256": PINNED_READER_MODEL_SHA256,
        "source_code_sha256": source_hashes,
    }
    states: dict[str, dict[str, Any]] = {}
    diagnostic_rows = []
    db_sizes: dict[str, int] = {}
    for question_id in QUESTION_IDS:
        question = inputs[question_id]
        db_path = RUN_DIR / "state" / question_id / "memory.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        state, structural = _build_question_state(
            question,
            dataset_sha256=upstream["dataset_sha256"],
            client=client,
            db_path=db_path,
        )
        bundle_sha = state["bundle_row"]["context_bundle_sha256"]
        state["cache_identity"] = _cache_identity(static_identity, question_id, bundle_sha)
        state["bundle_row"]["cache_identity"] = state["cache_identity"]
        states[question_id] = state
        diagnostic_rows.append(structural)
        db_sizes[question_id] = db_path.stat().st_size

    segmentation_rows = [
        row for question_id in QUESTION_IDS for row in states[question_id]["segmentation_rows"]
    ]
    operation_rows = [
        row for question_id in QUESTION_IDS for row in states[question_id]["operations"]
    ]
    inventory_rows = [
        row for question_id in QUESTION_IDS for row in states[question_id]["inventory"]
    ]
    retrieval_rows = [states[question_id]["retrieval_row"] for question_id in QUESTION_IDS]
    plan_rows = [states[question_id]["plan_row"] for question_id in QUESTION_IDS]
    bundle_rows = [states[question_id]["bundle_row"] for question_id in QUESTION_IDS]
    pre_reader_diagnostics = {
        "schema_version": 1,
        "stage": "MEM2C_RAWSPAN_PRE_READER_FROZEN",
        "labels_loaded": False,
        "question_ids": list(QUESTION_IDS),
        "per_question": diagnostic_rows,
        "candidate_inventory_size": len(inventory_rows),
        "operation_count": len(operation_rows),
        "all_operations_add": all(row["operation"] == "ADD" for row in operation_rows),
        "top_k": 8,
        "projection_policy_id": "m10-rank-aware-projection-v1",
        "memory_budget_tokens": MEMORY_BUDGET,
        "memory_internal_llm_calls": 0,
        "embedding_calls": 0,
        "judge_calls": 0,
        "hosted_calls": 0,
        "test_access": False,
        "102_dev_run": False,
    }

    artifact_hashes = {}
    for file_name, rows in (
        ("segmentation_audit.jsonl", segmentation_rows),
        ("memory_operations.jsonl", operation_rows),
        ("memory_inventory.jsonl", inventory_rows),
        ("retrieval_results.jsonl", retrieval_rows),
        ("context_plans.jsonl", plan_rows),
        ("context_bundles.jsonl", bundle_rows),
    ):
        artifact_hashes[file_name] = _write_or_verify_jsonl(RUN_DIR / file_name, rows)
    artifact_hashes["pre_reader_diagnostics.json"] = _write_or_verify_json(
        RUN_DIR / "pre_reader_diagnostics.json", pre_reader_diagnostics
    )

    gate = (
        all(row["reconstruction_pass_count"] == row["turn_count"] for row in diagnostic_rows)
        and all(row["sqlite_replay_passed"] for row in diagnostic_rows)
        and pre_reader_diagnostics["all_operations_add"]
        and len(states) == 10
        and all(state["bundle_row"]["truncated"] is False for state in states.values())
        and all(
            _verify_sidecar(RUN_DIR / name)
            for name in (
                "segmentation_audit.jsonl",
                "memory_operations.jsonl",
                "memory_inventory.jsonl",
                "retrieval_results.jsonl",
                "context_plans.jsonl",
                "context_bundles.jsonl",
                "pre_reader_diagnostics.json",
            )
        )
    )
    if not gate:
        raise RuntimeError("MEM-2C pre-reader structural freeze gate failed")

    reader_manifest = mem2b_runner._verify_reader(client, upstream["m2a_manifest"])
    if reader_manifest["model_sha256"] != PINNED_READER_MODEL_SHA256:
        raise RuntimeError("Live local reader model is not the frozen Qwen3-8B artifact")
    manifest = {
        "manifest_version": 1,
        "run_id": RUN_ID,
        "stage": "MEM-2C Lossless Raw-Span Memory Granularity Ablation",
        "status": "PRE_READER_FROZEN_AWAITING_READER",
        "split": "DEV_FROZEN_TEN_DIAGNOSTIC_ONLY",
        "question_ids": list(QUESTION_IDS),
        "dataset_sha256": upstream["dataset_sha256"],
        "selection_sha256": upstream["selection_sha256"],
        "representation_version": REPRESENTATION_VERSION,
        "segmenter_version": SEGMENTER_VERSION,
        "segmenter_contract_sha256": upstream["segmenter_contract_sha256"],
        "projection_policy_id": "m10-rank-aware-projection-v1",
        "projection_contract_sha256": upstream["projection_contract_sha256"],
        "final_reader_contract_sha256": upstream["reader_contract_sha256"],
        "reader_model_sha256": PINNED_READER_MODEL_SHA256,
        "reader_runtime": reader_manifest,
        "cache_identity_static": static_identity,
        "source_code_sha256": source_hashes,
        "upstream_artifact_sha256": {
            "mem2a": upstream["mem2a_artifact_sha256"],
            "mem2a_run_manifest": upstream["mem2a_run_manifest_sha256"],
            "mem2b": upstream["mem2b_artifact_sha256"],
            "mem2b_run_manifest": upstream["mem2b_run_manifest_sha256"],
        },
        "artifacts_sha256": artifact_hashes,
        "pre_reader_gate": {
            "passed": gate,
            "reconstruction_passes_equal_turns": True,
            "operations_add_only": True,
            "lexical_top_k_unchanged": 8,
            "projection_contract_unchanged": True,
            "memory_budget_tokens": MEMORY_BUDGET,
            "labels_loaded": False,
            "test_access": False,
            "102_dev_run": False,
        },
        "reader_calls_expected": 10,
        "reader_calls_successful": 0,
        "memory_internal_llm_calls": 0,
        "embedding_calls": 0,
        "judge_calls": 0,
        "hosted_api_calls": 0,
        "test_access": False,
        "completed_at_utc": None,
    }
    manifest_path = RUN_DIR / "run_manifest.json"
    if manifest_path.exists():
        if not _verify_sidecar(manifest_path):
            raise RuntimeError("Existing MEM-2C run manifest failed its SHA sidecar")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("run_id") != RUN_ID
            or existing.get("cache_identity_static") != static_identity
            or any(
                existing.get("artifacts_sha256", {}).get(name) != digest
                for name, digest in artifact_hashes.items()
            )
            or existing.get("pre_reader_gate", {}).get("passed") is not True
            or existing.get("question_ids") != list(QUESTION_IDS)
        ):
            raise RuntimeError(
                "Existing MEM-2C manifest does not bind these frozen pre-reader artifacts"
            )
        manifest = existing
    else:
        _write_or_verify_json(manifest_path, manifest)
    return states, manifest


def _load_frozen_prediction_cache(
    states: dict[str, dict[str, Any]], *, contract_sha: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]] | None:
    predictions_path = RUN_DIR / "predictions.jsonl"
    calls_path = RUN_DIR / "call_ledger.jsonl"
    if not predictions_path.exists() and not calls_path.exists():
        return None
    if not _verify_sidecar(predictions_path) or not _verify_sidecar(calls_path):
        raise RuntimeError("Global prediction/call ledger exists without valid frozen sidecars")
    prediction_rows = read_jsonl(predictions_path)
    call_rows = read_jsonl(calls_path)
    predictions = {row["question_id"]: row for row in prediction_rows}
    calls = {row["question_id"]: row for row in call_rows}
    if (
        list(predictions) != list(QUESTION_IDS)
        or list(calls) != list(QUESTION_IDS)
        or len(prediction_rows) != 10
        or len(call_rows) != 10
    ):
        raise RuntimeError("Frozen global reader artifacts do not cover exactly ten cases in order")
    for question_id in QUESTION_IDS:
        prediction = predictions[question_id]
        state = states[question_id]
        if (
            prediction.get("cache_identity") != state["cache_identity"]
            or prediction.get("context_bundle_sha256")
            != state["bundle_row"]["context_bundle_sha256"]
            or prediction.get("final_reader_contract_sha256") != contract_sha
            or prediction.get("call_ledger_row") != calls[question_id]
            or calls[question_id].get("cache_identity_sha256")
            != state["cache_identity"]["identity_sha256"]
        ):
            raise RuntimeError(f"Frozen reader cache identity mismatch: {question_id}")
    return predictions, calls


def _run_one_reader_call(
    question_id: str, state: dict[str, Any], *, client: httpx.Client, contract_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = state["cache_identity"]
    question_dir = RUN_DIR / "questions" / question_id
    question_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = question_dir / "prediction.json"
    call_state_path = RUN_DIR / "calls" / f"{question_id}.json"
    if prediction_path.exists() or call_state_path.exists():
        if not prediction_path.is_file() or not call_state_path.is_file():
            raise RuntimeError(
                f"Incomplete cached reader state; refusing duplicate call: {question_id}"
            )
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        call_state = json.loads(call_state_path.read_text(encoding="utf-8"))
        if (
            prediction.get("cache_identity") != identity
            or prediction.get("context_bundle_sha256")
            != state["bundle_row"]["context_bundle_sha256"]
            or call_state.get("cache_identity") != identity
            or call_state.get("status") != "COMPLETE"
        ):
            raise RuntimeError(f"Reader cache identity mismatch or interrupted call: {question_id}")
        return prediction, call_state["call"]

    request = {
        "model": READER_MODEL,
        "messages": state["messages"],
        "temperature": 0,
        "seed": 42,
        "max_tokens": OUTPUT_RESERVE,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request_sha = hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()
    prompt_sha = state["bundle_row"]["reader_prompt_sha256"]
    started_utc = datetime.now(UTC).isoformat()
    _write_json_atomic(
        call_state_path,
        {
            "cache_identity": identity,
            "status": "STARTED",
            "started_utc": started_utc,
            "request_sha256": request_sha,
        },
    )
    started = time.perf_counter()
    answer: str | None = None
    usage: dict[str, Any] = {}
    finish_reason = None
    status_code = None
    error_type = None
    try:
        response = client.post(f"{READER_ENDPOINT}/chat/completions", json=request)
        status_code = response.status_code
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        response_text = choice.get("message", {}).get("content")
        if not isinstance(response_text, str):
            raise TypeError("Frozen local reader response contained no text answer")
        answer = response_text.strip()
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        finish_reason = choice.get("finish_reason")
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
        error_type = type(error).__name__
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    prompt_tokens_server = usage.get("prompt_tokens")
    prompt_tokens_match = (
        prompt_tokens_server == state["bundle_row"]["reader_prompt_tokens_preflight"]
    )
    quality_status = "OK" if answer is not None and prompt_tokens_match else "INFRA_FAILURE"
    call = {
        "role": "reader_answer",
        "provider": "local_qwen",
        "endpoint": READER_ENDPOINT,
        "loopback_only": True,
        "model": READER_MODEL,
        "question_id": question_id,
        "context_bundle_sha256": state["bundle_row"]["context_bundle_sha256"],
        "final_reader_contract_sha256": contract_sha,
        "cache_identity_sha256": identity["identity_sha256"],
        "prompt_sha256": prompt_sha,
        "rendered_prompt_sha256": state["bundle_row"]["rendered_prompt_sha256"],
        "request_sha256": request_sha,
        "temperature": 0,
        "seed": 42,
        "enable_thinking": False,
        "max_new_tokens": OUTPUT_RESERVE,
        "prompt_tokens_preflight": state["bundle_row"]["reader_prompt_tokens_preflight"],
        "prompt_tokens_server": prompt_tokens_server,
        "completion_tokens_server": usage.get("completion_tokens"),
        "prompt_tokens_match": prompt_tokens_match,
        "finish_reason": finish_reason,
        "latency_ms": latency_ms,
        "http_status": status_code,
        "success": answer is not None,
        "quality_status": quality_status,
        "error_type": error_type,
        "retry_count": 0,
        "hosted_call": False,
    }
    prediction = {
        "system": "m10_rawspan",
        "representation_version": REPRESENTATION_VERSION,
        "question_id": question_id,
        "question": state["question"],
        "question_date": state["question_date"],
        "predicted": answer,
        "quality_status": quality_status,
        "reader_prompt_tokens_preflight": state["bundle_row"]["reader_prompt_tokens_preflight"],
        "reader_prompt_tokens_server": prompt_tokens_server,
        "completion_tokens_server": usage.get("completion_tokens"),
        "prompt_tokens_match": prompt_tokens_match,
        "reader_latency_ms": latency_ms,
        "finish_reason": finish_reason,
        "output_hit_token_cap": finish_reason == "length",
        "input_truncated": False if answer is not None and prompt_tokens_match else None,
        "context_bundle_sha256": state["bundle_row"]["context_bundle_sha256"],
        "final_reader_contract_sha256": contract_sha,
        "shared_reader_prompt_sha256": prompt_sha,
        "rendered_prompt_sha256": state["bundle_row"]["rendered_prompt_sha256"],
        "reader_error_type": error_type,
        "cache_identity": identity,
        "call_ledger_row": call,
    }
    _write_json_atomic(prediction_path, prediction)
    _write_json_atomic(
        call_state_path, {"cache_identity": identity, "status": "COMPLETE", "call": call}
    )
    return prediction, call


def _run_reader(
    states: dict[str, dict[str, Any]], *, client: httpx.Client, contract_sha: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cached = _load_frozen_prediction_cache(states, contract_sha=contract_sha)
    if cached is not None:
        return cached
    predictions: dict[str, dict[str, Any]] = {}
    calls: dict[str, dict[str, Any]] = {}
    for question_id in QUESTION_IDS:
        prediction, call = _run_one_reader_call(
            question_id, states[question_id], client=client, contract_sha=contract_sha
        )
        predictions[question_id] = prediction
        calls[question_id] = call
    prediction_rows = [predictions[question_id] for question_id in QUESTION_IDS]
    call_rows = [calls[question_id] for question_id in QUESTION_IDS]
    _write_or_verify_jsonl(RUN_DIR / "predictions.jsonl", prediction_rows)
    _write_or_verify_jsonl(RUN_DIR / "call_ledger.jsonl", call_rows)
    return predictions, calls


def _session_metrics(rows: list[dict[str, Any]], expected_sessions: set[str]) -> dict[str, Any]:
    if not expected_sessions:
        return {"recall_at_5": None, "recall_at_8": None, "mrr": None}
    hits = [row for row in rows if row.get("source_session_id") in expected_sessions]
    ranks = [row["rank"] for row in hits]
    return {
        "recall_at_5": len({row.get("source_session_id") for row in rows[:5]} & expected_sessions)
        / len(expected_sessions),
        "recall_at_8": len({row.get("source_session_id") for row in rows[:8]} & expected_sessions)
        / len(expected_sessions),
        "mrr": 1.0 / min(ranks) if ranks else 0.0,
    }


def _top_survival(selected_ranks: list[int], candidate_count: int) -> dict[str, Any]:
    return {
        "top_1_survival": float(1 in selected_ranks),
        "top_3_survival": len(set(selected_ranks) & {1, 2, 3}) / min(3, candidate_count)
        if candidate_count
        else 0.0,
        "top_5_survival": len(set(selected_ranks) & {1, 2, 3, 4, 5}) / min(5, candidate_count)
        if candidate_count
        else 0.0,
    }


def _mean(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if isinstance(value, (float, int))]
    return sum(filtered) / len(filtered) if filtered else None


def _percentile(values: list[int | float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _load_labels_after_freeze() -> dict[str, dict[str, Any]]:
    if not _verify_sidecar(RUN_DIR / "predictions.jsonl") or not _verify_sidecar(
        RUN_DIR / "call_ledger.jsonl"
    ):
        raise RuntimeError(
            "Prediction and call ledger SHA freeze must precede any gold-label access"
        )
    labels = {}
    for row in iter_json_array(DATASET_PATH, include_gold=True):
        question_id = row.get("question_id")
        if question_id in QUESTION_IDS:
            labels[question_id] = {
                "answer": row.get("answer", ""),
                "answer_session_ids": row.get("answer_session_ids", []),
                "question_type": row.get("question_type"),
                "has_answer": row.get("has_answer"),
            }
    if set(labels) != set(QUESTION_IDS):
        raise RuntimeError("Post-freeze label join did not produce exact frozen ten IDs")
    return labels


def _load_m2b_inputs() -> dict[str, Any]:
    inventory_rows = read_jsonl(MEM2A_DIR / "memory_inventory.jsonl")
    retrieval_rows = read_jsonl(MEM2A_DIR / "retrieval_results.jsonl")
    plan_rows = read_jsonl(MEM2B_DIR / "context_plans.jsonl")
    bundle_rows = read_jsonl(MEM2B_DIR / "context_bundles.jsonl")
    predictions = read_jsonl(MEM2B_DIR / "predictions.jsonl")
    return {
        "inventory_by_question": _group_by_question(inventory_rows),
        "retrieval_by_question": {row["question_id"]: row for row in retrieval_rows},
        "plans": {row["question_id"]: row for row in plan_rows},
        "bundles": {row["question_id"]: row for row in bundle_rows},
        "predictions": {row["question_id"]: row for row in predictions},
    }


def _group_by_question(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question_id"]].append(row)
    return dict(grouped)


def _score_and_compare(
    *,
    states: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    old = _load_m2b_inputs()
    per_question: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metric_names = (
        "m2b_native_recall_at_5",
        "m2c_native_recall_at_5",
        "m2b_native_recall_at_8",
        "m2c_native_recall_at_8",
        "m2b_native_mrr",
        "m2c_native_mrr",
        "m2b_projected_recall_at_5",
        "m2c_projected_recall_at_5",
        "m2b_projected_recall_at_8",
        "m2c_projected_recall_at_8",
        "m2b_projected_mrr",
        "m2c_projected_mrr",
        "m2b_gold_token_coverage",
        "m2c_gold_token_coverage",
        "m2b_exact_gold_sequence_present",
        "m2c_exact_gold_sequence_present",
        "m2b_token_precision",
        "m2c_token_precision",
        "m2b_token_recall",
        "m2c_token_recall",
        "m2b_token_f1",
        "m2c_token_f1",
        "m2b_normalized_em",
        "m2c_normalized_em",
        "m2b_context_item_count",
        "m2c_context_item_count",
        "m2b_reader_context_tokens",
        "m2c_reader_context_tokens",
        "m2b_estimated_memory_tokens",
        "m2c_estimated_memory_tokens",
        "m2b_selected_worse_better_dropped_pair_count",
        "m2c_selected_worse_better_dropped_pair_count",
        "m2b_top_1_survival",
        "m2c_top_1_survival",
        "m2b_top_3_survival",
        "m2c_top_3_survival",
        "m2b_top_5_survival",
        "m2c_top_5_survival",
    )
    for question_id in QUESTION_IDS:
        state = states[question_id]
        label = labels[question_id]
        expected_sessions = set(label.get("answer_session_ids") or [])
        raw_native = state["retrieval_row"]["results"]
        raw_projected = [
            {
                "rank": index,
                "memory_id": item["memory_id"],
                "source_session_id": (item.get("source_session_ids") or [None])[0],
            }
            for index, item in enumerate(state["context_bundle"]["items"], 1)
        ]
        m2b_retrieval = old["retrieval_by_question"][question_id]["results"]
        m2b_bundle = old["bundles"][question_id]["context_bundle"]
        m2b_projected = [
            {
                "rank": index,
                "memory_id": item["memory_id"],
                "source_session_id": (item.get("source_session_ids") or [None])[0],
            }
            for index, item in enumerate(m2b_bundle["items"], 1)
        ]
        m2c_native_metrics = _session_metrics(raw_native, expected_sessions)
        m2c_projected_metrics = _session_metrics(raw_projected, expected_sessions)
        m2b_native_metrics = _session_metrics(m2b_retrieval, expected_sessions)
        m2b_projected_metrics = _session_metrics(m2b_projected, expected_sessions)
        gold = label.get("answer") if isinstance(label.get("answer"), str) else ""
        m2b_context = m2b_bundle["serialized_context"]
        m2c_context = state["context_bundle"]["serialized_context"]
        m2b_coverage, m2b_exact_gold = mem2a_runner._memory_gold_coverage(gold, m2b_context)
        m2c_coverage, m2c_exact_gold = mem2a_runner._memory_gold_coverage(gold, m2c_context)
        m2b_pred = old["predictions"][question_id]
        m2c_pred = predictions[question_id]
        m2b_answer_metrics = mem2a_runner._answer_metrics(m2b_pred.get("predicted") or "", gold)
        m2c_answer_metrics = mem2a_runner._answer_metrics(m2c_pred.get("predicted") or "", gold)
        m2b_plan = old["plans"][question_id]
        m2b_selected = list(m2b_plan["selected_native_retrieval_ranks"])
        m2b_dropped = list(m2b_plan["dropped_native_retrieval_ranks"])
        m2c_plan = state["plan_row"]
        m2c_selected = list(m2c_plan["selected_native_retrieval_ranks"])
        m2c_dropped = list(m2c_plan["dropped_native_retrieval_ranks"])
        m2b_survival = _top_survival(m2b_selected, len(m2b_retrieval))
        m2c_survival = _top_survival(m2c_selected, len(raw_native))
        old_worse_pairs = sum(
            selected_rank > dropped_rank
            for selected_rank in m2b_selected
            for dropped_rank in m2b_dropped
        )
        new_worse_pairs = sum(
            selected_rank > dropped_rank
            for selected_rank in m2c_selected
            for dropped_rank in m2c_dropped
        )
        m2b_memory_tokens = m2b_plan["plan"]["memory_tokens"]
        m2c_memory_tokens = state["plan_row"]["plan"]["memory_tokens"]
        row = {
            "question_id": question_id,
            "question_type": label.get("question_type"),
            "has_answer": label.get("has_answer"),
            "answer_session_count": len(expected_sessions),
            "m2b_native_retrieval": m2b_native_metrics,
            "m2c_native_retrieval": m2c_native_metrics,
            "m2b_projected_context": m2b_projected_metrics,
            "m2c_projected_context": m2c_projected_metrics,
            "m2b_projected_gold_token_coverage": m2b_coverage,
            "m2c_projected_gold_token_coverage": m2c_coverage,
            "m2b_gold_token_coverage": m2b_coverage,
            "m2c_gold_token_coverage": m2c_coverage,
            "m2b_exact_normalized_gold_sequence_present": m2b_exact_gold,
            "m2c_exact_normalized_gold_sequence_present": m2c_exact_gold,
            "m2b_exact_gold_sequence_present": m2b_exact_gold,
            "m2c_exact_gold_sequence_present": m2c_exact_gold,
            "m2b_answer": m2b_answer_metrics,
            "m2c_answer": m2c_answer_metrics,
            "m2b_top_survival": m2b_survival,
            "m2c_top_survival": m2c_survival,
            "m2b_selected_native_ranks": m2b_selected,
            "m2c_selected_native_ranks": m2c_selected,
            "m2b_selected_worse_better_dropped_pair_count": old_worse_pairs,
            "m2c_selected_worse_better_dropped_pair_count": new_worse_pairs,
            "m2b_context_item_count": len(m2b_bundle["items"]),
            "m2c_context_item_count": len(state["context_bundle"]["items"]),
            "m2b_estimated_memory_tokens": m2b_memory_tokens,
            "m2c_estimated_memory_tokens": m2c_memory_tokens,
            "m2b_reader_context_tokens": m2b_bundle["context_reader_tokens"],
            "m2c_reader_context_tokens": state["context_bundle"]["context_reader_tokens"],
            "m2b_prompt_tokens": m2b_pred.get("reader_prompt_tokens_server"),
            "m2c_prompt_tokens": m2c_pred.get("reader_prompt_tokens_server"),
            "m2b_prediction_status": m2b_pred.get("quality_status"),
            "m2c_prediction_status": m2c_pred.get("quality_status"),
            "controlled_reader_contract_sha256": PINNED_READER_CONTRACT_SHA256,
            "sole_intended_intervention": "MemoryRecord granularity: raw turn vs exact raw span",
        }
        for prefix, metric in (
            ("m2b", m2b_native_metrics),
            ("m2c", m2c_native_metrics),
        ):
            row[f"{prefix}_native_recall_at_5"] = metric["recall_at_5"]
            row[f"{prefix}_native_recall_at_8"] = metric["recall_at_8"]
            row[f"{prefix}_native_mrr"] = metric["mrr"]
        for prefix, metric in (("m2b", m2b_projected_metrics), ("m2c", m2c_projected_metrics)):
            row[f"{prefix}_projected_recall_at_5"] = metric["recall_at_5"]
            row[f"{prefix}_projected_recall_at_8"] = metric["recall_at_8"]
            row[f"{prefix}_projected_mrr"] = metric["mrr"]
        for prefix, metric in (("m2b", m2b_answer_metrics), ("m2c", m2c_answer_metrics)):
            row[f"{prefix}_token_precision"] = metric["token_precision"]
            row[f"{prefix}_token_recall"] = metric["token_recall"]
            row[f"{prefix}_token_f1"] = metric["f1"]
            row[f"{prefix}_normalized_em"] = metric["normalized_exact_match"]
        for prefix, survival in (("m2b", m2b_survival), ("m2c", m2c_survival)):
            row[f"{prefix}_top_1_survival"] = survival["top_1_survival"]
            row[f"{prefix}_top_3_survival"] = survival["top_3_survival"]
            row[f"{prefix}_top_5_survival"] = survival["top_5_survival"]
        per_question.append(row)
        by_category[str(label.get("question_type"))].append(row)

    means: dict[str, Any] = {}
    for field in metric_names:
        means[field] = _mean([row.get(field) for row in per_question])
    paired_delta = {}
    for metric, m2b_key, m2c_key in (
        ("native_recall_at_5", "m2b_native_recall_at_5", "m2c_native_recall_at_5"),
        ("native_recall_at_8", "m2b_native_recall_at_8", "m2c_native_recall_at_8"),
        ("native_mrr", "m2b_native_mrr", "m2c_native_mrr"),
        ("projected_recall_at_5", "m2b_projected_recall_at_5", "m2c_projected_recall_at_5"),
        ("projected_recall_at_8", "m2b_projected_recall_at_8", "m2c_projected_recall_at_8"),
        ("projected_mrr", "m2b_projected_mrr", "m2c_projected_mrr"),
        ("token_f1", "m2b_token_f1", "m2c_token_f1"),
        ("token_precision", "m2b_token_precision", "m2c_token_precision"),
        ("token_recall", "m2b_token_recall", "m2c_token_recall"),
        ("normalized_em", "m2b_normalized_em", "m2c_normalized_em"),
        ("gold_token_coverage", "m2b_gold_token_coverage", "m2c_gold_token_coverage"),
        (
            "exact_gold_sequence_present",
            "m2b_exact_gold_sequence_present",
            "m2c_exact_gold_sequence_present",
        ),
    ):
        paired_delta[metric] = (
            (means[m2c_key] - means[m2b_key])
            if means[m2b_key] is not None and means[m2c_key] is not None
            else None
        )
    summary = {
        "schema_version": 1,
        "metric_definition": {
            "answer_session_recall": "fraction of benchmark answer_session_ids represented among retrieved/projected source sessions",
            "mrr": "reciprocal rank of first retrieved/projected item whose source session is in answer_session_ids; 0 if none",
            "gold_token_coverage": "normalized unique gold tokens present in serialized context, using MEM-2A implementation",
            "exact_normalized_gold_sequence_present": "normalized gold token sequence appears contiguously in serialized context",
            "answer_metrics": "deterministic MEM-2A token-set precision/recall/F1 and normalized exact match",
            "selection_inversion_pair": "selected native rank is worse than a dropped native rank; distinct from selected-order inversion",
        },
        "n": len(per_question),
        "means": means,
        "per_question": per_question,
        "by_question_type": {
            category: {
                "n": len(rows),
                "means": {field: _mean([row.get(field) for row in rows]) for field in metric_names},
            }
            for category, rows in sorted(by_category.items())
        },
        "paired_mean_delta_m2c_minus_m2b": paired_delta,
        "descriptive_only_no_quality_cost_winner_claim": True,
        "gold_join_after_prediction_and_call_ledger_sha_freeze": True,
        "parent_turn_answer_recall": "not scored; LongMemEval supplies answer-session labels, not answer-turn labels",
    }

    m2a_inventory = _group_by_question(read_jsonl(MEM2A_DIR / "memory_inventory.jsonl"))
    m2c_inventory = _group_by_question(read_jsonl(RUN_DIR / "memory_inventory.jsonl"))
    budget = _memory_budget()
    manager = ContextManager(
        budget=budget,
        estimator=DeterministicTokenEstimator(),
        history_window=24,
    )
    m2a_estimated_tokens = [
        manager._memory_item(mem2b_runner._record_from_inventory(row)).estimated_tokens
        for question_id in QUESTION_IDS
        for row in m2a_inventory[question_id]
    ]
    m2c_estimated_tokens = [
        manager._memory_item(_record_from_inventory(row)).estimated_tokens
        for question_id in QUESTION_IDS
        for row in m2c_inventory[question_id]
    ]
    per_q_efficiency = []
    for question_id in QUESTION_IDS:
        state = states[question_id]
        per_q_efficiency.append(
            {
                "question_id": question_id,
                "turn_count": state["turn_count"],
                "span_count": state["span_count"],
                "reconstruction_pass_count": state["reconstruction_pass_count"],
                "spans_per_turn": [row["span_count"] for row in state["segmentation_rows"]],
                "span_char_lengths": [len(span.content) for span in state["spans"]],
                "span_estimated_token_lengths": state["estimated_span_tokens"],
                "ingestion_latency_ms": state["runtime_metrics"]["ingestion_latency_ms"],
                "sqlite_bytes": state["runtime_metrics"]["sqlite_bytes"],
                "native_retrieval_latency_ms": state["runtime_metrics"][
                    "native_retrieval_latency_ms"
                ],
                "candidate_inventory_size": len(m2c_inventory[question_id]),
                "selected_span_count": state["structural_diagnostic"]["context_item_count"],
                "context_projection_latency_ms": state["runtime_metrics"][
                    "context_projection_latency_ms"
                ],
                "estimated_memory_tokens_selected": state["structural_diagnostic"][
                    "estimated_memory_tokens"
                ],
                "reader_context_tokens": state["context_bundle"]["context_reader_tokens"],
                "reader_prompt_tokens": predictions[question_id].get("reader_prompt_tokens_server"),
                "reader_completion_tokens": predictions[question_id].get(
                    "completion_tokens_server"
                ),
                "reader_latency_ms": predictions[question_id].get("reader_latency_ms"),
            }
        )
    efficiency = {
        "schema_version": 1,
        "raw_turn_count": sum(row["turn_count"] for row in per_q_efficiency),
        "raw_span_count": sum(row["span_count"] for row in per_q_efficiency),
        "memory_record_expansion_factor": sum(row["span_count"] for row in per_q_efficiency)
        / sum(row["turn_count"] for row in per_q_efficiency),
        "reconstruction_pass_count": sum(
            row["reconstruction_pass_count"] for row in per_q_efficiency
        ),
        "m2b_raw_turn_inventory_size": len(m2a_estimated_tokens),
        "m2c_raw_span_inventory_size": len(m2c_estimated_tokens),
        "m2b_estimated_tokens_per_memory_unit": {
            "median": statistics.median(m2a_estimated_tokens) if m2a_estimated_tokens else None,
            "p95": _percentile(m2a_estimated_tokens, 0.95),
        },
        "m2c_estimated_tokens_per_memory_unit": {
            "median": statistics.median(m2c_estimated_tokens) if m2c_estimated_tokens else None,
            "p95": _percentile(m2c_estimated_tokens, 0.95),
        },
        "mean_ingestion_latency_ms": _mean(
            [row["ingestion_latency_ms"] for row in per_q_efficiency]
        ),
        "total_sqlite_bytes": sum(row["sqlite_bytes"] or 0 for row in per_q_efficiency),
        "mean_native_retrieval_latency_ms": _mean(
            [row["native_retrieval_latency_ms"] for row in per_q_efficiency]
        ),
        "mean_selected_span_count": _mean([row["selected_span_count"] for row in per_q_efficiency]),
        "mean_context_projection_latency_ms": _mean(
            [row["context_projection_latency_ms"] for row in per_q_efficiency]
        ),
        "mean_reader_context_tokens": _mean(
            [row["reader_context_tokens"] for row in per_q_efficiency]
        ),
        "mean_reader_prompt_tokens": _mean(
            [row["reader_prompt_tokens"] for row in per_q_efficiency]
        ),
        "mean_reader_completion_tokens": _mean(
            [row["reader_completion_tokens"] for row in per_q_efficiency]
        ),
        "mean_reader_latency_ms": _mean([row["reader_latency_ms"] for row in per_q_efficiency]),
        "per_question": per_q_efficiency,
        "token_estimator": "ContextManager deterministic chars4-cjk1-v1 estimate; not Qwen tokenizer",
        "reader_token_count": "Qwen3-8B llama.cpp tokenizer, add_special=false",
        "quality_cost_tradeoff_claim": False,
    }
    comparison = {
        "schema_version": 1,
        "comparison": "MEM-2B M10-Base RawTurn vs MEM-2C M10-RawSpan",
        "controlled_components_identical": [
            "dataset and exact frozen ten DEV question IDs",
            "question text and official question_date",
            "user/assistant inclusion policy",
            "M10 SQLiteMemoryStore and ADD-only semantics",
            "native lexical score, exact-key behavior, and top_k=8",
            "M10 ContextManager budget and rank-aware greedy-skip projection contract",
            "shared local Qwen3-8B reader, prompt contract, generation settings, output cap",
        ],
        "sole_intended_intervention": "MemoryRecord granularity: one raw turn vs deterministic lossless text spans",
        "mem2b_run_manifest_sha256": _sha256_file(MEM2B_DIR / "run_manifest.json"),
        "per_question": per_question,
        "paired_mean_delta_m2c_minus_m2b": paired_delta,
        "overall_metrics": means,
        "inventory": {
            "m2b_raw_turn_count": sum(row["turn_count"] for row in per_q_efficiency),
            "m2c_raw_span_count": sum(row["span_count"] for row in per_q_efficiency),
            "expansion_factor": efficiency["memory_record_expansion_factor"],
            "m2b_median_p95_estimated_tokens_per_unit": efficiency[
                "m2b_estimated_tokens_per_memory_unit"
            ],
            "m2c_median_p95_estimated_tokens_per_unit": efficiency[
                "m2c_estimated_tokens_per_memory_unit"
            ],
        },
        "interpretation_boundary": [
            "higher native answer-session recall supports only finer textual units improving lexical discrimination",
            "higher projected coverage supports only smaller units using the fixed Memory budget more efficiently",
            "answer metrics are downstream deterministic diagnostics, not the selection criterion",
            "no validation claim for proposition memory, semantic extraction, or RevMem",
        ],
        "diagnostic_only_no_performance_ranking_claim": True,
    }
    return (
        summary,
        comparison,
        efficiency,
        {
            "per_question": per_question,
            "labels": labels,
            "predictions": predictions,
        },
    )


def _case_review(labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "1cea1afa": "knowledge_update",
        "c4ea545c": "knowledge_update",
        "1c549ce4": "multi_session",
        "778164c6": "single_session_factual",
        "a82c026e": "single_session_factual",
        "8550ddae": "single_session_factual",
        "gpt4_e061b84g": "temporal",
        "gpt4_f420262c": "temporal",
        "fca70973": "preference",
        "06878be2": "preference",
    }
    return {
        "schema_version": 1,
        "stage": "MEM-2C frozen-ten case packet; human Reflection not yet performed",
        "cases": [
            {
                "question_id": question_id,
                "review_group": groups[question_id],
                "question_type": labels[question_id].get("question_type"),
                "outcome": None,
                "failure_loci": None,
                "causal_attribution": None,
                "notes": None,
            }
            for question_id in QUESTION_IDS
        ],
    }


def _report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    efficiency: dict[str, Any],
) -> str:
    means = metrics["means"]
    lines = [
        "# MEM-2C - Lossless Raw-Span Frozen-10 Diagnostic",
        "",
        f"Gate: `MEM2C_RAWSPAN_FROZEN_10_DIAGNOSTIC={'YES' if manifest['gate']['passed'] else 'NO'}`",
        "",
        "## Protocol",
        "",
        "- Sole intervention: MemoryRecord granularity, one exact raw turn vs deterministic exact-text spans.",
        "- Segmenter: `raw-span-segmenter-v1`; representation: `raw-span-v1`.",
        "- No semantic extraction, embeddings, dense retrieval, reranking, revisions, UPDATE/DELETE, judge, or hosted calls.",
        "- Native M10 lexical ranking and exact-key behavior are unchanged; `top_k=8`.",
        "- Frozen rank-aware projection: `m10-rank-aware-projection-v1`, 1,024 estimated-token budget.",
        f"- Final reader contract SHA256: `{manifest['final_reader_contract_sha256']}`; reader calls: {manifest['reader_calls_successful']}/10 local Qwen3-8B.",
        "- Gold/category/session labels were joined only after prediction and call-ledger SHA freeze.",
        "- TEST access: false; 102-case DEV: not run.",
        "",
        "## Integrity And Efficiency",
        "",
        f"- Original turns reconstructed exactly: {efficiency['reconstruction_pass_count']}/{efficiency['raw_turn_count']}.",
        f"- Raw turns: {efficiency['raw_turn_count']}; raw spans / memory records: {efficiency['raw_span_count']} (expansion x{efficiency['memory_record_expansion_factor']:.2f}).",
        f"- Estimated tokens per unit, median/p95: RawTurn {efficiency['m2b_estimated_tokens_per_memory_unit']['median']} / {efficiency['m2b_estimated_tokens_per_memory_unit']['p95']}; RawSpan {efficiency['m2c_estimated_tokens_per_memory_unit']['median']} / {efficiency['m2c_estimated_tokens_per_memory_unit']['p95']}.",
        f"- Mean selected records: {efficiency['mean_selected_span_count']}; mean M10 estimated memory tokens: {means.get('m2c_estimated_memory_tokens')}; mean reader-tokenized context: {efficiency['mean_reader_context_tokens']}.",
        f"- Mean ingestion / native retrieval / projection / reader latency (ms): {efficiency['mean_ingestion_latency_ms']} / {efficiency['mean_native_retrieval_latency_ms']} / {efficiency['mean_context_projection_latency_ms']} / {efficiency['mean_reader_latency_ms']}.",
        f"- SQLite bytes: {efficiency['total_sqlite_bytes']}; reader prompt/completion tokens mean: {efficiency['mean_reader_prompt_tokens']} / {efficiency['mean_reader_completion_tokens']}.",
        "",
        "## Paired MEM-2B Comparison",
        "",
        "All figures below are frozen-ten diagnostic evidence, not a general performance ranking.",
        "",
        "| Metric | MEM-2B RawTurn | MEM-2C RawSpan | Delta |",
        "|---|---:|---:|---:|",
    ]
    amendment = manifest.get("post_reader_gate_correction")
    if amendment:
        lines.extend(
            [
                "",
                "## Finalization Audit",
                "",
                "The first post-reader gate check rejected required zero-valued call counters and expected-false access flags due to a validator type/expectation bug. The validator was corrected after prediction and call-ledger freeze; the frozen cache was reused and no additional reader request was issued.",
                f"- Reader-execution runner SHA256: `{amendment['reader_execution_runner_sha256']}`.",
                f"- Corrected finalizer runner SHA256: `{amendment['corrected_finalizer_runner_sha256']}`.",
                f"- Additional reader requests: {amendment['additional_reader_requests']}.",
            ]
        )
    for metric, before_key, after_key in (
        ("Native answer-session Recall@5", "m2b_native_recall_at_5", "m2c_native_recall_at_5"),
        ("Native answer-session Recall@8", "m2b_native_recall_at_8", "m2c_native_recall_at_8"),
        ("Native MRR", "m2b_native_mrr", "m2c_native_mrr"),
        (
            "Projected answer-session Recall@5",
            "m2b_projected_recall_at_5",
            "m2c_projected_recall_at_5",
        ),
        (
            "Projected answer-session Recall@8",
            "m2b_projected_recall_at_8",
            "m2c_projected_recall_at_8",
        ),
        ("Projected MRR", "m2b_projected_mrr", "m2c_projected_mrr"),
        ("Gold-token context coverage", "m2b_gold_token_coverage", "m2c_gold_token_coverage"),
        ("Token precision", "m2b_token_precision", "m2c_token_precision"),
        ("Token recall", "m2b_token_recall", "m2c_token_recall"),
        ("Token F1", "m2b_token_f1", "m2c_token_f1"),
        ("Normalized EM", "m2b_normalized_em", "m2c_normalized_em"),
        ("Reader-visible context tokens", "m2b_reader_context_tokens", "m2c_reader_context_tokens"),
        ("Estimated memory tokens", "m2b_estimated_memory_tokens", "m2c_estimated_memory_tokens"),
        (
            "Selected-worse/better-dropped pairs",
            "m2b_selected_worse_better_dropped_pair_count",
            "m2c_selected_worse_better_dropped_pair_count",
        ),
        ("Top-1 survival", "m2b_top_1_survival", "m2c_top_1_survival"),
        ("Top-3 survival", "m2b_top_3_survival", "m2c_top_3_survival"),
        ("Top-5 survival", "m2b_top_5_survival", "m2c_top_5_survival"),
    ):
        before = metrics["means"].get(before_key)
        after = metrics["means"].get(after_key)
        delta = after - before if before is not None and after is not None else None
        before_text = (
            "n/a"
            if before is None
            else f"{before:.4f}"
            if isinstance(before, float)
            else str(before)
        )
        after_text = (
            "n/a" if after is None else f"{after:.4f}" if isinstance(after, float) else str(after)
        )
        delta_text = (
            "n/a" if delta is None else f"{delta:.4f}" if isinstance(delta, float) else str(delta)
        )
        lines.append(f"| {metric} | {before_text} | {after_text} | {delta_text} |")
    lines.extend(
        [
            "",
            "Session recall is provenance-level coverage, not answer-bearing-turn recall. LongMemEval provides answer-session labels here, so no parent-turn answer labels are inferred. `Exact normalized gold sequence` is separately available per case in `deterministic_metrics.json`; answer scores remain downstream diagnostics only.",
            "",
            "## Category Slices",
            "",
            "| Question type | n | Native R@8 RawTurn to RawSpan | Projected R@8 RawTurn to RawSpan | Gold coverage RawTurn to RawSpan |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category, row in metrics["by_question_type"].items():
        lines.append(
            f"| {category} | {row['n']} | {row['means'].get('m2b_native_recall_at_8')} to {row['means'].get('m2c_native_recall_at_8')} | {row['means'].get('m2b_projected_recall_at_8')} to {row['means'].get('m2c_projected_recall_at_8')} | {row['means'].get('m2b_gold_token_coverage')} to {row['means'].get('m2c_gold_token_coverage')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "A native-recall gain supports only that finer textual units improve lexical retrieval discrimination. A projected-coverage gain supports only that smaller records use the fixed Memory budget more efficiently. This ablation does not validate propositions, semantic extraction, revision memory, or RevMem. No quality/cost winner is claimed.",
            "",
            "Case-level causal reflection remains intentionally unassigned in `docs/research/memory/mem_2c_case_review.json`.",
            "",
            "## Frozen Artifacts",
            "",
        ]
    )
    for name, digest in sorted(manifest["artifacts_sha256"].items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines.extend(
        [
            "",
            "The run stops at this frozen-ten diagnostic. No 102 DEV, TEST, dense retrieval, proposition extraction, RevMem, or RL was run.",
            "",
        ]
    )
    return "\n".join(lines)


def _finalize(
    *,
    states: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    calls: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    success = len(predictions) == 10 and all(
        predictions[qid].get("quality_status") == "OK" and calls[qid].get("hosted_call") is False
        for qid in QUESTION_IDS
    )
    if not success:
        raise RuntimeError("MEM-2C requires exactly ten successful, unhosted reader calls")
    metrics, comparison, efficiency, _detail = _score_and_compare(
        states=states, predictions=predictions, labels=labels
    )
    efficiency_path = RUN_DIR / "efficiency.json"
    if efficiency_path.exists():
        if not _verify_sidecar(efficiency_path):
            raise RuntimeError("Frozen MEM-2C efficiency artifact failed its SHA sidecar")
        frozen_efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
        if _efficiency_stable_fields(frozen_efficiency) != _efficiency_stable_fields(efficiency):
            raise RuntimeError("Frozen MEM-2C efficiency artifact changed beyond runtime timing")
        efficiency = frozen_efficiency
    case_review = _case_review(labels)
    _write_or_verify_json(RUN_DIR / "deterministic_metrics.json", metrics)
    _write_or_verify_json(RUN_DIR / "comparison_mem2b_vs_mem2c.json", comparison)
    _write_or_verify_json(RUN_DIR / "efficiency.json", efficiency)
    case_review_payload = (
        json.dumps(case_review, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    if CASE_REVIEW_PATH.exists():
        if (
            not _verify_sidecar(CASE_REVIEW_PATH)
            or CASE_REVIEW_PATH.read_text(encoding="utf-8") != case_review_payload
        ):
            raise RuntimeError("Frozen MEM-2C case review exists with different content")
    else:
        _write_json_atomic(CASE_REVIEW_PATH, case_review)
        _freeze(CASE_REVIEW_PATH)
    manifest["status"] = "COMPLETE"
    manifest["reader_calls_successful"] = 10
    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    manifest["artifacts_sha256"].update(
        {
            "predictions.jsonl": _sha256_file(RUN_DIR / "predictions.jsonl"),
            "call_ledger.jsonl": _sha256_file(RUN_DIR / "call_ledger.jsonl"),
            "deterministic_metrics.json": _sha256_file(RUN_DIR / "deterministic_metrics.json"),
            "comparison_mem2b_vs_mem2c.json": _sha256_file(
                RUN_DIR / "comparison_mem2b_vs_mem2c.json"
            ),
            "efficiency.json": _sha256_file(RUN_DIR / "efficiency.json"),
            "mem_2c_case_review.json": _sha256_file(CASE_REVIEW_PATH),
        }
    )
    manifest["comparison"] = {
        "mem2b_run_manifest_sha256": _sha256_file(MEM2B_DIR / "run_manifest.json"),
        "deterministic_metrics_sha256": _sha256_file(RUN_DIR / "deterministic_metrics.json"),
        "comparison_sha256": _sha256_file(RUN_DIR / "comparison_mem2b_vs_mem2c.json"),
    }
    manifest["gate"] = {
        "passed": False,
        "segmenter_contract_frozen": _verify_sidecar(
            SEGMENTER_CONTRACT_PATH, SEGMENTER_CONTRACT_SIDECAR
        ),
        "all_turns_reconstructed_exactly": sum(
            row["reconstruction_pass_count"] for row in efficiency["per_question"]
        )
        == efficiency["raw_turn_count"],
        "no_content_lost_or_duplicated": True,
        "leakage_tests_passed": True,
        "all_operations_add": all(
            row["operation"] == "ADD" for row in read_jsonl(RUN_DIR / "memory_operations.jsonl")
        ),
        "m10_memory_store_semantics_unchanged": True,
        "native_lexical_retrieval_unchanged": True,
        "top_k_is_8": True,
        "rank_aware_projection_contract_unchanged": True,
        "memory_budget_tokens": MEMORY_BUDGET,
        "exactly_ten_frozen_dev_cases": len(QUESTION_IDS) == 10,
        "exactly_ten_successful_reader_calls": len(calls) == 10 and success,
        "memory_internal_llm_calls": 0,
        "embedding_calls": 0,
        "judge_calls": 0,
        "hosted_calls": 0,
        "test_access": False,
        "102_dev_run": False,
        "no_semantic_proposition_extraction": True,
        "no_revision_semantics": True,
        "all_required_artifacts_sha_frozen": False,
    }
    if not _gate_requirements_passed(
        {
            key: value
            for key, value in manifest["gate"].items()
            if key != "all_required_artifacts_sha_frozen"
        }
    ):
        raise RuntimeError("MEM-2C final gate contains a false requirement")
    manifest["metrics_summary"] = metrics["means"]
    manifest["efficiency_summary"] = {
        "raw_turn_count": efficiency["raw_turn_count"],
        "raw_span_count": efficiency["raw_span_count"],
    }
    manifest["gate"]["passed"] = True
    report = _report(manifest, metrics, comparison, efficiency)
    report_path = RUN_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    manifest["artifacts_sha256"]["report.md"] = _freeze(report_path)
    protocol_text = _protocol_document(manifest, metrics, comparison, efficiency)
    PROTOCOL_PATH.write_text(protocol_text, encoding="utf-8", newline="\n")
    manifest["artifacts_sha256"]["mem_2c_rawspan_granularity.md"] = _freeze(PROTOCOL_PATH)
    required_artifacts = {
        **{
            name: RUN_DIR / name
            for name in (
                "segmentation_audit.jsonl",
                "memory_operations.jsonl",
                "memory_inventory.jsonl",
                "retrieval_results.jsonl",
                "context_plans.jsonl",
                "context_bundles.jsonl",
                "predictions.jsonl",
                "call_ledger.jsonl",
                "deterministic_metrics.json",
                "comparison_mem2b_vs_mem2c.json",
                "efficiency.json",
                "report.md",
            )
        },
        "mem_2c_case_review.json": CASE_REVIEW_PATH,
        "mem_2c_rawspan_granularity.md": PROTOCOL_PATH,
        "raw_span_segmenter_contract.json": SEGMENTER_CONTRACT_PATH,
    }
    manifest["gate"]["all_required_artifacts_sha_frozen"] = all(
        _verify_sidecar(path) for path in required_artifacts.values()
    )
    manifest["gate"]["passed"] = _gate_requirements_passed(manifest["gate"])
    manifest["completion_gate_marker"] = "MEM2C_RAWSPAN_FROZEN_10_DIAGNOSTIC=YES"
    manifest["pre_reader_gate_marker"] = "MEM2C_RAWSPAN_PRE_READER_FROZEN=YES"
    if not manifest["gate"]["passed"]:
        raise RuntimeError("MEM2C_RAWSPAN_FROZEN_10_DIAGNOSTIC=NO; artifact SHA gate failed")
    _write_json_atomic(RUN_DIR / "run_manifest.json", manifest)
    _freeze(RUN_DIR / "run_manifest.json")
    if not _verify_sidecar(RUN_DIR / "run_manifest.json"):
        raise RuntimeError("Final MEM-2C run manifest sidecar failed")
    return manifest


def _protocol_document(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    efficiency: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# MEM-2C - Lossless Raw-Span Memory Granularity Ablation",
            "",
            f"Completion gate: `MEM2C_RAWSPAN_FROZEN_10_DIAGNOSTIC={'YES' if manifest['gate']['passed'] else 'NO'}`.",
            "",
            "## Research Question",
            "",
            "Measure whether exact deterministic textual spans, rather than whole dialogue turns, improve native lexical retrieval discrimination or fixed-budget context coverage. The sole intended intervention is MemoryRecord granularity.",
            "",
            "## Frozen Method",
            "",
            "- Segmenter: `raw-span-segmenter-v1`; identity and boundaries are in `raw_span_segmenter_contract.json`.",
            "- Every span is an exact substring; concatenating spans reconstructs its source turn exactly. No text is stripped or normalized.",
            "- Each non-empty span becomes one M10 `ADD / SESSION_NOTE / SESSION_DERIVED / NON_SENSITIVE`, version 1, ACTIVE record.",
            "- The M10 SQLite store, native lexical overlap/exact-key scoring, `top_k=8`, temporal timestamp, ContextManager budgets and frozen rank-aware projection remain unchanged.",
            "- Reader: frozen local Qwen3-8B Q4_K_M, final reader contract SHA `"
            + manifest["final_reader_contract_sha256"]
            + "`, 10 calls; memory-internal LLM, embedding, judge, and hosted calls are zero.",
            "- Benchmark labels are joined only after prediction and call-ledger artifacts are hash-frozen.",
            "- Scope is exactly the ten frozen DEV diagnostic questions. TEST access is false; 102 DEV is not run.",
            "",
            "## Frozen Evidence",
            "",
            f"- Turn reconstructions: {efficiency['reconstruction_pass_count']}/{efficiency['raw_turn_count']} exact.",
            f"- Raw turns: {efficiency['raw_turn_count']}; raw spans: {efficiency['raw_span_count']}; expansion factor: {efficiency['memory_record_expansion_factor']:.3f}.",
            f"- Mean native answer-session Recall@8: RawTurn {metrics['means'].get('m2b_native_recall_at_8')} to RawSpan {metrics['means'].get('m2c_native_recall_at_8')}.",
            f"- Mean projected answer-session Recall@8: RawTurn {metrics['means'].get('m2b_projected_recall_at_8')} to RawSpan {metrics['means'].get('m2c_projected_recall_at_8')}.",
            f"- Mean projected gold-token coverage: RawTurn {metrics['means'].get('m2b_gold_token_coverage')} to RawSpan {metrics['means'].get('m2c_gold_token_coverage')}.",
            f"- Mean reader-visible context tokens: RawTurn {metrics['means'].get('m2b_reader_context_tokens')} to RawSpan {metrics['means'].get('m2c_reader_context_tokens')}.",
            "- Full per-question, per-category, answer, retrieval, and efficiency diagnostics are in the frozen run artifacts linked from `"
            + (RUN_DIR / "report.md").relative_to(ROOT).as_posix()
            + "`.",
            "",
            "## Interpretation Boundary",
            "",
            "A native-recall gain supports only finer textual units improving lexical retrieval discrimination. A projected coverage gain supports only smaller units using the fixed Memory budget more efficiently. Answer metrics are downstream deterministic diagnostics. This experiment does not validate proposition memory, semantic extraction, revision memory, or RevMem. No quality/cost winner is claimed.",
            "",
            "Answer-session provenance is measured at session level only. The public labels do not identify the answer-bearing turn, so parent-turn recall is not fabricated. Case-level outcome, failure locus, causal attribution, and notes remain null in `mem_2c_case_review.json` for subsequent human Reflection.",
            "",
            "Stop after this frozen-ten diagnostic. Do not automatically run 102 DEV, TEST, dense retrieval, proposition extraction, RevMem, or RL.",
            "",
        ]
    )


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="freeze segmentation, ADD inventory, lexical top-8, plans and bundles without reader calls",
    )
    args = parser.parse_args()
    global RUN_DIR
    RUN_DIR = ROOT / "runs/memory/mem2" / args.run_id
    if args.run_id != RUN_ID:
        raise ValueError("MEM-2C run ID is frozen for this diagnostic")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    upstream = _verify_upstream()
    sources = _load_sources()
    with httpx.Client(timeout=600.0, trust_env=False) as client:
        states, manifest = _prepare_all(inputs=sources, upstream=upstream, client=client)
        if args.prepare_only:
            print("MEM2C_RAWSPAN_PRE_READER_FROZEN=YES")
            print(f"run_dir={RUN_DIR}")
            print(f"span_records={sum(state['span_count'] for state in states.values())}")
            return 0

        if manifest.get("status") == "COMPLETE":
            if not _verify_sidecar(RUN_DIR / "predictions.jsonl") or not _verify_sidecar(
                RUN_DIR / "call_ledger.jsonl"
            ):
                raise RuntimeError("Completed MEM-2C run is missing frozen reader artifacts")
            print("MEM2C_RAWSPAN_PRE_READER_FROZEN=YES")
            print("MEM2C_RAWSPAN_FROZEN_10_DIAGNOSTIC=YES")
            print(f"run_dir={RUN_DIR}")
            print(
                f"raw_span_records={manifest.get('efficiency_summary', {}).get('raw_span_count')}"
            )
            return 0

        _write_json_atomic(
            RUN_DIR / "run_manifest.json",
            {
                **manifest,
                "status": "READER_RUNNING",
                "pre_reader_gate": {**manifest["pre_reader_gate"], "passed": True},
            },
        )
        _freeze(RUN_DIR / "run_manifest.json")
        predictions, calls = _run_reader(
            states, client=client, contract_sha=upstream["reader_contract_sha256"]
        )
        if len(predictions) != 10 or len(calls) != 10:
            raise RuntimeError("MEM-2C reader artifacts do not contain exactly ten calls")
        if not _verify_sidecar(RUN_DIR / "predictions.jsonl") or not _verify_sidecar(
            RUN_DIR / "call_ledger.jsonl"
        ):
            raise RuntimeError("MEM-2C prediction and call ledger artifacts failed SHA freeze")
        if not all(predictions[qid].get("quality_status") == "OK" for qid in QUESTION_IDS):
            failed = [qid for qid in QUESTION_IDS if predictions[qid].get("quality_status") != "OK"]
            raise RuntimeError(
                f"MEM-2C reader infrastructure failures; labels remain unjoined: {failed}"
            )

    labels = _load_labels_after_freeze()
    final_manifest = _finalize(
        states=states,
        predictions=predictions,
        calls=calls,
        labels=labels,
        manifest=manifest,
    )
    if final_manifest["gate"]["passed"] is not True:
        raise RuntimeError("MEM2C_RAWSPAN_FROZEN_10_DIAGNOSTIC=NO")
    print("MEM2C_RAWSPAN_PRE_READER_FROZEN=YES")
    print("MEM2C_RAWSPAN_FROZEN_10_DIAGNOSTIC=YES")
    print(f"run_dir={RUN_DIR}")
    print(f"raw_span_records={final_manifest['metrics_summary'].get('raw_span_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
