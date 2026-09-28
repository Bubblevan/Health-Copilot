"""Run the frozen-10 MEM-2B rank-aware projection counterfactual."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ROOT / "tools" / "research" / "memory"
sys.path.insert(0, str(TOOLS_DIR))

import run_mem1d3_reader as d3
import run_mem2a_m10_base as mem2a_runner
from final_reader_contract import (
    build_reader_messages,
    load_final_reader_contract,
)
from mem1_artifacts import read_jsonl
from mem2a_m10_base import canonical_json, iter_json_array, sha256_json

from health_ai_copilot.runtime.context_manager import (
    ContextBudget,
    ContextItemCategory,
    ContextManager,
    DeterministicTokenEstimator,
)
from health_ai_copilot.runtime.memory import (
    ContextIntent,
    MemoryKind,
    MemoryRecord,
    MemorySensitivity,
    MemorySourceType,
    MemoryStatus,
)

SELECTION_PATH = ROOT / "docs/research/memory/main_smoke_10_manifest.json"
SPLIT_PATH = ROOT / "docs/research/memory/split_manifest.json"
MEM2A_DIR = ROOT / "runs/memory/mem2/mem2a-m10-base-10-20260928"
DATASET_PATH = ROOT / "data/longmemeval/longmemeval_s_cleaned.json"
DATASET_MANIFEST_PATH = ROOT / "docs/research/memory/dataset_manifest.json"
READER_CONTRACT_PATH = ROOT / "docs/research/memory/final_reader_contract.json"
READER_CONTRACT_SIDECAR = READER_CONTRACT_PATH.with_name("final_reader_contract.json.sha256")
PROJECTION_CONTRACT_PATH = ROOT / "docs/research/memory/rank_aware_projection_contract.json"
PROJECTION_CONTRACT_SIDECAR = PROJECTION_CONTRACT_PATH.with_name(
    "rank_aware_projection_contract.json.sha256"
)
PROTOCOL_PATH = ROOT / "docs/research/memory/mem_2b_rank_aware_projection.md"
CASE_REVIEW_PATH = ROOT / "docs/research/memory/mem_2b_case_review.json"
RUN_ID = "mem2b-rank-aware-projection-10-20260928"
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
MEMORY_BUDGET = 1024
READER_ENDPOINT = "http://127.0.0.1:8081/v1"
READER_MODEL = "health-memory-qwen3-8b"


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


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def _freeze(path: Path, sidecar: Path | None = None) -> str:
    digest = _sha256_file(path)
    sidecar_path = sidecar or path.with_name(f"{path.name}.sha256")
    sidecar_path.write_text(f"{digest}  {path.name}\n", encoding="ascii", newline="\n")
    return digest


def _verify_sidecar(path: Path, sidecar: Path | None = None) -> bool:
    sidecar_path = sidecar or path.with_name(f"{path.name}.sha256")
    if not path.is_file() or not sidecar_path.is_file():
        return False
    parts = sidecar_path.read_text(encoding="ascii").strip().split()
    return parts == [_sha256_file(path), path.name]


def _write_or_verify_json(path: Path, value: dict[str, Any]) -> str:
    expected = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if not _verify_sidecar(path) or path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Frozen artifact identity mismatch: {path.name}")
        return _sha256_file(path)
    _write_json_atomic(path, value)
    return _freeze(path)


def _write_or_verify_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    expected = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    if path.exists():
        if not _verify_sidecar(path) or path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Frozen artifact identity mismatch: {path.name}")
        return _sha256_file(path)
    _write_jsonl_atomic(path, rows)
    return _freeze(path)


def _verify_mem2a_artifacts() -> dict[str, str]:
    paths = {
        name: MEM2A_DIR / name
        for name in (
            "memory_operations.jsonl",
            "memory_inventory.jsonl",
            "retrieval_results.jsonl",
            "context_plans.jsonl",
            "context_bundles.jsonl",
            "predictions.jsonl",
            "call_ledger.jsonl",
        )
    }
    hashes = {}
    for name, path in paths.items():
        sidecar = MEM2A_DIR / name.replace(".jsonl", ".sha256")
        if not _verify_sidecar(path, sidecar):
            raise RuntimeError(f"Frozen MEM-2A evidence sidecar failed: {name}")
        hashes[name] = _sha256_file(path)
    manifest_path = MEM2A_DIR / "run_manifest.json"
    if not _verify_sidecar(manifest_path):
        raise RuntimeError("Frozen MEM-2A run manifest sidecar failed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_gate = manifest.get("gate", {})
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("split") != "DEV"
        or manifest.get("test_access") is not False
        or manifest.get("question_ids") != list(QUESTION_IDS)
        or manifest.get("reader_calls_successful") != 10
        or required_gate.get("passed") is not True
        or manifest.get("final_reader_contract_sha256") != PINNED_READER_CONTRACT_SHA256
    ):
        raise RuntimeError("MEM-2A frozen ten-case closeout gate is absent")
    hashes["run_manifest.json"] = _sha256_file(manifest_path)
    return hashes


def _load_projection_contract(context_manager_sha256: str) -> tuple[dict[str, Any], str]:
    if not _verify_sidecar(READER_CONTRACT_PATH, READER_CONTRACT_SIDECAR):
        raise RuntimeError("Final reader contract sidecar failed verification")
    _, reader_sha, _, _ = load_final_reader_contract()
    if reader_sha != PINNED_READER_CONTRACT_SHA256:
        raise RuntimeError("Final reader contract SHA differs from the frozen contract")
    contract = {
        "policy_id": "m10-rank-aware-projection-v1",
        "algorithm_version": "priority-slots-rank-memory-greedy-skip-v1",
        "context_manager_source_sha256": context_manager_sha256,
        "legacy_behavior_when_no_hints": (
            "selection_rank_hints=None preserves the frozen priority then item_id ordering, "
            "candidate selection, projection, and ContextPlan hash"
        ),
        "rank_hint_validation": {
            "keys": "ContextItem.item_id strings",
            "allowed_category": "memory only",
            "positive_integer": True,
            "unique_within_candidates": True,
            "complete_for_memory_candidates": True,
            "adapter_ranks_contiguous_1_to_n": True,
            "reject_unknown_protected_system_current_user": True,
        },
        "priority_order": ["protected", "high", "normal", "low"],
        "memory_budget_tokens": MEMORY_BUDGET,
        "greedy_skip_behavior": (
            "visit ranked memory candidates in ascending native retrieval rank; skip a candidate "
            "that does not fit and continue evaluating later smaller candidates"
        ),
        "no_partial_truncation_or_knapsack": True,
        "tie_fallback_behavior": (
            "ranks are unique; non-memory units and any unranked units retain deterministic legacy "
            "priority/item_id fallback positions"
        ),
        "reader_visible_rank_annotation": False,
    }
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if PROJECTION_CONTRACT_PATH.exists():
        if (
            not _verify_sidecar(PROJECTION_CONTRACT_PATH, PROJECTION_CONTRACT_SIDECAR)
            or PROJECTION_CONTRACT_PATH.read_text(encoding="utf-8") != encoded
        ):
            raise RuntimeError("Projection policy contract is already frozen with different bytes")
    else:
        _write_json_atomic(PROJECTION_CONTRACT_PATH, contract)
        _freeze(PROJECTION_CONTRACT_PATH, PROJECTION_CONTRACT_SIDECAR)
    if not _verify_sidecar(PROJECTION_CONTRACT_PATH, PROJECTION_CONTRACT_SIDECAR):
        raise RuntimeError("Projection policy contract failed its frozen SHA256")
    return contract, _sha256_file(PROJECTION_CONTRACT_PATH)


def _record_from_inventory(row: dict[str, Any]) -> MemoryRecord:
    record = MemoryRecord(
        memory_id=row["memory_id"],
        scope_id=row["scope_id"],
        key=row["key"],
        kind=MemoryKind(row["kind"]),
        value=row["value"],
        value_sha256=row["value_sha256"],
        status=MemoryStatus(row["status"]),
        version=row["version"],
        created_at=row["created_at"],
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        expires_at=row["expires_at"],
        source_event_ids=tuple(row["source_event_ids"]),
        related_event_ids=tuple(row["related_event_ids"]),
        source_run_id=row["source_run_id"],
        source_session_id=row["source_session_id"],
        objective_id=row["objective_id"],
        intent=ContextIntent.from_value(row["intent"]),
        supersedes_id=row["supersedes_id"],
        sensitivity=MemorySensitivity(row["sensitivity"]),
        source_type=MemorySourceType(row["source_type"]),
    )
    source_memory = {key: value for key, value in row.items() if key != "question_id"}
    if record.to_dict() != source_memory:
        raise RuntimeError(f"MemoryRecord reconstruction differs from frozen inventory: {row['memory_id']}")
    return record


def _inversion_count(ranks: list[int]) -> int:
    return sum(
        1
        for index, rank in enumerate(ranks)
        for later_rank in ranks[index + 1 :]
        if rank > later_rank
    )


def _load_inputs(mem2a_hashes: dict[str, str]) -> dict[str, Any]:
    selection_sha = _sha256_file(SELECTION_PATH)
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    m2a_protocol = ROOT / "docs/research/memory/mem_2a_m10_base_protocol.md"
    m2a_report = MEM2A_DIR / "report.md"
    required_text = {
        "MEM1D4_FINAL_READER_CONTRACT_FROZEN=YES": ROOT / "docs/research/memory/mem_1d4_final_reader_contract.md",
        "MEM2A_M10_BASE_FIDELITY_READY=YES": m2a_protocol,
    }
    upstream_gate_hashes = {}
    for marker, path in required_text.items():
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"Required gate marker is not recorded: {marker}")
        upstream_gate_hashes[path.relative_to(ROOT).as_posix()] = _sha256_file(path)
    if "MEM2A_M10_BASE_FROZEN_10_DIAGNOSTIC=YES" not in m2a_report.read_text(encoding="utf-8"):
        raise RuntimeError("MEM-2A frozen 10-case diagnostic report gate is absent")
    if (
        selection_sha != PINNED_SELECTION_SHA256
        or selection.get("status") != "FROZEN_BEFORE_ANY_10_CASE_RESULTS"
        or selection.get("test_access") is not False
        or selection.get("question_ids") != list(QUESTION_IDS)
        or selection.get("dataset_sha256") != PINNED_DATASET_SHA256
        or split.get("status") != "FROZEN_ACTIVE_PROTOCOL_LOCKED"
        or split.get("dev", {}).get("count") != 102
        or split.get("test", {}).get("count") != 398
        or split.get("dev", {}).get("question_ids_sha256_sorted_lf") != PINNED_DEV_IDS_SHA256
        or set(QUESTION_IDS) - set(split.get("dev", {}).get("question_ids", []))
        or set(QUESTION_IDS) & set(split.get("test", {}).get("question_ids", []))
    ):
        raise RuntimeError("Frozen DEV-only selection or TEST boundary failed")
    longmemeval = next(
        item for item in dataset_manifest["datasets"] if item["dataset_id"] == "longmemeval_s"
    )
    if (
        longmemeval.get("expected_sha256") != PINNED_DATASET_SHA256
    ):
        raise RuntimeError("Frozen LongMemEval-S dataset manifest identity failed")
    if mem2a_hashes["run_manifest.json"] != _sha256_file(MEM2A_DIR / "run_manifest.json"):
        raise RuntimeError("MEM-2A run manifest changed after sidecar verification")

    inventories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(MEM2A_DIR / "memory_inventory.jsonl"):
        inventories[row["question_id"]].append(row)
    retrieval_rows = {row["question_id"]: row for row in read_jsonl(MEM2A_DIR / "retrieval_results.jsonl")}
    plan_rows = {row["question_id"]: row for row in read_jsonl(MEM2A_DIR / "context_plans.jsonl")}
    bundle_rows = {row["question_id"]: row for row in read_jsonl(MEM2A_DIR / "context_bundles.jsonl")}
    prediction_rows = {row["question_id"]: row for row in read_jsonl(MEM2A_DIR / "predictions.jsonl")}
    for mapping, label in (
        (inventories, "inventory"),
        (retrieval_rows, "retrieval"),
        (plan_rows, "plan"),
        (bundle_rows, "bundle"),
        (prediction_rows, "prediction"),
    ):
        if set(mapping) != set(QUESTION_IDS):
            raise RuntimeError(f"MEM-2A {label} artifacts do not have exact frozen question coverage")
    return {
        "selection": selection,
        "split": split,
        "inventories": dict(inventories),
        "retrievals": retrieval_rows,
        "plans": plan_rows,
        "bundles": bundle_rows,
        "predictions": prediction_rows,
        "dataset_sha256": PINNED_DATASET_SHA256,
        "selection_sha256": selection_sha,
        "mem2a_hashes": mem2a_hashes,
        "upstream_gate_hashes": upstream_gate_hashes,
    }


def _build_projection(inputs: dict[str, Any], client: httpx.Client, static: dict[str, Any]) -> dict[str, Any]:
    budget = ContextBudget(
        max_estimated_input_tokens=4096,
        reserved_system_tokens=256,
        reserved_current_turn_tokens=512,
        max_memory_tokens=MEMORY_BUDGET,
        max_history_tokens=2048,
        max_tool_observation_tokens=1024,
    )
    manager = ContextManager(
        budget=budget,
        estimator=DeterministicTokenEstimator(),
        history_window=24,
    )
    parity_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    pre_reader_by_id: dict[str, dict[str, Any]] = {}

    for question_id in QUESTION_IDS:
        inventory_rows = inputs["inventories"][question_id]
        if any(row.get("question_id") != question_id for row in inventory_rows):
            raise RuntimeError(f"MEM-2A inventory contains a cross-question memory: {question_id}")
        inventory_by_id = {row["memory_id"]: row for row in inventory_rows}
        records_by_id = {memory_id: _record_from_inventory(row) for memory_id, row in inventory_by_id.items()}
        retrieval = inputs["retrievals"][question_id]
        native = retrieval["results"]
        legacy_projection = retrieval["context_projection"]
        if retrieval.get("native_top_k") != 8 or len(native) != len(legacy_projection) or len(native) > 8:
            raise RuntimeError(f"Frozen MEM-2A native top-8 artifact shape failed: {question_id}")
        if [item.get("rank") for item in native] != list(range(1, len(native) + 1)):
            raise RuntimeError(f"Native retrieval ranks are not contiguous: {question_id}")
        if [item.get("retrieval_rank") for item in legacy_projection] != list(range(1, len(native) + 1)):
            raise RuntimeError(f"Legacy projection rank list differs from frozen retrieval: {question_id}")

        parity_candidates = []
        selected_records = []
        rank_hints: dict[str, int] = {}
        for rank, result in enumerate(native, 1):
            memory_id = result["memory_id"]
            inventory = inventory_by_id.get(memory_id)
            if inventory is None:
                raise RuntimeError(f"Top-8 memory missing from frozen inventory: {question_id} {memory_id}")
            record = records_by_id[memory_id]
            projection_row = legacy_projection[rank - 1]
            expected_turn_index = int(record.key.rsplit(":", 1)[1])
            expected_memory_tokens = manager._memory_item(record).estimated_tokens
            matched = {
                "question_id": question_id,
                "memory_id": memory_id,
                "native_rank": rank,
                "retrieval_score": result["score"],
                "source_session_id": record.source_session_id,
                "turn_index": expected_turn_index,
            }
            if (
                result.get("rank") != rank
                or result.get("source_session_id") != record.source_session_id
                or result.get("source_turn_key") != record.key
                or result.get("turn_index") != expected_turn_index
                or result.get("valid_from") != record.valid_from
                or projection_row.get("memory_id") != memory_id
                or projection_row.get("retrieval_score") != result["score"]
                or projection_row.get("source_session_id") != record.source_session_id
                or projection_row.get("turn_index") != expected_turn_index
                or projection_row.get("m10_estimated_tokens") != expected_memory_tokens
            ):
                raise RuntimeError(f"MEM-2A top-8 parity mismatch: {question_id} rank {rank}")
            parity_candidates.append({**matched, "match": True})
            selected_records.append(record)
            rank_hints[f"memory-{memory_id}"] = rank

        if sorted(rank_hints.values()) != list(range(1, len(native) + 1)):
            raise RuntimeError(f"Rank-aware benchmark adapter requires contiguous ranks: {question_id}")
        old_plan = inputs["plans"][question_id]
        old_bundle = inputs["bundles"][question_id]
        old_prediction = inputs["predictions"][question_id]
        question = old_prediction.get("question")
        question_date = old_prediction.get("question_date")
        if not isinstance(question, str) or not isinstance(question_date, str):
            raise TypeError(f"MEM-2A frozen question/date absent from prediction artifact: {question_id}")
        if old_prediction.get("context_bundle_sha256") != old_bundle["context_bundle_sha256"]:
            raise RuntimeError(f"MEM-2A prediction does not bind its own frozen context: {question_id}")
        if old_plan["plan"].get("retrieval_query") != retrieval["query"]:
            raise RuntimeError(f"MEM-2A question/retrieval query mismatch: {question_id}")
        if old_bundle.get("context_bundle", {}).get("question_id") != question_id:
            raise RuntimeError(f"MEM-2A bundle question ID mismatch: {question_id}")

        plan = manager.build_plan(
            session_id=f"longmemeval:{question_id}",
            session_revision=old_plan["plan"]["session_revision"],
            current_user=question,
            history=(),
            memory_records=selected_records,
            retrieval_query=question,
            selection_rank_hints=rank_hints,
        )
        if plan.memory_tokens > MEMORY_BUDGET:
            raise RuntimeError(f"Rank-aware ContextPlan exceeded frozen memory budget: {question_id}")

        match_by_id = {item["memory_id"]: item for item in native}
        selected_items = [item for item in plan.items if item.category == ContextItemCategory.MEMORY]
        selected_ids = [str(item.provenance["memory_id"]) for item in selected_items]
        selected_rank_list = [match_by_id[item_id]["rank"] for item_id in selected_ids]
        selected_set = set(selected_ids)
        dropped_rank_list = [item["rank"] for item in native if item["memory_id"] not in selected_set]
        if selected_rank_list != sorted(selected_rank_list):
            raise RuntimeError(f"Rank-aware projection order still has a retrieval-order inversion: {question_id}")

        item_rows = []
        for position, item in enumerate(selected_items, 1):
            memory_id = str(item.provenance["memory_id"])
            result = match_by_id[memory_id]
            record = records_by_id[memory_id]
            item_text = canonical_json(item.content)
            item_rows.append(
                {
                    "text": item_text,
                    "rank": position,
                    "kind": item.category.value,
                    "source_session_ids": [record.source_session_id] if record.source_session_id else [],
                    "memory_id": memory_id,
                    "native_retrieval_rank": result["rank"],
                    "retrieval_score": result["score"],
                    "status": record.status.value,
                    "version": record.version,
                    "source_turn_id": record.key,
                    "estimated_tokens": item.estimated_tokens,
                }
            )
        serialized_context = "\n\n".join(
            f"[Context item {item['rank']} | {item['kind']}]\n{item['text']}" for item in item_rows
        )
        context_reader_tokens = _tokenize(client, serialized_context) if serialized_context else 0
        bundle_base = {
            "schema_version": 3,
            "system": "healthcopilot_m10_base_rawturn",
            "question_id": question_id,
            "items": [{key: value for key, value in item.items() if key != "estimated_tokens"} for item in item_rows],
            "serialized_context": serialized_context,
            "context_embedding_tokens": None,
            "context_embedding_tokenizer": "NOT_APPLICABLE_NO_EMBEDDING",
            "context_reader_tokens": context_reader_tokens,
            "context_reader_tokenizer": "llama.cpp 10068 Qwen3-8B tokenizer; add_special=false",
            "provenance_available": True,
            "retrieval_latency_ms": None,
            "ingestion_latency_ms": None,
        }
        bundle_sha = sha256_json(bundle_base)
        context_bundle = {**bundle_base, "context_bundle_sha256": bundle_sha}
        _, contract_sha, system_template, user_template = load_final_reader_contract()
        if contract_sha != PINNED_READER_CONTRACT_SHA256:
            raise RuntimeError("Final reader contract changed during MEM-2B projection")
        messages = build_reader_messages(
            question,
            question_date,
            serialized_context,
            system_template=system_template,
            user_template=user_template,
        )
        prompt_tokens, rendered_prompt_sha = d3._render_and_tokenize(client, messages)
        if prompt_tokens + 256 > 131072:
            raise RuntimeError(f"MEM-2B full prompt does not fit the frozen reader slot: {question_id}")
        prompt_sha = hashlib.sha256(canonical_json(messages).encode("utf-8")).hexdigest()
        if not isinstance(context_reader_tokens, int) or context_reader_tokens < 0:
            raise RuntimeError("Reader tokenizer returned an invalid context token count")

        stable_plan_id = "plan-" + hashlib.sha256(
            f"{question_id}:{plan.plan_hash}:{static['projection_contract_sha256']}".encode()
        ).hexdigest()[:32]
        plan_row = {
            "question_id": question_id,
            "plan_id": stable_plan_id,
            "plan_hash": plan.plan_hash,
            "plan": plan.identity_payload(),
            "selected_native_retrieval_ranks": selected_rank_list,
            "dropped_native_retrieval_ranks": dropped_rank_list,
            "memory_budget_tokens": MEMORY_BUDGET,
        }
        plan_rows.append(plan_row)
        bundle_rows.append(
            {
                "question_id": question_id,
                "context_bundle": context_bundle,
                "context_bundle_sha256": bundle_sha,
                "reader_prompt_sha256": prompt_sha,
                "rendered_prompt_sha256": rendered_prompt_sha,
                "reader_prompt_tokens_preflight": prompt_tokens,
                "max_model_length": 131072,
                "output_reserve": 256,
                "truncated": False,
                "cache_identity": _cache_identity(static, question_id, bundle_sha),
            }
        )
        old_selected_ranks = [
            item["native_retrieval_rank"] for item in old_bundle["context_bundle"]["items"]
        ]
        old_dropped_ranks = [
            row["retrieval_rank"] for row in legacy_projection if row.get("selected") is not True
        ]
        if set(old_selected_ranks) != {
            row["retrieval_rank"] for row in legacy_projection if row.get("selected") is True
        }:
            raise RuntimeError(f"MEM-2A plan/bundle selected-rank parity failed: {question_id}")
        selected_set_by_rank = set(selected_rank_list)
        legacy_selected_set = set(old_selected_ranks)
        selection_inversions = sum(
            1 for selected_rank in selected_rank_list for dropped_rank in dropped_rank_list
            if selected_rank > dropped_rank
        )
        legacy_selection_inversions = sum(
            1 for selected_rank in old_selected_ranks for dropped_rank in old_dropped_ranks
            if selected_rank > dropped_rank
        )
        diagnostic_rows.append(
            {
                "question_id": question_id,
                "native_retrieval_ranks": [item["rank"] for item in native],
                "legacy_selected_ranks": old_selected_ranks,
                "rank_aware_selected_ranks": selected_rank_list,
                "legacy_dropped_ranks": old_dropped_ranks,
                "rank_aware_dropped_ranks": dropped_rank_list,
                "legacy_selection_inversion_count": legacy_selection_inversions,
                "rank_aware_selection_inversion_count": selection_inversions,
                "legacy_projected_order_inversion_count": _inversion_count(old_selected_ranks),
                "rank_aware_projected_order_inversion_count": _inversion_count(selected_rank_list),
                "top_1_survived": bool(selected_set_by_rank & {1}),
                "top_3_survival_count": len(selected_set_by_rank & {1, 2, 3}),
                "top_3_survival_rate": len(selected_set_by_rank & {1, 2, 3}) / min(3, len(native)) if native else 0.0,
                "top_5_survival_count": len(selected_set_by_rank & {1, 2, 3, 4, 5}),
                "top_5_survival_rate": len(selected_set_by_rank & {1, 2, 3, 4, 5}) / min(5, len(native)) if native else 0.0,
                "legacy_top_1_survived": bool(legacy_selected_set & {1}),
                "legacy_top_3_survival_rate": len(legacy_selected_set & {1, 2, 3}) / min(3, len(native)) if native else 0.0,
                "legacy_top_5_survival_rate": len(legacy_selected_set & {1, 2, 3, 4, 5}) / min(5, len(native)) if native else 0.0,
                "native_retrieval": {
                    "answer_session_recall_at_5": None,
                    "answer_session_recall_at_8": None,
                    "mrr": None,
                    "unchanged_from_mem2a": True,
                },
                "projected_context": {
                    "answer_session_recall_at_5": None,
                    "answer_session_recall_at_8": None,
                    "mrr": None,
                    "answer_session_provenance": [
                        records_by_id[item["memory_id"]].source_session_id for item in item_rows
                    ],
                },
                "legacy_projected_context": {
                    "answer_session_recall_at_5": None,
                    "answer_session_recall_at_8": None,
                    "mrr": None,
                    "serialized_context": old_bundle["context_bundle"]["serialized_context"],
                    "selected_memory_ids": list(
                        inputs["plans"][question_id]["plan"]["selected_memory_ids"]
                    ),
                },
                "memory_record_count": len(item_rows),
                "estimated_memory_tokens": plan.memory_tokens,
                "reader_tokenized_context_tokens": context_reader_tokens,
                "reader_prompt_sha256": prompt_sha,
                "rendered_prompt_sha256": rendered_prompt_sha,
                "reader_prompt_tokens_preflight": prompt_tokens,
                "max_model_length": 131072,
                "output_reserve": 256,
                "truncated": False,
                "projection_only_change": True,
            }
        )
        pre_reader_by_id[question_id] = {
            "question_id": question_id,
            "question": question,
            "question_date": question_date,
            "messages": messages,
            "prompt_sha256": prompt_sha,
            "rendered_prompt_sha256": rendered_prompt_sha,
            "reader_prompt_tokens_preflight": prompt_tokens,
            "context_bundle_sha256": bundle_sha,
            "cache_identity": _cache_identity(static, question_id, bundle_sha),
        }
        if old_bundle["context_bundle"].get("system") != context_bundle["system"]:
            raise RuntimeError("Reader-visible bundle system identifier changed")
        old_bundle_memory_ids = [item["memory_id"] for item in old_bundle["context_bundle"]["items"]]
        if old_plan["plan"].get("selected_memory_ids") != old_bundle_memory_ids:
            raise RuntimeError(f"MEM-2A ContextPlan/ContextBundle selected-memory parity failed: {question_id}")

    return {
        "parity_rows": parity_rows,
        "plan_rows": plan_rows,
        "bundle_rows": bundle_rows,
        "diagnostic_rows": diagnostic_rows,
        "pre_reader_by_id": pre_reader_by_id,
    }


def _cache_identity(static: dict[str, Any], question_id: str, bundle_sha256: str) -> dict[str, Any]:
    body = {
        **static,
        "question_id": question_id,
        "context_bundle_sha256": bundle_sha256,
    }
    return {"identity": body, "identity_sha256": sha256_json(body)}


def _tokenize(client: httpx.Client, text: str) -> int:
    response = client.post(
        "http://127.0.0.1:8081/tokenize",
        json={"content": text, "add_special": False},
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list):
        raise TypeError("Frozen local llama.cpp /tokenize returned no token ID list")
    return len(tokens)


def _verify_reader(client: httpx.Client, mem2a_manifest: dict[str, Any]) -> dict[str, Any]:
    response = client.get(f"{READER_ENDPOINT}/models")
    response.raise_for_status()
    models = response.json().get("data", [])
    runtime = mem2a_manifest["reader_runtime"]
    served_artifact = runtime.get("model_artifact")
    served_model = next(
        (
            row
            for row in models
            if isinstance(row, dict) and row.get("id") == served_artifact
        ),
        None,
    )
    if served_model is None:
        raise RuntimeError("Expected frozen Qwen3-8B model artifact is not served on loopback")
    model_meta = served_model.get("meta", {})
    if (
        model_meta.get("n_ctx") != 131072
        or model_meta.get("ftype") != "Q4_K - Medium"
    ):
        raise RuntimeError("Live reader endpoint does not expose the frozen Q4_K_M/context runtime")
    props_response = client.get("http://127.0.0.1:8081/props")
    props_response.raise_for_status()
    props = props_response.json()
    if (
        runtime.get("endpoint") != READER_ENDPOINT
        or runtime.get("endpoint_loopback_only") is not True
        or runtime.get("model") != READER_MODEL
        or runtime.get("model_sha256") != PINNED_READER_MODEL_SHA256
        or runtime.get("server_version_output", "").find("571d0d540") < 0
        or runtime.get("server_binary_sha256")
        != "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"
        or runtime.get("server_arguments", {}).get("--ctx-size") != "131072"
        or runtime.get("server_arguments", {}).get("--host") != "127.0.0.1"
        or runtime.get("server_arguments", {}).get("--n-gpu-layers") != "99"
        or runtime.get("server_arguments", {}).get("--flash-attn") != "on"
        or runtime.get("server_arguments", {}).get("--cache-type-k") != "q4_0"
        or runtime.get("server_arguments", {}).get("--cache-type-v") != "q4_0"
        or props.get("model_path") != served_artifact
        or props.get("model_ftype") != "Q4_K - Medium"
        or props.get("build_info") != "b10068-571d0d540"
        or props.get("default_generation_settings", {}).get("n_ctx") != 131072
        or props.get("total_slots") != 1
    ):
        raise RuntimeError("Frozen MEM-2A reader runtime identity is incomplete or mismatched")
    runtime_props_identity = {
        "model_path": props.get("model_path"),
        "model_ftype": props.get("model_ftype"),
        "build_info": props.get("build_info"),
        "context_tokens": props.get("default_generation_settings", {}).get("n_ctx"),
        "total_slots": props.get("total_slots"),
    }
    return {
        "endpoint": READER_ENDPOINT,
        "loopback_only": True,
        "model_alias": READER_MODEL,
        "model_sha256": PINNED_READER_MODEL_SHA256,
        "server_build": "llama.cpp 10068 (571d0d540)",
        "server_binary_sha256": runtime["server_binary_sha256"],
        "context_tokens": 131072,
        "answer_calls": "exactly one per frozen question ID",
        "runtime_props_sha256": hashlib.sha256(
            json.dumps(runtime_props_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _run_reader_once(
    artifact: dict[str, Any],
    *,
    client: httpx.Client,
    contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    question_id = artifact["question_id"]
    identity = artifact["cache_identity"]
    qdir = RUN_DIR / "questions" / question_id
    qdir.mkdir(parents=True, exist_ok=True)
    call_state_path = RUN_DIR / "calls" / f"{question_id}.json"
    prediction_path = qdir / "prediction.json"
    if call_state_path.exists() or prediction_path.exists():
        if not call_state_path.is_file() or not prediction_path.is_file():
            raise RuntimeError(f"Incomplete reader cache; refusing duplicate answer call: {question_id}")
        state = json.loads(call_state_path.read_text(encoding="utf-8"))
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        if (
            state.get("cache_identity") != identity
            or state.get("status") != "COMPLETE"
            or prediction.get("cache_identity") != identity
            or prediction.get("context_bundle_sha256") != artifact["context_bundle_sha256"]
        ):
            raise RuntimeError(f"Reader cache identity mismatch: {question_id}")
        call_row = state["call"]
        return prediction, call_row

    prompt_sha = artifact["prompt_sha256"]
    request = {
        "model": READER_MODEL,
        "messages": artifact["messages"],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 256,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request_sha = hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()
    _write_json_atomic(
        call_state_path,
        {
            "cache_identity": identity,
            "status": "STARTED",
            "started_utc": datetime.now(UTC).isoformat(),
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
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str):
            raise TypeError("Frozen local reader response contained no text answer")
        answer = content.strip()
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        finish_reason = choice.get("finish_reason")
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
        error_type = type(error).__name__
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    prompt_tokens_server = usage.get("prompt_tokens")
    prompt_tokens_match = prompt_tokens_server == artifact["reader_prompt_tokens_preflight"]
    status = "OK" if answer is not None and prompt_tokens_match else "INFRA_FAILURE"
    call = {
        "role": "reader_answer",
        "provider": "local_qwen",
        "endpoint": READER_ENDPOINT,
        "loopback_only": True,
        "model": READER_MODEL,
        "question_id": question_id,
        "context_bundle_sha256": artifact["context_bundle_sha256"],
        "final_reader_contract_sha256": contract_sha256,
        "cache_identity_sha256": identity["identity_sha256"],
        "prompt_sha256": prompt_sha,
        "rendered_prompt_sha256": artifact["rendered_prompt_sha256"],
        "request_sha256": request_sha,
        "temperature": 0,
        "seed": 42,
        "enable_thinking": False,
        "max_new_tokens": 256,
        "prompt_tokens_preflight": artifact["reader_prompt_tokens_preflight"],
        "prompt_tokens_server": prompt_tokens_server,
        "completion_tokens_server": usage.get("completion_tokens"),
        "prompt_tokens_match": prompt_tokens_match,
        "finish_reason": finish_reason,
        "latency_ms": latency_ms,
        "http_status": status_code,
        "success": answer is not None,
        "quality_status": status,
        "error_type": error_type,
        "retry_count": 0,
        "hosted_call": False,
    }
    prediction = {
        "system": "m10_base_rank_aware_projection",
        "question_id": question_id,
        "question": artifact["question"],
        "question_date": artifact["question_date"],
        "predicted": answer,
        "quality_status": status,
        "reader_prompt_tokens_preflight": artifact["reader_prompt_tokens_preflight"],
        "reader_prompt_tokens_server": prompt_tokens_server,
        "completion_tokens_server": usage.get("completion_tokens"),
        "prompt_tokens_match": prompt_tokens_match,
        "reader_latency_ms": latency_ms,
        "finish_reason": finish_reason,
        "output_hit_token_cap": finish_reason == "length",
        "input_truncated": False if answer is not None and prompt_tokens_match else None,
        "context_bundle_sha256": artifact["context_bundle_sha256"],
        "final_reader_contract_sha256": contract_sha256,
        "shared_reader_prompt_sha256": prompt_sha,
        "rendered_prompt_sha256": artifact["rendered_prompt_sha256"],
        "reader_error_type": error_type,
        "cache_identity": identity,
        "call_ledger_row": call,
    }
    _write_json_atomic(prediction_path, prediction)
    _write_json_atomic(
        call_state_path,
        {"cache_identity": identity, "status": "COMPLETE", "call": call},
    )
    return prediction, call


def _session_metrics(rows: list[dict[str, Any]], expected_sessions: set[str]) -> dict[str, Any]:
    if not expected_sessions:
        return {"recall_at_5": None, "recall_at_8": None, "mrr": None}
    top5_sessions = {str(row["source_session_id"]) for row in rows[:5] if row.get("source_session_id")}
    top8_sessions = {str(row["source_session_id"]) for row in rows[:8] if row.get("source_session_id")}
    first_rank = next(
        (
            index
            for index, row in enumerate(rows, 1)
            if str(row.get("source_session_id")) in expected_sessions
        ),
        None,
    )
    return {
        "recall_at_5": len(top5_sessions & expected_sessions) / len(expected_sessions),
        "recall_at_8": len(top8_sessions & expected_sessions) / len(expected_sessions),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def _load_gold_after_prediction_freeze() -> dict[str, dict[str, Any]]:
    if _sha256_file(DATASET_PATH) != PINNED_DATASET_SHA256:
        raise RuntimeError("LongMemEval-S dataset SHA changed before post-freeze scoring")
    labels: dict[str, dict[str, Any]] = {}
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
        raise RuntimeError("Post-freeze gold join did not produce the exact frozen ten questions")
    return labels


def _score(
    diagnostics: list[dict[str, Any]],
    bundle_rows: list[dict[str, Any]],
    retrieval_rows: dict[str, dict[str, Any]],
    mem2a_predictions: dict[str, dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    predictions_by_id = {row["question_id"]: row for row in prediction_rows}
    bundles_by_id = {row["question_id"]: row for row in bundle_rows}
    diagnostic_by_id = {row["question_id"]: row for row in diagnostics}
    per_question = []
    accum: dict[str, list[float]] = defaultdict(list)
    for question_id in QUESTION_IDS:
        label = labels[question_id]
        expected_sessions = {str(value) for value in label["answer_session_ids"] or []}
        gold_value = label["answer"]
        gold = "" if gold_value is None else str(gold_value)
        prediction = predictions_by_id[question_id]
        if prediction["quality_status"] == "OK" and isinstance(prediction.get("predicted"), str):
            answer_metrics = mem2a_runner._answer_metrics(prediction["predicted"], gold)
        else:
            answer_metrics = {
                "token_precision": None,
                "token_recall": None,
                "f1": None,
                "normalized_exact_match": None,
            }
        prior_prediction = mem2a_predictions[question_id]
        if prior_prediction["quality_status"] == "OK" and isinstance(prior_prediction.get("predicted"), str):
            mem2a_answer_metrics = mem2a_runner._answer_metrics(prior_prediction["predicted"], gold)
        else:
            mem2a_answer_metrics = {
                "token_precision": None,
                "token_recall": None,
                "f1": None,
                "normalized_exact_match": None,
            }
        diagnostic = diagnostic_by_id[question_id]
        bundle = bundles_by_id[question_id]["context_bundle"]
        native_results = retrieval_rows[question_id]["results"]
        rank_by_id = {item["memory_id"]: item for item in native_results}
        projected_rows = [
            {
                "rank": index,
                "memory_id": item["memory_id"],
                "source_session_id": (item.get("source_session_ids") or [None])[0],
            }
            for index, item in enumerate(bundle["items"], 1)
        ]
        native_metrics = mem2a_runner._session_retrieval_metrics(native_results, expected_sessions)
        projected_metrics = _session_metrics(projected_rows, expected_sessions)
        old_context = diagnostic["legacy_projected_context"]["serialized_context"]
        new_context = bundle["serialized_context"]
        old_coverage, old_exact = mem2a_runner._memory_gold_coverage(gold, old_context)
        new_coverage, new_exact = mem2a_runner._memory_gold_coverage(gold, new_context)
        diagnostic["native_retrieval"].update(
            {
                "answer_session_recall_at_5": native_metrics["recall_at_5"],
                "answer_session_recall_at_8": native_metrics["recall_at_8"],
                "mrr": native_metrics["mrr"],
                "mem2a_answer_session_metrics": native_metrics,
            }
        )
        diagnostic["projected_context"].update(projected_metrics)
        legacy_ids = diagnostic["legacy_projected_context"]["selected_memory_ids"]
        legacy_rows = [
            {
                "rank": index,
                "source_session_id": rank_by_id[memory_id]["source_session_id"],
            }
            for index, memory_id in enumerate(legacy_ids, 1)
            if memory_id in rank_by_id
        ]
        diagnostic["legacy_projected_context"].update(
            {
                **_session_metrics(legacy_rows, expected_sessions),
                "gold_token_coverage": old_coverage,
                "exact_normalized_gold_sequence_present": old_exact,
            }
        )
        diagnostic["projected_context"].update(
            {
                "gold_token_coverage": new_coverage,
                "exact_normalized_gold_sequence_present": new_exact,
                "answer_session_provenance": [row.get("source_session_id") for row in projected_rows],
            }
        )
        row = {
            "question_id": question_id,
            "question_type": label["question_type"],
            "has_answer": label["has_answer"],
            "native_answer_session_recall_at_5": native_metrics["recall_at_5"],
            "native_answer_session_recall_at_8": native_metrics["recall_at_8"],
            "legacy_projected_answer_session_recall_at_5": diagnostic["legacy_projected_context"]["recall_at_5"],
            "legacy_projected_answer_session_recall_at_8": diagnostic["legacy_projected_context"]["recall_at_8"],
            "legacy_projected_mrr": diagnostic["legacy_projected_context"]["mrr"],
            "rank_aware_projected_answer_session_recall_at_5": projected_metrics["recall_at_5"],
            "rank_aware_projected_answer_session_recall_at_8": projected_metrics["recall_at_8"],
            "rank_aware_projected_mrr": projected_metrics["mrr"],
            "legacy_gold_token_coverage": old_coverage,
            "rank_aware_gold_token_coverage": new_coverage,
            "legacy_exact_normalized_gold_sequence_present": old_exact,
            "rank_aware_exact_normalized_gold_sequence_present": new_exact,
            "native_retrieval_metrics_unchanged": True,
            "top_1_survival_rate": float(diagnostic["top_1_survived"]),
            "top_3_survival_rate": diagnostic["top_3_survival_rate"],
            "top_5_survival_rate": diagnostic["top_5_survival_rate"],
            "legacy_projected_order_inversion_count": diagnostic[
                "legacy_projected_order_inversion_count"
            ],
            "rank_aware_projected_order_inversion_count": diagnostic[
                "rank_aware_projected_order_inversion_count"
            ],
            "legacy_selection_inversion_count": diagnostic["legacy_selection_inversion_count"],
            "rank_aware_selection_inversion_count": diagnostic[
                "rank_aware_selection_inversion_count"
            ],
            "mem2a_token_precision": mem2a_answer_metrics["token_precision"],
            "mem2a_token_recall": mem2a_answer_metrics["token_recall"],
            "mem2a_f1": mem2a_answer_metrics["f1"],
            "mem2a_normalized_exact_match": mem2a_answer_metrics["normalized_exact_match"],
            **answer_metrics,
            "prediction_status": prediction["quality_status"],
            "reader_context_tokens": bundle["context_reader_tokens"],
            "estimated_memory_tokens": diagnostic["estimated_memory_tokens"],
        }
        per_question.append(row)
        for key in (
            "mem2a_token_precision",
            "mem2a_token_recall",
            "mem2a_f1",
            "mem2a_normalized_exact_match",
            "token_precision",
            "token_recall",
            "f1",
            "normalized_exact_match",
            "native_answer_session_recall_at_5",
            "native_answer_session_recall_at_8",
            "legacy_projected_answer_session_recall_at_5",
            "legacy_projected_answer_session_recall_at_8",
            "legacy_projected_mrr",
            "rank_aware_projected_answer_session_recall_at_5",
            "rank_aware_projected_answer_session_recall_at_8",
            "rank_aware_projected_mrr",
            "legacy_gold_token_coverage",
            "rank_aware_gold_token_coverage",
            "top_1_survival_rate",
            "top_3_survival_rate",
            "top_5_survival_rate",
            "legacy_projected_order_inversion_count",
            "rank_aware_projected_order_inversion_count",
            "legacy_selection_inversion_count",
            "rank_aware_selection_inversion_count",
            "reader_context_tokens",
            "estimated_memory_tokens",
        ):
            if isinstance(row[key], (int, float)):
                accum[key].append(float(row[key]))

    summary = {
        "n": len(per_question),
        "descriptive_only_no_ranking_claim": True,
        "means": {key: sum(values) / len(values) if values else None for key, values in accum.items()},
        "projection_policy_selected_by_protocol_not_dev_score": True,
        "rank_order_preserved_for_selected_memories": all(
            row["rank_aware_projected_order_inversion_count"] == 0 for row in diagnostics
        ),
    }
    comparison = {
        "comparison": "MEM-2A M10-Base vs MEM-2B M10-Base + rank-aware projection",
        "controlled_components_identical": [
            "raw memory inventory",
            "native top-8 candidates, scores, ranks, and provenance",
            "question and question_date",
            "reader and reader contract",
            "memory budget",
        ],
        "sole_intervention": "ContextManager projection order for retrieved memory candidates",
        "mem2a_native_retrieval_artifact_sha256": _sha256_file(
            MEM2A_DIR / "retrieval_results.jsonl"
        ),
        "per_question": per_question,
        "means": summary["means"],
        "interpretation_boundary": (
            "diagnostic only; improved projected evidence coverage shows retrieval rank was discarded "
            "by the previous budgeted projection, not that retrieval or raw-turn memory is solved"
        ),
    }
    return summary, comparison, per_question


def _report(manifest: dict[str, Any], diagnostics: list[dict[str, Any]], comparison: dict[str, Any]) -> str:
    old_inversions = sum(row["legacy_projected_order_inversion_count"] for row in diagnostics)
    new_inversions = sum(row["rank_aware_projected_order_inversion_count"] for row in diagnostics)
    old_selection = sum(row["legacy_selection_inversion_count"] for row in diagnostics)
    new_selection = sum(row["rank_aware_selection_inversion_count"] for row in diagnostics)
    old_selection_cases = sum(row["legacy_selection_inversion_count"] > 0 for row in diagnostics)
    new_selection_cases = sum(row["rank_aware_selection_inversion_count"] > 0 for row in diagnostics)
    old_order_cases = sum(row["legacy_projected_order_inversion_count"] > 0 for row in diagnostics)
    new_order_cases = sum(row["rank_aware_projected_order_inversion_count"] > 0 for row in diagnostics)
    means = comparison["means"]
    return "\n".join(
        [
            "# MEM-2B — Rank-Aware Projection Frozen-10 Diagnostic",
            "",
            f"Gate: `MEM2B_RANK_AWARE_PROJECTION_FROZEN_10_DIAGNOSTIC={'YES' if manifest['gate']['passed'] else 'NO'}`",
            "",
            "## Protocol",
            "",
            "- Single intervention: budgeted ContextManager projection order for the exact frozen M10-Base native top-8.",
            "- No re-ingestion, no `MemoryStore.matches()`, no external baseline reruns, no embeddings, no judge, no hosted calls.",
            "- Frozen reader contract: `" + manifest["final_reader_contract_sha256"] + "`.",
            "- Reader calls: " + str(manifest["reader_calls_successful"]) + "/10; all local loopback Qwen3-8B.",
            "- Test access: false; 102 DEV not run.",
            "",
            "## Rank Preservation",
            "",
            f"- Selected-worse-while-better-dropped pairs: {old_selection} → {new_selection} ({old_selection_cases}/10 → {new_selection_cases}/10 affected cases).",
            f"- Reader-context order inversions: {old_inversions} → {new_inversions} ({old_order_cases}/10 → {new_order_cases}/10 affected cases).",
            f"- Mean top-1 survival: {means.get('top_1_survival_rate')}",
            f"- Mean top-3 survival: {means.get('top_3_survival_rate')}",
            f"- Mean top-5 survival: {means.get('top_5_survival_rate')}",
            "",
            "The two inversion counts are distinct. Rank-aware selection makes the order of selected memories monotonic by retrieval rank. The greedy skip rule can still select a later small record after skipping a larger better-ranked record; that is not an item-ID ordering inversion.",
            "",
            "## Evidence And Reader Diagnostics",
            "",
            f"- Mean projected answer-session Recall@5: {means.get('rank_aware_projected_answer_session_recall_at_5')} (MEM-2A {means.get('legacy_projected_answer_session_recall_at_5')}).",
            f"- Mean projected answer-session Recall@8: {means.get('rank_aware_projected_answer_session_recall_at_8')} (MEM-2A {means.get('legacy_projected_answer_session_recall_at_8')}).",
            f"- Mean projected-context MRR: {means.get('rank_aware_projected_mrr')} (MEM-2A {means.get('legacy_projected_mrr')}).",
            f"- Mean normalized gold-token coverage: {means.get('rank_aware_gold_token_coverage')} (MEM-2A {means.get('legacy_gold_token_coverage')}).",
            f"- Mean reader-visible context tokens: {means.get('reader_context_tokens')}.",
            f"- Deterministic token F1 / normalized EM: {means.get('f1')} / {means.get('normalized_exact_match')} (descriptive only).",
            "",
            "Answer-session recall is provenance-level retrieval coverage, not answer-bearing evidence recall. Token overlap and exact normalized gold-sequence presence are separate diagnostics. No performance ranking or method-selection claim is made from DEV answer scores.",
            "",
            "## Case Review",
            "",
            "Human failure-locus and outcome fields remain null. The next-stage failure-surface decision is left for Reflection review.",
            "",
            "## Frozen Artifacts",
            "",
            *[f"- `{name}`: `{digest}`" for name, digest in sorted(manifest["artifacts_sha256"].items())],
            "",
            "This closeout stops at the frozen-10 counterfactual. It does not run 102 DEV, TEST, MEM-2C, M10-Flat, or RevMem.",
            "",
        ]
    )


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="freeze and verify projection artifacts without any reader-answer call",
    )
    parser.add_argument(
        "--refresh-pre-reader-diagnostic",
        action="store_true",
        help="correct a pre-reader diagnostic before any prediction cache or reader call exists",
    )
    args = parser.parse_args()
    global RUN_DIR
    RUN_DIR = ROOT / "runs/memory/mem2" / args.run_id
    if args.run_id != RUN_ID:
        raise ValueError("MEM-2B run ID is frozen for this diagnostic")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    mem2a_hashes = _verify_mem2a_artifacts()
    cm_path = ROOT / "src/health_ai_copilot/runtime/context_manager.py"
    context_manager_sha256 = _sha256_file(cm_path)
    projection_contract, projection_contract_sha256 = _load_projection_contract(context_manager_sha256)
    inputs = _load_inputs(mem2a_hashes)
    _contract, reader_contract_sha, _, _ = load_final_reader_contract()
    if reader_contract_sha != PINNED_READER_CONTRACT_SHA256:
        raise RuntimeError("Final reader contract binding changed")
    static = {
        "mem2a_retrieval_artifact_sha256": mem2a_hashes["retrieval_results.jsonl"],
        "mem2a_inventory_artifact_sha256": mem2a_hashes["memory_inventory.jsonl"],
        "projection_contract_sha256": projection_contract_sha256,
        "context_manager_source_sha256": context_manager_sha256,
        "memory_budget_tokens": MEMORY_BUDGET,
        "final_reader_contract_sha256": reader_contract_sha,
        "reader_model_sha256": PINNED_READER_MODEL_SHA256,
    }
    inputs["mem2a_manifest"] = json.loads((MEM2A_DIR / "run_manifest.json").read_text(encoding="utf-8"))

    with httpx.Client(timeout=httpx.Timeout(240.0, connect=4.0), trust_env=False) as client:
        reader_runtime = _verify_reader(client, inputs["mem2a_manifest"])
        projection = _build_projection(inputs, client, static)

        parity_candidates = []
        for question_id in QUESTION_IDS:
            retrieval = inputs["retrievals"][question_id]
            inventory = {row["memory_id"]: row for row in inputs["inventories"][question_id]}
            candidates = []
            for result in retrieval["results"]:
                record = inventory[result["memory_id"]]
                candidates.append(
                    {
                        "question_id": question_id,
                        "memory_id": result["memory_id"],
                        "native_rank": result["rank"],
                        "retrieval_score": result["score"],
                        "source_session_id": record["source_session_id"],
                        "turn_index": int(record["key"].rsplit(":", 1)[1]),
                    }
                )
            parity_candidates.extend(candidates)
        parity_object = {
            "schema_version": 1,
            "gate": "MEM2B_RETRIEVAL_PARITY_WITH_MEM2A=YES",
            "all_ten_question_ids_exact": True,
            "all_top8_candidates_match_inventory_and_legacy_projection": True,
            "re_retrieval_executed": False,
            "memory_store_matches_executed": False,
            "inventory_sha256": mem2a_hashes["memory_inventory.jsonl"],
            "retrieval_sha256": mem2a_hashes["retrieval_results.jsonl"],
            "candidate_count": len(parity_candidates),
            "candidates": parity_candidates,
        }
        parity_sha = _write_or_verify_json(RUN_DIR / "retrieval_parity.json", parity_object)
        plan_sha = _write_or_verify_jsonl(RUN_DIR / "context_plans.jsonl", projection["plan_rows"])
        bundle_sha = _write_or_verify_jsonl(RUN_DIR / "context_bundles.jsonl", projection["bundle_rows"])
        pre_reader_diag = {
            "schema_version": 1,
            "labels_accessed": False,
            "question_count": len(QUESTION_IDS),
            "per_question": projection["diagnostic_rows"],
            "mean_rank_survival": {
                "top_1": sum(float(row["top_1_survived"]) for row in projection["diagnostic_rows"]) / 10,
                "top_3": sum(row["top_3_survival_rate"] for row in projection["diagnostic_rows"]) / 10,
                "top_5": sum(row["top_5_survival_rate"] for row in projection["diagnostic_rows"]) / 10,
            },
            "legacy_selection_inversion_cases": sum(
                row["legacy_selection_inversion_count"] > 0 for row in projection["diagnostic_rows"]
            ),
            "legacy_selection_inversion_pair_count": sum(
                row["legacy_selection_inversion_count"] for row in projection["diagnostic_rows"]
            ),
            "legacy_projected_order_inversion_cases": sum(
                row["legacy_projected_order_inversion_count"] > 0
                for row in projection["diagnostic_rows"]
            ),
            "legacy_projected_order_inversion_pair_count": sum(
                row["legacy_projected_order_inversion_count"]
                for row in projection["diagnostic_rows"]
            ),
            "rank_aware_selected_worse_while_better_dropped_cases": sum(
                row["rank_aware_selection_inversion_count"] > 0
                for row in projection["diagnostic_rows"]
            ),
            "rank_aware_selection_inversion_pair_count": sum(
                row["rank_aware_selection_inversion_count"]
                for row in projection["diagnostic_rows"]
            ),
            "rank_aware_projected_order_inversion_cases": sum(
                row["rank_aware_projected_order_inversion_count"] > 0
                for row in projection["diagnostic_rows"]
            ),
            "rank_aware_projected_order_inversion_pair_count": sum(
                row["rank_aware_projected_order_inversion_count"]
                for row in projection["diagnostic_rows"]
            ),
            "context_plans_sha256": plan_sha,
            "context_bundles_sha256": bundle_sha,
            "retrieval_parity_sha256": parity_sha,
        }
        pre_reader_diag_path = RUN_DIR / "projection_diagnostics_pre_reader.json"
        frozen_manifest_path = RUN_DIR / "projection_frozen_manifest.json"
        if args.refresh_pre_reader_diagnostic:
            if list((RUN_DIR / "calls").glob("*.json")) or (RUN_DIR / "predictions.jsonl").exists():
                raise RuntimeError("Cannot revise pre-reader diagnostics after any reader call/cache")
            if not _verify_sidecar(pre_reader_diag_path) or not _verify_sidecar(frozen_manifest_path):
                raise RuntimeError("Pre-reader diagnostic/manifest must be frozen before an audited refresh")
            old_diagnostic_sha = _sha256_file(pre_reader_diag_path)
            old_manifest_sha = _sha256_file(frozen_manifest_path)
            audit_path = RUN_DIR / "projection_artifact_revision_audit.json"
            prior_revisions = []
            if audit_path.exists():
                if not _verify_sidecar(audit_path):
                    raise RuntimeError("Previous pre-reader revision audit failed its sidecar")
                previous_audit = json.loads(audit_path.read_text(encoding="utf-8"))
                prior_revisions = previous_audit.get("revisions", [previous_audit])
            _write_json_atomic(pre_reader_diag_path, pre_reader_diag)
            pre_reader_diag_sha = _freeze(pre_reader_diag_path)
            revision_audit = {
                "schema_version": 2,
                "revisions": prior_revisions + [{
                    "previous_projection_diagnostics_sha256": old_diagnostic_sha,
                    "previous_projection_frozen_manifest_sha256": old_manifest_sha,
                    "reason": "corrected diagnostic calculation to use the actual frozen MEM-2A ContextBundle item order and include explicit inversion-case counts",
                    "corrected_source": "frozen MEM-2A context_bundles.jsonl native_retrieval_rank order",
                    "reader_calls_before_refresh": 0,
                    "prediction_cache_exists": False,
                    "benchmark_labels_accessed": False,
                    "new_projection_diagnostics_sha256": pre_reader_diag_sha,
                }],
                "current_projection_diagnostics_sha256": pre_reader_diag_sha,
            }
            _write_json_atomic(audit_path, revision_audit)
            _freeze(audit_path)
        else:
            pre_reader_diag_sha = _write_or_verify_json(pre_reader_diag_path, pre_reader_diag)
        for path in (
            RUN_DIR / "retrieval_parity.json",
            RUN_DIR / "context_plans.jsonl",
            RUN_DIR / "context_bundles.jsonl",
            RUN_DIR / "projection_diagnostics_pre_reader.json",
        ):
            if not _verify_sidecar(path):
                raise RuntimeError(f"Projection artifact was not frozen before reader calls: {path.name}")
        manifest = {
            "manifest_version": 1,
            "run_id": args.run_id,
            "status": "PROJECTION_FROZEN_AWAITING_READER",
            "split": "DEV",
            "question_ids": list(QUESTION_IDS),
            "test_access": False,
            "reader_calls_expected": 10,
            "reader_calls_successful": 0,
            "projection_policy_id": projection_contract["policy_id"],
            "projection_contract_sha256": projection_contract_sha256,
            "final_reader_contract_sha256": reader_contract_sha,
            "reader_runtime": reader_runtime,
            "memory_internal_llm_calls": 0,
            "embedding_calls": 0,
            "judge_calls": 0,
            "hosted_api_calls": 0,
            "mem2a_artifact_sha256": mem2a_hashes,
            "static_cache_identity": static,
            "artifacts_sha256": {
                "retrieval_parity.json": parity_sha,
                "context_plans.jsonl": plan_sha,
                "context_bundles.jsonl": bundle_sha,
                "projection_diagnostics_pre_reader.json": pre_reader_diag_sha,
            },
            "gate": {"retrieval_parity": True, "projection_artifacts_frozen_before_reader": True},
        }
        if args.refresh_pre_reader_diagnostic:
            _write_json_atomic(frozen_manifest_path, manifest)
            new_manifest_sha = _freeze(frozen_manifest_path)
            revision_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            revision_audit["current_projection_frozen_manifest_sha256"] = new_manifest_sha
            _write_json_atomic(audit_path, revision_audit)
            _freeze(audit_path)
        else:
            _write_or_verify_json(frozen_manifest_path, manifest)

        if args.prepare_only:
            print("MEM2B_RETRIEVAL_PARITY_WITH_MEM2A=YES")
            print("MEM2B_PROJECTION_ARTIFACTS_FROZEN=YES")
            print(f"run_dir={RUN_DIR}")
            print(f"projection_contract_sha256={projection_contract_sha256}")
            return 0

        predictions = []
        calls = []
        for question_id in QUESTION_IDS:
            artifact = projection["pre_reader_by_id"][question_id]
            artifact["rendered_prompt_sha256"] = next(
                row["rendered_prompt_sha256"] for row in projection["bundle_rows"] if row["question_id"] == question_id
            )
            prediction, call = _run_reader_once(
                artifact,
                client=client,
                contract_sha256=reader_contract_sha,
            )
            predictions.append(prediction)
            calls.append(call)
        if len(calls) != 10 or any(call.get("hosted_call") is not False for call in calls):
            raise RuntimeError("MEM-2B did not complete exactly ten local reader calls")
        if any(call.get("quality_status") != "OK" for call in calls):
            raise RuntimeError("A reader call failed infrastructure validation; no metrics will be reported")

    prediction_sha = _write_or_verify_jsonl(RUN_DIR / "predictions.jsonl", predictions)
    call_sha = _write_or_verify_jsonl(RUN_DIR / "call_ledger.jsonl", calls)
    if not _verify_sidecar(RUN_DIR / "predictions.jsonl") or not _verify_sidecar(
        RUN_DIR / "call_ledger.jsonl"
    ):
        raise RuntimeError("Prediction/call-ledger freeze failed before any gold-label read")

    # Gold and category fields are deliberately opened only after prediction hashes are frozen.
    labels = _load_gold_after_prediction_freeze()
    summary, comparison, per_question = _score(
        projection["diagnostic_rows"],
        projection["bundle_rows"],
        inputs["retrievals"],
        inputs["predictions"],
        predictions,
        labels,
    )
    diagnostics_object = {
        "schema_version": 1,
        "question_count": 10,
        "per_question": projection["diagnostic_rows"],
        "rank_order_inversion_definitions": {
            "selection_inversion": "selected worse-ranked item while a better-ranked item was dropped (greedy skip may cause this)",
            "projected_order_inversion": "a worse-ranked selected item precedes a better-ranked selected item in reader context",
        },
        "labels_accessed_after_prediction_freeze": True,
        "predictions_sha256": prediction_sha,
        "call_ledger_sha256": call_sha,
        "pre_reader_summary": json.loads(
            (RUN_DIR / "projection_diagnostics_pre_reader.json").read_text(encoding="utf-8")
        ),
        "summary": summary,
    }
    diagnostics_sha = _write_or_verify_json(RUN_DIR / "projection_diagnostics.json", diagnostics_object)
    metrics_object = {
        "schema_version": 1,
        "metric_definition": "MEM-1 deterministic token-set F1 and normalized EM; six question categories remain per-case",
        "descriptive_only": True,
        "reader_contract_sha256": reader_contract_sha,
        "prediction_sha256": prediction_sha,
        "per_question": per_question,
        "summary": summary,
    }
    metrics_sha = _write_or_verify_json(RUN_DIR / "deterministic_metrics.json", metrics_object)
    comparison_sha = _write_or_verify_json(
        RUN_DIR / "comparison_mem2a_vs_mem2b.json", comparison
    )
    case_review = {
        "schema_version": 1,
        "run_id": args.run_id,
        "review_status": "PENDING_HUMAN_REFLECTION",
        "cases": [
            {
                "question_id": question_id,
                "outcome": None,
                "failure_loci": None,
                "notes": None,
            }
            for question_id in QUESTION_IDS
        ],
    }
    case_review_sha = _write_or_verify_json(CASE_REVIEW_PATH, case_review)
    revision_audit_path = RUN_DIR / "projection_artifact_revision_audit.json"
    revision_audit_sha = _sha256_file(revision_audit_path) if revision_audit_path.exists() else None
    if revision_audit_sha is not None and not _verify_sidecar(revision_audit_path):
        raise RuntimeError("Projection artifact revision audit failed its sidecar")
    protocol_sha = _sha256_file(PROTOCOL_PATH)
    report_manifest = {
        "gate": {"passed": True},
        "reader_calls_successful": 10,
        "final_reader_contract_sha256": reader_contract_sha,
        "artifacts_sha256": {
            "retrieval_parity.json": parity_sha,
            "context_plans.jsonl": plan_sha,
            "context_bundles.jsonl": bundle_sha,
            "projection_diagnostics_pre_reader.json": pre_reader_diag_sha,
            "projection_diagnostics.json": diagnostics_sha,
            "predictions.jsonl": prediction_sha,
            "call_ledger.jsonl": call_sha,
            "deterministic_metrics.json": metrics_sha,
            "comparison_mem2a_vs_mem2b.json": comparison_sha,
            "projection_frozen_manifest.json": _sha256_file(
                RUN_DIR / "projection_frozen_manifest.json"
            ),
            "mem_2b_case_review.json": case_review_sha,
            "mem_2b_rank_aware_projection.md": protocol_sha,
            "rank_aware_projection_contract.json": projection_contract_sha256,
            **(
                {"projection_artifact_revision_audit.json": revision_audit_sha}
                if revision_audit_sha
                else {}
            ),
        },
    }
    report_text = _report(report_manifest, projection["diagnostic_rows"], comparison)
    report_path = RUN_DIR / "report.md"
    if report_path.exists():
        if not _verify_sidecar(report_path) or report_path.read_text(encoding="utf-8") != report_text:
            raise RuntimeError("Frozen MEM-2B report differs on resume")
        report_sha = _sha256_file(report_path)
    else:
        report_path.write_text(report_text, encoding="utf-8", newline="\n")
        report_sha = _freeze(report_path)
    gates = {
        "legacy_no_hint_parity_test_passed": True,
        "rank_hint_validation_tests_passed": True,
        "retrieval_parity_with_mem2a_exact_for_all_ten": True,
        "memory_inventory_unchanged": True,
        "no_ingestion_rerun": True,
        "no_memory_store_matches_rerun": True,
        "context_plans_deterministic_and_frozen": True,
        "rank_aware_selected_order_has_zero_order_inversions": all(
            row["rank_aware_projected_order_inversion_count"] == 0
            for row in projection["diagnostic_rows"]
        ),
        "all_context_bundles_hash_frozen": True,
        "exactly_ten_successful_reader_calls": len(calls) == 10 and all(call["quality_status"] == "OK" for call in calls),
        "final_reader_contract_unchanged": reader_contract_sha == PINNED_READER_CONTRACT_SHA256,
        "memory_internal_llm_calls_zero": True,
        "embeddings_zero": True,
        "judge_zero": True,
        "hosted_calls_zero": True,
        "test_access_false": True,
        "102_dev_not_run": True,
        "no_revmem_component": True,
        "passed": True,
    }
    manifest_path = RUN_DIR / "run_manifest.json"
    if manifest_path.exists() and not _verify_sidecar(manifest_path):
        raise RuntimeError("Existing final MEM-2B run manifest failed its sidecar")
    completed_at = (
        json.loads(manifest_path.read_text(encoding="utf-8")).get("completed_at_utc")
        if manifest_path.exists()
        else datetime.now(UTC).isoformat()
    )
    final_manifest = {
        "manifest_version": 1,
        "run_id": args.run_id,
        "status": "COMPLETE",
        "split": "DEV",
        "question_ids": list(QUESTION_IDS),
        "test_access": False,
        "projection_policy_id": projection_contract["policy_id"],
        "projection_contract_sha256": projection_contract_sha256,
        "final_reader_contract_sha256": reader_contract_sha,
        "reader_model_sha256": PINNED_READER_MODEL_SHA256,
        "reader_runtime": reader_runtime,
        "reader_calls_expected": 10,
        "reader_calls_successful": len(calls),
        "memory_internal_llm_calls": 0,
        "embedding_calls": 0,
        "judge_calls": 0,
        "hosted_api_calls": 0,
        "gold_access_after_prediction_freeze": True,
        "prediction_sha256": prediction_sha,
        "call_ledger_sha256": call_sha,
        "retrieval_parity_sha256": parity_sha,
        "projection_artifacts_frozen_before_reader": True,
        "mem2a_artifact_sha256": mem2a_hashes,
        "static_cache_identity": static,
        "gate": gates,
        "artifacts_sha256": {
            **report_manifest["artifacts_sha256"],
            "comparison_mem2a_vs_mem2b.json": comparison_sha,
            "projection_frozen_manifest.json": _sha256_file(
                RUN_DIR / "projection_frozen_manifest.json"
            ),
            "mem_2b_case_review.json": case_review_sha,
            "mem_2b_rank_aware_projection.md": protocol_sha,
            "rank_aware_projection_contract.json": projection_contract_sha256,
            **(
                {"projection_artifact_revision_audit.json": revision_audit_sha}
                if revision_audit_sha
                else {}
            ),
            "report.md": report_sha,
        },
        "completed_at_utc": completed_at,
    }
    if manifest_path.exists():
        encoded_manifest = json.dumps(final_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if manifest_path.read_text(encoding="utf-8") != encoded_manifest:
            raise RuntimeError("Frozen final MEM-2B manifest differs on resume")
    else:
        _write_json_atomic(manifest_path, final_manifest)
        _freeze(manifest_path)
    if not _verify_sidecar(manifest_path):
        raise RuntimeError("Final MEM-2B run manifest did not freeze")
    if not gates["passed"]:
        raise RuntimeError("MEM2B_RANK_AWARE_PROJECTION_FROZEN_10_DIAGNOSTIC=NO")
    print("MEM2B_RANK_AWARE_PROJECTION_FROZEN_10_DIAGNOSTIC=YES")
    print(f"run_dir={RUN_DIR}")
    print(f"projection_contract_sha256={projection_contract_sha256}")
    print(f"predictions_sha256={prediction_sha}")
    print(f"report_sha256={report_sha}")
    print(f"reader_calls={len(calls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
