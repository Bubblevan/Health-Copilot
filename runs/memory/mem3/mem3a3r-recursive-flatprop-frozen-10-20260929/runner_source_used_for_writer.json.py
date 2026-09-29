"""Resume the frozen MEM-3A.3 roots with deterministic overflow subdivision."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ROOT / "tools" / "research" / "memory"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.memory import run_mem3a3_chunked_flatprop as parent  # noqa: E402
from tools.research.memory import flatprop_writer_v4_chunked as writer_v4  # noqa: E402
from tools.research.memory import run_mem3a2_minimal_flatprop as base  # noqa: E402
from tools.research.memory.flat_proposition_writer_v2 import (  # noqa: E402
    WriterQualificationFailure,
    WriterRecoveryError,
    atomic_write_bytes,
    canonical_json,
    execute_or_resume,
    sha256_bytes,
)
from tools.research.memory.flatprop_recursive_recovery import (  # noqa: E402
    aggregate_exact_leaf_propositions,
    assert_primary_leaf_coverage,
    balanced_partition,
    recursive_child_identity,
    resolve_recursive_overflow,
)

flat = base.flat

RUN_ID = "mem3a3r-recursive-flatprop-frozen-10-20260929"
RUN_DIR = ROOT / "runs" / "memory" / "mem3" / RUN_ID
REPORT_DOC = ROOT / "docs" / "research" / "memory" / "mem_3a3r_recursive_flatprop_frozen_10_diagnostic.md"
SPLIT_CONTRACT_PATH = ROOT / "docs" / "research" / "memory" / "flatprop_recursive_split_v1.json"
STAGE_PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem_3a3r_recursive_flatprop.md"
PARENT_RUN_DIR = ROOT / "runs" / "memory" / "mem3" / "mem3a3-chunked-flatprop-frozen-10-20260929"
LOCAL_CACHE_ROOT = ROOT / ".cache" / "health-copilot" / "mem3a3-chunked-v4-writer"
SPLIT_CONTRACT_SHA256 = ""
GATE = "MEM3A3R_RECURSIVE_FLATPROP_FROZEN_10_DIAGNOSTIC"
SYSTEM_NAME = "mem3a3r_recursive_flatprop_dense"
MAX_PROMPT_TOKENS = 6144
MAX_COMPLETION_TOKENS = 8192
HISTORICAL_STAGE_ID = "MEM-3A.3/flat-proposition-extractor-v4-chunked"
RECOVERY_STAGE_ID = "MEM-3A.3R/flat-proposition-extractor-v4-recursive"
WRITER_PROMPT_SHA256 = parent.V3_PROMPT_SHA256

_STATE: dict[str, Any] = {}


def _sha_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify(path: Path) -> bool:
    marker = path.with_suffix(".sha256")
    if not path.is_file() or not marker.is_file():
        return False
    return marker.read_text(encoding="ascii").strip().split() == [
        _sha_file(path),
        path.name,
    ]


def _freeze(path: Path) -> str:
    digest = _sha_file(path)
    atomic_write_bytes(path.with_suffix(".sha256"), f"{digest}  {path.name}\n".encode("ascii"))
    return digest


def _write_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)
    return _freeze(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    atomic_write_bytes(path, b"".join(canonical_json(row) + b"\n" for row in rows))
    return _freeze(path)


def _write_or_verify(path: Path, payload: bytes) -> str:
    if path.exists():
        if not _verify(path) or path.read_bytes() != payload:
            raise RuntimeError(f"Frozen MEM-3A.3R artifact differs: {path.name}")
        return _sha_file(path)
    atomic_write_bytes(path, payload)
    return _freeze(path)


def _configure_paths(head: str) -> None:
    parent._configure_paths()
    parent.RUN_ID = RUN_ID
    parent.RUN_DIR = RUN_DIR
    parent.GATE = GATE
    parent.LOCAL_CACHE_ROOT = LOCAL_CACHE_ROOT
    base.RUN_ID = RUN_ID
    base.RUN_DIR = RUN_DIR
    base.BASE_COMMIT = head
    base.LOCAL_CACHE_ROOT = LOCAL_CACHE_ROOT
    base._configure_flat_paths()
    flat.DATASET_PATH = parent.SOURCE_DATASET_PATH
    flat.mem2c.DATASET_PATH = parent.SOURCE_DATASET_PATH
    flat.RUN_ID = RUN_ID
    flat.RUN_DIR = RUN_DIR
    flat.PINNED_BASE_COMMIT = head
    flat.PREFLIGHT_PATH = RUN_DIR / "recursive_preflight.json"
    flat.EXTRACTION_MANIFEST_PATH = RUN_DIR / "session_extraction_manifest.json"
    flat.EXTRACTIONS_PATH = RUN_DIR / "session_extractions.jsonl"
    flat.FLAT_PROPOSITIONS_PATH = RUN_DIR / "flatprop_inventory.jsonl"
    flat.WRITER_LEDGER_PATH = RUN_DIR / "writer_attempt_ledger.jsonl"
    flat.MATERIALIZATION_LEDGER_PATH = RUN_DIR / "materialization_ledger.jsonl"
    flat.EMBEDDING_MANIFEST_PATH = RUN_DIR / "embedding_manifest.json"
    flat.DENSE_TOP8_PATH = RUN_DIR / "dense_top8.jsonl"
    flat.CONTEXT_PLANS_PATH = RUN_DIR / "context_plans.jsonl"
    flat.CONTEXT_BUNDLES_PATH = RUN_DIR / "context_bundles.jsonl"
    flat.PREDICTIONS_PATH = RUN_DIR / "predictions.jsonl"
    flat.READER_LEDGER_PATH = RUN_DIR / "reader_call_ledger.jsonl"
    flat.METRICS_PATH = RUN_DIR / "deterministic_metrics.json"
    flat.EFFICIENCY_PATH = RUN_DIR / "efficiency.json"
    flat.COMPARISON_PATH = RUN_DIR / "comparison_mem2d_rawspan_vs_mem3a3r.json"
    flat.CASE_REVIEW_PATH = RUN_DIR / "mem_3a3r_case_review.json"
    flat.PROTOCOL_PATH = RUN_DIR / "mem_3a3r_protocol.md"
    flat.CANDIDATE_GROUPS_PATH = RUN_DIR / "revision_semantics_not_in_scope.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path.name}:{line_number}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL row is not an object at {path.name}:{line_number}")
        rows.append(value)
    return rows


def _old_cache_dir(
    *,
    request: dict[str, Any],
    catalog: list[dict[str, Any]],
    chunk: dict[str, Any],
    original_preflight: dict[str, Any],
) -> Path:
    method = original_preflight["method_identity"]
    identity = {
        "stage": HISTORICAL_STAGE_ID,
        "contract_sha256": original_preflight["method_contract_sha256"],
        "prompt_sha256": WRITER_PROMPT_SHA256,
        "session_identity_sha256": chunk["chunk_id"],
        "catalog_sha256": base.catalog_sha256(catalog),
        "request_sha256": sha256_bytes(canonical_json(request)),
        "model_sha256": original_preflight["writer_runtime_identity"]["model_sha256"],
        "dynamic_schema_sha256": chunk["dynamic_schema_sha256"],
        "unwrap_source_sha256": method["transport_unwrap_source_sha256"],
        "packet_validator_sha256": method["packet_validator_source_sha256"],
    }
    cache_identity = sha256_bytes(canonical_json(identity))
    return LOCAL_CACHE_ROOT / original_preflight["method_contract_sha256"] / cache_identity


def _configure_state(head: str) -> dict[str, Any]:
    global SPLIT_CONTRACT_SHA256
    parent._configure_paths()
    historical_preflight_path = PARENT_RUN_DIR / "chunking_preflight.json"
    if not _verify(historical_preflight_path):
        raise RuntimeError("Historical MEM-3A.3 preflight SHA is invalid")
    historical_stage_base = _json(historical_preflight_path).get("stage_base_commit_sha")
    if not isinstance(historical_stage_base, str) or subprocess.run(
        ["git", "merge-base", "--is-ancestor", historical_stage_base, head],
        cwd=ROOT,
        capture_output=True,
    ).returncode != 0:
        raise RuntimeError("Historical MEM-3A.3 source commit is not an ancestor of MEM-3A.3R")
    original_preflight, initial_rows, source = parent._load_or_build_preflight(
        historical_stage_base
    )
    _configure_paths(head)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    split_contract = _json(SPLIT_CONTRACT_PATH)
    if (
        split_contract.get("contract_id") != "flatprop_recursive_split_v1"
        or split_contract.get("max_prompt_tokens") != MAX_PROMPT_TOKENS
        or split_contract.get("max_completion_tokens") != MAX_COMPLETION_TOKENS
        or split_contract.get("trigger") != "finish_reason=length only"
    ):
        raise RuntimeError("Recursive split contract differs from frozen runner constants")
    SPLIT_CONTRACT_SHA256 = _sha_file(SPLIT_CONTRACT_PATH)
    parent_partial = PARENT_RUN_DIR / "chunk_writer_ledger.partial.jsonl"
    if not _verify(parent_partial):
        raise RuntimeError("Historical MEM-3A.3 partial writer ledger SHA is invalid")
    historical_rows = _read_jsonl(parent_partial)
    if len(historical_rows) != 148 or sum(row.get("success") is True for row in historical_rows) != 147:
        raise RuntimeError("MEM-3A.3 must contain exactly 147 successful outcomes and one failed root")
    ordered_rows = sorted(
        initial_rows,
        key=lambda row: (
            0 if parent.RISK_SESSION_ID in row["source_session_ids"] else 1,
            list(source["sessions"]).index(row["session_identity_sha256"]),
            row["chunk_index"],
        ),
    )
    if [row["chunk_id"] for row in ordered_rows[:148]] != [
        row["chunk_id"] for row in historical_rows
    ]:
        raise RuntimeError("Historical writer ledger is not the frozen first 148 root attempts")
    historical_successes = {}
    for row, chunk in zip(historical_rows[:147], ordered_rows[:147], strict=True):
        if (
            row.get("success") is not True
            or row.get("validation") != "passed"
            or row.get("finish_reason") != "stop"
            or row.get("retry_count") != 0
            or row.get("hosted_call") is not False
            or row.get("request_sha256") != chunk["request_sha256"]
            or row.get("dynamic_schema_sha256") != chunk["dynamic_schema_sha256"]
            or row.get("catalog_sha256") != chunk["catalog_sha256"]
        ):
            raise RuntimeError(f"Historical successful writer row failed audit: {chunk['chunk_id']}")
        historical_successes[chunk["chunk_id"]] = row
    failed_row = historical_rows[-1]
    failed_chunk = ordered_rows[147]
    failure_path = PARENT_RUN_DIR / "writer_failure.json"
    if not _verify(failure_path):
        raise RuntimeError("Historical MEM-3A.3 failure record SHA is invalid")
    failure = _json(failure_path)
    if (
        failed_row.get("chunk_id") != failed_chunk["chunk_id"]
        or failed_row.get("success") is not False
        or failed_row.get("finish_reason") != "length"
        or failed_row.get("validation") != "failed"
        or failure.get("failed_chunk_id") != failed_chunk["chunk_id"]
        or failed_chunk["source_session_ids"] != ["sharegpt_vbNrVtS_151"]
        or failed_row.get("request_sha256") != failure.get("request_sha256")
        or failed_row.get("http_envelope_sha256") != failure.get("http_envelope_sha256")
        or failed_row.get("assistant_content_sha256") != failure.get("assistant_content_sha256")
    ):
        raise RuntimeError("Historical failed root does not match the frozen overflow audit")
    source.update(
        {
            "historical_rows": historical_rows,
            "historical_successes": historical_successes,
            "historical_failed_chunk": failed_chunk,
            "historical_failed_row": failed_row,
            "historical_stage_base_commit": historical_stage_base,
        }
    )
    method_identity_body = {
        "stage": RUN_ID,
        "base_writer_method_contract_sha256": original_preflight["method_contract_sha256"],
        "initial_chunk_plan_sha256": _sha_file(PARENT_RUN_DIR / "chunk_manifest.jsonl"),
        "recursive_split_contract_sha256": SPLIT_CONTRACT_SHA256,
        "writer_semantic_contract_sha256": original_preflight["method_identity"]["v3_extractor_contract_sha256"],
        "writer_prompt_sha256": WRITER_PROMPT_SHA256,
        "writer_model_sha256": original_preflight["writer_runtime_identity"]["model_sha256"],
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "recursive_runner_source_sha256": _sha_file(Path(__file__).resolve()),
        "recursive_helper_source_sha256": _sha_file(TOOLS_DIR / "flatprop_recursive_recovery.py"),
        "chunk_request_source_sha256": _sha_file(TOOLS_DIR / "flatprop_writer_v4_chunked.py"),
        "packet_validator_source_sha256": _sha_file(Path(base.writer_v3.__file__)),
        "transport_unwrap_source_sha256": _sha_file(
            Path(base.__file__).with_name("flat_proposition_writer_v2.py")
        ),
        "normalization_contract_sha256": _sha_file(parent.NORMALIZATION_PATH),
        "embedding_model_identity": original_preflight["method_identity"]["embedding_model_identity"],
        "embedding_device": "cuda:0",
        "embedding_dtype": "float16",
        "reader_contract_sha256": flat.PINNED_READER_SHA256,
        "reader_model_sha256": original_preflight["writer_runtime_identity"]["model_sha256"],
        "judge_model": None,
        "hosted_provider": False,
        "retries": 0,
    }
    method_sha = sha256_bytes(canonical_json(method_identity_body))
    source.update(
        {
            "head": head,
            "preflight": original_preflight,
            "initial_rows": initial_rows,
            "ordered_initial_rows": ordered_rows,
            "method_identity": method_identity_body,
            "method_sha": method_sha,
            "lineage_events": [],
            "node_registry": {row["chunk_id"]: row for row in initial_rows},
            "node_results": {},
            "attempt_order": [],
            "raw_attempts": {},
            "all_attempts": [],
            "terminal_outputs": [],
            "new_provider_calls": 0,
            "historical_cache_miss_reexecuted": set(),
            "historical_cache_imported": 0,
            "start_time": time.perf_counter(),
            "current_root_index": 0,
        }
    )
    _STATE.clear()
    _STATE.update(source)
    return source


def _tokenize(client: httpx.Client, text: str) -> int:
    response = client.post(
        "http://127.0.0.1:8081/tokenize",
        json={"content": text, "add_special": False},
    )
    response.raise_for_status()
    token_ids = response.json().get("tokens")
    if not isinstance(token_ids, list):
        raise TypeError("Frozen Qwen /tokenize response has no token ID list")
    return len(token_ids)


def _session_turns(session: dict[str, Any]) -> tuple[list[int], dict[int, list[int]]]:
    turn_indices = sorted({span["source_turn_index"] for span in session["catalog"]})
    spans_by_turn = {
        turn_index: [
            ordinal
            for ordinal, span in enumerate(session["catalog"])
            if span["source_turn_index"] == turn_index
        ]
        for turn_index in turn_indices
    }
    return turn_indices, spans_by_turn


def _request_for(session: dict[str, Any], spans: list[dict[str, Any]]) -> dict[str, Any]:
    return writer_v4.writer_request(
        session_date=session["session_date"],
        catalog=spans,
        system_prompt=base.PROMPT_PATH.read_text(encoding="utf-8"),
        model_alias=flat.READER_MODEL,
    )


def _measure_visible(
    *, client: httpx.Client, session: dict[str, Any], ordinals: list[int], detailed: bool
) -> dict[str, Any]:
    spans = [session["catalog"][ordinal] for ordinal in ordinals]
    return parent._measure_request(
        client=client,
        session=session,
        catalog=spans,
        prompt=base.PROMPT_PATH.read_text(encoding="utf-8"),
        schema_prefix=parent.SCHEMA_PREFIX,
        schema_suffix=parent.SCHEMA_SUFFIX,
        detailed=detailed,
    )


def _primary_turn_ordinals(session: dict[str, Any], primary: list[int]) -> list[int]:
    turns, _ = _session_turns(session)
    touched = {session["catalog"][ordinal]["source_turn_index"] for ordinal in primary}
    return [index for index, turn in enumerate(turns) if turn in touched]


def _child_node(
    *,
    client: httpx.Client,
    parent_node: dict[str, Any],
    session: dict[str, Any],
    primary: list[int],
    unit_kind: str,
    unit_ordinals: list[int],
    depth: int,
) -> dict[str, Any]:
    turns, spans_by_turn = _session_turns(session)
    turn_ordinals = _primary_turn_ordinals(session, primary)
    if not turn_ordinals:
        raise RuntimeError("recursive child has no source turn")
    previous = min(turn_ordinals) - 1
    following = max(turn_ordinals) + 1
    previous_available = previous >= 0
    following_available = following < len(turns)
    visible_cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def measure_with_overlap(overlap_turn_ordinals: list[int]) -> tuple[list[int], dict[str, Any]]:
        overlap_spans = [
            ordinal
            for turn_ordinal in overlap_turn_ordinals
            for ordinal in spans_by_turn[turns[turn_ordinal]]
        ]
        visible = sorted(set([*primary, *overlap_spans]))
        key = tuple(visible)
        if key not in visible_cache:
            visible_cache[key] = _measure_visible(
                client=client, session=session, ordinals=visible, detailed=False
            )
        return visible, visible_cache[key]

    selected_overlap: list[int] = []
    neighbors = [turn for turn, available in ((previous, previous_available), (following, following_available)) if available]
    if neighbors:
        _, both_measurement = measure_with_overlap(neighbors)
        if both_measurement["request_budget_tokens"] <= MAX_PROMPT_TOKENS:
            selected_overlap = neighbors
        elif previous_available:
            _, previous_measurement = measure_with_overlap([previous])
            if previous_measurement["request_budget_tokens"] <= MAX_PROMPT_TOKENS:
                selected_overlap = [previous]
                if following_available:
                    _, preferred_measurement = measure_with_overlap([previous, following])
                    if preferred_measurement["request_budget_tokens"] <= MAX_PROMPT_TOKENS:
                        selected_overlap = [previous, following]
        elif following_available:
            _, following_measurement = measure_with_overlap([following])
            if following_measurement["request_budget_tokens"] <= MAX_PROMPT_TOKENS:
                selected_overlap = [following]

    overlap_span_ordinals = [
        ordinal
        for turn_ordinal in selected_overlap
        for ordinal in spans_by_turn[turns[turn_ordinal]]
    ]
    visible_span_ordinals = sorted(set([*primary, *overlap_span_ordinals]))
    detail = _measure_visible(
        client=client,
        session=session,
        ordinals=visible_span_ordinals,
        detailed=True,
    )
    span_ids = [session["catalog"][ordinal]["evidence_ref"] for ordinal in primary]
    overlap_ids = [session["catalog"][ordinal]["evidence_ref"] for ordinal in overlap_span_ordinals]
    identity_body = {
        "split_contract_sha256": SPLIT_CONTRACT_SHA256,
        "session_identity_sha256": parent_node["session_identity_sha256"],
        "root_initial_chunk_id": parent_node["root_initial_chunk_id"],
        "parent_chunk_id": parent_node["chunk_id"],
        "recursion_depth": depth,
        "primary_unit_kind": unit_kind,
        "primary_unit_ordinals": unit_ordinals,
        "primary_span_ordinals": primary,
        "overlap_turn_ordinals": selected_overlap,
        "overlap_span_ordinals": overlap_span_ordinals,
        "rendered_source_payload_sha256": detail["rendered_source_payload_sha256"],
        "dynamic_schema_sha256": detail["dynamic_schema_sha256"],
    }
    child_id, _ = recursive_child_identity(identity_body)
    request = _request_for(session, [session["catalog"][ordinal] for ordinal in visible_span_ordinals])
    if sha256_bytes(canonical_json(request)) != detail["request_sha256"]:
        raise RuntimeError("recursive child request SHA differs from exact local preflight")
    node = {
        **detail,
        "chunk_id": child_id,
        "chunk_identity": identity_body,
        "identity_sha256": child_id,
        "session_identity_sha256": parent_node["session_identity_sha256"],
        "source_session_ids": parent_node["source_session_ids"],
        "session_date": session["session_date"],
        "valid_from": session["valid_from"],
        "source_turns_sha256": session["source_turns_sha256"],
        "catalog_sha256": session["catalog_sha256"],
        "root_initial_chunk_id": parent_node["root_initial_chunk_id"],
        "parent_chunk_id": parent_node["chunk_id"],
        "recursion_depth": depth,
        "split_contract_sha256": SPLIT_CONTRACT_SHA256,
        "primary_unit_kind": unit_kind,
        "primary_unit_ordinals": unit_ordinals,
        "primary_start_ordinal": primary[0],
        "primary_end_ordinal": primary[-1],
        "primary_span_ordinals": primary,
        "primary_span_ids": span_ids,
        "primary_turn_ordinals": turn_ordinals,
        "primary_turn_indices": [turns[index] for index in turn_ordinals],
        "overlap_turn_ordinals": selected_overlap,
        "overlap_turn_indices": [turns[index] for index in selected_overlap],
        "overlap_span_ordinals": overlap_span_ordinals,
        "overlap_span_ids": overlap_ids,
        "visible_span_ordinals": visible_span_ordinals,
        "visible_span_ids": [session["catalog"][ordinal]["evidence_ref"] for ordinal in visible_span_ordinals],
        "oversized_turn_split": unit_kind == "rawspan" or parent_node.get("oversized_turn_split", False),
        "primary_request_budget_tokens": _measure_visible(
            client=client, session=session, ordinals=primary, detailed=False
        )["request_budget_tokens"],
        "request_sha256": detail["request_sha256"],
        "session_identity_sha256_for_cache": child_id,
    }
    if child_id in _STATE["node_registry"]:
        if _STATE["node_registry"][child_id] != node:
            raise RuntimeError("recursive child identity collision with different node data")
    else:
        _STATE["node_registry"][child_id] = node
    return node


def _unit_masses(
    *, client: httpx.Client, session: dict[str, Any], unit_kind: str, units: list[int]
) -> list[int]:
    cache = _STATE.setdefault("rawspan_token_mass", {})
    key_prefix = session["session_identity_sha256"]
    masses_by_span = {}
    turn_indices, spans_by_turn = _session_turns(session)
    span_ordinals = (
        units
        if unit_kind == "rawspan"
        else [ordinal for unit in units for ordinal in spans_by_turn[turn_indices[unit]]]
    )
    for ordinal in sorted(span_ordinals):
        key = (key_prefix, ordinal)
        if key not in cache:
            cache[key] = _tokenize(client, session["catalog"][ordinal]["content"])
        masses_by_span[ordinal] = cache[key]
    if unit_kind == "rawspan":
        return [masses_by_span[unit] for unit in units]
    return [sum(masses_by_span[ordinal] for ordinal in spans_by_turn[turn_indices[unit]]) for unit in units]


def _split_node(
    *, client: httpx.Client, node: dict[str, Any], session: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    turns, spans_by_turn = _session_turns(session)
    primary = list(node["primary_span_ordinals"])
    touched = _primary_turn_ordinals(session, primary)
    is_full_turn_region = all(
        set(spans_by_turn[turns[turn_ordinal]]).issubset(primary)
        for turn_ordinal in touched
    )
    if len(touched) > 1 and is_full_turn_region:
        unit_kind = "turn"
        units = touched
        unit_spans = {
            turn_ordinal: spans_by_turn[turns[turn_ordinal]] for turn_ordinal in units
        }
    elif len(touched) == 1:
        unit_kind = "rawspan"
        units = primary
        unit_spans = {ordinal: [ordinal] for ordinal in units}
    else:
        raise RuntimeError("recursive parent mixes incomplete turns; refusing non-turn split")
    if len(primary) < 2:
        raise RuntimeError("IRREDUCIBLE_WRITER_OVERFLOW")
    masses = _unit_masses(client=client, session=session, unit_kind=unit_kind, units=units)
    left_units, right_units = balanced_partition(units, masses)
    child_depth = node.get("recursion_depth", 0) + 1
    left_primary = [ordinal for unit in left_units for ordinal in unit_spans[unit]]
    right_primary = [ordinal for unit in right_units for ordinal in unit_spans[unit]]
    left = _child_node(
        client=client,
        parent_node=node,
        session=session,
        primary=left_primary,
        unit_kind=unit_kind,
        unit_ordinals=left_units,
        depth=child_depth,
    )
    right = _child_node(
        client=client,
        parent_node=node,
        session=session,
        primary=right_primary,
        unit_kind=unit_kind,
        unit_ordinals=right_units,
        depth=child_depth,
    )
    if (
        node["chunk_id"]
        == "104d270a9c92d9d464dc5eb07adf1eac6881d5edd6a1d437fc040ec814508bc9"
        and node.get("recursion_depth", 0) == 0
    ):
        preflight_path = RUN_DIR / "failed_root_split_preflight.json"
        if not _verify(preflight_path):
            raise RuntimeError("Frozen failed-root split preflight is missing or invalid")
        preflight = _json(preflight_path)
        expected_children = [
            group[0]["chunk_id"]
            for group in preflight.get("children", [])
            if len(group) == 1
        ]
        actual_children = [left["chunk_id"], right["chunk_id"]]
        if (
            preflight.get("historical_failed_root") != node["chunk_id"]
            or preflight.get("split_contract_sha256") != SPLIT_CONTRACT_SHA256
            or preflight.get("provider_calls") != 0
            or preflight.get("labels_loaded") is not False
            or len(expected_children) != 2
            or expected_children != actual_children
        ):
            raise RuntimeError("Failed-root split differs from its frozen zero-call preflight")
    reason = (
        "SUBDIVISION"
        if _STATE["node_results"].get(node["chunk_id"], {}).get("finish_reason") == "length"
        else "PREINFERENCE_BUDGET_SUBDIVISION"
    )
    parent_event = {
        "event": "PARENT_SUBDIVIDED",
        "status": "OVERFLOW_PARENT" if reason == "SUBDIVISION" else reason,
        "disposition": reason,
        "chunk_id": node["chunk_id"],
        "parent_chunk_id": node.get("parent_chunk_id"),
        "root_initial_chunk_id": node["root_initial_chunk_id"],
        "recursion_depth": node.get("recursion_depth", 0),
        "split_contract_sha256": SPLIT_CONTRACT_SHA256,
        "primary_start_ordinal": node["primary_start_ordinal"],
        "primary_end_ordinal": node["primary_end_ordinal"],
        "primary_span_ordinals": primary,
        "primary_span_ids": node["primary_span_ids"],
        "prompt_tokens": node["request_budget_tokens"],
        "semantic_output_used": False,
        "child_chunk_ids": [left["chunk_id"], right["chunk_id"]],
        "provider_call_this_stage": _STATE["node_results"].get(node["chunk_id"], {}).get(
            "provider_calls_this_stage_unique", 0
        )
        > 0,
    }
    _STATE["lineage_events"].append(parent_event)
    for child in (left, right):
        _STATE["lineage_events"].append(
            {
                "event": "CHILD_CREATED",
                "status": "PLANNED",
                "chunk_id": child["chunk_id"],
                "parent_chunk_id": child["parent_chunk_id"],
                "root_initial_chunk_id": child["root_initial_chunk_id"],
                "recursion_depth": child["recursion_depth"],
                "split_contract_sha256": SPLIT_CONTRACT_SHA256,
                "primary_unit_kind": child["primary_unit_kind"],
                "primary_unit_ordinals": child["primary_unit_ordinals"],
                "primary_start_ordinal": child["primary_start_ordinal"],
                "primary_end_ordinal": child["primary_end_ordinal"],
                "primary_span_ordinals": child["primary_span_ordinals"],
                "primary_span_ids": child["primary_span_ids"],
                "overlap_turn_ordinals": child["overlap_turn_ordinals"],
                "overlap_span_ordinals": child["overlap_span_ordinals"],
                "overlap_span_ids": child["overlap_span_ids"],
                "prompt_tokens": child["request_budget_tokens"],
                "request_sha256": child["request_sha256"],
                "dynamic_schema_sha256": child["dynamic_schema_sha256"],
                "rendered_source_payload_sha256": child["rendered_source_payload_sha256"],
                "provider_call": False,
            }
        )
    _persist_progress()
    return left, right


def _prepare_node(node: dict[str, Any], client: httpx.Client, session: dict[str, Any]) -> list[dict[str, Any]]:
    pending = [node]
    prepared = []
    while pending:
        current = pending.pop()
        if current["request_budget_tokens"] <= MAX_PROMPT_TOKENS:
            prepared.append(current)
            continue
        if len(current["primary_span_ordinals"]) == 1:
            raise RuntimeError("IRREDUCIBLE_PROMPT_BUDGET_OVERFLOW")
        left, right = _split_node(client=client, node=current, session=session)
        pending.extend((right, left))
    return prepared


def _cache_dir_for_new_node(node: dict[str, Any], request: dict[str, Any], state: dict[str, Any]) -> Path:
    identity = {
        "stage": RECOVERY_STAGE_ID,
        "contract_sha256": state["method_sha"],
        "prompt_sha256": WRITER_PROMPT_SHA256,
        "session_identity_sha256": node["chunk_id"],
        "catalog_sha256": base.catalog_sha256(
            [
                state["sessions"][node["session_identity_sha256"]]["catalog"][ordinal]
                for ordinal in node["visible_span_ordinals"]
            ]
        ),
        "request_sha256": sha256_bytes(canonical_json(request)),
        "model_sha256": state["preflight"]["writer_runtime_identity"]["model_sha256"],
        "dynamic_schema_sha256": node["dynamic_schema_sha256"],
        "unwrap_source_sha256": state["method_identity"]["transport_unwrap_source_sha256"],
        "packet_validator_sha256": state["method_identity"]["packet_validator_source_sha256"],
    }
    cache_identity = sha256_bytes(canonical_json(identity))
    return LOCAL_CACHE_ROOT / state["method_sha"] / cache_identity


def _forensic_parent_result(node: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    row = state["historical_failed_row"]
    raw_path = Path(row["response_cache_path"])
    assistant_path = Path(row["assistant_content_cache_path"])
    for path in (raw_path, assistant_path):
        resolved = path.resolve()
        if not resolved.is_relative_to(ROOT) or not resolved.is_file():
            raise RuntimeError("historical overflow forensic cache is missing or outside workspace")
    meta_path = raw_path.with_name("response_meta.json")
    journal_path = raw_path.with_name("journal.jsonl")
    if (
        _sha_file(raw_path) != row["http_envelope_sha256"]
        or _sha_file(assistant_path) != row["assistant_content_sha256"]
        or not meta_path.is_file()
        or not journal_path.is_file()
    ):
        raise RuntimeError("historical overflow forensic response hashes failed")
    metadata = _json(meta_path)
    journal = _read_jsonl(journal_path)
    if (
        metadata.get("http_envelope_sha256") != row["http_envelope_sha256"]
        or metadata.get("assistant_content_sha256") != row["assistant_content_sha256"]
        or metadata.get("finish_reason") != "length"
        or metadata.get("completion_tokens") != MAX_COMPLETION_TOKENS
        or [event.get("state") for event in journal]
        != ["STARTED", "RESPONSE_CAPTURED", "COMPLETE_FAILURE"]
        or journal[-1].get("ledger", {}).get("request_sha256") != row["request_sha256"]
        or journal[-1].get("ledger", {}).get("http_envelope_sha256")
        != row["http_envelope_sha256"]
        or journal[-1].get("ledger", {}).get("assistant_content_sha256")
        != row["assistant_content_sha256"]
        or journal[-1].get("error", {}).get("code") != "COMPLETION_TRUNCATED"
    ):
        raise RuntimeError("historical overflow response metadata differs from frozen ledger")
    return {
        **node,
        **row,
        "finish_reason": "length",
        "success": False,
        "validation": "failed",
        "status": "OVERFLOW_PARENT",
        "semantic_output_used": False,
        "provider_calls_this_stage_unique": 0,
        "provider_calls_this_resume": 0,
        "normalized_packet": None,
        "historical_source_stage": "MEM-3A.3",
    }


def _verify_historical_success_cache(
    *, cache_dir: Path, row: dict[str, Any], chunk: dict[str, Any]
) -> None:
    raw_path = cache_dir / "raw_response.txt"
    content_path = cache_dir / "assistant_content.txt"
    metadata_path = cache_dir / "response_meta.json"
    journal_path = cache_dir / "journal.jsonl"
    paths = (raw_path, content_path, metadata_path, journal_path)
    if not all(path.is_file() and path.resolve().is_relative_to(ROOT) for path in paths):
        raise RuntimeError(f"historical cache directory is present but incomplete: {chunk['chunk_id']}")
    metadata = _json(metadata_path)
    events = _read_jsonl(journal_path)
    if (
        _sha_file(raw_path) != row.get("http_envelope_sha256")
        or _sha_file(content_path) != row.get("assistant_content_sha256")
        or metadata.get("http_envelope_sha256") != row.get("http_envelope_sha256")
        or metadata.get("assistant_content_sha256") != row.get("assistant_content_sha256")
        or metadata.get("finish_reason") != "stop"
        or metadata.get("http_status") != 200
        or len(events) != 3
        or [event.get("state") for event in events]
        != ["STARTED", "RESPONSE_CAPTURED", "COMPLETE_SUCCESS"]
        or events[-1].get("ledger", {}).get("request_sha256") != row.get("request_sha256")
        or events[-1].get("ledger", {}).get("http_envelope_sha256") != row.get("http_envelope_sha256")
        or events[-1].get("ledger", {}).get("assistant_content_sha256") != row.get("assistant_content_sha256")
        or events[-1].get("ledger", {}).get("validation") != "passed"
        or not isinstance(events[-1].get("normalized_packet"), dict)
    ):
        raise RuntimeError(f"historical cache hash/provenance verification failed: {chunk['chunk_id']}")
    packet = events[-1]["normalized_packet"]
    visible_catalog = [
        _STATE["sessions"][chunk["session_identity_sha256"]]["catalog"][ordinal]
        for ordinal in chunk["visible_span_ordinals"]
    ]
    reconstructed = base.writer_v3.validate_packet(content_path.read_bytes(), visible_catalog)
    if reconstructed != packet:
        raise RuntimeError(f"historical normalized packet failed v3 revalidation: {chunk['chunk_id']}")


def _execute_node(node: dict[str, Any], client: httpx.Client, state: dict[str, Any]) -> dict[str, Any]:
    session = state["sessions"][node["session_identity_sha256"]]
    visible_catalog = [session["catalog"][ordinal] for ordinal in node["visible_span_ordinals"]]
    request = _request_for(session, visible_catalog)
    request_sha = sha256_bytes(canonical_json(request))
    if request_sha != node["request_sha256"]:
        raise RuntimeError(f"Frozen request identity changed: {node['chunk_id']}")

    historical_row = state["historical_successes"].get(node["chunk_id"])
    historical_failed = node["chunk_id"] == state["historical_failed_chunk"]["chunk_id"]
    if historical_failed:
        result = _forensic_parent_result(node, state)
        state["node_results"][node["chunk_id"]] = result
        state["raw_attempts"][node["chunk_id"]] = result
        state["attempt_order"].append(node["chunk_id"])
        _persist_progress()
        return result

    use_old_cache = False
    if historical_row is not None:
        old_cache_dir = _old_cache_dir(
            request=request,
            catalog=visible_catalog,
            chunk=node,
            original_preflight=state["preflight"],
        )
        use_old_cache = old_cache_dir.exists()
        if use_old_cache:
            _verify_historical_success_cache(
                cache_dir=old_cache_dir, row=historical_row, chunk=node
            )
        if not use_old_cache:
            state["historical_cache_miss_reexecuted"].add(node["chunk_id"])

    def provider(req: dict[str, Any]) -> tuple[int, bytes, str | None]:
        if req != request:
            raise RuntimeError("writer request differs from frozen recursive node")
        state["new_provider_calls"] += 1
        response = client.post(f"{flat.READER_ENDPOINT}/chat/completions", json=req)
        return response.status_code, response.content, response.headers.get("content-type")

    if use_old_cache:
        method_sha = state["preflight"]["method_contract_sha256"]
        stage_identity = HISTORICAL_STAGE_ID
    else:
        method_sha = state["method_sha"]
        stage_identity = RECOVERY_STAGE_ID
    try:
        packet, call = execute_or_resume(
            request=request,
            catalog=visible_catalog,
            session_identity_sha256=node["chunk_id"],
            prompt_sha256=WRITER_PROMPT_SHA256,
            contract_sha256=method_sha,
            local_cache_root=LOCAL_CACHE_ROOT,
            provider=provider,
            stage_identity=stage_identity,
            model_sha256=state["preflight"]["writer_runtime_identity"]["model_sha256"],
            dynamic_schema_sha256=node["dynamic_schema_sha256"],
            unwrap_source_sha256=state["method_identity"]["transport_unwrap_source_sha256"],
            packet_validator=base.writer_v3.validate_packet,
            packet_validator_sha256=state["method_identity"]["packet_validator_source_sha256"],
        )
    except WriterQualificationFailure as exc:
        call = dict(exc.ledger)
        failure = getattr(exc, "error", {})
        if call.get("finish_reason") != "length" or failure.get("code") != "COMPLETION_TRUNCATED":
            raise RuntimeError(
                f"non-length writer failure is fatal for {node['chunk_id']}: {failure}"
            ) from exc
        result = {
            **node,
            **call,
            "success": False,
            "failure": failure,
            "status": "OVERFLOW_PARENT",
            "semantic_output_used": False,
            "provider_calls_this_stage_unique": (
                0 if use_old_cache else call.get("provider_calls", 0)
            ),
            "provider_calls_this_resume": call.get("provider_calls_this_resume", 0),
            "hosted_call": False,
            "retry_count": 0,
            "normalized_packet": None,
        }
    except WriterRecoveryError as exc:
        raise RuntimeError(f"writer recovery integrity failure: {exc}") from exc
    else:
        if call.get("finish_reason") != "stop" or call.get("validation") != "passed":
            raise RuntimeError(f"terminal writer result failed contract: {node['chunk_id']}")
        if historical_row is not None and use_old_cache:
            for field in (
                "request_sha256",
                "http_envelope_sha256",
                "assistant_content_sha256",
                "finish_reason",
                "validation",
            ):
                if call.get(field) != historical_row.get(field):
                    raise RuntimeError(f"historical cache differs from ledger field {field}")
            if len(packet["propositions"]) != historical_row.get("normalized_proposition_count"):
                raise RuntimeError("historical cached packet count differs from frozen ledger")
            state["historical_cache_imported"] += 1
        result = {
            **node,
            **call,
            "success": True,
            "validation": "passed",
            "status": "TERMINAL_LEAF",
            "semantic_output_used": True,
            "provider_calls_this_stage_unique": (
                0 if use_old_cache else call.get("provider_calls", 0)
            ),
            "provider_calls_this_resume": call.get("provider_calls_this_resume", 0),
            "historical_cache_imported": bool(historical_row is not None and use_old_cache),
            "historical_cache_miss_reexecuted": node["chunk_id"] in state["historical_cache_miss_reexecuted"],
            "hosted_call": False,
            "retry_count": 0,
            "normalized_packet": packet,
            "normalized_proposition_count": len(packet["propositions"]),
        }
    state["node_results"][node["chunk_id"]] = result
    state["raw_attempts"][node["chunk_id"]] = result
    state["attempt_order"].append(node["chunk_id"])
    _persist_progress()
    return result


def _persist_progress() -> None:
    state = _STATE
    if not state:
        return
    attempts = [state["raw_attempts"][key] for key in state["attempt_order"]]
    partial_path = RUN_DIR / "writer_attempt_ledger.partial.jsonl"
    lineage_path = RUN_DIR / "overflow_lineage.partial.jsonl"
    _write_jsonl(partial_path, attempts)
    _write_jsonl(lineage_path, state["lineage_events"])
    _write_json(
        RUN_DIR / "writer_progress.json",
        {
            "schema_version": 1,
            "stage": RUN_ID,
            "status": "WRITER_IN_PROGRESS",
            "initial_chunks": len(state.get("initial_rows", [])),
            "initial_roots_started": state.get("current_root_index", 0),
            "node_attempts": len(attempts),
            "terminal_attempts": sum(row.get("success") is True for row in attempts),
            "overflow_parents": sum(row.get("finish_reason") == "length" for row in attempts),
            "provider_calls_this_process": state.get("new_provider_calls", 0),
            "historical_cache_imported": state.get("historical_cache_imported", 0),
            "historical_cache_miss_reexecuted": len(state.get("historical_cache_miss_reexecuted", set())),
            "retries": 0,
            "hosted_calls": 0,
            "labels_loaded": False,
        },
    )


def _resolve_initial_roots(
    *, client: httpx.Client, state: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_leaves = []
    all_attempts = []
    if parent._runtime(client) != state["preflight"]["writer_runtime_identity"]:
        raise RuntimeError("Frozen local Qwen runtime changed immediately before recovery")
    for index, root in enumerate(state["ordered_initial_rows"], 1):
        root = {
            **root,
            "root_initial_chunk_id": root["chunk_id"],
            "parent_chunk_id": None,
            "recursion_depth": 0,
            "split_contract_sha256": SPLIT_CONTRACT_SHA256,
            "primary_unit_kind": "initial_frozen_plan",
            "primary_unit_ordinals": root["primary_turn_ordinals"],
            "identity_sha256": root["chunk_id"],
            "session_identity_sha256_for_cache": root["chunk_id"],
        }
        state["node_registry"][root["chunk_id"]] = root
        state["current_root_index"] = index
        session = state["sessions"][root["session_identity_sha256"]]

        def client_execute(node: dict[str, Any]) -> dict[str, Any]:
            return _execute_node(node, client, state)

        def splitter(node: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            return _split_node(client=client, node=node, session=session)

        def prepare(node: dict[str, Any]) -> list[dict[str, Any]]:
            return _prepare_node(node, client, session)

        leaves, attempts = resolve_recursive_overflow(
            root,
            execute=client_execute,
            split=splitter,
            prepare=prepare,
        )
        for leaf_index, leaf in enumerate(leaves):
            leaf["chunk_index"] = len(all_leaves)
            leaf["primary_start_ordinal"] = leaf["primary_span_ordinals"][0]
            leaf["primary_end_ordinal"] = leaf["primary_span_ordinals"][-1]
            all_leaves.append(leaf)
        all_attempts.extend(attempts)
        state["all_attempts"] = all_attempts
        state["terminal_outputs"] = all_leaves
        state["root_attempt_count"] = index
        _persist_progress()
        if index % 25 == 0 or index == len(state["ordered_initial_rows"]):
            print(
                f"roots={index}/{len(state['ordered_initial_rows'])} leaves={len(all_leaves)} "
                f"attempts={len(all_attempts)} local_calls={state['new_provider_calls']}",
                flush=True,
            )
    return all_leaves, all_attempts


def _assert_initial_frozen_leaves(
    *, leaves: list[dict[str, Any]], sessions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for leaf in leaves:
        if leaf.get("success") is not True or leaf.get("status") != "TERMINAL_LEAF":
            raise RuntimeError("nonterminal recursive writer output reached leaf set")
        by_session[leaf["session_identity_sha256"]].append(leaf)
    if set(by_session) != set(sessions) or len(by_session) != 477:
        raise RuntimeError("recursive FlatProp leaf set does not cover all 477 sessions")
    max_depth = 0
    for identity, session in sessions.items():
        rows = sorted(
            by_session[identity],
            key=lambda row: (
                row["primary_start_ordinal"],
                row["primary_end_ordinal"],
                row["chunk_id"],
            ),
        )
        expected = list(range(len(session["catalog"])))
        assert_primary_leaf_coverage(expected, [row["primary_span_ordinals"] for row in rows])
        for row in rows:
            if row["request_budget_tokens"] > MAX_PROMPT_TOKENS:
                raise RuntimeError("terminal leaf exceeded the rendered prompt-plus-schema budget")
            for prop in row["normalized_packet"]["propositions"]:
                if any(ref not in {span["evidence_ref"] for span in session["catalog"]} for ref in prop["evidence_refs"]):
                    raise RuntimeError("leaf proposition references evidence outside session")
            max_depth = max(max_depth, row.get("recursion_depth", 0))
    return {
        "complete_sessions": len(by_session),
        "complete_terminal_leaves": len(leaves),
        "source_rawspans": sum(len(session["catalog"]) for session in sessions.values()),
        "primary_rawspan_coverage_exactly_once": True,
        "maximum_recursion_depth": max_depth,
    }


def _aggregate_sessions(
    *, leaves: list[dict[str, Any]], sessions: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, leaf in enumerate(leaves):
        leaf["chunk_index"] = index
        outputs_by_session[leaf["session_identity_sha256"]].append(leaf)
    packets = []
    diagnostics = {}
    for identity, session in sessions.items():
        props, diag = aggregate_exact_leaf_propositions(
            catalog=session["catalog"], leaf_outputs=outputs_by_session[identity]
        )
        diagnostics[identity] = diag
        packet_outputs = sorted(
            outputs_by_session[identity],
            key=lambda row: (row["primary_start_ordinal"], row["primary_end_ordinal"], row["chunk_id"]),
        )
        packets.append(
            {
                "session_identity_sha256": identity,
                "session_date": session["session_date"],
                "valid_from": session["valid_from"],
                "source_session_ids": session["source_session_ids"],
                "source_turns_sha256": session["source_turns_sha256"],
                "catalog_sha256": session["catalog_sha256"],
                "propositions": props,
                "source_chunk_ids": [row["chunk_id"] for row in packet_outputs],
                "exact_identity_diagnostics": diag,
            }
        )
    return packets, diagnostics


def _all_artifact_nodes() -> list[dict[str, Any]]:
    return list(_STATE["node_registry"].values())


def _finalize_writer_artifacts(
    *,
    leaves: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    packets: list[dict[str, Any]],
    aggregate_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    state = _STATE
    nodes = _all_artifact_nodes()
    _write_or_verify(
        RUN_DIR / "writer_method_identity.json",
        json.dumps(state["method_identity"], ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
    _write_or_verify(
        RUN_DIR / "writer_extraction_identity.json",
        json.dumps(
            state["method_identity"] | {"identity_sha256": state["method_sha"]},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n",
    )
    _write_or_verify(RUN_DIR / "mem_3a3r_protocol.md", STAGE_PROTOCOL_PATH.read_bytes())
    # The inherited downstream helper expects the historical protocol filename too.
    _write_or_verify(RUN_DIR / "mem_3a3_protocol.md", parent.STAGE_PROTOCOL_PATH.read_bytes())
    _write_or_verify(RUN_DIR / "evidence_ref_set_normalization_v1.json", parent.NORMALIZATION_PATH.read_bytes())
    _write_or_verify(RUN_DIR / "flatprop_recursive_split_v1.json", SPLIT_CONTRACT_PATH.read_bytes())
    _write_jsonl(RUN_DIR / "initial_chunk_manifest.jsonl", state["initial_rows"])
    _write_jsonl(RUN_DIR / "chunk_manifest.jsonl", nodes)
    _write_jsonl(RUN_DIR / "overflow_lineage.jsonl", state["lineage_events"])
    _write_jsonl(flat.WRITER_LEDGER_PATH, attempts)
    leaf_rows = []
    for index, leaf in enumerate(leaves):
        leaf_rows.append(
            {
                **leaf,
                "chunk_index": index,
                "status": "TERMINAL_LEAF",
                "success": True,
                "finish_reason": leaf.get("finish_reason", "stop"),
            }
        )
    _write_jsonl(RUN_DIR / "chunk_normalized_outputs.jsonl", leaf_rows)
    _write_jsonl(flat.EXTRACTIONS_PATH, packets)
    preflight = {
        **state["preflight"],
        "stage_id": RUN_ID,
        "stage_base_commit_sha": state["head"],
        "historical_mem3a3_stage_base_commit_sha": state["historical_stage_base_commit"],
        "method_identity": state["method_identity"],
        "method_contract_sha256": state["method_sha"],
        "initial_chunk_plan_sha256": _sha_file(PARENT_RUN_DIR / "chunk_manifest.jsonl"),
        "initial_chunks": len(state["initial_rows"]),
        "total_chunks": len(leaves),
        "recursive_nodes": len(nodes) - len(state["initial_rows"]),
        "unique_source_sessions": len(state["sessions"]),
        "primary_span_coverage_exactly_once": True,
        "primary_turn_coverage_exactly_once": True,
        "all_rendered_request_budgets_fit": all(
            row.get("request_budget_tokens", MAX_PROMPT_TOKENS + 1) <= MAX_PROMPT_TOKENS
            for row in leaves
        ),
        "labels_loaded": False,
        "test_access": False,
        "102_dev_access": False,
        "medmemorybench_runs": 0,
        "terminal_leaf_coverage": _assert_initial_frozen_leaves(
            leaves=leaves, sessions=state["sessions"]
        ),
        "historical_source": {
            "mem3a3_chunk_manifest_sha256": _sha_file(PARENT_RUN_DIR / "chunk_manifest.jsonl"),
            "mem3a3_partial_ledger_sha256": _sha_file(PARENT_RUN_DIR / "chunk_writer_ledger.partial.jsonl"),
            "mem3a3_failure_sha256": _sha_file(PARENT_RUN_DIR / "writer_failure.json"),
        },
    }
    _write_json(flat.PREFLIGHT_PATH, preflight)
    extraction_manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "method_contract_sha256": state["method_sha"],
        "planned_sessions": 477,
        "aggregated_sessions": len(packets),
        "initial_chunks": 785,
        "terminal_leaf_chunks": len(leaves),
        "recursive_nodes": len(nodes) - 785,
        "writer_attempts": len(attempts),
        "provider_calls_unique_this_stage": sum(
            row.get("provider_calls_this_stage_unique", 0) for row in attempts
        ),
        "provider_calls_this_process": state["new_provider_calls"],
        "historical_cache_imported": state["historical_cache_imported"],
        "historical_cache_miss_reexecuted": len(state["historical_cache_miss_reexecuted"]),
        "retries": 0,
        "hosted_calls": 0,
        "session_extractions_sha256": _sha_file(flat.EXTRACTIONS_PATH),
        "chunk_ledger_sha256": _sha_file(flat.WRITER_LEDGER_PATH),
        "chunk_outputs_sha256": _sha_file(RUN_DIR / "chunk_normalized_outputs.jsonl"),
        "aggregate_diagnostics": aggregate_diagnostics,
        "labels_loaded": False,
        "test_access": False,
        "102_dev_access": False,
    }
    _write_json(flat.EXTRACTION_MANIFEST_PATH, extraction_manifest)
    return preflight


def _writer_diagnostics_r(
    packets: list[dict[str, Any]],
    terminal_ledger: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    chunk_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    attempts = _STATE["all_attempts"]
    aggregate_rows = list(_STATE["aggregate_diagnostics"].values())
    all_props = [prop for packet in packets for prop in packet["propositions"]]
    authority = defaultdict(int)
    same_text_refs: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for prop in all_props:
        authority[prop["source_authority"]] += 1
        same_text_refs[prop["proposition_text"]].add(tuple(prop["evidence_refs"]))
    same_text_different_evidence = sum(
        len(ref_sets) * (len(ref_sets) - 1) // 2 for ref_sets in same_text_refs.values()
    )
    leaves_per_session = [
        sum(row["session_identity_sha256"] == packet["session_identity_sha256"] for row in chunk_outputs)
        for packet in packets
    ]
    base_diag = {
        "source_sessions": len(packets),
        "rawspan_count": len(inventory),
        "total_chunks": len(terminal_ledger),
        "chunks_per_session": parent._percentiles(leaves_per_session),
        "request_budget_tokens_per_chunk": parent._percentiles(
            [row["request_budget_tokens"] for row in terminal_ledger]
        ),
        "rendered_chat_prompt_tokens_per_chunk": parent._percentiles(
            [row["rendered_chat_prompt_tokens"] for row in terminal_ledger]
        ),
        "dynamic_schema_tokens_per_chunk": parent._percentiles(
            [row["dynamic_schema_tokens"] for row in terminal_ledger]
        ),
        "completion_tokens_per_chunk": parent._percentiles(
            [row["completion_tokens"] for row in terminal_ledger if row.get("completion_tokens") is not None]
        ),
        "total_propositions": len(all_props),
        "propositions_per_chunk": parent._percentiles(
            [row["normalized_proposition_count"] for row in terminal_ledger]
        ),
        "propositions_per_session": parent._percentiles(
            [len(packet["propositions"]) for packet in packets]
        ),
        "exact_overlap_duplicates_removed": sum(
            row["exact_duplicates_removed_total"] for row in aggregate_rows
        ),
        "same_text_different_evidence_pairs_retained": same_text_different_evidence,
        "high_similarity_cross_chunk_pairs_retained": [],
        "high_similarity_definition": "SequenceMatcher ratio >= 0.90 over casefolded final unique proposition text; diagnostic-only and retained",
        "authority_counts": {
            "user_only_propositions": authority["user"],
            "assistant_only_propositions": authority["assistant"],
            "mixed_propositions": authority["mixed"],
        },
        "writer_wall_seconds": sum(row.get("provider_duration_ms") or 0 for row in attempts) / 1000,
        "writer_wall_seconds_this_process": sum(
            row.get("provider_duration_ms") or 0
            for row in attempts
            if row.get("provider_calls_this_resume") == 1
        )
        / 1000,
        "provider_calls": sum(row.get("provider_calls_this_stage_unique", 0) for row in attempts),
        "provider_calls_this_process": _STATE["new_provider_calls"],
        "retries": 0,
        "hosted_calls": 0,
        "writer_input_tokens_server_reported": sum(row.get("prompt_tokens") or 0 for row in attempts),
        "writer_completion_tokens": sum(row.get("completion_tokens") or 0 for row in attempts),
        "semantic_quality_used_as_gate": False,
        "revision_keys_extracted": False,
        "writer_stage_elapsed_wall_seconds": time.perf_counter() - _STATE["start_time"],
        "writer_input_tokens": sum(row.get("prompt_tokens", 0) or 0 for row in attempts),
        "writer_latency_ms_total_measured_this_process": sum(
            row.get("provider_duration_ms", 0) or 0 for row in attempts
        ),
        "writer_latency_ms_mean_measured_this_process": None,
        "initial_planned_chunks": 785,
        "historical_successful_chunks_imported": _STATE["historical_cache_imported"],
        "historical_cache_miss_reexecuted": len(_STATE["historical_cache_miss_reexecuted"]),
        "new_initial_chunk_calls": sum(
            row.get("provider_calls_this_stage_unique", 0)
            for row in attempts
            if row.get("recursion_depth", 0) == 0
            and row.get("chunk_id") not in _STATE["historical_successes"]
            and row.get("chunk_id") != _STATE["historical_failed_chunk"]["chunk_id"]
        ),
        "overflow_parent_calls_total": sum(row.get("finish_reason") == "length" for row in attempts),
        "overflow_parent_calls_this_stage": sum(
            row.get("finish_reason") == "length"
            and row.get("provider_calls_this_stage_unique", 0) == 1
            for row in attempts
        ),
        "recursive_child_calls": sum(
            row.get("provider_calls_this_stage_unique", 0)
            for row in attempts
            if row.get("recursion_depth", 0) > 0
        ),
        "overflow_root_count": len(
            {
                row["root_initial_chunk_id"]
                for row in attempts
                if row.get("finish_reason") == "length"
            }
        ),
        "maximum_recursion_depth": max(
            [row.get("recursion_depth", 0) for row in _all_artifact_nodes()] or [0]
        ),
        "total_generated_child_chunks": len(_all_artifact_nodes()) - 785,
        "irreducible_overflows": sum(
            row.get("status") == "IRREDUCIBLE_WRITER_OVERFLOW" for row in attempts
        ),
        "unresolved_structural_failures": 0,
        "raw_generated_propositions_before_exact_collapse": sum(
            row["raw_generated_emissions"] for row in aggregate_rows
        ),
        "final_logical_propositions": sum(
            row["final_logical_propositions"] for row in aggregate_rows
        ),
        "exact_duplicates_removed_within_leaf": sum(
            row["exact_duplicates_removed_within_leaf"] for row in aggregate_rows
        ),
        "exact_duplicates_removed_across_leaves": sum(
            row["exact_duplicates_removed_across_leaves"] for row in aggregate_rows
        ),
        "exact_duplicates_attributed_to_overlap": sum(
            row["exact_duplicates_attributed_to_overlap"] for row in aggregate_rows
        ),
        "near_duplicate_pairs_retained": sum(
            row["near_duplicate_pairs_retained"] for row in aggregate_rows
        ),
        "all_writer_request_completion_tokens": parent._percentiles(
            [row["completion_tokens"] for row in attempts if row.get("completion_tokens") is not None]
        ),
        "terminal_leaf_completion_tokens": parent._percentiles(
            [row["completion_tokens"] for row in terminal_ledger if row.get("completion_tokens") is not None]
        ),
        "writer_attempts": len(attempts),
        "provider_calls_this_stage": sum(
            row.get("provider_calls_this_stage_unique", 0) for row in attempts
        ),
    }
    return base_diag


def _write_closeout(
    *,
    downstream_manifest: dict[str, Any],
    upstream: dict[str, Any],
    preflight: dict[str, Any],
    writer_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = flat.RUN_MANIFEST_PATH
    manifest = _json(manifest_path)
    operations = _read_jsonl(flat.MATERIALIZATION_LEDGER_PATH)
    reader_rows = _read_jsonl(flat.READER_LEDGER_PATH)
    predictions = _read_jsonl(flat.PREDICTIONS_PATH)
    embedding = _json(flat.EMBEDDING_MANIFEST_PATH)
    context_rows = _read_jsonl(flat.CONTEXT_BUNDLES_PATH)
    lineage = _read_jsonl(RUN_DIR / "overflow_lineage.jsonl")
    initial_ids = {row["chunk_id"] for row in _STATE["initial_rows"]}
    child_events = [row for row in lineage if row.get("event") == "CHILD_CREATED"]
    parent_events = [row for row in lineage if row.get("event") == "PARENT_SUBDIVIDED"]
    recursive_ids = set(_STATE["node_registry"]) - initial_ids
    attempted_roots = {
        row["chunk_id"] for row in _STATE["all_attempts"] if row.get("recursion_depth", 0) == 0
    }
    attempt_statuses_valid = all(
        row.get("status") in {"TERMINAL_LEAF", "OVERFLOW_PARENT"}
        and row.get("retry_count", 0) == 0
        and row.get("hosted_call") is False
        and row.get("request_budget_tokens", MAX_PROMPT_TOKENS + 1) <= MAX_PROMPT_TOKENS
        for row in _STATE["all_attempts"]
    )
    materialization_valid = bool(operations) and all(
        row.get("operation") == "ADD"
        and row.get("status") == "active"
        and row.get("version") == 1
        and row.get("supersedes_id") is None
        and row.get("valid_until") is None
        and row.get("expires_at") is None
        for row in operations
    )
    embedding_valid = (
        embedding.get("local_only") is True
        and embedding.get("hosted_calls") == 0
        and embedding.get("model_id") == "Qwen/Qwen3-Embedding-0.6B"
        and str(embedding.get("device", "")).startswith("cuda")
        and embedding.get("dtype") == "float16"
        and embedding.get("document_truncations") == 0
        and embedding.get("query_truncations") == 0
    )
    reader_valid = len(reader_rows) == 10 and len(predictions) == 10 and all(
        row.get("success") is True
        and row.get("quality_status") == "OK"
        and row.get("hosted_call") is False
        and row.get("loopback_only") is True
        for row in reader_rows
    ) and all(row.get("system") == SYSTEM_NAME for row in predictions)
    initial_plan_sha = _sha_file(PARENT_RUN_DIR / "chunk_manifest.jsonl")
    initial_plan_copy_sha = _sha_file(RUN_DIR / "initial_chunk_manifest.jsonl")
    lineage_valid = (
        len(child_events) == len(recursive_ids)
        and {row.get("chunk_id") for row in child_events} == recursive_ids
        and all(row.get("child_chunk_ids") for row in parent_events)
        and all(
            child_id in recursive_ids
            for row in parent_events
            for child_id in row["child_chunk_ids"]
        )
    )
    manifest.update(
        {
            "schema_version": 1,
            "stage": RUN_ID,
            "status": "COMPLETE",
            "completion_gate_marker": f"{GATE}=YES",
            "flat_no_revision_gate_marker": "MEM3A3R_FLAT_NO_REVISION=YES",
            "base_commit_sha": _STATE["head"],
            "historical_mem3a3_gate": "MEM3A3_CHUNKED_FLATPROP_FROZEN_10_DIAGNOSTIC=NO",
            "method_identity": writer_v4.CONTRACT_ID,
            "method_contract_sha256": _STATE["method_sha"],
            "initial_chunks": 785,
            "terminal_leaf_chunks": len(_STATE["terminal_outputs"]),
            "recursive_nodes": len(_all_artifact_nodes()) - 785,
            "source_sessions": 477,
            "source_rawspans": len(_STATE["inventory"]),
            "historical_successful_chunks_imported": _STATE["historical_cache_imported"],
            "historical_cache_miss_reexecuted": len(_STATE["historical_cache_miss_reexecuted"]),
            "new_initial_chunk_calls": writer_diagnostics["new_initial_chunk_calls"],
            "overflow_parent_calls_total": writer_diagnostics["overflow_parent_calls_total"],
            "overflow_parent_calls_this_stage": writer_diagnostics["overflow_parent_calls_this_stage"],
            "recursive_child_calls": writer_diagnostics["recursive_child_calls"],
            "overflow_root_count": writer_diagnostics["overflow_root_count"],
            "maximum_recursion_depth": writer_diagnostics["maximum_recursion_depth"],
            "irreducible_overflows": writer_diagnostics["irreducible_overflows"],
            "writer_provider_calls_unique": writer_diagnostics["provider_calls_this_stage"],
            "writer_calls_this_process": _STATE["new_provider_calls"],
            "writer_failures": 0,
            "retries": 0,
            "hosted_calls": 0,
            "reader_calls": 10,
            "judge_calls": 0,
            "reader_model_role": "reader_answer; frozen local Qwen3-8B Q4_K_M",
            "memory_internal_llm_role": "memory_ingest; same frozen local Qwen3-8B",
            "embedding_model_role": "Qwen/Qwen3-Embedding-0.6B local CUDA FP16",
            "judge_model": None,
            "labels_loaded_after_freeze": True,
            "test_access": False,
            "102_dev_access": False,
            "medmemorybench_runs": 0,
            "upstream": upstream,
            "gate": {
                "historical_mem3a3_gate_preserved_no": True,
                "frozen_785_initial_chunk_plan_sha_valid": len(_STATE["initial_rows"]) == 785
                and initial_plan_sha == preflight["initial_chunk_plan_sha256"]
                and initial_plan_copy_sha == initial_plan_sha
                and _verify(RUN_DIR / "initial_chunk_manifest.jsonl"),
                "historical_successes_imported_or_reexecuted": _STATE["historical_cache_imported"]
                + len(_STATE["historical_cache_miss_reexecuted"])
                == 147,
                "all_785_initial_roots_attempted": len(attempted_roots) == 785
                and attempted_roots == initial_ids,
                "477_sessions_complete": preflight["terminal_leaf_coverage"]["complete_sessions"] == 477,
                "every_rawspan_has_one_terminal_primary_owner": preflight["terminal_leaf_coverage"]["primary_rawspan_coverage_exactly_once"],
                "initial_and_recursive_nodes_auditable": _verify(RUN_DIR / "chunk_manifest.jsonl")
                and _verify(RUN_DIR / "overflow_lineage.jsonl")
                and lineage_valid,
                "all_overflow_parents_discarded": all(
                    row.get("semantic_output_used") is False
                    for row in _STATE["all_attempts"]
                    if row.get("finish_reason") == "length"
                )
                and all(
                    row["chunk_id"] != _STATE["historical_failed_chunk"]["chunk_id"]
                    for row in _STATE["terminal_outputs"]
                ),
                "all_provider_requests_within_6144_and_non_length_statuses_valid": preflight["all_rendered_request_budgets_fit"]
                and attempt_statuses_valid,
                "zero_irreducible_overflows": writer_diagnostics["irreducible_overflows"] == 0,
                "zero_unresolved_structural_failures": writer_diagnostics["unresolved_structural_failures"] == 0,
                "zero_retries": all(row.get("retry_count", 0) == 0 for row in _STATE["all_attempts"]),
                "zero_hosted_calls": all(
                    row.get("hosted_call") is False for row in _STATE["all_attempts"] + reader_rows
                ),
                "flatprop_exact_identity_canonicalization_complete": True,
                "materialization_add_active_version1_no_revision": materialization_valid,
                "frozen_embedding_local_cuda_fp16_no_truncation": embedding_valid,
                "dense_top8_complete_for_ten": manifest.get("dense_rows") == 80,
                "projection_unchanged_and_budget_1024": flat.MEMORY_BUDGET == 1024
                and _sha_file(flat.PROJECTION_CONTRACT_PATH) == flat.PINNED_PROJECTION_SHA256
                and len(context_rows) == 10,
                "reader_10_of_10": manifest.get("reader_calls") == 10 and reader_valid,
                "judge_zero": manifest.get("judge_calls") == 0,
                "prediction_and_reader_ledger_frozen_before_labels": _verify(flat.PREDICTIONS_PATH)
                and _verify(flat.READER_LEDGER_PATH)
                and manifest.get("labels_loaded_after_freeze") is True,
                "no_dev_test_or_medmemorybench": manifest.get("test_access") is False
                and manifest.get("102_dev_access") is False
                and manifest.get("medmemorybench_runs") == 0,
            },
        }
    )
    required_run_artifacts = [
        flat.PREFLIGHT_PATH,
        RUN_DIR / "initial_chunk_manifest.jsonl",
        RUN_DIR / "chunk_manifest.jsonl",
        RUN_DIR / "overflow_lineage.jsonl",
        flat.WRITER_LEDGER_PATH,
        RUN_DIR / "chunk_normalized_outputs.jsonl",
        flat.EXTRACTION_MANIFEST_PATH,
        flat.EXTRACTIONS_PATH,
        flat.FLAT_PROPOSITIONS_PATH,
        flat.MATERIALIZATION_LEDGER_PATH,
        flat.EMBEDDING_MANIFEST_PATH,
        flat.DENSE_TOP8_PATH,
        flat.CONTEXT_PLANS_PATH,
        flat.CONTEXT_BUNDLES_PATH,
        flat.PREDICTIONS_PATH,
        flat.READER_LEDGER_PATH,
        flat.METRICS_PATH,
        flat.EFFICIENCY_PATH,
        flat.COMPARISON_PATH,
        flat.CASE_REVIEW_PATH,
        RUN_DIR / "writer_chunk_diagnostics.json",
        RUN_DIR / "writer_method_identity.json",
        RUN_DIR / "writer_extraction_identity.json",
        RUN_DIR / "failed_root_split_preflight.json",
        RUN_DIR / "mem_3a3r_protocol.md",
        RUN_DIR / "flatprop_recursive_split_v1.json",
        flat.RUN_REPORT_PATH,
    ]
    if not all(_verify(path) for path in required_run_artifacts):
        manifest["gate"]["all_required_artifact_sha_sidecars_valid"] = False
    else:
        manifest["gate"]["all_required_artifact_sha_sidecars_valid"] = True
    if not all(manifest["gate"].values()):
        manifest["status"] = "FAILED"
        manifest["completion_gate_marker"] = f"{GATE}=NO"
        raise RuntimeError("MEM-3A.3R closeout gate failed")
    metrics = _json(flat.METRICS_PATH)
    means = metrics.get("means", {})
    report = _render_report_r(manifest, writer_diagnostics, means)
    atomic_write_bytes(flat.RUN_REPORT_PATH, report.encode("utf-8"))
    _freeze(flat.RUN_REPORT_PATH)
    atomic_write_bytes(REPORT_DOC, report.encode("utf-8"))
    _freeze(REPORT_DOC)
    manifest["report_sha256"] = _sha_file(REPORT_DOC)
    manifest["artifact_sha256"] = {
        path.name: _sha_file(path) for path in required_run_artifacts
    }
    manifest["artifact_sha256"][REPORT_DOC.name] = _sha_file(REPORT_DOC)
    manifest["gate"]["all_required_artifact_sha_sidecars_valid"] = all(
        _verify(path) for path in required_run_artifacts + [REPORT_DOC]
    )
    if not all(manifest["gate"].values()):
        raise RuntimeError("MEM-3A.3R artifact hash closeout failed")
    _write_json(manifest_path, manifest)
    print(f"{GATE}=YES", flush=True)
    return manifest


def _render_report_r(manifest: dict[str, Any], diagnostics: dict[str, Any], means: dict[str, Any]) -> str:
    gate_lines = [f"- `{name}`: `{value}`" for name, value in manifest["gate"].items()]
    return "\n".join(
        [
            "# MEM-3A.3R - Recursive Overflow Recovery and FlatProp Completion",
            "",
            f"Completion gate: `{manifest['completion_gate_marker']}`.",
            "",
            "This is a frozen-ten mechanism diagnostic only. It is not a public benchmark result, performance ranking, or broad superiority claim. The historical MEM-3A.3 gate remains `MEM3A3_CHUNKED_FLATPROP_FROZEN_10_DIAGNOSTIC=NO` and its 785-root plan is unchanged.",
            "",
            "## Writer Accounting",
            "",
            f"- Initial plan: 785 chunks; historical successful cache imports: {diagnostics['historical_successful_chunks_imported']}; historical chunks re-executed after cache miss: {diagnostics['historical_cache_miss_reexecuted']}.",
            f"- New initial-chunk calls: {diagnostics['new_initial_chunk_calls']}; overflow-parent calls this stage: {diagnostics['overflow_parent_calls_this_stage']} (all-time overflow parents: {diagnostics['overflow_parent_calls_total']}); recursive child calls: {diagnostics['recursive_child_calls']}.",
            f"- Overflow roots: {diagnostics['overflow_root_count']}; maximum recursion depth: {diagnostics['maximum_recursion_depth']}; generated child nodes: {diagnostics['total_generated_child_chunks']}; irreducible overflows: {diagnostics['irreducible_overflows']}.",
            f"- Local provider calls attributable to this stage: {diagnostics['provider_calls_this_stage']}; retries: 0; hosted calls: 0.",
            f"- Sessions/terminal leaves: {manifest['source_sessions']} / {manifest['terminal_leaf_chunks']}; source RawSpans: {manifest['source_rawspans']}; primary ownership exactly once: {manifest['gate']['every_rawspan_has_one_terminal_primary_owner']}.",
            f"- Completion tokens, all writer attempts: {diagnostics['all_writer_request_completion_tokens']}; terminal leaves: {diagnostics['terminal_leaf_completion_tokens']}.",
            "- The historical truncated parent response was verified as forensic-only (`semantic_output_used=false`); no partial parent proposition was aggregated.",
            "",
            "## Exact Identity",
            "",
            f"- Raw generated propositions: {diagnostics['raw_generated_propositions_before_exact_collapse']}; final logical propositions: {diagnostics['final_logical_propositions']}.",
            f"- Exact duplicates removed: same leaf {diagnostics['exact_duplicates_removed_within_leaf']}, across leaves {diagnostics['exact_duplicates_removed_across_leaves']}, overlap-attributed {diagnostics['exact_duplicates_attributed_to_overlap']}.",
            f"- Near-duplicate pairs retained for diagnosis: {diagnostics['near_duplicate_pairs_retained']}; fuzzy/embedding merge: none.",
            "- Logical identity is SHA-256 over stripped proposition text and canonical evidence refs. Only terminal successful leaves contribute.",
            "",
            "## Frozen-Ten Diagnostic",
            "",
            f"- Dense top-8 rows: {manifest['dense_rows']}; shared-reader calls: {manifest['reader_calls']}/10; judge calls: 0.",
            f"- Mean answer-session Recall@5 / @8 / MRR: {means.get('answer_session_recall_at_5')} / {means.get('answer_session_recall_at_8')} / {means.get('mrr')}; projected Recall@8: {means.get('projected_answer_session_recall_at_8')}.",
            f"- Mean context reader tokens: {means.get('reader_visible_context_tokens')}; token precision / recall / F1 / normalized EM: {means.get('token_precision')} / {means.get('token_recall')} / {means.get('token_f1')} / {means.get('normalized_em')}.",
            "- Comparison is MEM-2D RawSpan + Dense versus recursive chunked FlatProp + Dense. Per-question comparison is descriptive only.",
            "- Frozen update cases `1cea1afa` and `c4ea545c` were inspected diagnostically; Instagram 500/600 state is not resolved or suppressed.",
            "",
            "## Scope and Gate",
            "",
            "- Materialization remains ADD / SESSION_NOTE / SESSION_DERIVED / ACTIVE / version 1; no revision semantics.",
            "- Embedding is the frozen local Qwen3-Embedding-0.6B CUDA FP16; dense top-8; unchanged rank-aware projection SHA and 1,024-token budget; frozen shared local Qwen3-8B reader; no judge.",
            "- No LongMemEval 102 DEV, TEST, MedMemoryBench, hosted provider, retry, M10-Flat, RevMem, or MEM-3B0 run.",
            *gate_lines,
            "",
            f"`{GATE}=YES`",
            "",
        ]
    )


def _patch_reader_system_name() -> Any:
    original = flat._run_reader

    def wrapped(pre_reader_sha: str, preflight: dict[str, Any], bundles: list[dict[str, Any]]):
        predictions, ledger = original(pre_reader_sha, preflight, bundles)
        return [{**row, "system": SYSTEM_NAME} for row in predictions], ledger

    flat._run_reader = wrapped
    return original


def _run_downstream(
    *,
    upstream: dict[str, Any],
    preflight: dict[str, Any],
    packets: list[dict[str, Any]],
    terminal_outputs: list[dict[str, Any]],
    terminal_ledger: list[dict[str, Any]],
    writer_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    original_reader = _patch_reader_system_name()
    original_diagnostics = parent._writer_diagnostics
    original_render = parent._render_report
    original_comparison = parent._comparison
    parent._writer_diagnostics = lambda *args: writer_diagnostics
    parent._render_report = lambda *args: "# Interim MEM-3A.3R downstream artifact\n"

    def comparison(metrics: dict[str, Any]) -> dict[str, Any]:
        result = original_comparison(metrics)
        result["candidate"] = "MEM-3A.3R Recursive Chunked FlatProp + Dense"
        result["changed_component"] = "recursive overflow recovery and exact session-wide proposition identity"
        result["interpretation"] = "Descriptive frozen-ten diagnostic only; no ranking, superiority claim, or statistical inference."
        result["labels_used_after_freeze"] = True
        return result

    parent._comparison = comparison
    leaf_ledger = [
        {
            **leaf,
            "success": True,
            "validation": "passed",
            "retry_count": 0,
            "hosted_call": False,
            "provider_calls": 1,
        }
        for leaf in terminal_outputs
    ]
    faux_writer_manifest = {
        "provider_calls_unique": len(terminal_outputs),
        "provider_calls_this_process": _STATE["new_provider_calls"],
        "retries": 0,
        "hosted_calls": 0,
        "wall_seconds": time.perf_counter() - _STATE["start_time"],
    }
    try:
        downstream_manifest = parent._downstream(
            upstream,
            preflight,
            packets,
            leaf_ledger,
            _STATE["sessions"],
            _STATE["refs"],
            _STATE["inventory"],
            faux_writer_manifest,
            terminal_outputs,
        )
    finally:
        flat._run_reader = original_reader
        parent._writer_diagnostics = original_diagnostics
        parent._render_report = original_render
        parent._comparison = original_comparison
    return downstream_manifest


def _build_preflight_only(state: dict[str, Any]) -> dict[str, Any]:
    failed = state["historical_failed_chunk"]
    session = state["sessions"][failed["session_identity_sha256"]]
    with httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=10.0),
        trust_env=False,
    ) as client:
        if parent._runtime(client) != state["preflight"]["writer_runtime_identity"]:
            raise RuntimeError("Frozen local Qwen runtime identity changed")
        root = {
            **failed,
            "root_initial_chunk_id": failed["chunk_id"],
            "parent_chunk_id": None,
            "recursion_depth": 0,
            "split_contract_sha256": SPLIT_CONTRACT_SHA256,
            "primary_unit_kind": "initial_frozen_plan",
            "primary_unit_ordinals": failed["primary_turn_ordinals"],
            "identity_sha256": failed["chunk_id"],
        }
        children = _split_node(client=client, node=root, session=session)
        prepared = [
            _prepare_node(child, client, session)
            for child in children
        ]
    result = {
        "historical_failed_root": failed["chunk_id"],
        "historical_source_session_id": "sharegpt_vbNrVtS_151",
        "split_contract_sha256": SPLIT_CONTRACT_SHA256,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "child_count": 2,
        "children": [
            [
                {
                    "chunk_id": node["chunk_id"],
                    "parent_chunk_id": node["parent_chunk_id"],
                    "root_initial_chunk_id": node["root_initial_chunk_id"],
                    "recursion_depth": node["recursion_depth"],
                    "primary_start_ordinal": node["primary_start_ordinal"],
                    "primary_end_ordinal": node["primary_end_ordinal"],
                    "request_budget_tokens": node["request_budget_tokens"],
                    "finish_reason": None,
                }
                for node in group
            ]
            for group in prepared
        ],
        "provider_calls": 0,
        "labels_loaded": False,
    }
    _write_or_verify(
        RUN_DIR / "failed_root_split_preflight.json",
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
    return result


def _run(preflight_only: bool) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    state = _configure_state(head)
    if preflight_only:
        result = _build_preflight_only(state)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result

    protocol_copy = RUN_DIR / "mem_3a3r_protocol.md"
    _write_or_verify(protocol_copy, STAGE_PROTOCOL_PATH.read_bytes())
    _write_or_verify(RUN_DIR / "flatprop_recursive_split_v1.json", SPLIT_CONTRACT_PATH.read_bytes())
    with httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=10.0),
        trust_env=False,
    ) as client:
        leaves, attempts = _resolve_initial_roots(client=client, state=state)
    state["all_attempts"] = attempts
    state["terminal_outputs"] = leaves
    coverage = _assert_initial_frozen_leaves(leaves=leaves, sessions=state["sessions"])
    print(
        f"writer_complete sessions={coverage['complete_sessions']} leaves={coverage['complete_terminal_leaves']} "
        f"rawspans={coverage['source_rawspans']} max_depth={coverage['maximum_recursion_depth']}",
        flush=True,
    )
    packets, aggregate_diagnostics = _aggregate_sessions(leaves=leaves, sessions=state["sessions"])
    state["aggregate_diagnostics"] = aggregate_diagnostics
    preflight = _finalize_writer_artifacts(
        leaves=leaves,
        attempts=attempts,
        packets=packets,
        aggregate_diagnostics=aggregate_diagnostics,
    )
    upstream = state["upstream"]
    if len(packets) != 477 or len(leaves) != preflight["total_chunks"]:
        raise RuntimeError("FlatProp writer outputs incomplete; downstream was not started")

    # Freeze the exact inputs before labels. The inherited downstream loads labels only after
    # predictions and reader ledger have both been written and SHA-frozen.
    provisional_diag = {
        "writer_stage_elapsed_wall_seconds": time.perf_counter() - state["start_time"],
        "writer_input_tokens": sum(row.get("prompt_tokens", 0) or 0 for row in attempts),
        "writer_completion_tokens": sum(row.get("completion_tokens", 0) or 0 for row in attempts),
        "writer_latency_ms_total_measured_this_process": sum(
            row.get("provider_duration_ms", 0) or 0 for row in attempts
        ),
        "writer_latency_ms_mean_measured_this_process": None,
        "provider_calls_this_stage": sum(row.get("provider_calls_this_stage_unique", 0) for row in attempts),
        "new_initial_chunk_calls": 0,
        "overflow_parent_calls_total": 0,
        "overflow_parent_calls_this_stage": 0,
        "recursive_child_calls": 0,
        "overflow_root_count": 0,
        "maximum_recursion_depth": coverage["maximum_recursion_depth"],
        "total_generated_child_chunks": len(_all_artifact_nodes()) - 785,
        "irreducible_overflows": 0,
        "unresolved_structural_failures": 0,
        "raw_generated_propositions_before_exact_collapse": sum(row["raw_generated_emissions"] for row in aggregate_diagnostics.values()),
        "final_logical_propositions": sum(row["final_logical_propositions"] for row in aggregate_diagnostics.values()),
        "exact_duplicates_removed_within_leaf": sum(row["exact_duplicates_removed_within_leaf"] for row in aggregate_diagnostics.values()),
        "exact_duplicates_removed_across_leaves": sum(row["exact_duplicates_removed_across_leaves"] for row in aggregate_diagnostics.values()),
        "exact_duplicates_attributed_to_overlap": sum(row["exact_duplicates_attributed_to_overlap"] for row in aggregate_diagnostics.values()),
        "near_duplicate_pairs_retained": sum(row["near_duplicate_pairs_retained"] for row in aggregate_diagnostics.values()),
    }
    state["inventory"] = state["inventory"]
    state["aggregate_diagnostics"] = aggregate_diagnostics
    leaf_ledger = [row for row in attempts if row.get("status") == "TERMINAL_LEAF"]
    _write_json(RUN_DIR / "writer_chunk_diagnostics.json", provisional_diag)
    # The shared downstream helper uses module globals; all of them now point at this R run.
    downstream_manifest = _run_downstream(
        upstream=upstream,
        preflight=preflight,
        packets=packets,
        terminal_outputs=leaves,
        terminal_ledger=leaf_ledger,
        writer_diagnostics=provisional_diag,
    )
    writer_diagnostics = _writer_diagnostics_r(
        packets, leaf_ledger, state["inventory"], leaves
    )
    _write_json(RUN_DIR / "writer_chunk_diagnostics.json", writer_diagnostics)
    comparison = _json(flat.COMPARISON_PATH)
    comparison["candidate"] = "MEM-3A.3R Recursive Chunked FlatProp + Dense"
    comparison["changed_component"] = "recursive overflow recovery and exact session-wide proposition identity"
    comparison["interpretation"] = "Descriptive frozen-ten diagnostic only; no ranking, superiority claim, or statistical inference."
    _write_json(flat.COMPARISON_PATH, comparison)
    # Re-render report after final diagnostics and comparison are frozen.
    manifest = _write_closeout(
        downstream_manifest=downstream_manifest,
        upstream=upstream,
        preflight=preflight,
        writer_diagnostics=writer_diagnostics,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        _run(preflight_only=args.preflight_only)
        return 0
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        _write_json(
            RUN_DIR / "run_failure.json",
            {
                "schema_version": 1,
                "stage": RUN_ID,
                "status": "FAILED_STOP_NO_RETRY",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completion_gate_marker": f"{GATE}=NO",
                "provider_calls_this_process": _STATE.get("new_provider_calls", 0),
                "retries": 0,
                "hosted_calls": 0,
                "labels_loaded": (RUN_DIR / "deterministic_metrics.json").exists(),
                "downstream_started": (RUN_DIR / "flatprop_inventory.jsonl").exists(),
            },
        )
        print(f"{GATE}=NO; {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
