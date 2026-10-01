"""Replay one completed local response and post only the two unsent cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from tools.research.memory import run_mem3b0q_r4_cardinality_devset_v2 as scorer
from tools.research.memory import run_mem3b0q_r4_candidate_local_smoke_v1 as smoke
from tools.research.memory import run_mem3b0q_r4_event_value_confirmation_v1 as engine
from tools.research.memory import run_mem3b0q_r4_event_value_confirmation_v1r2 as v1r2
from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate


ROOT = engine.ROOT
OUTPUT_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-event-value-confirmation-v1r3"
LOCK_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_event_value_confirmation_v1r3_lock.json"
LOCK_SHA_PATH = LOCK_PATH.with_suffix(LOCK_PATH.suffix + ".sha256")
PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem3b0q_r4_event_value_confirmation_v1r3_protocol.md"
FREEZE_PATH = ROOT / "tools" / "research" / "memory" / "freeze_mem3b0q_r4_event_value_confirmation_v1r3.py"
PRIOR_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-event-value-confirmation-v1r2"
REPLAY_CASE = "EVTCONF-01"

LOCK_STATIC_FIELDS = {
    **v1r2.LOCK_STATIC_FIELDS,
    "lock_id": "mem3b0q-r4-event-value-confirmation-v1r3-lock",
    "run_id": OUTPUT_ROOT.name,
    "max_completion_posts": 2,
    "process_snapshot_execution": "elevated_read_only_listener_process_query_with_exact_prior_response_replay",
}
DEPENDENCY_PATHS = tuple(
    dict.fromkeys(
        (
            *v1r2.DEPENDENCY_PATHS,
            "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1r3_protocol.md",
            "tools/research/memory/run_mem3b0q_r4_event_value_confirmation_v1r3.py",
            "tools/research/memory/freeze_mem3b0q_r4_event_value_confirmation_v1r3.py",
            "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1r2_lock.json",
            "docs/research/memory/mem3b0q_r4_event_value_confirmation_v1r2_lock.json.sha256",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r2/run_reservation.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r2/EVTCONF-01/request.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r2/EVTCONF-01/reservation.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r2/EVTCONF-01/response_body.bin",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r2/EVTCONF-01/case_result.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r2/confirmation_result.json",
            "runs/memory/mem3/mem3b0q-r4-event-value-confirmation-v1r2/run_manifest.json",
        )
    )
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Any) -> None:
    _write_exclusive(path, _canonical_json(value) + b"\n")


@contextmanager
def _configured_engine() -> Iterator[None]:
    with v1r2._configured_engine():
        saved_build_inputs = engine._build_inputs
        names = ("OUTPUT_ROOT", "LOCK_PATH", "LOCK_SHA_PATH", "PROTOCOL_PATH", "FREEZE_PATH", "LOCK_STATIC_FIELDS", "DEPENDENCY_PATHS")
        saved = {name: getattr(engine, name) for name in names}

        def build_inputs_with_scoring_adapter():
            pack, requests, scoring_requests = saved_build_inputs()
            return pack, requests, {case_id: {"request": value} for case_id, value in scoring_requests.items()}

        try:
            engine.OUTPUT_ROOT = OUTPUT_ROOT
            engine.LOCK_PATH = LOCK_PATH
            engine.LOCK_SHA_PATH = LOCK_SHA_PATH
            engine.PROTOCOL_PATH = PROTOCOL_PATH
            engine.FREEZE_PATH = FREEZE_PATH
            engine.LOCK_STATIC_FIELDS = LOCK_STATIC_FIELDS
            engine.DEPENDENCY_PATHS = DEPENDENCY_PATHS
            engine._build_inputs = build_inputs_with_scoring_adapter
            yield
        finally:
            engine._build_inputs = saved_build_inputs
            for name, value in saved.items():
                setattr(engine, name, value)


def build_lock_payload() -> dict[str, Any]:
    with _configured_engine():
        return engine.build_lock_payload()


def _replay_case(case: dict[str, Any], request: dict[str, Any], scoring_request: dict[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    if case_id != REPLAY_CASE:
        raise engine.ConfirmationRunError("unexpected_replay_case")
    prior_dir = PRIOR_ROOT / case_id
    prior_result = json.loads((prior_dir / "case_result.json").read_text(encoding="utf-8"))
    prior_request = (prior_dir / "request.json").read_bytes()
    response_bytes = (prior_dir / "response_body.bin").read_bytes()
    if prior_request != request["body"]:
        raise engine.ConfirmationRunError("replay_request_bytes_mismatch")
    if prior_result.get("request_sha256") != request["request_sha256"]:
        raise engine.ConfirmationRunError("replay_request_sha_mismatch")
    if prior_result.get("completion_post_attempts") != 1 or prior_result.get("retry_count") != 0:
        raise engine.ConfirmationRunError("prior_post_cardinality_mismatch")
    if prior_result.get("http_status") != 200 or prior_result.get("hosted_calls") != 0:
        raise engine.ConfirmationRunError("prior_response_not_replayable")

    case_dir = OUTPUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    response_sha = _sha(response_bytes)
    _write_json_exclusive(case_dir / "reservation.json", {
        "status": "REPLAYED_PRIOR_RESPONSE_NO_POST",
        "run_id": OUTPUT_ROOT.name,
        "case_id": case_id,
        "source_run_id": PRIOR_ROOT.name,
        "request_sha256": request["request_sha256"],
        "response_sha256": response_sha,
    })
    _write_exclusive(case_dir / "request.json", prior_request)
    _write_exclusive(case_dir / "response_body.bin", response_bytes)
    envelope, content = smoke._response_content(response_bytes)
    row = scorer._score_response(
        case,
        content,
        v2_request=request["request"],
        v1_request=scoring_request["request"],
        response_sha256=response_sha,
        usage=envelope.get("usage"),
        latency_ms=prior_result.get("latency_ms"),
        finish_reason=envelope.get("choices", [{}])[0].get("finish_reason"),
    )
    row.update({
        "run_id": OUTPUT_ROOT.name,
        "request_sha256": request["request_sha256"],
        "http_status": 200,
        "completion_post_attempts": 0,
        "retry_count": 0,
        "hosted_calls": 0,
        "memory_store_mutations": 0,
        "runtime_checkpoints": prior_result.get("runtime_checkpoints", []),
        "evidence_type": "REPLAYED_PRIOR_LOCAL_RESPONSE_NO_NEW_POST",
        "replay_source_run_id": PRIOR_ROOT.name,
        "replay_source_case_result_sha256": _sha((prior_dir / "case_result.json").read_bytes()),
        "replay_source_response_sha256": response_sha,
        "prior_infra_failure_preserved": prior_result.get("status") == "INFRA_FAILURE",
    })
    if row.get("admitted_normalized_atoms"):
        _write_json_exclusive(case_dir / "admitted_atoms.json", {
            "source_id": case_id,
            "scope_id": engine.v1_runner.frozen_r4.FROZEN_SCOPE_ID,
            "atoms": row["admitted_normalized_atoms"],
        })
    _write_json_exclusive(case_dir / "case_result.json", row)
    return row


def run_once() -> dict[str, Any]:
    with _configured_engine():
        original_run_case = engine._run_case

        def run_case(case: dict[str, Any], request: dict[str, Any], scoring_request: dict[str, Any]) -> dict[str, Any]:
            if case["case_id"] == REPLAY_CASE:
                return _replay_case(case, request, scoring_request)
            return original_run_case(case, request, scoring_request)

        engine._run_case = run_case
        try:
            return engine.run_once()
        finally:
            engine._run_case = original_run_case


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-confirmation-once", action="store_true")
    args = parser.parse_args()
    if not args.run_confirmation_once:
        parser.error("explicit --run-confirmation-once is required")
    result = run_once()
    print(json.dumps({"status": result["status"], "aggregate": result["aggregate"], "cases": [{"case_id": row["case_id"], "status": row["status"], "matched": row.get("matched_expected_atoms"), "expected": row.get("expected_atoms"), "failure": row.get("failure"), "evidence_type": row.get("evidence_type")} for row in result["cases"]]}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
