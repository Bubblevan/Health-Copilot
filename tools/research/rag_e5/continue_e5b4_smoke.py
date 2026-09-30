"""Continue only the unattempted path smoke actions after a shared-slot timeout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.rag_e5.b4_execution import (ACTION_ORDER, DEFAULT_B4_PRIVATE_ROOT,
                                      DEFAULT_PROTOCOL_PATH, LlamaServerClient,
                                      _read_json, _wait_for_idle, _write_json,
                                      load_verified_b2, make_execution_plan,
                                      verify_b3_identity, verify_frozen_code,
                                      verify_protocol_lock, verify_server)
from eval.rag_e5.b4_materializer import extract_citations
from eval.rag_e5.e5b3_recovery import sha256_file


def _output_row(action: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("response")
    if (
        payload.get("action_label_for_smoke_only") != action
        or not isinstance(response, dict)
        or not isinstance(response.get("text"), str)
        or not response["text"].strip()
    ):
        raise ValueError(f"existing {action} smoke artifact is incomplete or empty")
    return {
        "action": action,
        "non_empty": True,
        "finish_reason": response.get("finish_reason"),
        "alias_map_rows": len(payload.get("alias_map", [])),
        "citation_regex_exercised": True,
        "output_file_sha256": sha256_file(path),
    }


def continue_smoke(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    private_root: Path = DEFAULT_B4_PRIVATE_ROOT,
) -> dict[str, Any]:
    """Reuse the already completed OFF smoke and execute only STANDARD and STRONG."""
    lock = _read_json(protocol_path)
    lock_sha = verify_protocol_lock(lock)
    verify_frozen_code(lock)
    client = LlamaServerClient(lock["runtime"]["server_url"])
    server = verify_server(client, lock)
    b3_identity = verify_b3_identity(lock)
    frozen = load_verified_b2(lock)
    plan = make_execution_plan(frozen=frozen, client=client, private_root=private_root)
    sample = next(row for row in plan if row["task_kind"] == "GUIDANCE_ONLY")
    smoke_dir = private_root / "smoke"
    report_path = smoke_dir / "smoke_report.json"
    if report_path.exists():
        raise FileExistsError("a B4 smoke report already exists; smoke artifacts are immutable")

    off_path = smoke_dir / "OFF.json"
    off_payload = _read_json(off_path)
    off_plan_row = next(
        row for row in plan if row["case_id"] == sample["case_id"] and row["action"] == "OFF"
    )
    if (
        off_payload.get("case_id") != sample["case_id"]
        or off_payload.get("prompt_sha256") != off_plan_row["guidance_prompt_sha256"]
        or off_payload.get("alias_map") != off_plan_row["evidence_aliases"]
    ):
        raise ValueError("preserved OFF smoke artifact does not match the frozen plan")
    outputs = [_output_row("OFF", off_path, off_payload)]

    for action in ACTION_ORDER[1:]:
        row = next(
            item
            for item in plan
            if item["case_id"] == sample["case_id"] and item["action"] == action
        )
        path = smoke_dir / f"{action}.json"
        if path.exists():
            payload = _read_json(path)
            if (
                payload.get("case_id") != row["case_id"]
                or payload.get("prompt_sha256") != row["guidance_prompt_sha256"]
                or payload.get("alias_map") != row["evidence_aliases"]
            ):
                raise ValueError(f"existing {action} smoke artifact does not match frozen plan")
        else:
            _wait_for_idle(client)
            completion = client.complete(row["guidance_prompt"])
            citations, invented = extract_citations(completion.text, row["evidence_aliases"])
            payload = {
                "action_label_for_smoke_only": action,
                "case_id": row["case_id"],
                "prompt_sha256": row["guidance_prompt_sha256"],
                "request": dict(completion.request_payload),
                "response": {
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "input_tokens": completion.input_tokens,
                    "output_tokens": completion.output_tokens,
                    "latency_ms": completion.latency_ms,
                },
                "alias_map": row["evidence_aliases"],
                "resolved_citation_chunk_ids": citations,
                "invented_evidence_aliases": invented,
            }
            _write_json(path, payload)
        outputs.append(_output_row(action, path, payload))

    result = {
        "schema_version": "rag-e5-e5b4-smoke-v1",
        "status": "PASS_PATH_ONLY_NOT_QUALITY",
        "protocol_lock_sha256": lock_sha,
        "server": server,
        "b3_unchanged_identity": b3_identity,
        "model_calls": 3,
        "outputs": outputs,
        "quality_scored": False,
        "retrieval_calls": 0,
        "bridge_calls": 0,
        "teacher_opened": False,
        "202608_opened": False,
        "continued_after_shared_slot_timeout": True,
        "continuation_driver_sha256": sha256_file(Path(__file__)),
    }
    _write_json(report_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_B4_PRIVATE_ROOT)
    args = parser.parse_args()
    result = continue_smoke(protocol_path=args.protocol, private_root=args.private_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
