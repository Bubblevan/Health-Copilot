"""Score deterministic event-slot projection against saved v1r3 responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tools.research.memory import run_mem3b0q_r4_cardinality_devset_v2 as scorer
from tools.research.memory import run_mem3b0q_r4_candidate_local_smoke_v1 as smoke
from tools.research.memory import run_mem3b0q_r4_event_value_confirmation_v1 as engine
from tools.research.memory.mem3b0q_r4_event_value_projection_v1 import (
    POLICY_VERSION,
    project_event_value_spans,
)


ROOT = engine.ROOT
SOURCE_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-event-value-confirmation-v1r3"
OUTPUT_ROOT = ROOT / "runs" / "memory" / "mem3" / "mem3b0q-r4-event-projection-offline-v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_source_manifest() -> tuple[dict[str, Any], str]:
    manifest_path = SOURCE_ROOT / "run_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    for relative, expected_sha in manifest["artifact_sha256"].items():
        if _sha((SOURCE_ROOT / relative).read_bytes()) != expected_sha:
            raise RuntimeError(f"source_artifact_hash_mismatch:{relative}")
    return manifest, _sha(manifest_bytes)


def run_offline() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise RuntimeError("offline_projection_output_exists")
    source_manifest, source_manifest_sha = _verify_source_manifest()
    pack, requests, v1_requests = engine._build_inputs()
    case_rows = []
    for case in pack["cases"]:
        case_id = case["case_id"]
        case_dir = SOURCE_ROOT / case_id
        request_bytes = (case_dir / "request.json").read_bytes()
        response_bytes = (case_dir / "response_body.bin").read_bytes()
        prior = json.loads((case_dir / "case_result.json").read_text(encoding="utf-8"))
        if request_bytes != requests[case_id]["body"]:
            raise RuntimeError(f"source_request_bytes_mismatch:{case_id}")
        if prior.get("request_sha256") != _sha(request_bytes):
            raise RuntimeError(f"source_request_hash_mismatch:{case_id}")
        if prior.get("response_sha256") != _sha(response_bytes):
            raise RuntimeError(f"source_response_hash_mismatch:{case_id}")

        envelope, content = smoke._response_content(response_bytes)
        proposition = {"source_id": case_id, "proposition_text": case["proposition_text"]}
        projected, projection_audit = project_event_value_spans(content, proposition)
        scored = scorer._score_response(
            case,
            projected,
            v2_request=requests[case_id]["request"],
            v1_request=v1_requests[case_id],
            response_sha256=_sha(response_bytes),
            usage=envelope.get("usage"),
            latency_ms=prior.get("latency_ms"),
            finish_reason=envelope.get("choices", [{}])[0].get("finish_reason"),
        )
        case_rows.append(
            {
                "case_id": case_id,
                "source_status": prior["status"],
                "source_response_sha256": _sha(response_bytes),
                "raw_model_value_span": projection_audit[0]["prior_model_value_span"],
                "raw_model_owner_candidate_id": projection_audit[0]["prior_model_owner_candidate_id"],
                "projection": projection_audit,
                "projected_status": scored["status"],
                "projected_failure": scored.get("failure"),
                "matched_expected_atoms": scored.get("matched_expected_atoms", 0),
                "admitted_atoms": scored.get("admitted_atoms", 0),
                "quarantined_atoms": scored.get("quarantined_atoms", []),
            }
        )
    result = {
        "schema_version": 1,
        "run_id": OUTPUT_ROOT.name,
        "status": "COMPLETE",
        "evaluation_scope": "OFFLINE_COUNTERFACTUAL_ABLATION_ON_FROZEN_SYNTHETIC_RESPONSES",
        "source_run_id": source_manifest["manifest_id"],
        "source_run_manifest_sha256": source_manifest_sha,
        "dataset_sha256": source_manifest["dataset_sha256"],
        "projection_policy": POLICY_VERSION,
        "new_model_calls": 0,
        "hosted_calls": 0,
        "cases": case_rows,
        "aggregate": {
            "cases": len(case_rows),
            "raw_exact_matches": sum(row["source_status"] == "QUALITY_PASS" for row in case_rows),
            "projected_exact_matches": sum(row["projected_status"] == "QUALITY_PASS" for row in case_rows),
            "projected_admitted_atoms": sum(row["admitted_atoms"] for row in case_rows),
            "expected_atoms": len(case_rows),
        },
        "limitations": [
            "This is a deterministic offline counterfactual, not a fresh independent confirmation.",
            "The cue registry is intentionally narrow and currently covers one synthetic bicycle-purchase slot.",
            "The projection supplies owner occurrence and event value from frozen lexical policies; resulting passes are harness-assisted admission, not raw model extraction accuracy.",
            "No public benchmark, memory materialization, revision resolution, or clinical content was evaluated.",
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    result_bytes = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    _write_exclusive(OUTPUT_ROOT / "projection_result.json", result_bytes)
    manifest = {
        "manifest_id": OUTPUT_ROOT.name,
        "status": result["status"],
        "source_run_manifest_sha256": source_manifest_sha,
        "projection_policy": POLICY_VERSION,
        "artifact_sha256": {"projection_result.json": _sha(result_bytes)},
    }
    _write_exclusive(
        OUTPUT_ROOT / "run_manifest.json",
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "aggregate": result["aggregate"],
                "cases": [
                    {
                        "case_id": row["case_id"],
                        "raw_model_owner_candidate_id": row["raw_model_owner_candidate_id"],
                        "projected_owner_candidate_id": row["projection"][0]["projected_owner_candidate_id"],
                        "raw_model_value_span": row["raw_model_value_span"],
                        "projected_value_span": row["projection"][0]["projected_value_span"],
                        "projected_status": row["projected_status"],
                        "failure": row["projected_failure"],
                    }
                    for row in case_rows
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--score-offline", action="store_true")
    args = parser.parse_args()
    if not args.score_offline:
        parser.error("explicit --score-offline is required")
    run_offline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
