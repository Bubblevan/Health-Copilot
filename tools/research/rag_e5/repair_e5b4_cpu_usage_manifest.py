"""Pre-score correction for the E5-B4 CPU manifest's usage aggregation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from eval.rag_e5.b4_execution import (DEFAULT_B4_PRIVATE_ROOT,
                                      DEFAULT_EXECUTION_MANIFEST_PATH,
                                      DEFAULT_PROTOCOL_PATH, _read_json,
                                      verify_frozen_code, verify_protocol_lock)
from eval.rag_e5.counterfactual import canonical_sha256
from eval.rag_e5.e5b3_recovery import sha256_file

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BACKUP_PATH = ROOT / "runs/rag_e5/e5b4_cpu_execution_manifest_pre_erratum.json"
DEFAULT_ERRATUM_PATH = ROOT / "runs/rag_e5/e5b4_cpu_manifest_erratum.json"
ERRATUM_ID = "E5B4-CPU-USAGE-AGGREGATION-01"


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def derive_usage_totals(
    arm_payloads: list[dict[str, Any]],
    ledger_rows: list[dict[str, Any]],
    *,
    expected_calls: int = 120,
) -> dict[str, int | bool]:
    """Derive usage from completed model-call arms, excluding zero-call arms."""
    call_arms = [row for row in arm_payloads if row.get("new_model_calls") == 1]
    if len(call_arms) != expected_calls or any(
        row.get("new_model_calls") not in {0, 1} for row in arm_payloads
    ):
        raise ValueError("arm set does not contain the expected one-call generation arms")

    started = [row for row in ledger_rows if row.get("status") == "STARTED"]
    completed = [row for row in ledger_rows if row.get("status") == "COMPLETED"]
    failures = [row for row in ledger_rows if row.get("status") == "FAILED"]
    start_by_id = {row.get("call_id"): row for row in started}
    done_by_id = {row.get("call_id"): row for row in completed}
    if (
        failures
        or len(started) != expected_calls
        or len(completed) != expected_calls
        or len(start_by_id) != expected_calls
        or set(start_by_id) != set(done_by_id)
    ):
        raise ValueError("call ledger is not a complete, unique, failure-free call set")

    arms_by_identity = {
        (row.get("case_id"), row.get("action"), row.get("run_id")): row
        for row in call_arms
    }
    if len(arms_by_identity) != expected_calls:
        raise ValueError("generation arm identities are duplicated")

    input_total = 0
    output_total = 0
    for call_id, start_row in start_by_id.items():
        done_row = done_by_id[call_id]
        identity_fields = ("case_id", "action", "run_id", "prompt_sha256", "call_ordinal")
        if any(start_row.get(field) != done_row.get(field) for field in identity_fields):
            raise ValueError("call ledger start/completion identity mismatch")
        identity = (done_row.get("case_id"), done_row.get("action"), done_row.get("run_id"))
        arm = arms_by_identity.get(identity)
        if arm is None:
            raise ValueError("completed call has no matching frozen generation arm")
        response = arm.get("guidance_raw_response")
        if not isinstance(response, dict):
            raise ValueError("generation arm has no raw completion record")
        input_tokens = response.get("input_tokens")
        output_tokens = response.get("output_tokens")
        if not _is_nonnegative_int(input_tokens) or not _is_nonnegative_int(output_tokens):
            raise ValueError("a completed generation arm is missing token usage")
        if (
            arm.get("guidance_input_tokens") != input_tokens
            or arm.get("guidance_output_tokens") != output_tokens
            or done_row.get("input_tokens") != input_tokens
            or done_row.get("output_tokens") != output_tokens
        ):
            raise ValueError("arm and append-only ledger token usage disagree")
        input_total += input_tokens
        output_total += output_tokens

    return {
        "model_calls": expected_calls,
        "input_tokens": input_total,
        "output_tokens": output_total,
        "token_usage_complete": True,
    }


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def repair_manifest(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    manifest_path: Path = DEFAULT_EXECUTION_MANIFEST_PATH,
    private_root: Path = DEFAULT_B4_PRIVATE_ROOT,
    backup_path: Path = DEFAULT_BACKUP_PATH,
    erratum_path: Path = DEFAULT_ERRATUM_PATH,
) -> dict[str, Any]:
    lock = _read_json(protocol_path)
    lock_sha = verify_protocol_lock(lock)
    verify_frozen_code(lock)
    expected_manifest_rel = lock["execution_manifest_path"]
    if manifest_path.resolve() != (ROOT / expected_manifest_rel).resolve():
        raise ValueError("usage correction only accepts the frozen canonical CPU manifest path")
    try:
        manifest_path.resolve().relative_to(ROOT.resolve())
        backup_path.resolve().relative_to(ROOT.resolve())
        erratum_path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("manifest and erratum artifacts must be inside the repository") from exc
    if not manifest_path.is_file():
        raise FileNotFoundError("completed CPU execution manifest is required")
    if erratum_path.exists():
        raise FileExistsError("E5-B4 usage erratum already exists; refusing to rewrite it")

    original_bytes = manifest_path.read_bytes()
    original_file_sha = hashlib.sha256(original_bytes).hexdigest()
    manifest = json.loads(original_bytes)
    original_body = {
        key: value for key, value in manifest.items() if key != "execution_manifest_sha256"
    }
    if canonical_sha256(original_body) != manifest.get("execution_manifest_sha256"):
        raise ValueError("original execution manifest self-hash mismatch")
    if (
        manifest.get("status") != "ALL_B4_ARMS_FROZEN_BEFORE_SCORING"
        or manifest.get("protocol_lock_sha256") != lock_sha
        or manifest.get("execution_id") != lock.get("execution_id")
        or manifest.get("arms_expected") != 180
        or manifest.get("arms_completed") != 180
        or manifest.get("guidance_calls_expected") != 120
        or manifest.get("guidance_calls_completed") != 120
        or manifest.get("empty_outputs") != 0
        or manifest.get("finish_reason_length") != 0
        or manifest.get("transport_failures") != 0
        or manifest.get("teacher_opened") is not False
        or manifest.get("202608_opened") is not False
        or manifest.get("token_usage_complete") is not False
        or manifest.get("total_input_tokens") is not None
        or manifest.get("total_output_tokens") is not None
    ):
        raise ValueError("manifest is not the exact complete run with the known usage-aggregation defect")
    if manifest.get("external_arm_root") != str(private_root):
        raise ValueError("manifest private arm root differs from the requested CPU run")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or len(entries) != 180:
        raise ValueError("manifest does not bind exactly 180 arms")
    if canonical_sha256(entries) != manifest.get("artifact_set_sha256"):
        raise ValueError("manifest artifact-set hash mismatch")

    arm_payloads: list[dict[str, Any]] = []
    for entry in entries:
        arm_path = (private_root / entry["relative_path"]).resolve()
        try:
            arm_path.relative_to(private_root.resolve())
        except ValueError as exc:
            raise ValueError("manifest arm path escapes the private CPU run root") from exc
        if sha256_file(arm_path) != entry.get("file_sha256"):
            raise ValueError("a frozen arm file changed after execution")
        payload = _read_json(arm_path)
        completion_sha = payload.pop("completion_sha256", None)
        if (
            completion_sha != entry.get("completion_sha256")
            or canonical_sha256(payload) != completion_sha
            or payload.get("case_id") != entry.get("case_id")
            or payload.get("action") != entry.get("action")
            or payload.get("run_id") != entry.get("run_id")
        ):
            raise ValueError("a frozen arm payload identity/hash mismatch")
        payload["completion_sha256"] = completion_sha
        arm_payloads.append(payload)

    ledger_path = private_root / "call_ledger.jsonl"
    if sha256_file(ledger_path) != manifest.get("call_ledger_sha256"):
        raise ValueError("append-only generation ledger changed after the run")
    ledger_rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    totals = derive_usage_totals(arm_payloads, ledger_rows)
    if backup_path.exists():
        raise FileExistsError("pre-erratum manifest backup already exists")

    corrected = deepcopy(manifest)
    corrected["total_input_tokens"] = totals["input_tokens"]
    corrected["total_output_tokens"] = totals["output_tokens"]
    corrected["token_usage_complete"] = True
    corrected["usage_aggregation_erratum"] = {
        "erratum_id": ERRATUM_ID,
        "record_path": erratum_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "pre_erratum_manifest_file_sha256": original_file_sha,
        "source": "120 immutable generation-arm responses cross-checked against their 120 append-only COMPLETED ledger rows",
        "scope": ["total_input_tokens", "total_output_tokens", "token_usage_complete"],
        "does_not_change": ["prompts", "model responses", "arms", "actions", "state claims", "citations", "retrievals", "scores"],
    }
    corrected.pop("execution_manifest_sha256", None)
    corrected["execution_manifest_sha256"] = canonical_sha256(corrected)

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with backup_path.open("xb") as stream:
        stream.write(original_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError("unexpected partial manifest file exists")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(corrected, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_path)

    erratum = {
        "schema_version": "rag-e5-e5b4-cpu-manifest-erratum-v1",
        "erratum_id": ERRATUM_ID,
        "status": "PRE_SCORE_AGGREGATE_CORRECTION_ONLY",
        "execution_id": lock["execution_id"],
        "protocol_lock_sha256": lock_sha,
        "original_manifest_path": backup_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "original_manifest_file_sha256": original_file_sha,
        "original_execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "corrected_manifest_path": manifest_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "corrected_manifest_file_sha256": sha256_file(manifest_path),
        "corrected_execution_manifest_sha256": corrected["execution_manifest_sha256"],
        "arms_cross_checked": len(arm_payloads),
        "generation_calls_cross_checked": totals["model_calls"],
        "ledger_rows": len(ledger_rows),
        "ledger_started": sum(row.get("status") == "STARTED" for row in ledger_rows),
        "ledger_completed": sum(row.get("status") == "COMPLETED" for row in ledger_rows),
        "ledger_failed": sum(row.get("status") == "FAILED" for row in ledger_rows),
        "call_ledger_sha256": manifest["call_ledger_sha256"],
        "artifact_set_sha256": manifest["artifact_set_sha256"],
        "old_values": {
            "token_usage_complete": manifest["token_usage_complete"],
            "total_input_tokens": manifest["total_input_tokens"],
            "total_output_tokens": manifest["total_output_tokens"],
        },
        "new_values": totals,
        "cause": "The original accumulator included no-model-call arms (whose usage fields are zero) in its list-length check, then compared the 180-arm list length to 120 generation calls. The 120 model-call arms themselves all have usage values cross-checked between arm payloads and the append-only ledger.",
        "teacher_opened_before_correction": False,
        "scoring_run_before_correction": False,
        "202608_opened": False,
        "qrels_or_teacher_data_read_by_correction": False,
    }
    _write_json_exclusive(erratum_path, erratum)
    return erratum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_EXECUTION_MANIFEST_PATH)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_B4_PRIVATE_ROOT)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP_PATH)
    parser.add_argument("--erratum", type=Path, default=DEFAULT_ERRATUM_PATH)
    args = parser.parse_args()
    result = repair_manifest(
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        private_root=args.private_root,
        backup_path=args.backup,
        erratum_path=args.erratum,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
