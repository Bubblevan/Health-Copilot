"""Resume E5-B2 after the nullable bridge-metadata logging defect.

This adapter changes no prompts, model settings, retrieval, scorer, or run IDs. It
only treats a null bridge field as empty metadata in memory; the immutable arm
JSON remains unchanged. The adapter and the first already completed arm are
identified in the execution manifest and an external erratum sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b2")
LOCK_PATH = ROOT / "runs/rag_e5/e5b2_protocol_lock.json"
ERRATUM_ID = "E5B2-OPS-001"


class _NullableBridgePayload(dict[str, Any]):
    """Use an empty mapping for progress logging while preserving serialized null."""

    def get(self, key: str, default: Any = None) -> Any:
        if key == "bridge" and dict.get(self, key) is None:
            return {}
        return super().get(key, default)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_arm_bridge(arm: dict[str, Any]) -> dict[str, Any]:
    """Normalize nullable non-STRONG metadata after the on-disk hash is verified."""
    normalized = dict(arm)
    if normalized.get("bridge") is None:
        normalized["bridge"] = {}
    return normalized


def _erratum_record() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "operational_erratum.json"
    checksum_path = ARTIFACT_ROOT / "operational_erratum.sha256"
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock_sha = lock["counterfactual_lock_sha256"]
    adapter_sha = _sha256_file(Path(__file__).resolve())
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        digest = _sha256_file(path)
        if (
            record.get("erratum_id") != ERRATUM_ID
            or record.get("protocol_lock_sha256") != lock_sha
            or record.get("recovery_adapter_sha256") != adapter_sha
            or not checksum_path.is_file()
            or checksum_path.read_text(encoding="ascii").strip() != digest
        ):
            raise ValueError("existing operational erratum does not match this recovery adapter")
        record["operational_erratum_sha256"] = digest
        return record

    arms_root = ARTIFACT_ROOT / "arms"
    completed_arms = sorted(arms_root.rglob("*.json")) if arms_root.exists() else []
    ledger = ARTIFACT_ROOT / "call_ledger.jsonl"
    if len(completed_arms) != 1 or not ledger.is_file():
        raise ValueError("OPS-001 is valid only for the single completed arm from the first attempt")
    ledger_rows = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(ledger_rows) != 2 or [row.get("status") for row in ledger_rows] != ["STARTED", "COMPLETED"]:
        raise ValueError("the pre-recovery call ledger is not exactly one completed inference")
    if any(row.get("phase") != "reader" or row.get("action") != "OFF" for row in ledger_rows):
        raise ValueError("the sole pre-recovery call is not the expected first OFF reader arm")
    recovery_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    record = {
        "schema_version": "rag-e5-e5b2-operational-erratum-v1",
        "erratum_id": ERRATUM_ID,
        "reason": "progress logging dereferenced nullable bridge metadata after a completed arm was atomically written",
        "protocol_lock_sha256": lock_sha,
        "locked_runner_source_unchanged": True,
        "pre_recovery_completed_arms": 1,
        "pre_recovery_model_calls": 1,
        "pre_recovery_call_ledger_sha256": _sha256_file(ledger),
        "pre_recovery_arm_sha256": _sha256_file(completed_arms[0]),
        "recovery_commit": recovery_commit,
        "recovery_adapter_path": "tools/research/rag_e5/run_e5b2_recovery.py",
        "recovery_adapter_sha256": adapter_sha,
        "scope": "in-memory nullable-metadata compatibility only; no inference is replayed and no prompt, retrieval, model, scorer, or run ID changes",
        "teacher_or_outcome_scoring_opened": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    digest = _sha256_file(path)
    checksum_path.write_text(digest + "\n", encoding="ascii")
    record["operational_erratum_sha256"] = digest
    return record


def _install_runner_adapter(record: dict[str, Any]) -> None:
    from tools.research.rag_e5 import run_e5b2_counterfactual as runner

    original_execute = runner._execute_arm

    def execute_arm(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], int]:
        payload, ordinal = original_execute(*args, **kwargs)
        return _NullableBridgePayload(payload), ordinal

    original_verify = runner.verify_complete_arm

    def verify_arm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return normalize_arm_bridge(original_verify(*args, **kwargs))

    original_write_json = runner._write_json

    def write_json(path: Path, value: dict[str, Any]) -> None:
        if path == runner.EXECUTION_MANIFEST_PATH:
            value["operational_compatibility_erratum"] = record
            value.pop("execution_manifest_sha256", None)
            value["execution_manifest_sha256"] = runner.canonical_sha256(value)
        original_write_json(path, value)

    runner._execute_arm = execute_arm
    runner.verify_complete_arm = verify_arm
    runner._write_json = write_json


def _install_evaluator_adapter() -> None:
    from eval.rag_e5 import e5b2_evaluator as evaluator

    original_verify = evaluator.verify_complete_arm

    def verify_arm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return normalize_arm_bridge(original_verify(*args, **kwargs))

    evaluator.verify_complete_arm = verify_arm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", action="store_true", help="score a fully frozen run using the same null-metadata adapter")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    record = _erratum_record()
    if args.score:
        _install_evaluator_adapter()
        from tools.research.rag_e5.score_e5b2_counterfactual import main as score_main

        score_main()
        return
    _install_runner_adapter(record)
    from tools.research.rag_e5.run_e5b2_counterfactual import main as run_main

    run_main()


if __name__ == "__main__":
    main()
