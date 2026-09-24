"""Create the aggregate-only E1.2 DEV selection lock before TEST."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.e1_2_analysis import select_dev_retrieval_cost_reference
from eval.e1_2_runner import file_sha256, read_jsonl
from tools.run_e1_2_mirage import load_frozen_config

DEFAULT_SCRATCH = Path(r"E:\Health-Copilot-E1.2")
LOCK_PATH = ROOT / "runs/e1_2/dev_selection_lock.json"


def lock_dev(*, scratch_root: Path) -> dict[str, Any]:
    config, config_hash = load_frozen_config()
    split_path = ROOT / config["benchmark"]["split_manifest"]
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    expected_cases = sum(
        int(split_manifest["counts"][subset]["DEV"])
        for subset in config["benchmark"]["clean_test_subdatasets"]
    )
    selection_path = ROOT / "runs/e1_2/generator_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("config_sha256") != config_hash:
        raise ValueError("generator selection artifact does not match the frozen config")
    selected_candidate = str(selection["selected_candidate"])

    dev_rows: dict[str, list[dict[str, Any]]] = {}
    arm_identities: dict[str, str] = {}
    for arm in ("rag_bm25", "rag_medcpt"):
        arm_dir = scratch_root / "runs/e1_2/dev" / arm
        manifest_path = arm_dir / "manifest.json"
        rows_path = arm_dir / "case_results.jsonl"
        if not manifest_path.is_file() or not rows_path.is_file():
            raise FileNotFoundError(f"missing completed DEV arm artifact: {arm}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "COMPLETED"
            or manifest.get("partition") != "DEV"
            or manifest.get("arm") != arm
            or manifest.get("config_sha256") != config_hash
            or manifest.get("selected_candidate") != selected_candidate
            or int(manifest.get("case_count", -1)) != expected_cases
        ):
            raise ValueError(f"DEV arm manifest is incomplete or incompatible: {arm}")
        rows = read_jsonl(rows_path)
        identity = str(manifest.get("result_identity") or "")
        if (
            len(rows) != expected_cases
            or len({str(row.get("case_id")) for row in rows}) != expected_cases
            or any(
                row.get("result_identity") != identity
                or row.get("partition") != "DEV"
                or row.get("arm") != arm
                or row.get("subdataset") not in config["benchmark"]["clean_test_subdatasets"]
                for row in rows
            )
        ):
            raise ValueError(f"DEV arm case rows are incomplete or incompatible: {arm}")
        dev_rows[arm] = rows
        arm_identities[arm] = identity

    selected = select_dev_retrieval_cost_reference(dev_rows)
    lock = {
        "schema_version": "e1-2-dev-selection-lock-v1",
        "status": "LOCKED_BEFORE_TEST",
        "config_sha256": config_hash,
        "code_base_commit": config["code_base_commit"],
        "normalized_cases_sha256": config["benchmark"]["normalized_cases_sha256"],
        "split_manifest_sha256": config["benchmark"]["split_manifest_sha256"],
        "generator_candidate": selected_candidate,
        "generator_selection_sha256": file_sha256(selection_path),
        "dev_arm_result_identities": arm_identities,
        **selected,
        "test_opened": False,
        "locked_at_utc": datetime.now(UTC).isoformat(),
    }
    if LOCK_PATH.exists():
        existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        comparable_existing = {key: value for key, value in existing.items() if key != "locked_at_utc"}
        comparable_current = {key: value for key, value in lock.items() if key != "locked_at_utc"}
        if comparable_existing == comparable_current:
            return existing
        raise FileExistsError(f"refusing to replace the existing DEV selection lock: {LOCK_PATH}")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = LOCK_PATH.with_name(LOCK_PATH.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary lock artifact already exists: {temporary}")
    temporary.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(LOCK_PATH)
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    args = parser.parse_args()
    result = lock_dev(scratch_root=args.scratch_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_retrieval_cost_reference": result["selected_retrieval_cost_reference"],
                "dev_case_count": result["dev_case_count"],
                "lock": str(LOCK_PATH),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
