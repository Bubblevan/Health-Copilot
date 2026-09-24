"""Write the aggregate-only held-out QA analysis after the DEV lock is committed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.e1_2_analysis import analyze_arms, load_completed_test_arms
from eval.e1_2_protocol import assert_committed_artifact
from tools.run_e1_2_mirage import DEV_SELECTION_LOCK_PATH, load_frozen_config

DEFAULT_SCRATCH = Path(r"E:\Health-Copilot-E1.2")
DEFAULT_OUTPUT = ROOT / "runs/e1_2/test_analysis.json"


def analyze_test(*, scratch_root: Path, output_path: Path) -> dict[str, Any]:
    config, config_hash = load_frozen_config()
    if not DEV_SELECTION_LOCK_PATH.is_file():
        raise FileNotFoundError("QA TEST analysis requires a committed DEV selection lock")
    assert_committed_artifact(DEV_SELECTION_LOCK_PATH, repo_root=ROOT)
    lock = json.loads(DEV_SELECTION_LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_TEST" or lock.get("config_sha256") != config_hash:
        raise ValueError("QA DEV lock does not match the frozen test identity")
    expected = int(config["benchmark"]["clean_test_case_count"])
    arms = load_completed_test_arms(scratch_root, config_hash, expected)
    report = analyze_arms(
        arms,
        dev_selected_retrieval_cost_reference=str(lock["selected_retrieval_cost_reference"]),
    )
    if report["case_count"] != expected:
        raise ValueError(f"QA TEST arm case count differs from the frozen count: {report['case_count']} != {expected}")
    report.update(
        {
            "config_sha256": config_hash,
            "dev_selection_lock_sha256": hashlib.sha256(DEV_SELECTION_LOCK_PATH.read_bytes()).hexdigest(),
            "dev_selected_retrieval_cost_reference": lock["selected_retrieval_cost_reference"],
            "test_records_emitted": False,
            "data_boundary": "Aggregate metrics only; no question text, options, gold labels, or per-case predictions are included.",
        }
    )
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing == report:
            return existing
        raise FileExistsError(f"refusing to replace the existing QA TEST analysis: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary analysis artifact already exists: {temporary}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze_test(scratch_root=args.scratch_root, output_path=args.output)
    print(
        json.dumps(
            {
                "status": "COMPLETED",
                "test_case_count": report["case_count"],
                "headline_eligibility": report["headline_eligibility"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
