"""Summarize the committed R2MED final public TEST evaluation without reopening qrels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.r2med_final_test import CANDIDATE_PATH, REPORT_PATH


def analyze_final_report(report_path: Path, candidate_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    summaries = report["metrics"]["arm_summaries"]
    table = {
        arm: {
            subset: values["ndcg@10"]
            for subset, values in summary["by_subset"].items()
        }
        | {"macro": summary["macro_equal_subset_weight"]["ndcg@10"]}
        for arm, summary in summaries.items()
    }
    return {
        "test_status": report["test_status"],
        "test_config_drift": report["test_config_drift"],
        "ndcg_at_10": table,
        "paired_bootstrap": report["paired_bootstrap"]["comparisons"],
        "candidate_complementarity": candidate["macro_equal_subset_weight"],
        "gates": report["gates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / REPORT_PATH)
    parser.add_argument("--candidate-analysis", type=Path, default=ROOT / CANDIDATE_PATH)
    args = parser.parse_args()
    print(json.dumps(analyze_final_report(args.report, args.candidate_analysis), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
