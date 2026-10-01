"""Score FROZEN_DEV only after all execution artifacts pass frozen checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.rag_e6.scoring import score_frozen_dev


def main() -> None:
    report = score_frozen_dev()
    print(json.dumps({
        "partition": report["partition"],
        "episode_count": report["episode_count"],
        "capability_class_counts": report["capability_class_counts"],
        "primary_comparison": report["primary_comparison"],
        "frozen_dev_gate": report["frozen_dev_gate"],
        "evaluator_truth_opened_after_execution_freeze": report[
            "evaluator_truth_opened_after_execution_freeze"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
