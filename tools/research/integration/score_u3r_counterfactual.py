"""Open U3-R DEV evaluator truth only after validating the frozen execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.u3r_rag_scoring import score_u3r
from eval.u3r_rag_transfer import U2F_ROOT, U3R_RUN_ROOT, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u2f-root", type=Path, default=ROOT / U2F_ROOT)
    parser.add_argument("--run-root", type=Path, default=ROOT / U3R_RUN_ROOT)
    args = parser.parse_args()
    scoring_manifest_path = args.run_root / "u3r_scoring_manifest.json"
    if scoring_manifest_path.exists():
        scoring_manifest = json.loads(scoring_manifest_path.read_text(encoding="utf-8"))
        expected_outputs = {
            "u3r_scored_outcomes.jsonl",
            "u3r_fixed_action_report.json",
            "u3r_minimal_action_labels.json",
            "u3r_cost_frontier.json",
            "u3r_failure_attribution.json",
            "u3r_predictability_probe.json",
        }
        output_hashes = scoring_manifest.get("output_sha256")
        if (
            scoring_manifest.get("execution_freeze_sha256")
            != sha256_file(args.run_root / "u3r_counterfactual_manifest.json")
            or not isinstance(output_hashes, dict)
            or set(output_hashes) != expected_outputs
            or any(
                not (args.run_root / name).is_file()
                or output_hashes[name] != sha256_file(args.run_root / name)
                for name in expected_outputs
            )
        ):
            raise ValueError("existing post-freeze scoring outputs failed hash validation")
        fixed_path = args.run_root / "u3r_fixed_action_report.json"
        result = json.loads(fixed_path.read_text(encoding="utf-8"))
        print(json.dumps({
            "scoring_reused": True,
            "primary_slice_count": result["primary_slice_count"],
            "fixed_action_metrics": result["fixed_action_metrics"],
            "best_fixed_action": result["best_fixed_action"],
            "quality_oracle": result["quality_oracle"],
        }, sort_keys=True))
        return
    result = score_u3r(u2f_root=args.u2f_root, run_root=args.run_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
