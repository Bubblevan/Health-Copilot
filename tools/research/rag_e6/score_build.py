"""Score the frozen E6A BUILD run; only BUILD evaluator truth is decoded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.rag_e6.scoring import (
    BUILD_ROOT,
    CORPUS_ROOT,
    REPORT_JSON,
    SCORED_ROWS,
    SPLIT_MANIFEST,
    U2F_ROOT,
    score_build,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u2f-root", type=Path, default=U2F_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--build-root", type=Path, default=BUILD_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT_JSON)
    parser.add_argument("--scored-rows", type=Path, default=SCORED_ROWS)
    args = parser.parse_args()
    report = score_build(
        u2f_root=args.u2f_root,
        split_manifest_path=args.split_manifest,
        corpus_root=args.corpus_root,
        build_root=args.build_root,
        report_path=args.report,
        scored_rows_path=args.scored_rows,
    )
    print(json.dumps({
        "partition": report["partition"],
        "episode_count": report["episode_count"],
        "capability_class_counts": report["capability_class_counts"],
        "primary_comparison": report["primary_comparison"],
        "development_gate": report["development_gate"],
        "evaluator_truth_opened_after_execution_freeze": report[
            "evaluator_truth_opened_after_execution_freeze"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
