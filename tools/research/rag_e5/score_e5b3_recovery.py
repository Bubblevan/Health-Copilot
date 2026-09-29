"""Score E5-B3 only after its complete execution manifest is frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.rag_e5.e5b3_evaluator import (
    DEFAULT_B2_CALL_LEDGER,
    DEFAULT_B2_REPORT,
    DEFAULT_LOCK,
    DEFAULT_MANIFEST,
    DEFAULT_PRIVATE_LEDGER,
    DEFAULT_PRIVATE_ROOT,
    DEFAULT_REPORT_JSON,
    DEFAULT_REPORT_MD,
    score_recovery,
)
from eval.rag_e5.e5b3_recovery import DEFAULT_B2_ARTIFACT_ROOT, DEFAULT_CORPUS_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--b2-artifact-root", type=Path, default=DEFAULT_B2_ARTIFACT_ROOT)
    parser.add_argument("--b2-call-ledger", type=Path, default=DEFAULT_B2_CALL_LEDGER)
    parser.add_argument("--b2-report", type=Path, default=DEFAULT_B2_REPORT)
    parser.add_argument("--private-ledger", type=Path, default=DEFAULT_PRIVATE_LEDGER)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-markdown", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    report = score_recovery(
        lock_path=args.lock,
        manifest_path=args.manifest,
        private_root=args.private_root,
        corpus_root=args.corpus_root,
        b2_artifact_root=args.b2_artifact_root,
        b2_call_ledger=args.b2_call_ledger,
        b2_report_path=args.b2_report,
        private_score_path=args.private_ledger,
        report_json_path=args.report_json,
        report_markdown_path=args.report_markdown,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "measurement_recovery_gate": report["MEASUREMENT_RECOVERY_GATE"],
                "reader_json_valid_rate": report["reader_measurement_comparison"]["B3"][
                    "reader_json_valid_rate"
                ],
                "finish_reason_length": report["reader_measurement_comparison"]["B3"][
                    "finish_reason_length"
                ],
                "teacher_opened": report["teacher_opened"],
                "report_sha256": report["report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
