"""Freeze the selected DEV method only after the predeclared success gate passes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.r2med_crb_data import sha256_file

PROTOCOL_PATH = ROOT / "runs/rag_r2med_rerank/protocol.json"
REPORT_PATH = ROOT / "runs/rag_r2med_rerank/dev_report.json"
LOCK_PATH = ROOT / "runs/rag_r2med_rerank/final_method_lock.json"
LOCKED_CODE_PATHS = (
    "eval/r2med_candidate_union.py",
    "eval/r2med_reranker.py",
    "eval/r2med_crb_data.py",
    "eval/r2med_crb_evaluator.py",
    "eval/r2med_multiview.py",
    "tools/analyze_r2med_candidate_complementarity.py",
    "tools/run_r2med_reranker_dev.py",
    "tools/analyze_r2med_reranker_dev.py",
    "tools/freeze_r2med_reranker.py",
    "runs/rag_r2med_rerank/protocol.json",
)


def freeze_method(
    *, report_path: Path = REPORT_PATH, protocol_path: Path = PROTOCOL_PATH, lock_path: Path = LOCK_PATH
) -> dict[str, Any]:
    if lock_path.exists():
        raise FileExistsError(f"final method lock already exists: {lock_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    selection = report.get("selection", {})
    if selection.get("dev_signal") != "POSITIVE" or not selection.get("test_allowed"):
        raise ValueError("DEV success gate is negative; TEST freeze is prohibited")
    if report.get("test_accessed") is not False or report.get("partition") != "DEV":
        raise ValueError("method lock requires a DEV-only report")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if subprocess.run(
        ["git", "diff", "--quiet", report["code_commit"], head, "--", *LOCKED_CODE_PATHS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).returncode:
        raise ValueError("DEV method source changed after evaluation; refusing to freeze mismatched method")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *LOCKED_CODE_PATHS],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("reranking implementation has uncommitted changes after DEV evaluation")
    lock = {
        "schema_version": "r2med-reranker-final-method-lock-v1",
        "status": "FROZEN_BEFORE_TEST",
        "partition_selected_on": "DEV",
        "method": selection["best_ours"],
        "config": selection["best_ours_config"],
        "primary_metric": "equal-subset macro nDCG@10",
        "strongest_non_ours": selection["strongest_non_ours"],
        "dev_delta_vs_strongest_non_ours": selection["delta_vs_strongest_non_ours"],
        "protocol_sha256": sha256_file(protocol_path),
        "dev_report_sha256": sha256_file(report_path),
        "candidate_analysis_sha256": report["candidate_analysis_sha256"],
        "reranker": report["reranker"],
        "source_ranking_hashes": protocol["candidate_sources"],
        "baseline_ranking_hashes": protocol["baseline_rankings"],
        "reranked_ranking_hashes": report["reranked_ranking_sha256"],
        "per_query_metrics_sha256": report["per_query_metrics"]["sha256"],
        "code_commit": head,
        "dev_execution_code_commit": report["code_commit"],
        "test_policy": protocol["test_policy"],
        "test_run_started": False,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    args = parser.parse_args()
    lock = freeze_method(report_path=args.report, protocol_path=args.protocol, lock_path=args.lock)
    print(json.dumps({"method": lock["method"], "config": lock["config"], "test_run_started": False}, indent=2))


if __name__ == "__main__":
    main()
