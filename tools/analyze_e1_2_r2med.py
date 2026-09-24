"""Lock the R2MED DEV comparator and score the untouched TEST once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.e1_2_protocol import assert_committed_artifact
from eval.e1_2_r2med_metrics import (
    FIXED_ARMS,
    METRICS,
    paired_macro_bootstrap,
    select_best_fixed_baseline,
    summarize_metrics,
)
from tools.run_e1_2_r2med_retrieval import (
    ARMS,
    DEV_LOCK_PATH,
    arm_paths,
    dataset_entries,
    load_protocol,
    load_rank_rows,
    read_jsonl,
    write_json_atomic,
)


def collect_arm(
    partition: str,
    arm: str,
    *,
    source_manifest: dict[str, Any],
    config_hash: str,
    source_hash: str,
    scratch_root: Path,
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for entry in dataset_entries(source_manifest, partition):
        result_path, manifest_path = arm_paths(scratch_root, partition, entry, arm)
        if not result_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"missing {partition} output for {entry['name']} / {arm}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "COMPLETED"
            or manifest.get("config_sha256") != config_hash
            or manifest.get("source_manifest_sha256") != source_hash
            or manifest.get("subset") != entry["name"]
            or manifest.get("arm") != arm
        ):
            raise ValueError(f"incomplete or incompatible {partition} run for {entry['name']} / {arm}")
        rows = read_jsonl(result_path)
        checked = load_rank_rows(result_path, str(manifest["result_identity"]))
        if len(rows) != entry["query_count"] or len(checked) != entry["query_count"]:
            raise ValueError(f"wrong number of query results for {entry['name']} / {arm}")
        all_rows.extend(rows)
    if len({str(row["query_id"]) for row in all_rows}) != len(all_rows):
        raise ValueError(f"duplicate query IDs across {partition} subsets for {arm}")
    return all_rows


def _summaries(rows_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {arm: summarize_metrics(rows) for arm, rows in rows_by_arm.items()}


def lock_dev(*, scratch_root: Path) -> dict[str, Any]:
    _, source_manifest, config_hash, source_hash = load_protocol()
    dev_rows = {arm: collect_arm("DEV", arm, source_manifest=source_manifest, config_hash=config_hash, source_hash=source_hash, scratch_root=scratch_root) for arm in ARMS}
    summaries = _summaries(dev_rows)
    selected = select_best_fixed_baseline({arm: summaries[arm] for arm in FIXED_ARMS})
    method_rows = {str(row["query_id"]): row for row in dev_rows["rrf_medcpt_rerank"]}
    baseline_rows = {str(row["query_id"]): row for row in dev_rows[selected]}
    subset_by_query = {str(row["query_id"]): str(row["subset"]) for row in dev_rows[selected]}
    dev_difference = paired_macro_bootstrap(
        method_rows, baseline_rows, subset_by_query, metric="ndcg@10", resamples=10_000
    )
    lock = {
        "schema_version": "e1-2-r2med-dev-lock-v1",
        "status": "LOCKED_BEFORE_TEST",
        "config_sha256": config_hash,
        "source_manifest_sha256": source_hash,
        "dev_query_count": sum(entry["query_count"] for entry in source_manifest["dev_subsets"]),
        "dev_metrics": summaries,
        "selected_fixed_baseline": selected,
        "selection_rule": "highest equal-weight macro DEV nDCG@10 among bm25, bge_dense, and medcpt_dense; ties favor the fixed order in the protocol.",
        "custom_method_dev_difference_vs_selected_baseline": dev_difference,
        "custom_method_dev_complete": True,
        "test_opened": False,
        "test_selection_uses_dev_only": True,
    }
    if DEV_LOCK_PATH.exists():
        existing = json.loads(DEV_LOCK_PATH.read_text(encoding="utf-8"))
        if existing == lock:
            return existing
        raise FileExistsError(f"refusing to replace the existing R2MED DEV lock: {DEV_LOCK_PATH}")
    write_json_atomic(DEV_LOCK_PATH, lock)
    return lock


def analyze_test(*, scratch_root: Path, output_path: Path) -> dict[str, Any]:
    _, source_manifest, config_hash, source_hash = load_protocol()
    if not DEV_LOCK_PATH.is_file():
        raise FileNotFoundError("R2MED TEST requires a committed DEV selection lock")
    lock = json.loads(DEV_LOCK_PATH.read_text(encoding="utf-8"))
    assert_committed_artifact(DEV_LOCK_PATH, repo_root=ROOT)
    if (
        lock.get("status") != "LOCKED_BEFORE_TEST"
        or lock.get("config_sha256") != config_hash
        or lock.get("source_manifest_sha256") != source_hash
    ):
        raise ValueError("R2MED DEV lock does not match the frozen test identity")
    test_rows = {arm: collect_arm("TEST", arm, source_manifest=source_manifest, config_hash=config_hash, source_hash=source_hash, scratch_root=scratch_root) for arm in ARMS}
    summaries = _summaries(test_rows)
    baseline_arm = str(lock["selected_fixed_baseline"])
    baseline = {str(row["query_id"]): row for row in test_rows[baseline_arm]}
    subset_by_query = {str(row["query_id"]): str(row["subset"]) for row in test_rows[baseline_arm]}
    paired: dict[str, Any] = {}
    for arm in ARMS:
        if arm == baseline_arm:
            continue
        candidates = {str(row["query_id"]): row for row in test_rows[arm]}
        paired[arm] = {
            metric: paired_macro_bootstrap(
                candidates,
                baseline,
                subset_by_query,
                metric=metric,
                resamples=10_000,
            )
            for metric in METRICS
        }
    custom = paired["rrf_medcpt_rerank"]["ndcg@10"]
    report = {
        "schema_version": "e1-2-r2med-test-report-v1",
        "status": "COMPLETED",
        "config_sha256": config_hash,
        "source_manifest_sha256": source_hash,
        "test_query_count": sum(entry["query_count"] for entry in source_manifest["test_subsets"]),
        "metrics": summaries,
        "selected_fixed_baseline": baseline_arm,
        "paired_differences_vs_dev_selected_fixed_baseline": paired,
        "custom_method_primary_result": custom,
        "headline_gate": {
            "eligible": bool(custom["mean_difference"] > 0 and custom["lower_95"] > 0),
            "metric": "equal-weight macro nDCG@10 difference versus the strongest fixed DEV baseline",
            "criterion": "positive mean difference and paired 95% interval lower bound above zero",
        },
        "metric_semantics": "nDCG@10 uses gain=2^relevance-1; the published R2MED primary qrels are binary, so this is standard binary gain. Also report MRR@10 and Recall@5/10; all queries included with equal subset weights for the primary macro score.",
        "data_boundary": "Aggregate metrics only; source questions, qrels, and document text are not included.",
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing == report:
            return existing
        raise FileExistsError(f"refusing to overwrite the existing R2MED TEST report: {output_path}")
    write_json_atomic(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lock-dev", action="store_true")
    group.add_argument("--analyze-test", action="store_true")
    parser.add_argument("--scratch-root", type=Path, default=Path(r"E:\Health-Copilot-E1.2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/e1_2/r2med_test_report.json",
    )
    args = parser.parse_args()
    result = lock_dev(scratch_root=args.scratch_root) if args.lock_dev else analyze_test(
        scratch_root=args.scratch_root, output_path=args.output
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_fixed_baseline": result.get("selected_fixed_baseline"),
                "output": str(DEV_LOCK_PATH if args.lock_dev else args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
