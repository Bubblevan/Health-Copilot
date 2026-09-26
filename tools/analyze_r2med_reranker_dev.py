"""Print the frozen DEV reranker report without touching TEST data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "runs/rag_r2med_rerank/dev_report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    selection = report["selection"]
    output = {
        "partition": report["partition"],
        "query_count": report["query_count"],
        "test_accessed": report["test_accessed"],
        "reranker": report["reranker"],
        "selection": selection,
        "arms": {
            name: {
                "ndcg@10": summary["macro_equal_subset_weight"]["ndcg@10"],
                "mrr@10": summary["macro_equal_subset_weight"]["mrr@10"],
                "recall@10": summary["macro_equal_subset_weight"]["recall@10"],
                "recall@100": summary["macro_equal_subset_weight"]["recall@100"],
                "by_subset_ndcg@10": {
                    subset: values["ndcg@10"] for subset, values in summary["by_subset"].items()
                },
            }
            for name, summary in report["arm_summaries"].items()
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
