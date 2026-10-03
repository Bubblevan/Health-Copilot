"""One-shot scoring of frozen RAG-E6B reserved executions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.rag_e6b.scoring import score_reserved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = score_reserved(repository_root=args.repository_root)
    print(json.dumps({
        "status": "SCORED_ONCE",
        "reserved_total_episodes": report["reserved_total_episodes"],
        "reserved_total_subjects": report["reserved_total_subjects"],
        "iid_gate": report["iid_gate"],
        "iid_test_rag_population": report["iid_test_rag_population"],
        "iid_grounded_delta_pp": report["pool_reports"]["IID_TEST"][
            "grounded_success_delta"]["delta_pp"],
        "pooled_ood_delta_pp": report["pooled_ood_rag"]["grounded_success_delta"]["delta_pp"],
        "ood_generalization_supported": report["ood_generalization_supported"],
        "no_match_parity": report["rsel_match_no_match"]["no_match_parity"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
