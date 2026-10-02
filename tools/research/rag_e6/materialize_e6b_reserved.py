"""One-time materialization of the frozen RAG-E6B reserved pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.rag_e6b.materialize import materialize_reserved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = materialize_reserved(repository_root=args.repository_root)
    print(json.dumps({
        "status": "MATERIALIZED",
        "pool_episode_counts": manifest["actual_episodes_per_pool"],
        "pool_subject_counts": manifest["actual_subjects_per_pool"],
        "reserved_total_episodes": len(manifest["capability_blind_episode_ids"]),
        "reserved_total_subjects": manifest["pool_subject_id_union_count"],
        "evaluator_truth_opened": manifest["evaluator_truth_opened"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
