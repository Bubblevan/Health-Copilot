"""Freeze the RAG-E6A subject split without opening evaluator truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.rag_e6.split import build_split_manifest, write_immutable_json

ROOT = Path(__file__).resolve().parents[3]
U2F_ROOT = ROOT / "runs/integration/u2f-owned-v1-55955b2eff38"
OUTPUT = ROOT / "runs/rag_e6/split_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u2f-root", type=Path, default=U2F_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = build_split_manifest(
        episodes_path=args.u2f_root / "train/episodes.jsonl",
        source_manifest_path=args.u2f_root / "manifest.json",
    )
    write_immutable_json(args.output, manifest)
    print(json.dumps({
        "output": str(args.output),
        "partition_subject_counts": manifest["partition_subject_counts"],
        "partition_episode_counts": manifest["partition_episode_counts"],
        "evaluator_truth_opened": manifest["evaluator_truth_opened"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
