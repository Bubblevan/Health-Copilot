"""Build one gold-blind E6A corpus partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.rag_e6.data import materialize_partition_corpus

U2F_ROOT = ROOT / "runs/integration/u2f-owned-v1-55955b2eff38"
SPLIT_MANIFEST = ROOT / "runs/rag_e6/split_manifest.json"
OUTPUT_ROOT = ROOT / "runs/rag_e6/corpus"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", choices=("BUILD", "FROZEN_DEV"))
    parser.add_argument("--u2f-root", type=Path, default=U2F_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    manifest = materialize_partition_corpus(
        u2f_root=args.u2f_root,
        split_manifest_path=args.split_manifest,
        partition=args.partition,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
