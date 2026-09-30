"""Materialize the gold-blind DEV-only corpus consumed by the U3-R runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.u3r_rag_transfer import (
    U2F_ROOT,
    U3R_RUN_ROOT,
    materialize_runtime_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u2f-root", type=Path, default=ROOT / U2F_ROOT)
    parser.add_argument("--run-root", type=Path, default=ROOT / U3R_RUN_ROOT)
    args = parser.parse_args()
    manifest = materialize_runtime_corpus(u2f_root=args.u2f_root, run_root=args.run_root)
    print(json.dumps({
        "episode_count": manifest["episode_count"],
        "split_counts": manifest["split_counts"],
        "visible_document_count": manifest["visible_document_count"],
        "runtime_corpus_sha256": manifest["runtime_corpus_sha256"],
        "evaluator_truth_opened": manifest["evaluator_truth_opened"],
        "reserved_test_ood_opened": manifest["reserved_test_ood_opened"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
