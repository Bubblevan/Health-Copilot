"""Evaluate E5-B2 only after the complete 180-arm artifact set is frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.rag_e5.e5b2_evaluator import analyze_frozen_run

ROOT = Path(__file__).resolve().parents[3]
PRIVATE_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b1")
CORPUS_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5a3\corpus\public_health_plus_guideline")
ARTIFACT_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b2")


def _analyze(
    *,
    private_root: Path,
    corpus_root: Path,
    artifact_root: Path,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    destinations = (
        artifact_root / "e5b2_private_eval.json",
        ROOT / "runs/rag_e5/e5b2_counterfactual_report.json",
        ROOT / "runs/rag_e5/e5b2_counterfactual_matrix_summary.json",
        ROOT / "docs/research/rag_e5/e5b2_counterfactual_report.md",
    )
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "B2 scoring outputs already exist; refusing to overwrite frozen analysis artifacts"
        )
    return analyze_frozen_run(
        lock_path=ROOT / "runs/rag_e5/e5b2_protocol_lock.json",
        execution_manifest_path=ROOT / "runs/rag_e5/e5b2_execution_manifest.json",
        private_root=private_root,
        corpus_root=corpus_root,
        artifact_root=artifact_root,
        private_output_path=destinations[0],
        report_json_path=destinations[1],
        matrix_json_path=destinations[2],
        report_markdown_path=destinations[3],
        seed=seed,
        bootstrap_samples=samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=PRIVATE_ROOT)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    report = _analyze(
        private_root=args.private_root,
        corpus_root=args.corpus_root,
        artifact_root=args.artifact_root,
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                "cases": report["cases"],
                "arms": report["arms"],
                "gates": report["gates"],
                "E5C_AUTHORIZED": report["E5C_AUTHORIZED"],
                "E5C_STARTED": report["E5C_STARTED"],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
