"""Run LuceneBM25Model and BGE index builds in isolated Python processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_ROOT = Path("D:/MyLab/Jianli/external/rag_e5")
DEFAULT_MODEL_ROOT = Path("E:/Health-Copilot-Models/models/bge-large-en-v1.5")
DEFAULT_JAVA_HOME = Path("D:/jdk-21.0.4")
DEFAULT_PYSERINI_ROOT = Path("E:/Health-Copilot-Models/cache/python-packages/pyserini")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--java-home", type=Path, default=DEFAULT_JAVA_HOME)
    parser.add_argument("--pyserini-root", type=Path, default=DEFAULT_PYSERINI_ROOT)
    args = parser.parse_args()
    tasks = (
        [
            sys.executable,
            str(args.repo_root / "tools/research/rag_e5/build_e5a3_bm25.py"),
            "--repo-root",
            str(args.repo_root),
            "--external-root",
            str(args.external_root),
            "--java-home",
            str(args.java_home),
            "--pyserini-root",
            str(args.pyserini_root),
        ],
        [
            sys.executable,
            str(args.repo_root / "tools/research/rag_e5/build_e5a3_dense.py"),
            "--external-root",
            str(args.external_root),
            "--model-root",
            str(args.model_root),
        ],
    )
    for command in tasks:
        subprocess.run(command, cwd=args.repo_root, check=True)
    report = json.loads(
        (args.external_root / "e5a3/index_qualification_report.json").read_text(encoding="utf-8")
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
