"""Run frozen E1.2 BM25 or MedCPT retrieval into external scratch storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_DEFAULT = Path(r"F:\Health-Copilot-E1.2")
CORPUS_DEFAULT = ROOT / "data/raw/medrag_textbooks"
MODEL_DEFAULT = ROOT.parent / "models"
CLEAN_SUBDATASETS = ("medqa", "medmcqa", "mmlu")
MIN_FREE_GIB = 186.3
NEW_USE_CAP_GIB = 50.0


def ensure_storage(scratch_root: Path, *, reserve_gib: float = 5.0) -> None:
    if not scratch_root.is_dir():
        raise FileNotFoundError(f"E1.2 scratch root is unavailable: {scratch_root}")
    free_gib = shutil.disk_usage(scratch_root).free / (1024**3)
    if free_gib - reserve_gib < MIN_FREE_GIB:
        raise OSError(
            f"scratch volume would cross the {MIN_FREE_GIB:.1f} GiB free-space floor"
        )
    used_bytes = sum(path.stat().st_size for path in scratch_root.rglob("*") if path.is_file())
    predicted_use_gib = used_bytes / (1024**3) + reserve_gib
    if predicted_use_gib > NEW_USE_CAP_GIB:
        raise OSError(
            f"predicted E1.2 scratch use {predicted_use_gib:.2f} GiB exceeds the 50 GiB cap"
        )


def verify_retrieval_input(scratch_root: Path, split: str) -> None:
    config_path = ROOT / "runs/e1_2/frozen_test_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_path = ROOT / config["benchmark"]["split_manifest"]
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_hash != config["benchmark"]["split_manifest_sha256"]:
        raise ValueError("split manifest hash does not match the frozen E1.2 config")
    input_path = scratch_root / "inputs" / f"benchmark_{split.lower()}.json"
    sidecar_path = input_path.with_name(input_path.name + ".manifest.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"retrieval input sidecar is missing: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("sha256") != hashlib.sha256(input_path.read_bytes()).hexdigest():
        raise ValueError("retrieval input file does not match its sidecar hash")
    if sidecar.get("split_manifest_sha256") != split_hash:
        raise ValueError("retrieval input was materialized from a different split manifest")
    if sidecar.get("normalized_cases_sha256") != config["benchmark"]["normalized_cases_sha256"]:
        raise ValueError("retrieval input was materialized from different normalized MIRAGE data")
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    expected_counts = {
        name: int(manifest.get("counts", {}).get(name, {}).get(split.upper(), 0))
        for name in CLEAN_SUBDATASETS
    }
    if sidecar.get("counts") != expected_counts:
        raise ValueError("retrieval input counts do not match the frozen partition")
    benchmark = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(benchmark, dict) or set(benchmark) != set(CLEAN_SUBDATASETS):
        raise ValueError("retrieval input has unexpected subdatasets")
    actual_counts: dict[str, int] = {}
    for subset, rows in benchmark.items():
        if not isinstance(rows, dict) or any(
            not isinstance(row, dict) or set(row) != {"question", "options"}
            for row in rows.values()
        ):
            raise ValueError("retrieval input must contain only question/options per case")
        actual_counts[subset] = len(rows)
    if actual_counts != expected_counts:
        raise ValueError("retrieval input case counts differ from the frozen partition")


def build_command(
    *,
    retriever: str,
    split: str,
    scratch_root: Path,
    benchmark_path: Path,
    corpus_dir: Path,
    model_root: Path,
) -> list[str]:
    index_path = scratch_root / "index" / "medrag_textbooks_fts5.sqlite3"
    output_path = scratch_root / "retrieval" / split.lower() / retriever
    common = [
        sys.executable,
        str(ROOT / "tools" / ("run_medrag_textbooks_retrieval.py" if retriever == "bm25" else "run_medcpt_textbooks_retrieval.py")),
        "--benchmark-json",
        str(benchmark_path),
        "--corpus-dir",
        str(corpus_dir),
        "--index-path",
        str(index_path),
        "--output-dir",
        str(output_path),
        "--subdatasets",
        *CLEAN_SUBDATASETS,
        "--top-k",
        "5",
    ]
    if retriever == "bm25":
        return [*common, "--expected-corpus-chunks", "125847"]
    if retriever == "medcpt":
        cache_path = scratch_root / "cache" / "medcpt_textbooks"
        return [
            *common,
            "--model-root",
            str(model_root),
            "--cache-dir",
            str(cache_path),
            "--dense-candidate-depth",
            "100",
            "--expected-corpus-chunks",
            "125847",
            "--article-batch-size",
            "64",
            "--query-batch-size",
            "32",
            "--rerank-batch-size",
            "32",
        ]
    raise ValueError(f"unsupported retriever: {retriever}")


def run_retrieval(
    *,
    retriever: str,
    split: str,
    scratch_root: Path = SCRATCH_DEFAULT,
    corpus_dir: Path = CORPUS_DEFAULT,
    model_root: Path = MODEL_DEFAULT,
) -> int:
    split_name = split.upper()
    if split_name not in {"DEV", "TEST"}:
        raise ValueError("split must be DEV or TEST")
    if retriever not in {"bm25", "medcpt"}:
        raise ValueError("retriever must be bm25 or medcpt")
    ensure_storage(scratch_root)
    benchmark_path = scratch_root / "inputs" / f"benchmark_{split_name.lower()}.json"
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"split-specific retrieval input is missing: {benchmark_path}")
    verify_retrieval_input(scratch_root, split_name)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"MedRAG Textbooks corpus is missing: {corpus_dir}")
    if retriever == "medcpt" and not model_root.is_dir():
        raise FileNotFoundError(f"local MedCPT models are missing: {model_root}")

    command = build_command(
        retriever=retriever,
        split=split_name,
        scratch_root=scratch_root,
        benchmark_path=benchmark_path,
        corpus_dir=corpus_dir,
        model_root=model_root,
    )
    environment = os.environ.copy()
    for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME"):
        environment[name] = str(scratch_root / "hf_cache")
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    print(json.dumps({"retriever": retriever, "split": split_name, "output": command[command.index("--output-dir") + 1]}), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=("bm25", "medcpt"), required=True)
    parser.add_argument("--split", choices=("DEV", "TEST"), required=True)
    parser.add_argument("--scratch-root", type=Path, default=SCRATCH_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--model-root", type=Path, default=MODEL_DEFAULT)
    args = parser.parse_args()
    return run_retrieval(
        retriever=args.retriever,
        split=args.split,
        scratch_root=args.scratch_root,
        corpus_dir=args.corpus_dir,
        model_root=args.model_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
