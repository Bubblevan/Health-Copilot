"""Execute all BUILD subjects under the frozen five-arm E6A reader protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.rag_e6.reader_executor import (
    DEFAULT_LLAMA_URL,
    DEFAULT_RUNTIME_CORPUS_ROOT,
    DEFAULT_SERVER_MANIFEST,
    execute_partition,
)
from eval.rag_e6.split import U2F_ROOT_SHA256

U2F_ROOT = ROOT / "runs/integration/u2f-owned-v1-55955b2eff38"
SPLIT_MANIFEST = ROOT / "runs/rag_e6/split_manifest.json"
OUTPUT_ROOT = ROOT / "runs/rag_e6/build_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u2f-root", type=Path, default=U2F_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_RUNTIME_CORPUS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--bge-path", type=Path)
    parser.add_argument("--qwen-path", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--llama-url", default=DEFAULT_LLAMA_URL)
    parser.add_argument("--server-manifest", type=Path, default=DEFAULT_SERVER_MANIFEST)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split_manifest.get("dataset_root_sha256") != U2F_ROOT_SHA256:
        raise ValueError("E6A subject split does not match the frozen U2-F root")
    if split_manifest.get("evaluator_truth_opened") is not False:
        raise ValueError("BUILD runtime cannot start after evaluator-truth access")
    manifest = execute_partition(
        partition="BUILD",
        u2f_root=args.u2f_root,
        split_manifest_path=args.split_manifest,
        corpus_root=args.corpus_root,
        output_root=args.output_root,
        **({"bge_path": args.bge_path} if args.bge_path else {}),
        **({"qwen_path": args.qwen_path} if args.qwen_path else {}),
        **({"upstream_root": args.upstream_root} if args.upstream_root else {}),
        llama_url=args.llama_url,
        server_manifest_path=args.server_manifest,
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps({
        "partition": manifest["partition"],
        "episode_count": manifest["episode_count"],
        "arm_execution_count": manifest["arm_execution_count"],
        "reader_output_sha256": manifest["reader_output_sha256"],
        "generation_calls_attempted": manifest["generator"]["generation_calls_attempted"],
        "evaluator_truth_opened": manifest["evaluator_truth_opened"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
