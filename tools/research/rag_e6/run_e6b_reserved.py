"""Gold-blind paired execution over every reserved E6B pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.rag_e6b.runner import execute_reserved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--bge-path", type=Path)
    parser.add_argument("--qwen-path", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--llama-url", default="http://127.0.0.1:8092/v1")
    parser.add_argument("--server-manifest", type=Path,
                        default=Path("runs/rag_e6/runtime/gpu_server_manifest.json"))
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    kwargs = {key: value for key, value in {
        "bge_path": args.bge_path,
        "qwen_path": args.qwen_path,
        "upstream_root": args.upstream_root,
    }.items() if value is not None}
    manifest = execute_reserved(
        repository_root=args.repository_root,
        llama_url=args.llama_url,
        server_manifest_path=args.server_manifest,
        cpu_threads=args.cpu_threads,
        **kwargs,
    )
    print(json.dumps({
        "status": "EXECUTED",
        "total_episode_count": manifest["total_episode_count"],
        "generation_calls": manifest["generation_calls"],
        "provider_calls": manifest["provider_calls"],
        "failed_generation_calls": manifest["failed_generation_calls"],
        "truncated_generation_calls": manifest["truncated_generation_calls"],
        "rsel_actions": manifest["rsel_actions"],
        "evaluator_truth_opened": manifest["evaluator_truth_opened"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
