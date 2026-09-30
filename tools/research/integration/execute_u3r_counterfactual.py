"""Run frozen U3-R OFF/STANDARD/STRONG arms without opening evaluator truth."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.r2med_crb_data import PINNED_UPSTREAM_COMMIT
from eval.r2med_gar_generation import EXPECTED_UPSTREAM_FILES
from eval.u3r_rag_transfer import (
    ACTION_ORDER,
    DEFAULT_BGE_PATH,
    DEFAULT_LLAMA_URL,
    DEFAULT_QWEN_PATH,
    DEFAULT_UPSTREAM_ROOT,
    LLAMA_MAX_OUTPUT_TOKENS,
    LLAMA_MODEL_NAME,
    LLAMA_REASONING_ENABLED,
    LLAMA_SYSTEM_PROMPT,
    LLAMA_TEMPERATURE,
    U2F_DATASET_ROOT_SHA256,
    U2F_MANIFEST_SHA256,
    U2F_ROOT,
    U3R_RUN_ROOT,
    DenseDocumentCache,
    DeterministicIntegrationExecutor,
    LocalCpuLlamaClient,
    U3RRetriever,
    build_lamer_generation_record,
    canonical_json_bytes,
    fixed_arm_runtime_row,
    lamer_prompt,
    load_dev_runtime_episodes,
    load_runtime_corpora,
    load_upstream_lamer_prompt,
    sha256_bytes,
    sha256_file,
)

BGE_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
BGE_WEIGHTS_SHA256 = "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7"
QWEN_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
BRIDGE_CHECKPOINT = "u3r_bridge_checkpoint.jsonl"
RUNTIME_CHECKPOINT = "u3r_counterfactual_checkpoint.jsonl"
BRIDGE_OUTPUT = "u3r_lamer_bridges.jsonl"
RUNTIME_OUTPUT = "u3r_counterfactuals.jsonl"
FREEZE_MANIFEST = "u3r_counterfactual_manifest.json"
CPU_SERVER_MANIFEST = "u3r_cpu_server_manifest.json"


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _llama_cpp_version(server_path: Path) -> str:
    result = subprocess.run(
        [str(server_path), "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    version = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0 or not version:
        raise RuntimeError("could not verify the local llama.cpp binary version")
    return version


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = path.read_bytes()
    complete_end = raw.rfind(b"\n") + 1
    if complete_end != len(raw):
        with path.open("r+b") as handle:
            handle.truncate(complete_end)
    result: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(raw[:complete_end].splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid checkpoint row {path.name}:{number}") from exc
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or episode_id in result:
            raise ValueError(f"invalid or duplicate checkpoint ID in {path.name}")
        result[episode_id] = row
    return result


def _append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    payload = canonical_json_bytes(row) + b"\n"
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite changed frozen artifact: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary artifact requires inspection: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _require_cpu_server(base_url: str, manifest_path: Path, model_sha: str) -> dict[str, Any]:
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise ValueError("U3-R generation endpoint must be loopback-only")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("host") != "127.0.0.1"
        or manifest.get("n_gpu_layers") != 0
        or manifest.get("model_sha256") != model_sha
        or manifest.get("cpu_only") is not True
    ):
        raise ValueError("llama.cpp server manifest does not prove the frozen CPU runtime")
    health_url = f"{parsed.scheme}://{parsed.netloc}/health"
    with urllib.request.urlopen(health_url, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("local llama.cpp health endpoint is not ready")
        health_body = response.read(4096).decode("utf-8", errors="replace")
    return {"manifest": manifest, "health_response": health_body.strip()}


def _load_bge_cpu(model_path: Path, threads: int):
    if sha256_file(model_path / "model.safetensors") != BGE_WEIGHTS_SHA256:
        raise ValueError("BGE-large weights SHA-256 mismatch")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    model.eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("BGE-large unexpectedly loaded outside CPU")
    return model


def _verify_bridge_checkpoint(
    bridge: dict[str, Any], *, episode, docs, original_bm25, upstream_root: Path,
    expected_model_api_id: str,
) -> None:
    passage_by_id = {item.doc_id: item.text for item in docs}
    feedback_ids = [item.doc_id for item in original_bm25[:10]]
    feedback_text = [passage_by_id[item] for item in feedback_ids]
    _prompt, prompt_sha = lamer_prompt(episode.query, feedback_text, upstream_root)
    if (
        bridge.get("query_sha256") != episode.query_sha256
        or bridge.get("feedback_doc_ids") != feedback_ids
        or bridge.get("prompt_sha256") != prompt_sha
        or bridge.get("generator_model") != LLAMA_MODEL_NAME
        or bridge.get("generator_model_api_id") != expected_model_api_id
        or bridge.get("generator_max_output_tokens") != LLAMA_MAX_OUTPUT_TOKENS
        or bridge.get("temperature") != LLAMA_TEMPERATURE
        or bridge.get("reasoning") != "disabled"
        or bridge.get("provider_calls") != 1
        or not isinstance(bridge.get("generated_text"), str)
        or bridge.get("generated_text_sha256")
        != sha256_bytes(bridge["generated_text"].encode("utf-8"))
    ):
        raise ValueError(f"bridge checkpoint does not match frozen inputs: {episode.episode_id}")


def _verify_runtime_checkpoint(row: dict[str, Any], *, episode, bridge) -> None:
    if (
        row.get("episode_id") != episode.episode_id
        or row.get("query_sha256") != episode.query_sha256
        or row.get("split") != episode.split
        or row.get("bridge_sha256")
        != sha256_bytes(bridge["generated_text"].encode("utf-8"))
        or set(row.get("actions", {})) != set(ACTION_ORDER)
    ):
        raise ValueError(f"counterfactual checkpoint identity mismatch: {episode.episode_id}")


def _code_hashes() -> dict[str, str]:
    paths = (
        "eval/u3r_rag_transfer.py",
        "eval/u3r_rag_scoring.py",
        "tools/research/integration/prepare_u3r_runtime_corpus.py",
        "tools/research/integration/execute_u3r_counterfactual.py",
        "tools/research/integration/score_u3r_counterfactual.py",
        "tools/research/integration/record_u3r_cpu_llama_server.ps1",
        "tests/test_u3r_rag_transfer.py",
        "eval/r2med_crb.py",
        "eval/r2med_crb_data.py",
        "eval/r2med_gar_generation.py",
        "eval/r2med_multiview.py",
        "src/health_ai_copilot/research/integration/actions.py",
        "src/health_ai_copilot/research/integration/contracts.py",
        "src/health_ai_copilot/research/integration/evidence_world.py",
        "src/health_ai_copilot/research/integration/executor.py",
        "src/health_ai_copilot/research/integration/tools.py",
        "src/health_ai_copilot/research/integration/owned_universe/evaluator_truth.py",
        "src/health_ai_copilot/research/integration/owned_universe/schema.py",
    )
    return {name: sha256_file(ROOT / name) for name in paths}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u2f-root", type=Path, default=ROOT / U2F_ROOT)
    parser.add_argument("--run-root", type=Path, default=ROOT / U3R_RUN_ROOT)
    parser.add_argument("--bge-path", type=Path, default=DEFAULT_BGE_PATH)
    parser.add_argument("--qwen-path", type=Path, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--llama-url", default=DEFAULT_LLAMA_URL)
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("CPU thread count must be positive")
    args.run_root.mkdir(parents=True, exist_ok=True)
    server_manifest_path = args.run_root / CPU_SERVER_MANIFEST
    qwen_sha = sha256_file(args.qwen_path)
    if qwen_sha != QWEN_SHA256:
        raise ValueError("Qwen3-8B GGUF SHA-256 mismatch")
    server_identity = _require_cpu_server(args.llama_url, server_manifest_path, qwen_sha)
    server_identity["llama_cpp_version"] = _llama_cpp_version(
        Path(server_identity["manifest"]["llama_server_path"])
    )
    episodes = load_dev_runtime_episodes(args.u2f_root)
    corpus_path = args.run_root / "u3r_runtime_visible_corpus.jsonl"
    corpus_manifest_path = args.run_root / "u3r_runtime_corpus_manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    if (
        corpus_manifest.get("dataset_root_sha256") != U2F_DATASET_ROOT_SHA256
        or corpus_manifest.get("u2f_manifest_sha256") != U2F_MANIFEST_SHA256
        or corpus_manifest.get("runtime_corpus_sha256") != sha256_file(corpus_path)
        or corpus_manifest.get("episode_count") != 1024
        or corpus_manifest.get("evaluator_truth_opened") is not False
        or corpus_manifest.get("reserved_test_ood_opened") is not False
    ):
        raise ValueError("runtime corpus does not match the frozen DEV-only manifest")
    corpora = load_runtime_corpora(corpus_path)
    if set(corpora) != {item.episode_id for item in episodes}:
        raise ValueError("runtime corpus and DEV episode IDs differ")

    bridge_path = args.run_root / BRIDGE_CHECKPOINT
    runtime_path = args.run_root / RUNTIME_CHECKPOINT
    bridges = _load_checkpoint(bridge_path)
    runtime_rows = _load_checkpoint(runtime_path)
    episode_by_id = {item.episode_id: item for item in episodes}
    for episode_id, row in runtime_rows.items():
        if episode_id not in bridges or episode_id not in episode_by_id:
            raise ValueError("runtime checkpoint is missing its episode or bridge")
        _verify_runtime_checkpoint(row, episode=episode_by_id[episode_id], bridge=bridges[episode_id])

    missing_ids = set(episode_by_id) - set(runtime_rows)
    model = _load_bge_cpu(args.bge_path, args.cpu_threads) if missing_ids else None
    retriever = U3RRetriever(DenseDocumentCache(model)) if model is not None else None
    client = (
        LocalCpuLlamaClient(
            args.llama_url,
            model_name=str(server_identity["manifest"]["model_api_id"]),
        )
        if missing_ids else None
    )
    deterministic_executor = DeterministicIntegrationExecutor()
    started = time.monotonic()
    completed_now = 0

    for index, episode in enumerate(episodes, start=1):
        if episode.episode_id in runtime_rows:
            continue
        if retriever is None or client is None:
            raise RuntimeError("retrieval runtime was not initialized for unfinished episodes")
        docs = corpora[episode.episode_id]
        doc_by_id = {item.doc_id: item for item in docs}
        original_bm25 = retriever._bm25(docs, episode.query)

        bridge = bridges.get(episode.episode_id)
        if bridge is None:
            bridge = build_lamer_generation_record(
                episode, docs, retriever, client,
                original_bm25=original_bm25,
                upstream_root=args.upstream_root,
            )
            bridge["generated_text_sha256"] = sha256_bytes(
                bridge["generated_text"].encode("utf-8")
            )
            _append_checkpoint(bridge_path, bridge)
            bridges[episode.episode_id] = bridge
        else:
            _verify_bridge_checkpoint(
                bridge, episode=episode, docs=docs, original_bm25=original_bm25,
                upstream_root=args.upstream_root,
                expected_model_api_id=str(server_identity["manifest"]["model_api_id"]),
            )

        for doc_id in bridge["feedback_doc_ids"]:
            if doc_id not in doc_by_id:
                raise ValueError("LameR feedback refers to a document outside the runtime corpus")
        standard_started = time.perf_counter()
        standard = retriever.standard(episode, docs)
        standard_latency_ms = round((time.perf_counter() - standard_started) * 1000)
        strong_started = time.perf_counter()
        strong = retriever.strong(
            episode, docs, str(bridge["generated_text"]), original_bm25
        )
        strong_latency_ms = round((time.perf_counter() - strong_started) * 1000)
        row = fixed_arm_runtime_row(
            episode, docs, standard=standard, strong=strong, bridge=bridge,
            executor=deterministic_executor,
            retrieval_latency_ms={
                "STANDARD": standard_latency_ms,
                "STRONG": strong_latency_ms,
            },
        )
        _verify_runtime_checkpoint(row, episode=episode, bridge=bridge)
        _append_checkpoint(runtime_path, row)
        runtime_rows[episode.episode_id] = row
        completed_now += 1
        if completed_now % 8 == 0 or index == len(episodes):
            elapsed = time.monotonic() - started
            print(json.dumps({
                "completed_this_process": completed_now,
                "completed_total": len(runtime_rows),
                "episodes_total": len(episodes),
                "elapsed_seconds": round(elapsed, 1),
            }, sort_keys=True), flush=True)

    if set(bridges) != set(episode_by_id) or set(runtime_rows) != set(episode_by_id):
        raise ValueError("U3-R full-DEV counterfactual execution is incomplete")
    ordered_bridges = [bridges[item.episode_id] for item in episodes]
    ordered_runtime = [runtime_rows[item.episode_id] for item in episodes]
    bridge_content = _jsonl_bytes(ordered_bridges)
    runtime_content = _jsonl_bytes(ordered_runtime)
    _write_immutable(args.run_root / BRIDGE_OUTPUT, bridge_content)
    _write_immutable(args.run_root / RUNTIME_OUTPUT, runtime_content)

    upstream = load_upstream_lamer_prompt(args.upstream_root)
    upstream["pinned_commit"] = PINNED_UPSTREAM_COMMIT
    upstream["verified_source_sha256"] = dict(sorted(EXPECTED_UPSTREAM_FILES.items()))
    manifest = {
        "schema_version": "u3r-counterfactual-freeze-v1",
        "stage": "U3-R",
        "dataset_id": "health-copilot-owned-longitudinal-v1",
        "dataset_root_sha256": U2F_DATASET_ROOT_SHA256,
        "u2f_manifest_sha256": U2F_MANIFEST_SHA256,
        "split_counts": {
            split: sum(item.split == split for item in episodes)
            for split in ("DEV_IID", "DEV_STRUCTURAL")
        },
        "episode_count": len(episodes),
        "arms_per_episode": list(ACTION_ORDER),
        "arm_count": len(episodes) * len(ACTION_ORDER),
        "primary_evaluation_slice_applied_during_execution": False,
        "query_order_sha256": sha256_bytes(canonical_json_bytes([
            [item.episode_id, item.query_sha256] for item in episodes
        ])),
        "runtime_corpus_sha256": sha256_file(corpus_path),
        "bridge_artifact_sha256": sha256_bytes(bridge_content),
        "counterfactual_artifact_sha256": sha256_bytes(runtime_content),
        "bridge_generation": {
            "method": "pinned R2MED LameR prompt family + local Qwen3-8B",
            "upstream": upstream,
            "generator_model": LLAMA_MODEL_NAME,
            "generator_model_api_id": server_identity["manifest"]["model_api_id"],
            "generator_gguf_sha256": qwen_sha,
            "generator_max_output_tokens": LLAMA_MAX_OUTPUT_TOKENS,
            "temperature": LLAMA_TEMPERATURE,
            "reasoning": "disabled",
            "chat_template_kwargs": {"enable_thinking": LLAMA_REASONING_ENABLED},
            "top_p": 1,
            "system_prompt_sha256": sha256_bytes(LLAMA_SYSTEM_PROMPT.encode("utf-8")),
            "attempted_calls": sum(int(row["provider_calls"]) for row in ordered_bridges),
            "valid_nonempty": sum(bool(row["valid"]) for row in ordered_bridges),
            "completed": sum(bool(row["completed"]) for row in ordered_bridges),
            "truncated": sum(bool(row["truncated"]) for row in ordered_bridges),
            "fallback_original": sum(bool(row["fallback_original"]) for row in ordered_bridges),
            "input_tokens": sum(int(row["input_tokens"]) for row in ordered_bridges),
            "output_tokens": sum(int(row["output_tokens"]) for row in ordered_bridges),
            "output_text_sha256": sha256_bytes(canonical_json_bytes([
                [row["episode_id"], row["generated_text_sha256"]]
                for row in ordered_bridges
            ])),
        },
        "retrieval": {
            "standard": {"bm25_k1": 0.9, "bm25_b": 0.4, "bge_revision": BGE_REVISION,
                         "bge_weights_sha256": BGE_WEIGHTS_SHA256,
                         "rrf_k": 60, "weights": [1, 1], "top_k": 10},
            "strong": {"method": "LameR-MV", "bm25_feedback_top_k": 10,
                       "retrieval_channels": ["BM25_QUERY", "BM25_QUERY_PLUS_BRIDGE",
                                              "BGE_QUERY", "BGE_BRIDGE"],
                       "rrf_k": 20, "weights": [1, 2, 1, 2], "top_k": 10,
                       "feedback_bm25_reused_as_query_channel": True},
        },
        "runtime": {
            "llama_cpu_server_manifest_sha256": sha256_file(server_manifest_path),
            "llama_cpu_server": server_identity,
            "bge_device": "cpu",
            "cpu_threads": args.cpu_threads,
            "python": sys.version.split()[0],
            "packages": {
                name: _package_version(name)
                for name in ("torch", "sentence-transformers", "pyserini", "gensim")
            },
        },
        "code_commit": _git_commit(),
        "code_sha256": _code_hashes(),
        "evaluator_truth_opened": False,
        "train_outcomes_opened": False,
        "reserved_test_ood_opened": False,
        "reserved_test_ood_materialized": False,
        "memory_capability": "UNCHANGED_UNAVAILABLE",
        "team_capability": "DISABLED",
        "llm_authority": ["retrieval_bridge_text_only"],
        "llm_cannot_select_actions": True,
        "llm_cannot_write_answers": True,
        "llm_cannot_access_evaluator_truth": True,
        "training_started": False,
    }
    freeze_bytes = canonical_json_bytes(manifest) + b"\n"
    _write_immutable(args.run_root / FREEZE_MANIFEST, freeze_bytes)
    print(json.dumps({
        "frozen": True,
        "episode_count": len(episodes),
        "arm_count": len(episodes) * len(ACTION_ORDER),
        "bridge_valid": manifest["bridge_generation"]["valid_nonempty"],
        "bridge_fallback": manifest["bridge_generation"]["fallback_original"],
        "manifest_sha256": sha256_bytes(freeze_bytes),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
