from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from eval.parsers import extract_mcq_answer, final_response
from eval.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_SHA256

DATA_ROOT = Path(os.environ.get("PT_E0_DATA_ROOT", "/root/gpufree-data/Health-Copilot-PT-E0-data"))
MODEL_DIR = Path(os.environ.get("PT_E0_BASE_MODEL", "/root/gpufree-share/data/Qwen3-8B"))
RUNS = Path(os.environ.get("PT_E0_RUNS", str(DATA_ROOT / "runs/posttrain/pt-e0")))
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
BENCHMARKS = {"diagnosisarena", "cmb", "hbpro"}
MAX_NEW_TOKENS = 2048
SEED = 20261004
BATCH_SIZE = 16
MAX_MODEL_LEN = 16384
MAX_NUM_BATCHED_TOKENS = 16384
MAX_NUM_SEQS = 16
GPU_MEMORY_UTILIZATION = 0.88


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_artifact_manifest(model_path: Path) -> dict[str, Any]:
    frozen_base_manifest = Path(__file__).resolve().parents[1] / "manifests/model/qwen3-8b-base.json"
    if model_path.resolve() == MODEL_DIR.resolve():
        base_manifest = json.loads(frozen_base_manifest.read_text(encoding="utf-8"))
        for filename, info in base_manifest["all_verified_files"].items():
            path = model_path / filename
            if not path.is_file() or path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
                raise ValueError(f"Frozen Qwen3-8B base file changed: {filename}")
        return {
            "kind": "frozen_base_model_manifest",
            "manifest_sha256": sha256_file(frozen_base_manifest),
            "files": base_manifest["all_verified_files"],
        }
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    files = {}
    for path in sorted(item for item in model_path.rglob("*") if item.is_file() and ".git" not in item.parts):
        relative = str(path.relative_to(model_path))
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if not files:
        raise ValueError(f"No files found in checkpoint path: {model_path}")
    return {"kind": "local_hf_checkpoint_files", "files": files}


def build_messages(row: dict[str, Any], benchmark: str) -> list[dict[str, str]]:
    return row["messages"] if benchmark == "hbpro" else [{"role": "user", "content": row["prompt"]}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--checkpoint-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--model-path", default=str(MODEL_DIR))
    parser.add_argument("--model-revision", default=None)
    args = parser.parse_args()
    benchmark = args.benchmark
    model_path = Path(args.model_path)
    model_revision = args.model_revision or (MODEL_REVISION if model_path.resolve() == MODEL_DIR.resolve() else "local")
    if args.checkpoint_name == "Qwen/Qwen3-8B" and (model_path.resolve() != MODEL_DIR.resolve() or model_revision != MODEL_REVISION):
        raise ValueError("Qwen/Qwen3-8B must use the frozen local Base path and revision")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    model_artifacts = model_artifact_manifest(model_path)
    base_manifest_path = Path(__file__).resolve().parents[1] / "manifests/model/qwen3-8b-base.json"
    base_manifest_sha = sha256_file(base_manifest_path)
    project_root = Path(__file__).resolve().parents[1]
    protocol_path = project_root / "manifests/eval/eval_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(protocol_path)
    frozen_source_hashes = {
        "prompt_builder_sha256": sha256_file(project_root / "eval/prompts.py"),
        "parser_sha256": sha256_file(project_root / "eval/parsers.py"),
        "candidate_generation_code_sha256": sha256_file(Path(__file__)),
    }
    if any(protocol.get(key) != value for key, value in frozen_source_hashes.items()):
        raise ValueError("Candidate prompt/parser/generation code differs from the frozen protocol")
    expected_generation = protocol["generation"]
    inference_runtime = protocol["inference_runtime"]
    expected_runtime = {
        "engine": "vllm",
        "version": importlib.metadata.version("vllm"),
        "flashinfer_python_version": importlib.metadata.version("flashinfer-python"),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "batch_size": BATCH_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "prefix_caching": False,
        "attention_backend": "auto",
        "seed": SEED,
    }
    if inference_runtime != expected_runtime:
        raise ValueError("vLLM inference runtime differs from the frozen evaluation protocol")
    if (
        expected_generation["temperature"] != 0
        or expected_generation["do_sample"] is not False
        or expected_generation["top_p"] != 1
        or expected_generation["max_new_tokens"] != MAX_NEW_TOKENS
        or expected_generation["enable_thinking"] is not True
        or protocol["candidate_system_prompt_sha256"] != SYSTEM_PROMPT_SHA256
    ):
        raise ValueError("Inference settings differ from the frozen evaluation protocol")
    candidate_path = DATA_ROOT / "eval/prepared" / benchmark / "candidate_view.jsonl"
    out_dir = RUNS / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "predictions.jsonl"
    partial_path = out_dir / "predictions.jsonl.partial"
    partial_meta_path = out_dir / "partial_run_manifest.json"
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen predictions: {final_path}")
    candidates = read_jsonl(candidate_path)
    if not candidates:
        raise ValueError(f"No candidates at {candidate_path}")
    candidate_ids = [str(row["id"]) for row in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate IDs are not unique")
    candidate_hash = sha256_file(candidate_path)
    generation = {
        "temperature": 0,
        "do_sample": False,
        "top_p": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "enable_thinking": True,
        "backend": "vllm-greedy",
        "batch_size": BATCH_SIZE,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
    }
    partial_identity = {
        "checkpoint": args.checkpoint_name,
        "model_revision": model_revision,
        "base_model_revision": MODEL_REVISION,
        "model_path": str(model_path),
        "model_artifacts": model_artifacts,
        "base_model_manifest_sha256": base_manifest_sha,
        "adapter": None,
        "candidate_view_sha256": candidate_hash,
        "eval_protocol_sha256": protocol_hash,
        "generation": generation,
        "inference_runtime": inference_runtime,
    }
    if partial_meta_path.exists():
        prior_identity = json.loads(partial_meta_path.read_text(encoding="utf-8"))
        if prior_identity != partial_identity:
            raise ValueError("Partial run protocol differs from frozen candidate/model/generation identity")
    else:
        write_json(partial_meta_path, partial_identity)
    existing_rows = read_jsonl(partial_path) if partial_path.exists() else []
    done_ids = {str(row["id"]) for row in existing_rows}
    if not done_ids.issubset(set(candidate_ids)):
        raise ValueError("Partial predictions contain IDs outside the candidate view")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process; no baseline prediction was started")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 20 * 1024**3:
        raise RuntimeError(f"Refusing to load Qwen3-8B: only {free_bytes / 1024**3:.1f} GiB GPU memory is free")

    started_at = datetime.now(timezone.utc).isoformat()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    chat_template_hash = hashlib.sha256((tokenizer.chat_template or "").encode("utf-8")).hexdigest()
    if chat_template_hash != protocol["candidate_chat_template_sha256"]:
        raise ValueError("Checkpoint chat template differs from the frozen Qwen3 evaluation template")
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        seed=SEED,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=False,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED,
        skip_special_tokens=False,
    )
    latencies = [float(row["generation_latency_seconds"]) for row in existing_rows]
    completed_this_run = 0
    pending_rows = [row for row in candidates if str(row["id"]) not in done_ids]
    with partial_path.open("a" if partial_path.exists() else "w", encoding="utf-8", buffering=1) as output:
        for batch_start in range(0, len(pending_rows), BATCH_SIZE):
            batch = pending_rows[batch_start : batch_start + BATCH_SIZE]
            prompts = []
            for row in batch:
                prompt_text = tokenizer.apply_chat_template(
                    build_messages(row, benchmark),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                prompts.append({"prompt_token_ids": token_ids})
            batch_started = time.perf_counter()
            requests = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            batch_elapsed = time.perf_counter() - batch_started
            if len(requests) != len(batch):
                raise RuntimeError("vLLM returned a different number of outputs than prompts")
            for row, request in zip(batch, requests, strict=True):
                if len(request.outputs) != 1:
                    raise RuntimeError("Frozen greedy evaluation expects exactly one completion per prompt")
                output_item = request.outputs[0]
                new_ids = list(output_item.token_ids)
                raw_generation = tokenizer.decode(new_ids, skip_special_tokens=False)
                final = final_response(raw_generation)
                if benchmark in {"diagnosisarena", "cmb"}:
                    multi = benchmark == "cmb" and row.get("question_type") == "多项选择题"
                    parsed = extract_mcq_answer(raw_generation, set("ABCDEF"), allow_multiple=multi)
                else:
                    parsed = final
                metrics = request.metrics
                if metrics is not None and metrics.last_token_ts is not None:
                    latency = max(0.0, metrics.last_token_ts - metrics.arrival_time)
                else:
                    latency = batch_elapsed
                row_id = str(row["id"])
                result = {
                    "id": row_id,
                    "raw_generation": raw_generation,
                    "final_answer": final,
                    "parse_success": bool(parsed),
                    "parsed_answer": parsed,
                    "raw_output_hash": hashlib.sha256(raw_generation.encode("utf-8")).hexdigest(),
                    "output_tokens": len(new_ids),
                    "generation_latency_seconds": latency,
                }
                output.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                done_ids.add(row_id)
                latencies.append(latency)
                completed_this_run += 1
            output.flush()
            print(json.dumps({
                "benchmark": benchmark,
                "completed_this_run": completed_this_run,
                "completed_total": len(done_ids),
                "total": len(candidates),
                "batch_elapsed_seconds": round(batch_elapsed, 2),
            }), flush=True)
    if done_ids != set(candidate_ids):
        raise RuntimeError(f"Prediction IDs do not match candidate IDs: {len(done_ids)} != {len(candidate_ids)}")
    partial_path.replace(final_path)
    prediction_hash = sha256_file(final_path)
    root = Path(__file__).resolve().parent.parent
    prompt_template_hash = sha256_file(root / "eval/prompts.py")
    parser_hash = sha256_file(root / "eval/parsers.py")
    environment = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "trl", "peft", "unsloth", "vllm", "flashinfer-python")
    }
    manifest = {
        "schema_version": "pt-e0-prediction-v1",
        **partial_identity,
        "predictions_sha256": prediction_hash,
        "n": len(candidates),
        "prompt_template_sha256": prompt_template_hash,
        "parser_sha256": parser_hash,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "resumed_from_partial": bool(existing_rows),
        "gpu_free_gib_at_start": round(free_bytes / 1024**3, 3),
        "gpu_total_gib": round(total_bytes / 1024**3, 3),
        "environment": environment,
        "generation_latency_seconds": {
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
        },
    }
    write_json(out_dir / "prediction_manifest.json", manifest)
    (out_dir / "predictions.sha256").write_text(f"{prediction_hash}  predictions.jsonl\n", encoding="ascii")
    print(json.dumps({"benchmark": benchmark, "n": len(candidates), "predictions_sha256": prediction_hash, "path": str(final_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
