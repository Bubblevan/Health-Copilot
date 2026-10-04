from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_base_eval as base
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams


POSTTRAIN_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = base.DATA_ROOT
RUNS = base.RUNS
MODEL_DIR = base.MODEL_DIR
PROTOCOL_PATH = POSTTRAIN_ROOT / "manifests/eval/eval_protocol.json"
CORE_PATH = POSTTRAIN_ROOT / "manifests/eval/common_eval_core.json"
IDS_PATH = POSTTRAIN_ROOT / "manifests/eval/cmb_common1024_ids.json"
DATASET_MANIFEST_PATH = POSTTRAIN_ROOT / "manifests/eval/eval_dataset_manifest.json"
COMMON_CANDIDATE_PATH = DATA_ROOT / "eval/prepared/cmb_common1024/candidate_view.jsonl"
FULL_CANDIDATE_PATH = DATA_ROOT / "eval/prepared/cmb/candidate_view.jsonl"
FULL_PARTIAL_DIR = RUNS / "cmb_full_partial_stopped_after_352"
OUT_DIR = RUNS / "cmb_common1024"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_contract() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    core = read_json(CORE_PATH)
    ids_manifest = read_json(IDS_PATH)
    dataset_manifest = read_json(DATASET_MANIFEST_PATH)
    runtime = protocol["inference_runtime"]

    runtime_now = {
        "engine": "vllm",
        "version": importlib.metadata.version("vllm"),
        "flashinfer_python_version": importlib.metadata.version("flashinfer-python"),
        "cuda_home": base.CUDA_HOME,
        "ninja_package_version": importlib.metadata.version("ninja"),
        "ninja_binary_version": subprocess.check_output(["ninja", "--version"], text=True).strip(),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "batch_size": base.BATCH_SIZE,
        "max_model_len": base.MAX_MODEL_LEN,
        "max_num_seqs": base.MAX_NUM_SEQS,
        "max_num_batched_tokens": base.MAX_NUM_BATCHED_TOKENS,
        "gpu_memory_utilization": base.GPU_MEMORY_UTILIZATION,
        "prefix_caching": False,
        "attention_backend": "auto",
        "seed": base.SEED,
    }
    if runtime != runtime_now:
        raise ValueError("Runtime differs from the frozen PT-E0 v4 evaluation protocol")
    if protocol["schema_version"] != "pt-e0-eval-protocol-v4-half-vram-common-core":
        raise ValueError("Expected frozen PT-E0 protocol v4")
    if protocol["candidate_generation_code_sha256"] != sha256_file(POSTTRAIN_ROOT / "scripts/run_base_eval.py"):
        raise ValueError("Shared Base candidate generator has changed since protocol freeze")
    if protocol["prompt_builder_sha256"] != sha256_file(POSTTRAIN_ROOT / "eval/prompts.py"):
        raise ValueError("Frozen prompt builder hash mismatch")
    if protocol["parser_sha256"] != sha256_file(POSTTRAIN_ROOT / "eval/parsers.py"):
        raise ValueError("Frozen parser hash mismatch")
    if protocol["cmb_scoring_code_sha256"] != sha256_file(POSTTRAIN_ROOT / "eval/cmb_scoring.py"):
        raise ValueError("Frozen CMB scorer hash mismatch")

    common_hash = core["benchmarks"]["CMB-COMMON-1024"]["candidate_view_sha256"]
    full_hash = dataset_manifest["prepared_artifacts"]["cmb/candidate_view.jsonl"]
    if sha256_file(COMMON_CANDIDATE_PATH) != common_hash:
        raise ValueError("CMB common candidate view does not match its frozen hash")
    if sha256_file(FULL_CANDIDATE_PATH) != full_hash:
        raise ValueError("CMB full candidate view does not match its frozen hash")
    if sha256_file(IDS_PATH) != core["benchmarks"]["CMB-COMMON-1024"]["ids_manifest_sha256"]:
        raise ValueError("CMB common ID manifest does not match its frozen hash")

    common_rows = base.read_jsonl(COMMON_CANDIDATE_PATH)
    full_rows = base.read_jsonl(FULL_CANDIDATE_PATH)
    target_ids = {str(value) for value in ids_manifest["ids"]}
    common_by_id = {str(row["id"]): row for row in common_rows}
    full_by_id = {str(row["id"]): row for row in full_rows}
    if len(target_ids) != 1024 or set(common_by_id) != target_ids:
        raise ValueError("Common candidate view does not cover the frozen 1,024 IDs")
    if len(full_by_id) != len(full_rows) or not target_ids.issubset(full_by_id):
        raise ValueError("CMB full candidate view has duplicate IDs or is missing common IDs")
    for row_id in target_ids:
        if common_by_id[row_id] != full_by_id[row_id]:
            raise ValueError(f"Common candidate changed the full-view prompt/metadata for {row_id}")
        row = common_by_id[row_id]
        if any(key in row for key in ("answer", "gold", "reference", "right_option", "label")):
            raise ValueError(f"Common candidate view contains a scorer-only field: {row_id}")
    return common_rows, common_by_id, full_by_id, (protocol, core, ids_manifest, dataset_manifest)


def generation_config() -> dict[str, Any]:
    return {
        "temperature": 0,
        "do_sample": False,
        "top_p": 1,
        "max_new_tokens": base.MAX_NEW_TOKENS,
        "enable_thinking": True,
        "backend": "vllm-greedy",
        "batch_size": base.BATCH_SIZE,
        "system_prompt": base.SYSTEM_PROMPT,
        "system_prompt_sha256": base.SYSTEM_PROMPT_SHA256,
    }


def runtime_identity() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    return protocol["inference_runtime"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-partial", type=Path, default=FULL_PARTIAL_DIR / "predictions.jsonl.partial")
    parser.add_argument("--source-manifest", type=Path, default=FULL_PARTIAL_DIR / "partial_run_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    candidates, candidate_by_id, full_by_id, frozen = load_contract()
    protocol, core, ids_manifest, dataset_manifest = frozen
    source_manifest = read_json(args.source_manifest)
    source_path = args.source_partial
    source_hash = sha256_file(source_path)
    source_manifest_hash = sha256_file(args.source_manifest)
    full_candidate_hash = dataset_manifest["prepared_artifacts"]["cmb/candidate_view.jsonl"]
    protocol_hash = sha256_file(PROTOCOL_PATH)
    if source_manifest.get("candidate_view_sha256") != full_candidate_hash:
        raise ValueError("Reusable partial was not generated from the frozen full CMB candidate view")
    if source_manifest.get("eval_protocol_sha256") != protocol_hash:
        raise ValueError("Reusable partial was not generated under the currently frozen protocol")
    if source_manifest.get("checkpoint") != "Qwen/Qwen3-8B" or source_manifest.get("model_revision") != base.MODEL_REVISION:
        raise ValueError("Reusable partial checkpoint identity differs from Qwen3-8B Base")
    if source_manifest.get("adapter") is not None:
        raise ValueError("Reusable partial unexpectedly has an adapter")
    if source_manifest.get("generation") != generation_config():
        raise ValueError("Reusable partial generation settings differ from the common evaluation")

    source_rows = base.read_jsonl(source_path)
    common_ids = set(candidate_by_id)
    source_by_id: dict[str, dict[str, Any]] = {}
    for pred in source_rows:
        row_id = str(pred["id"])
        if row_id not in full_by_id or row_id in source_by_id:
            raise ValueError(f"Reusable full-CMB predictions have invalid/duplicate ID {row_id}")
        source_by_id[row_id] = pred
        raw = pred["raw_generation"]
        if pred["raw_output_hash"] != sha256_bytes(raw.encode("utf-8")):
            raise ValueError(f"Reusable prediction output hash mismatch for {row_id}")
        candidate = full_by_id[row_id]
        parsed = base.extract_mcq_answer(
            raw,
            set(candidate["valid_options"]),
            allow_multiple=candidate["question_type"] == "多项选择题",
        )
        if parsed != pred.get("parsed_answer") or bool(parsed) != bool(pred.get("parse_success")):
            raise ValueError(f"Reusable prediction parse result differs from frozen parser for {row_id}")
    reused = {row_id: source_by_id[row_id] for row_id in common_ids.intersection(source_by_id)}
    if not reused:
        raise ValueError("No stopped full-CMB predictions overlap the frozen Common-1024 subset")

    model_path = MODEL_DIR
    model_manifest_path = POSTTRAIN_ROOT / "manifests/model/qwen3-8b-base.json"
    model_manifest_hash = sha256_file(model_manifest_path)
    model_artifacts = base.model_artifact_manifest(model_path)
    generation = generation_config()
    runtime = runtime_identity()
    runner_hash = sha256_file(Path(__file__))
    partial_identity = {
        "schema_version": "pt-e0-cmb-common-subset-run-v1",
        "checkpoint": "Qwen/Qwen3-8B",
        "model_revision": base.MODEL_REVISION,
        "base_model_revision": base.MODEL_REVISION,
        "model_path": str(model_path),
        "model_artifacts": model_artifacts,
        "base_model_manifest_sha256": model_manifest_hash,
        "adapter": None,
        "benchmark": "CMB-COMMON-1024",
        "candidate_view_sha256": core["benchmarks"]["CMB-COMMON-1024"]["candidate_view_sha256"],
        "full_candidate_view_sha256": full_candidate_hash,
        "evaluation_ids_sha256": ids_manifest["ids_sequence_sha256"],
        "ids_manifest_sha256": sha256_file(IDS_PATH),
        "eval_protocol_sha256": protocol_hash,
        "generation": generation,
        "inference_runtime": runtime,
        "runner_sha256": runner_hash,
        "reused_source_prediction_sha256": source_hash,
        "reused_source_partial_manifest_sha256": source_manifest_hash,
        "reused_prediction_count": len(reused),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "predictions.jsonl"
    partial_path = args.output_dir / "predictions.jsonl.partial"
    partial_meta_path = args.output_dir / "partial_run_manifest.json"
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite common predictions: {final_path}")
    if partial_meta_path.exists():
        if read_json(partial_meta_path) != partial_identity:
            raise ValueError("Existing Common-1024 partial run identity differs")
        if not partial_path.is_file():
            raise ValueError("Common-1024 partial manifest exists without partial predictions")
    else:
        if partial_path.exists():
            raise ValueError("Common-1024 partial predictions exist without a frozen partial manifest")
        ordered_reused = [reused[str(row["id"])] for row in candidates if str(row["id"]) in reused]
        write_jsonl(partial_path, ordered_reused)
        write_json(partial_meta_path, partial_identity)

    existing_rows = base.read_jsonl(partial_path)
    existing_by_id = {str(row["id"]): row for row in existing_rows}
    candidate_ids = [str(row["id"]) for row in candidates]
    if len(existing_by_id) != len(existing_rows) or not set(existing_by_id).issubset(set(candidate_ids)):
        raise ValueError("Common-1024 partial contains duplicate or unexpected IDs")
    done_ids = set(existing_by_id)
    if not set(reused).issubset(done_ids):
        raise ValueError("Previously completed overlapping full-CMB rows were not reused")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 20 * 1024**3:
        raise RuntimeError(f"Refusing to load Qwen3-8B: only {free_bytes / 1024**3:.1f} GiB GPU memory is free")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    chat_template_hash = hashlib.sha256((tokenizer.chat_template or "").encode("utf-8")).hexdigest()
    if chat_template_hash != protocol["candidate_chat_template_sha256"]:
        raise ValueError("Checkpoint chat template differs from frozen Qwen3 evaluation template")

    started_at = datetime.now(timezone.utc).isoformat()
    llm = base.LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        seed=base.SEED,
        gpu_memory_utilization=base.GPU_MEMORY_UTILIZATION,
        max_model_len=base.MAX_MODEL_LEN,
        max_num_seqs=base.MAX_NUM_SEQS,
        max_num_batched_tokens=base.MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=False,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=base.MAX_NEW_TOKENS,
        seed=base.SEED,
        skip_special_tokens=False,
    )
    latencies = [float(row["generation_latency_seconds"]) for row in existing_rows]
    pending = [row for row in candidates if str(row["id"]) not in done_ids]
    completed_this_run = 0
    with partial_path.open("a", encoding="utf-8", buffering=1) as output:
        for batch_start in range(0, len(pending), base.BATCH_SIZE):
            batch = pending[batch_start : batch_start + base.BATCH_SIZE]
            prompts = []
            for row in batch:
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
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
                raw = tokenizer.decode(new_ids, skip_special_tokens=False)
                final = base.final_response(raw)
                valid_options = row["valid_options"]
                if not isinstance(valid_options, list) or not valid_options:
                    raise ValueError(f"Candidate {row['id']} lacks valid option metadata")
                parsed = base.extract_mcq_answer(
                    raw,
                    set(valid_options),
                    allow_multiple=row["question_type"] == "多项选择题",
                )
                metrics = request.metrics
                if metrics is not None and metrics.last_token_ts is not None:
                    latency = max(0.0, metrics.last_token_ts - metrics.arrival_time)
                else:
                    latency = batch_elapsed
                result = {
                    "id": str(row["id"]),
                    "raw_generation": raw,
                    "final_answer": final,
                    "parse_success": bool(parsed),
                    "parsed_answer": parsed,
                    "raw_output_hash": sha256_bytes(raw.encode("utf-8")),
                    "output_tokens": len(new_ids),
                    "generation_latency_seconds": latency,
                }
                output.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                done_ids.add(str(row["id"]))
                latencies.append(latency)
                completed_this_run += 1
            output.flush()
            print(json.dumps({
                "benchmark": "CMB-COMMON-1024",
                "completed_this_run": completed_this_run,
                "completed_total": len(done_ids),
                "reused_from_full_partial": len(reused),
                "total": len(candidates),
                "batch_elapsed_seconds": round(batch_elapsed, 2),
            }), flush=True)

    if done_ids != set(candidate_ids):
        raise RuntimeError(f"Common prediction IDs do not match frozen IDs: {len(done_ids)} != {len(candidate_ids)}")
    ordered_rows = [existing_by_id.get(row_id) for row_id in candidate_ids]
    generated_rows = base.read_jsonl(partial_path)
    generated_by_id = {str(row["id"]): row for row in generated_rows}
    ordered_rows = [generated_by_id[row_id] for row_id in candidate_ids]
    write_jsonl(partial_path, ordered_rows)
    partial_path.replace(final_path)
    prediction_hash = sha256_file(final_path)
    environment = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "trl", "peft", "unsloth", "vllm", "flashinfer-python")
    }
    manifest = {
        **partial_identity,
        "predictions_sha256": prediction_hash,
        "n": len(candidates),
        "prompt_template_sha256": sha256_file(POSTTRAIN_ROOT / "eval/prompts.py"),
        "parser_sha256": sha256_file(POSTTRAIN_ROOT / "eval/parsers.py"),
        "chat_template_sha256": chat_template_hash,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "resumed_from_partial": bool(existing_rows),
        "reused_prediction_ids": sorted(reused),
        "gpu_free_gib_at_start": round(free_bytes / 1024**3, 3),
        "gpu_total_gib": round(total_bytes / 1024**3, 3),
        "environment": environment,
        "generation_latency_seconds": {
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)],
        },
    }
    write_json(args.output_dir / "prediction_manifest.json", manifest)
    (args.output_dir / "predictions.sha256").write_text(
        f"{prediction_hash}  predictions.jsonl\n", encoding="ascii"
    )
    print(json.dumps({
        "benchmark": "CMB-COMMON-1024",
        "n": len(candidates),
        "reused_from_full_partial": len(reused),
        "predictions_sha256": prediction_hash,
        "path": str(final_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
