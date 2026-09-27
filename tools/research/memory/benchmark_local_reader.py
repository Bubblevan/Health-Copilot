"""Measure a local llama.cpp reader with a synthetic, loopback-only prompt."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx


def _gpu_memory_used_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(completed.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.median(values) if values else None


def run(
    *,
    mode: str,
    runs: int,
    warmup_runs: int,
    target_prompt_tokens: int,
    output_length_cap_tokens: int,
    base_url: str,
    hold_embedding_model: bool = False,
) -> dict[str, Any]:
    parsed = httpx.URL(base_url)
    if parsed.scheme != "http" or parsed.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Reader benchmark accepts a loopback HTTP endpoint only")
    base = base_url.rstrip("/")
    api_base = base if base.endswith("/v1") else f"{base}/v1"
    server_base = api_base[:-3]
    memory = httpx.Client(timeout=3600, trust_env=False)
    slots = memory.get(f"{server_base}/slots")
    slots.raise_for_status()
    slot_data = slots.json()
    if isinstance(slot_data, dict):
        slot_data = [slot_data]
    contexts = [
        slot.get("n_ctx", slot.get("n_ctx_slot", slot.get("context_length")))
        for slot in slot_data if isinstance(slot, dict)
    ]
    max_model_length = min((value for value in contexts if isinstance(value, int)), default=None)

    system = "Write an extended paragraph about swimming safety, using full sentences and no preamble."
    sentence = "Synthetic note: Alice likes swimming; this text is only for local throughput measurement. "
    def make_prompt(run_index: int) -> tuple[list[dict[str, str]], int]:
        low, high = 0, max(target_prompt_tokens * 3, 1)
        selected = None
        selected_tokens = 0
        while low <= high:
            count = (low + high) // 2
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Synthetic trial {run_index:04d}.\nMemory context:\n"
                        f"{sentence * count}\n\n"
                        "Task: Explain healthy swimming habits with practical detail."
                    ),
                },
            ]
            rendered = memory.post(
                f"{server_base}/apply-template",
                json={"messages": messages, "add_generation_prompt": True,
                      "chat_template_kwargs": {"enable_thinking": False}},
            )
            rendered.raise_for_status()
            tokenized = memory.post(
                f"{server_base}/tokenize",
                json={"content": rendered.json()["prompt"], "add_special": False},
            )
            tokenized.raise_for_status()
            token_count = len(tokenized.json()["tokens"])
            if token_count <= target_prompt_tokens:
                selected = messages
                selected_tokens = token_count
                low = count + 1
            else:
                high = count - 1
        if selected is None or selected_tokens < max(256, target_prompt_tokens // 2):
            raise RuntimeError(f"Synthetic prompt reached only {selected_tokens} tokens")
        return selected, selected_tokens

    first_prompt, selected_prompt_tokens = make_prompt(0)
    embedding_runtime = None
    embedding_probe = None
    if hold_embedding_model:
        memeval_src = Path(__file__).resolve().parents[3].parent / "external" / "memory" / "MemEval" / "src"
        sys.path.insert(0, str(memeval_src))
        from agents_memory.healthcopilot_embedding import get_local_embedding_runtime

        embedding_path = os.environ.get("HC_LOCAL_EMBEDDING_PATH")
        if not embedding_path:
            raise RuntimeError("HC_LOCAL_EMBEDDING_PATH is required for the concurrent GPU probe")
        embedding_runtime = get_local_embedding_runtime(embedding_path, "cuda:0", "float16")
        sample_texts = [
            "Synthetic memory note: Alice likes swimming after work. " * 50
            for _ in range(8)
        ]
        _vectors, sample_tokens, sample_truncated, embedding_ms = embedding_runtime.encode(
            sample_texts, "document"
        )
        embedding_probe = {
            "model": embedding_runtime.artifact.repo,
            "device": embedding_runtime.artifact.device,
            "dtype": embedding_runtime.artifact.dtype,
            "batch_items": len(sample_texts),
            "input_tokens": sample_tokens,
            "truncated": sample_truncated,
            "encode_latency_ms": round(embedding_ms, 3),
            "gpu_memory_after_load_and_encode_mib": _gpu_memory_used_mib(),
        }
    results = []
    peak_gpu = _gpu_memory_used_mib()
    peak_lock = threading.Lock()
    stop_monitor = threading.Event()

    def monitor_gpu() -> None:
        nonlocal peak_gpu
        while not stop_monitor.wait(1.0):
            current = _gpu_memory_used_mib()
            if current is not None:
                with peak_lock:
                    peak_gpu = max(peak_gpu or 0, current)

    monitor = threading.Thread(target=monitor_gpu, daemon=True)
    monitor.start()
    try:
        for run_index in range(runs + warmup_runs):
            prompt, actual_prompt_tokens = (
                (first_prompt, selected_prompt_tokens)
                if run_index == 0 else make_prompt(run_index)
            )
            started = time.perf_counter()
            first_content_at = None
            chunks = []
            usage = None
            finish_reason = None
            with memory.stream(
                "POST",
                f"{api_base}/chat/completions",
                json={
                    "model": "health-memory-qwen3-8b",
                    "messages": prompt,
                    "temperature": 0,
                    "seed": 42,
                    "max_tokens": output_length_cap_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        continue
                    event = json.loads(payload)
                    if event.get("usage"):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if choices:
                        finish_reason = choices[0].get("finish_reason") or finish_reason
                        text = choices[0].get("delta", {}).get("content") or ""
                        if text:
                            if first_content_at is None:
                                first_content_at = time.perf_counter()
                            chunks.append(text)
            ended = time.perf_counter()
            first_content_at = first_content_at or ended
            completion_tokens = (
                usage.get("completion_tokens") if isinstance(usage, dict) else None
            )
            reported_prompt_tokens = (
                usage.get("prompt_tokens") if isinstance(usage, dict) else None
            )
            usage_source = "llama.cpp usage"
            if completion_tokens is None and chunks:
                tokenized_output = memory.post(
                    f"{server_base}/tokenize",
                    json={"content": "".join(chunks), "add_special": False},
                )
                tokenized_output.raise_for_status()
                completion_tokens = len(tokenized_output.json()["tokens"])
                usage_source = "llama.cpp tokenizer"
            inference_ms = (ended - started) * 1000
            generation_ms = max((ended - first_content_at) * 1000, 1.0)
            results.append({
                "run": run_index,
                "warmup": run_index < warmup_runs,
                "prompt_tokens": actual_prompt_tokens,
                "server_reported_prompt_tokens": reported_prompt_tokens,
                "prompt_token_count_matches_server": (
                    reported_prompt_tokens == actual_prompt_tokens
                    if isinstance(reported_prompt_tokens, int) else None
                ),
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason,
                "generation_truncated": finish_reason == "length",
                "token_usage_source": usage_source if completion_tokens is not None else "NOT_CAPTURED",
                "ttft_ms": round((first_content_at - started) * 1000, 3),
                "total_ms": round(inference_ms, 3),
                "generation_tokens_per_second": (
                    round(completion_tokens / (generation_ms / 1000), 3)
                    if isinstance(completion_tokens, int) and completion_tokens > 0 else None
                ),
                "text": "".join(chunks),
            })
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)
        if embedding_runtime is not None:
            embedding_runtime.close()
        memory.close()

    measured = [row for row in results if not row["warmup"]]
    prompt_counts_verified = bool(results) and max_model_length is not None and all(
        row.get("prompt_token_count_matches_server") is True
        and row["prompt_tokens"] + output_length_cap_tokens <= max_model_length
        for row in results
    )
    return {
        "schema_version": 1,
        "mode": mode,
        "kv_cache_type": "q4_0" if mode.endswith("-q4") else "q8_0",
        "endpoint": base_url,
        "prompt": "synthetic; no benchmark question or user data",
        "prompt_tokens": selected_prompt_tokens,
        "prompt_tokens_per_run": [row["prompt_tokens"] for row in results],
        "max_model_length": max_model_length,
        "max_prompt_plus_output_tokens": max(
            row["prompt_tokens"] for row in results
        ) + output_length_cap_tokens,
        "truncated": False if prompt_counts_verified else None,
        "prompt_counts_verified": prompt_counts_verified,
        "generation_truncated": any(row["generation_truncated"] for row in results),
        "warmup_runs": warmup_runs,
        "measured_runs": runs,
        "output_length_cap_tokens": output_length_cap_tokens,
        "fixed_length_measurement_valid": all(
            row.get("completion_tokens") == output_length_cap_tokens
            and row.get("finish_reason") == "length"
            for row in measured
        ),
        "median_ttft_ms": _median(measured, "ttft_ms"),
        "median_total_ms": _median(measured, "total_ms"),
        "median_generation_tokens_per_second": _median(measured, "generation_tokens_per_second"),
        "gpu_memory_peak_mib": peak_gpu,
        "embedding_resident_during_reader": hold_embedding_model,
        "embedding_probe": embedding_probe,
        "runs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cpu-kv", "cpu-kv-q4", "gpu-kv", "gpu-kv-q4"),
        required=True,
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--target-prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-length-cap-tokens", type=int, default=96)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hold-embedding-model", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup_runs < 0:
        parser.error("--runs must be positive and --warmup-runs nonnegative")
    result = run(
        mode=args.mode,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        target_prompt_tokens=args.target_prompt_tokens,
        output_length_cap_tokens=args.output_length_cap_tokens,
        base_url=args.base_url,
        hold_embedding_model=args.hold_embedding_model,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
