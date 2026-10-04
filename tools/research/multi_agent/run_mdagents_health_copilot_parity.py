#!/usr/bin/env python3
"""Compare the opt-in Health-Copilot graft with frozen local MDAgents outputs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from health_ai_copilot.multi_agent import (
    ClinicalReasoningSkill,
    LocalVllmProvider,
    MDAgentsStyleConfig,
    MedicalAgentRequest,
    MedicalAgentRuntime,
)

DEFAULT_REFERENCE = Path("/root/gpufree-share/results/mdagents-local/full_medqa/adaptive.jsonl")
DEFAULT_REFERENCE_MANIFEST = Path("/root/gpufree-share/results/mdagents-local/full_medqa/manifest.json")
DEFAULT_OUTPUT = Path("runs/multi_agent/mdagents-graft-parity-20261004")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accuracy(rows: list[dict], option_key: str) -> tuple[int, float]:
    correct = sum(row.get(option_key) == row.get("label") for row in rows)
    return correct, correct / len(rows) if rows else 0.0


def make_report(metrics: dict, output_dir: Path) -> str:
    paired = metrics["paired"]
    slices_table = "".join(
        f"| {name} | {row['n']} | {row['reference_accuracy']:.2%} | "
        f"{row['health_copilot_accuracy']:.2%} | {row['answer_agreement']:.2%} |\n"
        for name, row in metrics["complexity_slices"].items()
    )
    return f"""# Health-Copilot MDAgents Graft Parity Sample

Date (UTC): {metrics['completed_at_utc']}

This is a deterministic {metrics['sample_size']}-case sample from the pinned MedQA US test split. The comparison uses the same shuffled `model_question` strings and labels as the frozen local MDAgents adaptive JSONL. It does not rerun the reference arm or the full test split.

| Metric | Frozen MDAgents Adaptive | Health-Copilot graft | Delta |
|---|---:|---:|---:|
| Accuracy | {paired['reference_accuracy']:.2%} ({paired['reference_correct']}/{metrics['sample_size']}) | {paired['health_copilot_accuracy']:.2%} ({paired['health_copilot_correct']}/{metrics['sample_size']}) | {paired['accuracy_delta_pp']:+.2f} pp |
| Parse success | {paired['reference_parse_success_rate']:.2%} | {paired['health_copilot_parse_success_rate']:.2%} | {paired['parse_success_delta_pp']:+.2f} pp |
| Exact parsed-answer agreement | — | {paired['answer_agreement_all_cases']:.2%} | {paired['answer_agreement_count']}/{metrics['sample_size']} |
| Agreement when both parse | — | {paired['answer_agreement_both_parse']:.2%} | {paired['answer_agreement_both_parse_count']}/{paired['both_parse_count']} |
| Complexity-route agreement | — | {paired['complexity_route_agreement']:.2%} | {paired['route_agreement_count']}/{metrics['sample_size']} |

## Graft runtime

- Complexity distribution: `{json.dumps(metrics['health_copilot_complexity_distribution'], sort_keys=True)}`
- Mean model calls per case: {metrics['health_copilot_runtime']['mean_provider_calls']:.2f}
- Mean tokens per case: {metrics['health_copilot_runtime']['mean_total_tokens']:.1f}
- Mean latency: {metrics['health_copilot_runtime']['mean_latency_seconds']:.2f} s
- Wall time: {metrics['health_copilot_runtime']['wall_clock_seconds']:.1f} s; throughput {metrics['health_copilot_runtime']['cases_per_second']:.3f} cases/s
- Runtime failures: {metrics['health_copilot_runtime']['failure_count']}

## Route slices

| Complexity | n | Reference accuracy | Health-Copilot accuracy | Answer agreement |
|---|---:|---:|---:|---:|
{slices_table}

## Engineering parity targets

The prior handoff set sample-level targets of absolute accuracy delta ≤ 2 percentage points, parsed-answer agreement ≥ 85%, and complexity-route agreement ≥ 90%.

- Accuracy delta target: **{'PASS' if paired['accuracy_delta_abs_pp'] <= 2 else 'MISS'}**
- Answer agreement target: **{'PASS' if paired['answer_agreement_all_cases'] >= 0.85 else 'MISS'}**
- Route agreement target: **{'PASS' if paired['complexity_route_agreement'] >= 0.90 else 'MISS'}**
- Combined: **{'PASS' if paired['parity_targets_met'] else 'MISS'}**

A pass is evidence of parity on these sampled cases only. It does not prove all 1,273 cases would match; this run deliberately stops at the sample and does not rerun full MedQA in Health-Copilot.

## Artifacts

- `health_copilot.jsonl`: one checkpoint row per completed question, including raw answer and parsed option.
- `paired_results.json`: case-level comparison against frozen reference rows.
- `metrics.json`: denominators, metric definitions, and run configuration.
- `manifest.json`: model, data, and code identities.
"""


async def run(args: argparse.Namespace) -> None:
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "health_copilot.jsonl"
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics.json"
    paired_path = output_dir / "paired_results.json"
    report_path = output_dir / "report.md"

    reference_rows = read_jsonl(args.reference)
    if len(reference_rows) != 1273:
        raise SystemExit(f"expected 1273 frozen reference rows, found {len(reference_rows)}")
    if args.sample_size < 1 or args.sample_size > len(reference_rows):
        raise SystemExit("sample size must be between 1 and 1273")
    selected = sorted(random.Random(args.seed).sample(range(len(reference_rows)), args.sample_size))
    expected_config = {
        "sample_size": args.sample_size,
        "seed": args.seed,
        "served_model": args.served_model,
        "base_url": args.base_url,
        "temperature": 0.0,
        "no_think": True,
        "concurrency": args.concurrency,
        "selected_dataset_indices": selected,
        "reference_sha256": sha256_file(args.reference),
    }

    completed: dict[int, dict] = {}
    if result_path.exists():
        if not args.resume:
            raise SystemExit(f"output exists; pass --resume: {result_path}")
        for row in read_jsonl(result_path):
            completed[int(row["dataset_index"])] = row
        if not manifest_path.exists():
            raise SystemExit("cannot resume without the original manifest")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("configuration") != expected_config:
            raise SystemExit("resume configuration differs from the original run")
    elif args.resume:
        raise SystemExit("cannot resume because the checkpoint JSONL does not exist")

    ref_manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    code_files = [
        Path("src/health_ai_copilot/multi_agent/mdagents_style.py"),
        Path("src/health_ai_copilot/multi_agent/runtime.py"),
        Path("src/health_ai_copilot/multi_agent/providers.py"),
    ]
    manifest = {
        "experiment": "HEALTH_COPILOT_MDAGENTS_STYLE_PARITY_SAMPLE",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "configuration": expected_config,
        "reference": {
            "arm": "LOCAL_QWEN_MDAGENTS frozen Adaptive",
            "upstream_sha": ref_manifest["mdagents"]["upstream_sha"],
            "model_revision": ref_manifest["model"]["revision"],
            "full_rows": len(reference_rows),
            "reference_accuracy_full": ref_manifest["results"]["adaptive"]["metrics"]["accuracy"],
        },
        "health_copilot": {
            "branch": "multi-agent-mdagents-graft-20261003",
            "base_sha": "277a0c1897109110408de39d77e8741c8bb18f4c",
            "code_sha256": {str(path): sha256_file(path) for path in code_files},
        },
        "checkpoint_path": str(result_path),
        "checkpoint_rows_before_run": len(completed),
    }
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    provider = LocalVllmProvider(base_url=args.base_url, model=args.served_model)
    available_models = await asyncio.wait_for(provider.healthcheck(), timeout=15)
    if args.served_model not in available_models:
        await provider.close()
        raise SystemExit(f"server does not expose {args.served_model!r}: {available_models}")
    skill = ClinicalReasoningSkill(
        provider,
        config=MDAgentsStyleConfig(model_name=args.served_model, max_output_tokens=1024),
    )
    runtime = MedicalAgentRuntime(provider, clinical_reasoning_skill=skill)
    semaphore = asyncio.Semaphore(args.concurrency)
    began = time.perf_counter()

    async def evaluate(index: int) -> dict:
        reference = reference_rows[index]
        async with semaphore:
            started = time.perf_counter()
            try:
                execution = await runtime.execute(MedicalAgentRequest(
                    query=reference["model_question"],
                    request_id=reference["case_id"],
                ))
                trajectory = execution.trajectory
                return {
                    "case_id": reference["case_id"],
                    "dataset_index": index,
                    "seed": args.seed,
                    "served_model": args.served_model,
                    "temperature": 0.0,
                    "label": reference["label"],
                    "model_question": reference["model_question"],
                    "reference_option": reference.get("parsed_option"),
                    "reference_complexity": reference.get("complexity"),
                    "raw_model_output": trajectory.get("raw_model_output", ""),
                    "parsed_option": trajectory.get("parsed_option"),
                    "complexity": trajectory.get("complexity"),
                    "provider_calls": execution.provider_calls,
                    "tool_calls": execution.tool_calls,
                    "input_tokens": execution.input_tokens,
                    "output_tokens": execution.output_tokens,
                    "total_tokens": execution.input_tokens + execution.output_tokens,
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "failure_reason": trajectory.get("failure_reason"),
                    "runtime_status": "error" if trajectory.get("failure_reason") else "ok",
                }
            except Exception as exc:  # noqa: BLE001 - checkpoint every failed case explicitly
                return {
                    "case_id": reference["case_id"],
                    "dataset_index": index,
                    "seed": args.seed,
                    "served_model": args.served_model,
                    "temperature": 0.0,
                    "label": reference["label"],
                    "model_question": reference["model_question"],
                    "reference_option": reference.get("parsed_option"),
                    "reference_complexity": reference.get("complexity"),
                    "raw_model_output": "",
                    "parsed_option": None,
                    "complexity": "failed",
                    "provider_calls": 0,
                    "tool_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "failure_reason": type(exc).__name__,
                    "runtime_status": "error",
                }

    pending = [index for index in selected if index not in completed]
    append_mode = "a" if result_path.exists() else "w"
    with result_path.open(append_mode, encoding="utf-8") as stream:
        tasks = [asyncio.create_task(evaluate(index)) for index in pending]
        for future in asyncio.as_completed(tasks):
            row = await future
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            completed[int(row["dataset_index"])] = row
            print(json.dumps({
                "completed": len(completed), "total": args.sample_size,
                "index": row["dataset_index"], "status": row["runtime_status"],
                "calls": row["provider_calls"], "seconds": row["latency_seconds"],
            }, ensure_ascii=False), flush=True)
    wall = time.perf_counter() - began
    await provider.close()

    rows = [completed[index] for index in selected]
    paired = []
    for row in rows:
        paired.append({
            "case_id": row["case_id"],
            "dataset_index": row["dataset_index"],
            "label": row["label"],
            "reference_option": row.get("reference_option"),
            "health_copilot_option": row.get("parsed_option"),
            "reference_correct": row.get("reference_option") == row["label"],
            "health_copilot_correct": row.get("parsed_option") == row["label"],
            "answer_agreement": (row.get("reference_option") is not None and row.get("reference_option") == row.get("parsed_option")),
            "reference_complexity": row.get("reference_complexity"),
            "health_copilot_complexity": row.get("complexity"),
            "route_agreement": row.get("reference_complexity") == row.get("complexity"),
        })
    n = len(rows)
    ref_correct = sum(row["reference_correct"] for row in paired)
    hc_correct = sum(row["health_copilot_correct"] for row in paired)
    ref_parse = sum(row.get("reference_option") in {"A", "B", "C", "D"} for row in rows)
    hc_parse = sum(row.get("parsed_option") in {"A", "B", "C", "D"} for row in rows)
    answer_agreement = sum(row["answer_agreement"] for row in paired)
    both_parse_rows = [row for row in paired if row["reference_option"] in {"A", "B", "C", "D"} and row["health_copilot_option"] in {"A", "B", "C", "D"}]
    answer_agreement_both = sum(row["answer_agreement"] for row in both_parse_rows)
    route_agreement = sum(row["route_agreement"] for row in paired)
    ref_acc = ref_correct / n
    hc_acc = hc_correct / n
    delta_pp = (hc_acc - ref_acc) * 100
    by_complexity = {}
    for name in ("basic", "intermediate", "advanced"):
        group = [row for row in paired if row["reference_complexity"] == name]
        if group:
            by_complexity[name] = {
                "n": len(group),
                "reference_accuracy": sum(row["reference_correct"] for row in group) / len(group),
                "health_copilot_accuracy": sum(row["health_copilot_correct"] for row in group) / len(group),
                "answer_agreement": sum(row["answer_agreement"] for row in group) / len(group),
            }
    complexities = {name: sum(row.get("complexity") == name for row in rows)
                    for name in ("basic", "intermediate", "advanced", "safety_routed", "failed")}
    latencies = [row["latency_seconds"] for row in rows]
    total_tokens = [row["total_tokens"] for row in rows]
    provider_calls = [row["provider_calls"] for row in rows]
    metrics = {
        "sample_size": n,
        "created_at_utc": manifest["created_at_utc"],
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "configuration": expected_config,
        "paired": {
            "reference_correct": ref_correct,
            "reference_accuracy": ref_acc,
            "health_copilot_correct": hc_correct,
            "health_copilot_accuracy": hc_acc,
            "accuracy_delta_pp": delta_pp,
            "accuracy_delta_abs_pp": abs(delta_pp),
            "reference_parse_success_count": ref_parse,
            "reference_parse_success_rate": ref_parse / n,
            "health_copilot_parse_success_count": hc_parse,
            "health_copilot_parse_success_rate": hc_parse / n,
            "parse_success_delta_pp": (hc_parse / n - ref_parse / n) * 100,
            "answer_agreement_count": answer_agreement,
            "answer_agreement_all_cases": answer_agreement / n,
            "both_parse_count": len(both_parse_rows),
            "answer_agreement_both_parse_count": answer_agreement_both,
            "answer_agreement_both_parse": answer_agreement_both / len(both_parse_rows) if both_parse_rows else 0.0,
            "route_agreement_count": route_agreement,
            "complexity_route_agreement": route_agreement / n,
            "parity_targets_met": abs(delta_pp) <= 2 and answer_agreement / n >= 0.85 and route_agreement / n >= 0.90,
        },
        "health_copilot_complexity_distribution": complexities,
        "complexity_slices": by_complexity,
        "health_copilot_runtime": {
            "failure_count": sum(row["runtime_status"] == "error" for row in rows),
            "mean_provider_calls": statistics.fmean(provider_calls) if rows else 0.0,
            "mean_total_tokens": statistics.fmean(total_tokens) if rows else 0.0,
            "mean_latency_seconds": statistics.fmean(latencies) if rows else 0.0,
            "wall_clock_seconds": wall,
            "cases_per_second": n / wall if wall else 0.0,
        },
        "metric_definitions": {
            "accuracy": "correct parsed option count divided by all sampled cases; parse failures count incorrect",
            "parse_success_rate": "cases with parsed_option in A-D divided by all sampled cases",
            "answer_agreement_all_cases": "exact parsed-option equality over all cases; any parse failure counts as disagreement",
            "answer_agreement_both_parse": "exact option equality over cases where both arms parse A-D",
            "complexity_route_agreement": "exact basic/intermediate/advanced equality over all cases; failures count as disagreement unless both match",
        },
    }
    paired_path.write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(make_report(metrics, output_dir), encoding="utf-8")
    manifest["completed_at_utc"] = metrics["completed_at_utc"]
    manifest["result_sha256"] = {
        "health_copilot.jsonl": sha256_file(result_path),
        "paired_results.json": sha256_file(paired_path),
        "metrics.json": sha256_file(metrics_path),
        "report.md": sha256_file(report_path),
    }
    manifest["metrics"] = metrics["paired"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["paired"], ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model", default="qwen3-8b-local")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.concurrency < 1:
        raise SystemExit("concurrency must be positive")
    asyncio.run(run(arguments))
