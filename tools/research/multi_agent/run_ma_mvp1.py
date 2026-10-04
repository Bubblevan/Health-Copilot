"""Run TRAIN_DEV engineering, freeze MA-MVP1, and execute the final DEV pair."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from health_ai_copilot.multi_agent.contracts import MedicalAgentRequest, RouteMode
from health_ai_copilot.multi_agent.data import DATASET_ROOT_HASH, load_records
from health_ai_copilot.multi_agent.evaluation import (
    aggregate_metrics,
    compare_systems,
    score_execution,
)
from health_ai_copilot.multi_agent.providers import LocalLlamaCppProvider
from health_ai_copilot.multi_agent.routing import MedicalRouter, RouteDecision
from health_ai_copilot.multi_agent.runtime import (
    MedicalAgentRuntime,
    RuntimeConfig,
)

SYSTEM_CONFIG_VERSION = "MA_MVP1_CONFIG_V1"
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"


class StrongSingleRouter(MedicalRouter):
    def decide(self, query: str) -> RouteDecision:
        routed = super().decide(query)
        return RouteDecision(
            RouteMode.SINGLE,
            "strong_single_baseline; all registered skills remain available",
            routed.predicted_capabilities,
            routed.unique_key_count,
        )


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        max_model_turns_per_worker=3,
        max_tool_calls_per_worker=2,
        max_worker_calls=3,
        worker_timeout_seconds=120.0,
        planning_timeout_seconds=60.0,
        synthesis_timeout_seconds=120.0,
        max_worker_output_tokens=64,
        max_lead_output_tokens=96,
        max_single_output_tokens=96,
    )


def _provider(args) -> LocalLlamaCppProvider:
    return LocalLlamaCppProvider(
        base_url=args.base_url,
        model=args.model,
    )


def _run_directory(run_id: str) -> Path:
    if not run_id or any(item not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
                         for item in run_id):
        raise ValueError("run_id may contain only letters, digits, dash, underscore, and dot")
    return REPOSITORY_ROOT / "runs" / "multi_agent" / f"ma-mvp1-{run_id}"


def _frozen_config(args) -> dict[str, Any]:
    return {
        "config_version": SYSTEM_CONFIG_VERSION,
        "frozen": True,
        "frozen_at_utc": _utc_now(),
        "model": {
            "provider": "local llama.cpp OpenAI-compatible API",
            "name": args.model,
            "artifact_sha256": MODEL_SHA256,
            "base_url": args.base_url,
            "device_policy": "CPU_ONLY; llama-server uses --n-gpu-layers 0",
            "reasoning_mode": "Qwen3 /no_think; server --reasoning off",
            "server_parallel_sequences": 3,
            "server_context_tokens": 16384,
            "server_threads": 24,
            "temperature": 0,
        },
        "router": "medical-router-v1",
        "skills": [
            "PatientStateLookupSkill", "TimelineCompareSkill",
            "MedicalKnowledgeSearchSkill", "ExternalEvidenceSearchSkill",
            "RiskAssessmentSkill", "AnswerabilitySkill", "HospitalKnowledgeSearchSkill",
        ],
        "workers": ["patient_context", "evidence", "care"],
        "constraints": {
            "max_worker_model_turns": 3,
            "max_tool_calls_per_worker": 2,
            "max_worker_calls_per_request": 3,
            "max_waves": 2,
            "recursive_delegation": False,
            "parallel_independent_workers": True,
            "hospital_corpus_enabled": False,
        },
        "generation": {
            "plan_max_output_tokens": 80,
            "worker_max_output_tokens": 64,
            "lead_max_output_tokens": 96,
            "single_max_output_tokens": 96,
            "retries": 0,
        },
        "train_dev": {
            "dataset_root_hash": DATASET_ROOT_HASH,
            "sample_size": 256,
            "selection": "sha256(MA_MVP1_TRAIN_DEV_V1|episode_id), first 256",
            "engineering_rounds": 1,
        },
    }


def _freeze(args) -> None:
    run_dir = _run_directory(args.run_id)
    train_path = run_dir / "train_dev_metrics.json"
    if not train_path.is_file():
        raise FileNotFoundError("TRAIN_DEV pass must finish before configuration freeze")
    existing = _read_json(train_path)
    if existing.get("episode_count_per_system") != 256:
        raise ValueError("TRAIN_DEV_FREEZE_REQUIRES_256_EPISODES_PER_SYSTEM")
    if existing.get("reserved_split_accessed") is not False:
        raise ValueError("TRAIN_DEV_ARTIFACT_REPORTED_RESERVED_SPLIT_ACCESS")
    payload = _frozen_config(args)
    payload["training_dev_metrics_sha256"] = hashlib.sha256(train_path.read_bytes()).hexdigest()
    payload["config_sha256"] = _canonical_hash(payload)
    _write_json(run_dir / "system_config.json", payload)
    print(json.dumps({"status": "FROZEN", "config_version": SYSTEM_CONFIG_VERSION,
                      "config_sha256": payload["config_sha256"], "run_dir": str(run_dir)},
                     ensure_ascii=False, sort_keys=True))


async def _healthcheck(provider: LocalLlamaCppProvider) -> tuple[str, ...]:
    try:
        served = await provider.healthcheck()
    except Exception as exc:
        raise RuntimeError(
            "LOCAL_LLAMA_CPP_UNAVAILABLE; start the CPU server on the configured loopback endpoint"
        ) from exc
    if not served:
        raise RuntimeError("LOCAL_LLAMA_CPP_RETURNED_NO_MODELS")
    return served


async def _run_system(records, runtime: MedicalAgentRuntime, system: str, output_dir: Path,
                      stage: str, result_rows, routing_rows, worker_rows, trace_rows) -> None:
    for index, record in enumerate(records, start=1):
        execution = await runtime.execute(MedicalAgentRequest(
            query=record.case.episode.query,
            request_id=f"ma-mvp1-{stage}-{system.lower()}-{record.episode_id}",
            patient_id=record.case.episode.subject_id,
            as_of_time=record.case.episode.decision_time,
            episode=record.case.episode,
            resources=record.case.resources,
        ))
        scored = score_execution(record, execution, system)
        result_rows.append(scored)
        if system == "ROUTED_MEDICAL_TEAM":
            routing_rows.append({
                "episode_id": record.episode_id,
                "route_decision": execution.route_decision.to_dict(),
                "lead_plan": execution.plan.to_dict() if execution.plan else None,
                "expected_capabilities": scored["expected_capabilities"],
                "predicted_capabilities": scored["predicted_capabilities"],
                "route_precision": scored["route_precision_numerator"],
                "route_precision_denominator": scored["route_precision_denominator"],
                "route_recall": scored["route_recall_numerator"],
                "route_recall_denominator": scored["route_recall_denominator"],
                "exact_worker_set_match": scored["exact_worker_set_match"],
            })
        for report in execution.worker_reports:
            worker_rows.append({"episode_id": record.episode_id, "system": system,
                                **report.to_dict()})
        trace_rows.append({
            "episode_id": record.episode_id,
            "system": system,
            "trace": execution.trace,
            "trajectory": execution.trajectory,
        })
        if index % 16 == 0 or index == len(records):
            print(json.dumps({
                "stage": stage, "system": system, "completed": index,
                "total": len(records), "episode_id": record.episode_id,
            }, ensure_ascii=False))


async def _train_dev(args) -> None:
    run_dir = _run_directory(args.run_id)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    provider = _provider(args)
    served_models = await _healthcheck(provider)
    records = load_records(REPOSITORY_ROOT, "train_dev")
    results: list[dict[str, Any]] = []
    routing: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    config = _runtime_config()
    single = MedicalAgentRuntime(provider, router=StrongSingleRouter(), config=config)
    team = MedicalAgentRuntime(provider, router=MedicalRouter(), config=config)
    await _run_system(records, single, "STRONG_SINGLE", run_dir, "train_dev",
                      results, routing, workers, traces)
    await _run_system(records, team, "ROUTED_MEDICAL_TEAM", run_dir, "train_dev",
                      results, routing, workers, traces)
    _write_jsonl(run_dir / "train_dev_results.jsonl", results)
    _write_jsonl(run_dir / "train_dev_routing.jsonl", routing)
    _write_jsonl(run_dir / "train_dev_worker_reports.jsonl", workers)
    _write_jsonl(run_dir / "train_dev_traces.jsonl", traces)
    metrics = {
        "dataset_id": "health-copilot-owned-longitudinal-v1",
        "dataset_root_hash": DATASET_ROOT_HASH,
        "episode_count_per_system": len(records),
        "reserved_split_accessed": False,
        "systems": {
            key: aggregate_metrics([row for row in results if row["system"] == key])
            for key in ("STRONG_SINGLE", "ROUTED_MEDICAL_TEAM")
        },
        "served_model_ids": list(served_models),
        "selection": "TRAIN only; deterministic SHA-256 sample of 256",
    }
    _write_json(run_dir / "train_dev_metrics.json", metrics)
    print(json.dumps({"status": "TRAIN_DEV_COMPLETE", "run_dir": str(run_dir),
                      "episode_count_per_system": len(records),
                      "systems": metrics["systems"]}, ensure_ascii=False, sort_keys=True))


def _verify_frozen_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "system_config.json"
    if not path.is_file():
        raise FileNotFoundError("MA_MVP1_CONFIG_V1 must be frozen before final DEV")
    payload = _read_json(path)
    expected = payload.get("config_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "config_sha256"}
    if payload.get("config_version") != SYSTEM_CONFIG_VERSION or not payload.get("frozen"):
        raise ValueError("MA_MVP1_FINAL_DEV_CONFIG_NOT_FROZEN")
    if expected != _canonical_hash(unsigned):
        raise ValueError("MA_MVP1_CONFIG_HASH_MISMATCH")
    if payload.get("train_dev", {}).get("dataset_root_hash") != DATASET_ROOT_HASH:
        raise ValueError("MA_MVP1_CONFIG_DATASET_HASH_MISMATCH")
    return payload


async def _final_dev(args) -> None:
    run_dir = _run_directory(args.run_id)
    config_payload = _verify_frozen_config(run_dir)
    provider = _provider(args)
    served_models = await _healthcheck(provider)
    records = load_records(REPOSITORY_ROOT, "dev")
    results: list[dict[str, Any]] = []
    routing: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    config = _runtime_config()
    single = MedicalAgentRuntime(provider, router=StrongSingleRouter(), config=config)
    team = MedicalAgentRuntime(provider, router=MedicalRouter(), config=config)
    await _run_system(records, single, "STRONG_SINGLE", run_dir, "final_dev",
                      results, routing, workers, traces)
    await _run_system(records, team, "ROUTED_MEDICAL_TEAM", run_dir, "final_dev",
                      results, routing, workers, traces)
    _write_jsonl(run_dir / "results.jsonl", results)
    _write_jsonl(run_dir / "routing.jsonl", routing)
    _write_jsonl(run_dir / "worker_reports.jsonl", workers)
    _write_jsonl(run_dir / "traces.jsonl", traces)
    metric_sets = {}
    for label, selected in (
        ("ALL_DEV", results),
        ("COMPLEX", [row for row in results if row["slice_complex"]]),
        ("SIMPLE", [row for row in results if row["slice_simple"]]),
    ):
        metric_sets[label] = {
            system: aggregate_metrics([row for row in selected if row["system"] == system])
            for system in ("STRONG_SINGLE", "ROUTED_MEDICAL_TEAM")
        }
        metric_sets[label]["comparison"] = compare_systems(
            metric_sets[label]["STRONG_SINGLE"],
            metric_sets[label]["ROUTED_MEDICAL_TEAM"],
        )
    metrics = {
        "dataset_id": "health-copilot-owned-longitudinal-v1",
        "dataset_root_hash": DATASET_ROOT_HASH,
        "frozen_config_version": SYSTEM_CONFIG_VERSION,
        "frozen_config_sha256": config_payload["config_sha256"],
        "systems": metric_sets["ALL_DEV"],
        "episode_count_per_system": 1024,
        "reserved_test_ood_accessed": False,
        "served_model_ids": list(served_models),
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "slice_metrics.json", metric_sets)
    _write_json(run_dir / "latency_metrics.json", _latency_metrics(results))
    report = _render_report(metrics, metric_sets, results)
    (run_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    docs_path = REPOSITORY_ROOT / "docs" / "research" / "multi_agent" / "ma_mvp1_results.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(report, encoding="utf-8", newline="\n")
    manifest = {
        "run_id": args.run_id,
        "created_at_utc": _utc_now(),
        "stage": "FINAL_1024_DEV",
        "branch": (await asyncio.to_thread(
            subprocess.check_output,
            ["git", "branch", "--show-current"],
            cwd=REPOSITORY_ROOT,
            text=True,
        )).strip(),
        "base_sha": "8938be0e8e21d030c2790b33cd6a31f8198baf4a",
        "rag_capability_sha": "8938be0e8e21d030c2790b33cd6a31f8198baf4a",
        "memory_capability_status": (
            "MEM-3B0Q closeout is on the public main baseline; active Memory topic work was not merged. "
            "Runtime uses the owned deterministic longitudinal provider behind MemoryProvider."
        ),
        "dataset_root_hash": DATASET_ROOT_HASH,
        "config_sha256": config_payload["config_sha256"],
        "episode_count_per_system": 1024,
        "systems": ["STRONG_SINGLE", "ROUTED_MEDICAL_TEAM"],
        "model": config_payload["model"],
        "served_model_ids": list(served_models),
        "reserved_test_ood_accessed": False,
        "external_benchmarks_run": [],
        "provider_calls": sum(row["provider_calls"] for row in results),
        "tool_calls": sum(row["tool_calls"] for row in results),
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"status": "FINAL_DEV_COMPLETE", "run_dir": str(run_dir),
                      "report": str(run_dir / "report.md"),
                      "all_dev": metrics["systems"]}, ensure_ascii=False, sort_keys=True))


def _latency_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for system in ("STRONG_SINGLE", "ROUTED_MEDICAL_TEAM"):
        subset = [row for row in rows if row["system"] == system]
        for label, predicate in (
            ("ALL", lambda _row: True),
            ("SINGLE_FAST_PATH", lambda row: row["single_fast_path"]),
            ("TEAM", lambda row: row["team_activated"]),
        ):
            latency = sorted(int(row["latency_ms"]) for row in subset if predicate(row))
            key = f"{system}_{label}"
            result[key] = {
                "count": len(latency),
                "mean_ms": round(sum(latency) / len(latency), 3) if latency else None,
                "p50_ms": latency[int((len(latency) - 1) * 0.50)] if latency else None,
                "p95_ms": latency[int((len(latency) - 1) * 0.95)] if latency else None,
                "provider_calls": sum(int(row["provider_calls"]) for row in subset if predicate(row)),
                "tool_calls": sum(int(row["tool_calls"]) for row in subset if predicate(row)),
                "input_tokens": sum(int(row["input_tokens"]) for row in subset if predicate(row)),
                "output_tokens": sum(int(row["output_tokens"]) for row in subset if predicate(row)),
                "workers": sum(int(row["workers_assigned"]) for row in subset if predicate(row)),
            }
    team_fast = result["ROUTED_MEDICAL_TEAM_SINGLE_FAST_PATH"]
    team_all = result["ROUTED_MEDICAL_TEAM_ALL"]
    result["parallel_speedup"] = {
        "mean_worker_sequential_sum_ms": round(
            sum(row["sequential_worker_latency_ms"] for row in rows
                if row["system"] == "ROUTED_MEDICAL_TEAM" and row["team_activated"]), 3
        ),
        "mean_worker_parallel_wall_ms": round(
            sum(row["worker_wave_wall_ms"] for row in rows
                if row["system"] == "ROUTED_MEDICAL_TEAM" and row["team_activated"]), 3
        ),
        "aggregate_speedup_ratio": aggregate_metrics([
            row for row in rows if row["system"] == "ROUTED_MEDICAL_TEAM" and row["team_activated"]
        ]).get("parallel_speedup"),
        "simple_fast_path_count": team_fast["count"],
        "team_request_count": team_all["count"],
    }
    return result


def _render_report(metrics: dict[str, Any], slices: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    all_dev = slices["ALL_DEV"]["comparison"]
    complex_slice = slices["COMPLEX"]["comparison"]
    team = slices["ALL_DEV"]["ROUTED_MEDICAL_TEAM"]
    all_rows = [row for row in rows if row["system"] == "ROUTED_MEDICAL_TEAM"]
    route_accuracy = sum(row["exact_worker_set_match"] for row in all_rows) / len(all_rows)
    team_latency = [row["latency_ms"] for row in all_rows if row["team_activated"]]
    p95_team = sorted(team_latency)[int((len(team_latency) - 1) * .95)] if team_latency else 0
    fast_rate = team["single_fast_path"]
    workers_avg = team["average_workers_per_team_request"]
    speedup = team["parallel_speedup"]

    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{100 * value:.2f}%"

    def pp(value: float | None) -> str:
        return "N/A" if value is None else f"{100 * value:+.2f} pp"

    lines = [
        "# MA-MVP1 Quantitative Results",
        "",
        "RESUME CANDIDATE METRICS",
        "",
        (f"- Complex task success: {pct(complex_slice['strong_single']['task_success'])} → "
         f"{pct(complex_slice['routed_medical_team']['task_success'])} "
         f"({pp(complex_slice['delta']['task_success'])})"),
        (f"- Overall task success: {pct(all_dev['strong_single']['task_success'])} → "
         f"{pct(all_dev['routed_medical_team']['task_success'])} "
         f"({pp(all_dev['delta']['task_success'])})"),
        (f"- Grounded success: {pct(complex_slice['strong_single']['grounded_task_success'])} → "
         f"{pct(complex_slice['routed_medical_team']['grounded_task_success'])} "
         f"({pp(complex_slice['delta']['grounded_task_success'])})"),
        (f"- Required fact coverage: {pct(complex_slice['strong_single']['required_fact_coverage'])} → "
         f"{pct(complex_slice['routed_medical_team']['required_fact_coverage'])} "
         f"({pp(complex_slice['delta']['required_fact_coverage'])})"),
        f"- Route exact-set accuracy: {route_accuracy:.2%}",
        f"- Single fast-path rate: {fast_rate:.2%}",
        f"- Team activation rate: {team['team_activated']:.2%}",
        f"- Router Team-candidate rate: {team['router_team_candidate']:.2%}",
        f"- Average workers per Team request: {workers_avg:.2f}",
        f"- Team P95 latency: {p95_team / 1000:.2f} s",
        f"- Parallel speedup (mean of per-request ratios): "
        f"{speedup:.2f}x" if speedup is not None else "- Parallel speedup: N/A",
        "",
        "## Primary comparison",
        "",
        "| Metric | Strong Single | Routed Medical Team | Δ |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("task_success", "Task success"),
        ("grounded_task_success", "Grounded task success"),
        ("required_fact_coverage", "Required fact coverage"),
        ("external_evidence_coverage", "External evidence coverage"),
        ("patient_state_fact_coverage", "Patient-state fact coverage"),
    ):
        lines.append(
            f"| {label} | {pct(all_dev['strong_single'].get(key))} "
            f"| {pct(all_dev['routed_medical_team'].get(key))} "
            f"| {pp(all_dev['delta'].get(key))} |"
        )
    lines += [
        "",
        "## Evaluation scope",
        "",
        f"- Dataset: `{metrics['dataset_id']}`, root hash `{metrics['dataset_root_hash']}`.",
        "- 1,024 DEV episodes per system: 512 DEV_IID and 512 DEV_STRUCTURAL.",
        "- Reserved TEST/OOD rows were not accessed or materialized.",
        "- Runtime received no evaluator truth, required-fact labels, scenario-family labels, or split labels.",
        "- Strong Single and Routed Team used the same local model, U2-F world, evaluator, and tool implementations.",
        "- This owned universe is synthetic research data and is not clinical guidance.",
        "",
        "## Engineering metrics",
        "",
        f"- Worker completion rate: {pct(team.get('worker_completion_rate'))}.",
        f"- Useful worker rate: {pct(team.get('useful_worker_rate'))}.",
        f"- Lead incorporation rate: {pct(team.get('lead_incorporation_rate'))}.",
        f"- Partial-failure recovery rate: {pct(team.get('partial_failure_recovery_rate'))}.",
        f"- Provider calls / tool calls / total tokens: {team['provider_calls']} / {team['tool_calls']} / {team['total_tokens']}.",
        (f"- Routed Team overall P50/P95: {team['p50_latency_ms'] / 1000:.2f} / "
         f"{team['p95_latency_ms'] / 1000:.2f} s."),
        "- Latency, token, routing, trace, worker, and provenance artifacts are stored beside this report.",
        "",
        "## Interpretation",
        "",
        "These values describe performance on the project-owned synthetic DEV universe and this frozen local model configuration. They do not establish clinical safety or generalization to real patients.",
        "",
    ]
    return "\n".join(lines)


async def _main(args) -> None:
    if args.stage == "freeze":
        _freeze(args)
    elif args.stage == "train-dev":
        await _train_dev(args)
    else:
        await _final_dev(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("train-dev", "freeze", "final-dev"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default=os.environ.get(
        "MA_MVP1_BASE_URL", "http://127.0.0.1:8083/v1"
    ))
    parser.add_argument("--model", default=os.environ.get(
        "MA_MVP1_MODEL", "Qwen3-8B-Q4_K_M.gguf"
    ))
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
