"""Run research-only Single, team, and Jev routing architecture comparisons."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

from eval.research_architecture_team import run_single, run_team
from eval.resume_experiment_utils import (
    append_jsonl,
    case_identity,
    classify_failure,
    config_identity,
    mean,
    percentile,
    read_json,
    safe_output,
    write_json,
)
from health_ai_copilot.routing import (
    WORKER_FAMILY,
    JevArchitectureRouter,
    JevTaskIntentRouter,
)


def load_provider() -> Any:
    module_name = os.environ.get("HEALTH_COPILOT_MULTIAGENT_PROVIDER")
    if not module_name:
        raise RuntimeError("set HEALTH_COPILOT_MULTIAGENT_PROVIDER to a module exposing async adapter functions")
    return importlib.import_module(module_name)


def load_cases(benchmark_dir: Path, split: str) -> list[dict[str, Any]]:
    candidates = [benchmark_dir / f"{split.lower()}.jsonl", benchmark_dir / "cases.jsonl"]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"no cases.jsonl or {split.lower()}.jsonl under {benchmark_dir}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.setdefault("case_id", f"research-architecture-{split.lower()}-{index:04d}")
        result.append(item)
    return result


def validate_test_freeze(benchmark_dir: Path, split: str) -> None:
    if split != "TEST":
        return
    manifest = read_json(benchmark_dir / "manifest.json")
    if manifest.get("status") != "FROZEN" or manifest.get("decision") != "APPROVED":
        raise RuntimeError("TEST requires a human-frozen v2 manifest with status=FROZEN and decision=APPROVED")


async def run_case(
    case: dict[str, Any],
    architecture: str,
    provider: Any,
    budget_mode: str,
    *,
    jev_router: JevArchitectureRouter | None = None,
    jev_data_classification: str | None = None,
    route_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if architecture == "single":
        return await run_single(case, provider.single_answer)
    deadline_ms = int(case.get("deadline_ms", 30000)) if budget_mode == "deadline" else None
    if architecture == "team":
        return await run_team(
            case,
            provider.worker_retrieve,
            provider.lead_synthesize,
            budget_mode,
            deadline_ms,
        )
    is_jev_mode = architecture in {"jev-routed", "jev-intent-routed"}
    if is_jev_mode and (jev_router is None or jev_data_classification is None):
        raise ValueError("Jev routing modes require a router and explicit data classification")
    if not is_jev_mode:
        raise ValueError(f"unsupported architecture: {architecture}")
    decision = await jev_router.route(
        question=str(case.get("question", "")),
        data_classification=jev_data_classification,
    )
    route_details = {
        "routing_decision": decision.metadata(),
        "jev_api_calls": 1,
        "jev_input_tokens": decision.input_tokens,
        "jev_output_tokens": decision.output_tokens,
        "jev_latency_ms": decision.latency_ms,
        "provider_calls": 1,
    }
    if route_artifact is not None:
        route_artifact.update(route_details)
    if decision.architecture.value == "single":
        result = await run_single(case, provider.single_answer)
    else:
        routed_case = {
            **case,
            "required_source_families": [WORKER_FAMILY[role] for role in decision.worker_roles],
        }
        result = await run_team(
            routed_case,
            provider.worker_retrieve,
            provider.lead_synthesize,
            budget_mode,
            deadline_ms,
        )
    total_provider_calls = int(result.get("provider_calls", 0)) + 1
    result.update(route_details)
    result["provider_calls"] = total_provider_calls
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverage = [float(row["evidence_group_coverage"]) for row in rows if isinstance(row.get("evidence_group_coverage"), (int, float))]
    latencies = [float(row["latency_ms"]) for row in rows if isinstance(row.get("latency_ms"), (int, float))]
    jev_call_count = sum(int(row.get("jev_api_calls", 0)) for row in rows)
    jev_output_token_values = [
        row.get("jev_output_tokens")
        for row in rows
        if int(row.get("jev_api_calls", 0)) > 0 and row.get("jev_output_tokens") is not None
    ]
    jev_output_token_coverage = (
        len(jev_output_token_values) / jev_call_count if jev_call_count else 1.0
    )
    return {
        "cases": len(rows),
        "EvidenceGroupCoverage": mean(coverage),
        "AllRequiredGroupsCoveredRate": mean(float(row["all_required_groups_covered"]) for row in rows if isinstance(row.get("all_required_groups_covered"), bool)),
        "CitationIntegrity": mean(float(row["citation_integrity"]) for row in rows if isinstance(row.get("citation_integrity"), bool)),
        "unsafe_ood_answer_rate": mean(float(row["unsafe_ood_answer"]) for row in rows if isinstance(row.get("unsafe_ood_answer"), bool)),
        "worker_completion_rate": mean(float(row["worker_completion_rate"]) for row in rows if isinstance(row.get("worker_completion_rate"), (int, float))),
        "productive_worker_rate": mean(float(row["productive_worker_rate"]) for row in rows if isinstance(row.get("productive_worker_rate"), (int, float))),
        "unique_worker_evidence_contribution": mean(float(row["unique_worker_evidence_contribution"]) for row in rows if isinstance(row.get("unique_worker_evidence_contribution"), (int, float))),
        "pairwise_worker_overlap": mean(float(row["pairwise_worker_overlap"]) for row in rows if isinstance(row.get("pairwise_worker_overlap"), (int, float))),
        "merge_loss_rate": mean(float(row.get("merge_loss", {}).get("merge_loss", False)) for row in rows if isinstance(row.get("merge_loss"), dict)),
        "provider_calls": sum(int(row.get("provider_calls", 0)) for row in rows),
        "jev_api_calls": sum(int(row.get("jev_api_calls", 0)) for row in rows),
        "jev_input_tokens": sum(int(row.get("jev_input_tokens", 0)) for row in rows),
        "jev_output_tokens": (
            sum(int(value) for value in jev_output_token_values)
            if jev_output_token_coverage == 1.0
            else None
        ),
        "jev_output_token_measurement_coverage": jev_output_token_coverage,
        "tool_calls": sum(int(row.get("tool_calls", 0)) for row in rows),
        "input_tokens": mean(float(row["lead_input_tokens"]) for row in rows if isinstance(row.get("lead_input_tokens"), (int, float))),
        "output_tokens": mean(float(row["lead_output_tokens"]) for row in rows if isinstance(row.get("lead_output_tokens"), (int, float))),
        "p50_latency_ms": percentile(latencies, 0.5),
        "p95_latency_ms": percentile(latencies, 0.95),
    }


def summarize_task_intent(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_counts: dict[str, int] = {}
    facets: dict[str, list[float]] = {}
    for row in rows:
        routing = row.get("routing_decision")
        if not isinstance(routing, dict) or not isinstance(routing.get("primary_task_intent"), str):
            continue
        label = routing["primary_task_intent"]
        primary_counts[label] = primary_counts.get(label, 0) + 1
        probabilities = routing.get("intent_facet_probabilities", {})
        if isinstance(probabilities, dict):
            for name, value in probabilities.items():
                if isinstance(value, (int, float)):
                    facets.setdefault(str(name), []).append(float(value))
    return {
        "cases_with_intent_output": sum(primary_counts.values()),
        "primary_task_intent_counts": dict(sorted(primary_counts.items())),
        "mean_intent_facet_probabilities": {
            name: mean(values) for name, values in sorted(facets.items())
        },
    }


async def async_main(args: argparse.Namespace) -> int:
    validate_test_freeze(args.benchmark_dir, args.split)
    jev_modes = {"jev-routed", "jev-intent-routed"}
    if (
        args.architecture in jev_modes
        and args.jev_data_classification not in {"public", "synthetic"}
    ):
        raise ValueError("Jev routing requires --jev-data-classification public or synthetic")
    if args.architecture == "jev-routed":
        jev_router = JevArchitectureRouter()
    elif args.architecture == "jev-intent-routed":
        jev_router = JevTaskIntentRouter()
    else:
        jev_router = None
    config = {
        "benchmark": "research_architecture_v2",
        "architecture": args.architecture,
        "split": args.split,
        "budget_mode": args.budget_mode,
        "model": args.model,
        "answer_extraction": "research_architecture_v2_v1",
    }
    if jev_router is not None:
        config.update(
            {
                "jev_router_version": jev_router.version,
                "jev_model": jev_router.client.config.model,
                "jev_team_threshold": jev_router.team_threshold,
                "jev_worker_threshold": jev_router.worker_threshold,
                "jev_data_classification": args.jev_data_classification,
            }
        )
        if isinstance(jev_router, JevTaskIntentRouter):
            config["jev_route_policy_version"] = jev_router.policy_version
    identity = config_identity(config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"{args.output_dir} is non-empty; choose a new directory or pass --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists() and not args.resume:
        raise FileExistsError(f"{args.output_dir} exists; pass --resume only for identical config")
    if config_path.exists() and read_json(config_path).get("config_identity") != identity:
        raise RuntimeError("incompatible resume config")
    write_json(config_path, {**config, "config_identity": identity})
    write_json(args.output_dir / "manifest.json", {"status": "RUNNING", "config_identity": identity})
    for filename in ("case_results.jsonl", "worker_reports.jsonl", "evidence_ledger.jsonl", "failures.jsonl"):
        (args.output_dir / filename).touch(exist_ok=True)
    cases = load_cases(args.benchmark_dir, args.split)
    cap = args.max_cases or args.limit
    if args.smoke:
        cap = min(cap or 5, 5)
    if cap:
        cases = cases[:cap]
    completed: set[str] = set()
    result_path = args.output_dir / "case_results.jsonl"
    if args.resume and result_path.exists():
        with result_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("config_identity") != identity:
                        raise RuntimeError("incompatible case result")
                    completed.add(row["case_identity"])
    provider = load_provider()
    for case in cases:
        cid = case_identity(case, config)
        if cid in completed:
            continue
        started = time.perf_counter()
        try:
            route_artifact: dict[str, Any] = {}
            result = await run_case(
                case,
                args.architecture,
                provider,
                args.budget_mode,
                jev_router=jev_router,
                jev_data_classification=args.jev_data_classification,
                route_artifact=route_artifact,
            )
            result = {"case_id": case["case_id"], "task_family": case.get("task_family", "uncategorized"), "budget_mode": args.budget_mode, "case_identity": cid, "config_identity": identity, "status": "completed", **result}
            result["latency_ms"] = round((time.perf_counter() - started) * 1000)
            append_jsonl(result_path, result)
            if args.architecture == "team" or args.architecture in jev_modes:
                for report in result.get("worker_reports", []):
                    append_jsonl(args.output_dir / "worker_reports.jsonl", {"case_id": case["case_id"], "config_identity": identity, **report})
                for ledger_row in result.get("evidence_ledger", []):
                    append_jsonl(args.output_dir / "evidence_ledger.jsonl", {"case_id": case["case_id"], "config_identity": identity, **ledger_row})
        # Preserve unexpected per-case failures as resumable records.
        except Exception as exc:  # noqa: BLE001
            failure = {"case_id": case["case_id"], "task_family": case.get("task_family", "uncategorized"), "case_identity": cid, "config_identity": identity, "status": "failed", "failure_classification": classify_failure(exc), "error": safe_output(exc), "latency_ms": round((time.perf_counter() - started) * 1000)}
            failure.update(route_artifact)
            append_jsonl(args.output_dir / "failures.jsonl", failure)
            append_jsonl(result_path, failure)
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()] if result_path.exists() else []
    metrics = {"architecture": args.architecture, "split": args.split, "budget_mode": args.budget_mode, "summary": summarize(rows)}
    if args.architecture == "jev-intent-routed":
        metrics["task_intent"] = summarize_task_intent(rows)
    category_metrics: dict[str, Any] = {}
    for category in sorted({str(row.get("task_family", "uncategorized")) for row in rows}):
        category_rows = [
            row for row in rows if str(row.get("task_family", "uncategorized")) == category
        ]
        category_metrics[category] = summarize(category_rows)
        if args.architecture == "jev-intent-routed":
            category_metrics[category]["task_intent"] = summarize_task_intent(category_rows)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "category_metrics.json", category_metrics)
    (args.output_dir / "report.md").write_text(
        "# Research Architecture v2 Resume Experiment\n\n"
        f"Architecture: `{args.architecture}`\n\nSplit: `{args.split}`\n\nBudget mode: `{args.budget_mode}`\n\n"
        "Results are category-specific; no claim is made that the team always beats Single Agent.\n",
        encoding="utf-8",
    )
    write_json(args.output_dir / "manifest.json", {"status": "COMPLETED", "config_identity": identity, "case_count": len(rows)})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=("single", "team", "jev-routed", "jev-intent-routed"),
        required=True,
    )
    parser.add_argument("--jev-data-classification", choices=("public", "synthetic"))
    parser.add_argument("--split", choices=("DEV", "TEST"), required=True)
    parser.add_argument("--budget-mode", choices=("native", "deadline", "cost"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
