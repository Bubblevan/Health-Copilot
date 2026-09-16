"""Run the small live M1 recovery pack and persist diagnostic evidence.

This script intentionally requires the configured OpenAI-compatible provider.
It never substitutes a fake model and never writes a synthetic live metric.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from health_ai_copilot.agent.loop import AgentRunResult
from health_ai_copilot.agent.model import OpenAICompatibleAgentModel
from health_ai_copilot.config import load_openai_config
from health_ai_copilot.contracts import Evidence, Route
from health_ai_copilot.eval.m1 import summarize_m1_runs
from health_ai_copilot.eval.runner import load_cases
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.pipeline import HealthCopilotPipeline
from health_ai_copilot.retrieval.bm25 import BM25Retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the focused live M1 recovery evaluation")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--dataset", default="evals/m1_recovery.jsonl")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--initial-top-k", type=int, default=5)
    parser.add_argument("--recovery-top-k", type=int, default=3)
    parser.add_argument("--output-root", default="runs/m1")
    args = parser.parse_args(argv)
    if args.trials <= 0:
        parser.error("--trials must be greater than zero")

    config = load_openai_config()
    cards = load_knowledge_cards(args.knowledge_dir)
    cases = load_cases(args.dataset)
    retriever = BM25Retriever(cards)
    run_dir = Path(args.output_root) / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir.mkdir(parents=True, exist_ok=False)

    commit_sha = _git_sha()
    eval_pack_hash = _sha256(Path(args.dataset))
    run_config = {
        "commit_sha": commit_sha,
        "knowledge_pack_version": "m0.2-2026-09-15",
        "eval_pack_version": f"m1_recovery.jsonl:sha256:{eval_pack_hash}",
        "model": config.model,
        "temperature": config.temperature,
        "base_url": config.base_url,
        "trials": args.trials,
        "max_model_turns": 2,
        "max_tool_calls": 1,
        "initial_top_k": args.initial_top_k,
        "recovery_top_k": args.recovery_top_k,
        "live_api_key": "configured-but-not-recorded",
    }
    (run_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "cases.jsonl").write_text(
        Path(args.dataset).read_text(encoding="utf-8"), encoding="utf-8"
    )

    all_runs: dict[str, AgentRunResult] = {}
    all_responses: dict[str, Any] = {}
    expanded_cases: list[dict[str, Any]] = []
    safety_short_circuit_ids: set[str] = set()
    trajectory_rows: list[dict[str, Any]] = []

    for trial in range(1, args.trials + 1):
        agent_model = OpenAICompatibleAgentModel()
        for case in cases:
            case_key = f"trial-{trial}:{case['id']}"
            pipeline = HealthCopilotPipeline(
                retriever,
                top_k=args.initial_top_k,
                recovery_top_k=args.recovery_top_k,
                agent_model=agent_model,
            )
            response = pipeline.answer(case["question"])
            run = pipeline.last_agent_run
            expanded_case = dict(case)
            expanded_case["id"] = case_key
            expanded_cases.append(expanded_case)
            all_responses[case_key] = response
            if run is not None:
                all_runs[case_key] = run
            if (
                run is None
                and case.get("expected_route") in {"urgent", "prescription"}
                and response.route
                in {Route.URGENT_CARE, Route.HUMAN_REVIEW}
            ):
                safety_short_circuit_ids.add(case_key)
            trajectory_rows.append(_trajectory_row(case_key, trial, case, response, run))

    metrics = summarize_m1_runs(
        expanded_cases,
        all_runs,
        all_responses,
        safety_short_circuit_ids=safety_short_circuit_ids,
    )
    metrics.update(
        {
            "trial_count": args.trials,
            "pack_case_count": len(cases),
            "trajectory_count": len(trajectory_rows),
        }
    )
    failures = _failures(trajectory_rows)
    (run_dir / "trajectories.jsonl").write_text(
        _jsonl(trajectory_rows), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "failures.jsonl").write_text(_jsonl(failures), encoding="utf-8")
    (run_dir / "report.md").write_text(
        _report(run_dir.name, run_config, metrics, failures), encoding="utf-8"
    )
    print(run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def _trajectory_row(
    case_key: str,
    trial: int,
    case: dict[str, Any],
    response: Any,
    run: AgentRunResult | None,
) -> dict[str, Any]:
    return {
        "case_key": case_key,
        "trial": trial,
        "case_id": case["id"],
        "question": case["question"],
        "category": case.get("category"),
        "expected_source_ids": case.get("expected_source_ids", []),
        "route": response.route.value,
        "safety_reasons": response.safety_reasons,
        "citations": [citation.source_id for citation in response.citations],
        "agent_ran": run is not None,
        "initial_ranked_evidence": _evidence_rows(run.initial_ranked_evidence) if run else [],
        "recovery_ranked_evidence": _evidence_rows(run.recovery_ranked_evidence) if run else [],
        "observed_evidence": _evidence_rows(run.observed_evidence) if run else [],
        "model_turns_used": run.state.model_turns_used if run else 0,
        "tool_calls_used": run.state.tool_calls_used if run else 0,
        "stop_reason": run.stop_reason.value if run and run.stop_reason else None,
        "final_abstain": run.draft.abstain if run and run.draft else None,
    }


def _evidence_rows(evidence: list[Evidence]) -> list[dict[str, Any]]:
    return [asdict(item) for item in evidence]


def _failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        expected = set(row["expected_source_ids"])
        observed = {item["source_id"] for item in row["observed_evidence"]}
        recovery = {item["source_id"] for item in row["recovery_ranked_evidence"][:3]}
        initial = {item["source_id"] for item in row["initial_ranked_evidence"][:3]}
        if expected and not observed.intersection(expected):
            failures.append({"case_key": row["case_key"], "type": "post_recovery_miss"})
        if row["category"] == "ood_false_retrieval" and row["route"] == "answer":
            failures.append({"case_key": row["case_key"], "type": "ood_answer"})
        if row["category"] in {"urgent", "prescription"} and row["agent_ran"]:
            failures.append({"case_key": row["case_key"], "type": "safety_not_short_circuited"})
        if (
            row["category"] == "synonym_paraphrase"
            and not initial.intersection(expected)
            and not recovery.intersection(expected)
        ):
            failures.append({"case_key": row["case_key"], "type": "recovery_top3_miss"})
    return failures


def _report(
    run_name: str,
    config: dict[str, Any],
    metrics: dict[str, float | int],
    failures: list[dict[str, Any]],
) -> str:
    lines = [
        f"# M1 focused regression run: `{run_name}`",
        "",
        "这是同一模型配置下的 focused regression / diagnostic result，不是泛化性能声明。",
        "",
        "## Configuration",
        "",
        f"- commit: `{config['commit_sha']}`",
        f"- model: `{config['model']}`",
        f"- temperature: `{config['temperature']}`",
        f"- trials: `{config['trials']}`",
        f"- initial_top_k: `{config['initial_top_k']}`",
        f"- recovery_top_k: `{config['recovery_top_k']}`",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in metrics.items())
    lines.extend(
        [
            "",
            "## Failures",
            "",
            f"- failure records: `{len(failures)}`",
            "",
            (
                "Evidence stages are preserved separately in `trajectories.jsonl`; final citation "
                "verification uses the observed union."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
