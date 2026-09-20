"""Run same-pack, three-trial M2 versus M3 live diagnostics without new tools."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from health_ai_copilot.agent.messages import AssistantToolCallMessage
from health_ai_copilot.agent.model import AgentOutputMode, OpenAICompatibleAgentModel
from health_ai_copilot.config import load_openai_config
from health_ai_copilot.eval.m2 import summarize_m2_runs
from health_ai_copilot.eval.runner import load_cases
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.knowledge.scope import load_knowledge_scope
from health_ai_copilot.pipeline import HealthCopilotPipeline
from health_ai_copilot.policy.model import OpenAICompatibleEvidencePolicy
from health_ai_copilot.retrieval.bm25 import BM25Retriever
from health_ai_copilot.verification.grounding import (
    OpenAICompatibleClaimSupportVerifier,
    OpenAICompatibleGroundingVerifier,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run focused M2-vs-M3 live diagnostic")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--knowledge-scope", default="data/knowledge_scope.json")
    parser.add_argument("--dataset", default="evals/m1_recovery.jsonl")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--initial-top-k", type=int, default=5)
    parser.add_argument("--recovery-top-k", type=int, default=3)
    parser.add_argument("--output-root", default="runs/m3")
    args = parser.parse_args(argv)
    if args.trials <= 0:
        parser.error("--trials must be greater than zero")
    config = load_openai_config()
    cards = load_knowledge_cards(args.knowledge_dir)
    scope = load_knowledge_scope(args.knowledge_scope, cards)
    cases = load_cases(args.dataset)
    retriever = BM25Retriever(cards)
    run_dir = Path(args.output_root) / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = _config(args, config, scope, len(cases))
    _write_json(run_dir / "config.json", run_config)
    (run_dir / "cases.jsonl").write_text(Path(args.dataset).read_text(encoding="utf-8"), encoding="utf-8")

    m2_runs: dict[str, Any] = {}
    m2_responses: dict[str, Any] = {}
    m3_runs: dict[str, Any] = {}
    m3_responses: dict[str, Any] = {}
    expanded: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    capability_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    for trial in range(1, args.trials + 1):
        for case in cases:
            key = f"trial-{trial}:{case['id']}"
            expanded.append({**case, "id": key})
            _write_progress(run_dir, trial, key, "m2", "started")
            m2 = HealthCopilotPipeline(
                retriever,
                top_k=args.initial_top_k,
                recovery_top_k=args.recovery_top_k,
                agent_model=OpenAICompatibleAgentModel(
                    output_mode=AgentOutputMode.M2_GROUNDED
                ),
                evidence_policy=OpenAICompatibleEvidencePolicy(),
                grounding_verifier=OpenAICompatibleGroundingVerifier(),
            )
            m2_response = m2.answer(case["question"])
            m2_responses[key] = m2_response
            if m2.last_agent_run is not None:
                m2_runs[key] = m2.last_agent_run
            trajectories.append(_trajectory("m2", key, trial, case, m2_response, m2, None))
            if m2.last_agent_run and m2.last_agent_run.state.policy_decision:
                capability_rows.append(_capability_row("m2", key, m2.last_agent_run, None))
            if m2.last_grounding_result:
                claim_rows.append(
                    {
                        "arm": "m2",
                        "case_key": key,
                        "verifier_type": "coverage_and_claim_grounding",
                        "claim_results": [asdict(item) for item in m2.last_grounding_result.claim_results],
                    }
                )
            _write_progress(run_dir, trial, key, "m2", "completed")

            _write_progress(run_dir, trial, key, "m3", "started")
            m3 = HealthCopilotPipeline(
                retriever,
                top_k=args.initial_top_k,
                recovery_top_k=args.recovery_top_k,
                agent_model=OpenAICompatibleAgentModel(
                    output_mode=AgentOutputMode.M3_CLAIM_FIRST
                ),
                evidence_policy=OpenAICompatibleEvidencePolicy(knowledge_scope=scope),
                knowledge_scope=scope,
                claim_support_verifier=OpenAICompatibleClaimSupportVerifier(),
            )
            m3_response = m3.answer(case["question"])
            m3_responses[key] = m3_response
            if m3.last_agent_run is not None:
                m3_runs[key] = m3.last_agent_run
            trajectories.append(_trajectory("m3", key, trial, case, m3_response, m3, scope))
            if m3.last_agent_run and m3.last_agent_run.state.policy_decision:
                capability_rows.append(_capability_row("m3", key, m3.last_agent_run, scope))
            if m3.last_claim_support_result:
                claim_rows.append(
                    {
                        "arm": "m3",
                        "case_key": key,
                        "verifier_type": "claim_support_only",
                        "claim_results": [asdict(item) for item in m3.last_claim_support_result.claim_results],
                    }
                )
            _write_progress(run_dir, trial, key, "m3", "completed")

    m2_metrics = summarize_m2_runs(expanded, m2_runs, m2_responses)
    m3_metrics = summarize_m2_runs(expanded, m3_runs, m3_responses)
    m3_metrics["claim_support_rejection_rate"] = _ratio_or_none(
        sum(response.safety_reasons == ["claim_support_failed"] for response in m3_responses.values()),
        len(m3_runs),
    )
    metrics = {
        "m2": m2_metrics,
        "m3": m3_metrics,
        "frozen_m2_baseline": {
            "ood_tool_execution_rate": 8 / 12,
            "expected_answer_rate": 17 / 18,
        },
        "trial_count": args.trials,
        "pack_case_count": len(cases),
        "trajectory_count": len(trajectories),
    }
    failures = _failures(trajectories)
    _write_jsonl(run_dir / "trajectories.jsonl", trajectories)
    _write_jsonl(run_dir / "capability_decisions.jsonl", capability_rows)
    _write_jsonl(run_dir / "claim_results.jsonl", claim_rows)
    _write_jsonl(run_dir / "failures.jsonl", failures)
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "report.md").write_text(_report(run_dir.name, metrics, failures), encoding="utf-8")
    print(run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def _trajectory(arm, key, trial, case, response, pipeline, scope):
    run = pipeline.last_agent_run
    state = run.state if run else None
    proposed_query = None
    if state:
        for message in state.session.messages:
            if isinstance(message, AssistantToolCallMessage) and message.tool_calls:
                arguments = message.tool_calls[0].arguments
                if isinstance(arguments, dict):
                    proposed_query = arguments.get("query")
                break
    claim_results = (
        [asdict(item) for item in pipeline.last_claim_support_result.claim_results]
        if pipeline.last_claim_support_result
        else []
    )
    return {
        "arm": arm,
        "case_id": case["id"],
        "case_key": key,
        "trial": trial,
        "question": case["question"],
        "category": case.get("category"),
        "initial_ranked_evidence": [asdict(item) for item in run.initial_ranked_evidence] if run else [],
        "tool_proposed": bool(state and state.tool_proposals_used),
        "proposed_search_query": proposed_query,
        "knowledge_scope_id": scope.scope_id if scope else None,
        "knowledge_scope_version": scope.version if scope else None,
        "policy_decision": state.policy_decision if state else None,
        "policy_reason_codes": list(state.policy_reason_codes) if state else [],
        "policy_matched_topic_ids": list(state.policy_matched_topic_ids) if state else [],
        "tool_executed": bool(state and state.tool_calls_used),
        "recovery_ranked_evidence": [asdict(item) for item in run.recovery_ranked_evidence] if run else [],
        "observed_evidence": [asdict(item) for item in run.observed_evidence] if run else [],
        "final_claims": [asdict(item) for item in run.claims] if run else [],
        "claim_support_results": claim_results,
        "materialized_answer": response.message if arm == "m3" and response.route.value == "answer" else None,
        "route": response.route.value,
        "agent_stop_reason": run.stop_reason.value if run and run.stop_reason else None,
        "harness_disposition": pipeline.last_harness_disposition,
        "model_turns_used": state.model_turns_used if state else 0,
        "tool_proposals_used": state.tool_proposals_used if state else 0,
        "tool_calls_used": state.tool_calls_used if state else 0,
        "policy_calls_used": state.policy_calls_used if state else 0,
        "verifier_calls_used": state.verifier_calls_used if state else 0,
    }


def _capability_row(arm, key, run, scope):
    state = run.state
    return {
        "arm": arm,
        "case_key": key,
        "scope_id": scope.scope_id if scope else None,
        "scope_version": scope.version if scope else None,
        "decision": state.policy_decision,
        "reason_codes": list(state.policy_reason_codes),
        "matched_topic_ids": list(state.policy_matched_topic_ids),
        "supporting_source_ids": list(state.policy_supporting_source_ids),
    }


def _failures(rows):
    return [
        {
            "arm": row["arm"],
            "case_key": row["case_key"],
            "type": "ood_tool_executed",
        }
        for row in rows
        if row["category"] == "ood_false_retrieval" and row["tool_executed"]
    ] + [
        {
            "arm": row["arm"],
            "case_key": row["case_key"],
            "type": "unexpected_abstain",
        }
        for row in rows
        if row["category"] in {"synonym_paraphrase", "direct_hit"} and row["route"] != "answer"
    ]


def _config(args, config, scope, case_count):
    return {
        "commit_sha": _git_sha(),
        "model": config.model,
        "policy_model": config.model,
        "verifier_model": config.model,
        "same_model_verification": True,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "policy_temperature": 0,
        "verifier_temperature": 0,
        "dataset_path": str(Path(args.dataset)),
        "dataset_sha256": _sha256(Path(args.dataset)),
        "knowledge_pack_version": scope.knowledge_pack_version,
        "knowledge_scope_id": scope.scope_id,
        "knowledge_scope_version": scope.version,
        "knowledge_scope_sha256": _sha256(Path(args.knowledge_scope)),
        "case_count": case_count,
        "trial_count": args.trials,
        "timeout": 30.0,
        "retries": 0,
        "max_model_turns": 2,
        "max_tool_calls": 1,
        "initial_top_k": args.initial_top_k,
        "recovery_top_k": args.recovery_top_k,
    }


def _report(name, metrics, failures):
    return "\n".join(
        [
            f"# M3 focused M2-vs-M3 run: `{name}`",
            "",
            "M3 materializes user-visible output from verified claims and does not use semantic coverage_ok.",
            "",
            "```json",
            json.dumps(metrics, ensure_ascii=False, indent=2),
            "```",
            "",
            f"- failure records: `{len(failures)}`",
            "- Same-model verification is not an independent judge.",
            "",
        ]
    )


def _ratio_or_none(numerator, denominator):
    return numerator / denominator if denominator else None


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_progress(run_dir, trial, case_key, arm, status):
    record = {
        "trial": trial,
        "case_key": case_key,
        "arm": arm,
        "status": status,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    with (run_dir / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
