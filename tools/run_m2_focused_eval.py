"""Run the same focused pack through frozen-style M1 and M2, without fake live results."""

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from health_ai_copilot.agent.model import OpenAICompatibleAgentModel
from health_ai_copilot.config import load_openai_config
from health_ai_copilot.eval.m1 import summarize_m1_runs
from health_ai_copilot.eval.m2 import summarize_m2_runs
from health_ai_copilot.eval.runner import load_cases
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.pipeline import HealthCopilotPipeline
from health_ai_copilot.policy.model import OpenAICompatibleEvidencePolicy
from health_ai_copilot.retrieval.bm25 import BM25Retriever
from health_ai_copilot.verification.grounding import OpenAICompatibleGroundingVerifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run focused M1-vs-M2 live diagnostic")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--dataset", default="evals/m1_recovery.jsonl")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--initial-top-k", type=int, default=5)
    parser.add_argument("--recovery-top-k", type=int, default=3)
    parser.add_argument("--output-root", default="runs/m2")
    args = parser.parse_args(argv)
    if args.trials <= 0:
        parser.error("--trials must be greater than zero")

    config = load_openai_config()
    cards = load_knowledge_cards(args.knowledge_dir)
    cases = load_cases(args.dataset)
    run_dir = Path(args.output_root) / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir.mkdir(parents=True, exist_ok=False)
    policy_model = os.getenv("HEALTH_COPILOT_POLICY_MODEL") or config.model
    verifier_model = os.getenv("HEALTH_COPILOT_VERIFIER_MODEL") or config.model
    run_config = {
        "commit_sha": _git_sha(), "knowledge_pack_version": "m0.2-2026-09-15",
        "eval_pack_sha256": _sha256(Path(args.dataset)), "model": config.model,
        "policy_model": policy_model, "verifier_model": verifier_model,
        "same_model_verification": policy_model == verifier_model == config.model,
        "temperature": config.temperature, "policy_temperature": 0, "verifier_temperature": 0,
        "base_url": config.base_url, "trials": args.trials, "max_model_turns": 2,
        "max_tool_calls": 1, "initial_top_k": args.initial_top_k, "recovery_top_k": args.recovery_top_k,
        "live_api_key": "configured-but-not-recorded",
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "cases.jsonl").write_text(Path(args.dataset).read_text(encoding="utf-8"), encoding="utf-8")

    m1_runs: dict[str, Any] = {}
    m1_responses: dict[str, Any] = {}
    m2_runs: dict[str, Any] = {}
    m2_responses: dict[str, Any] = {}
    expanded: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []
    retriever = BM25Retriever(cards)
    for trial in range(1, args.trials + 1):
        for case in cases:
            key = f"trial-{trial}:{case['id']}"
            expanded_case = {**case, "id": key}
            expanded.append(expanded_case)
            m1 = HealthCopilotPipeline(retriever, top_k=args.initial_top_k, recovery_top_k=args.recovery_top_k, agent_model=OpenAICompatibleAgentModel())
            m1_responses[key] = m1.answer(case["question"])
            if m1.last_agent_run:
                m1_runs[key] = m1.last_agent_run
            m2 = HealthCopilotPipeline(retriever, top_k=args.initial_top_k, recovery_top_k=args.recovery_top_k, agent_model=OpenAICompatibleAgentModel(), evidence_policy=OpenAICompatibleEvidencePolicy(), grounding_verifier=OpenAICompatibleGroundingVerifier())
            m2_responses[key] = m2.answer(case["question"])
            run = m2.last_agent_run
            if run:
                m2_runs[key] = run
            trajectories.append(_trajectory(key, trial, case, m2_responses[key], run, m2.last_grounding_result))
            if run and run.state.policy_decision:
                policy_rows.append({"case_key": key, "decision": run.state.policy_decision, "reason_codes": list(run.state.policy_reason_codes), "supporting_source_ids": list(run.state.policy_supporting_source_ids)})
            if m2.last_grounding_result:
                grounding_rows.append({"case_key": key, "coverage_ok": m2.last_grounding_result.coverage_ok, "claim_results": [asdict(item) for item in m2.last_grounding_result.claim_results]})
    m1_metrics = summarize_m1_runs(expanded, m1_runs, m1_responses)
    metrics = {"m1": m1_metrics, "m2": summarize_m2_runs(expanded, m2_runs, m2_responses), "trial_count": args.trials, "pack_case_count": len(cases), "trajectory_count": len(trajectories)}
    failures = [row for row in trajectories if row["category"] == "ood_false_retrieval" and row["route"] == "answer"]
    _write_jsonl(run_dir / "trajectories.jsonl", trajectories)
    _write_jsonl(run_dir / "policy_decisions.jsonl", policy_rows)
    _write_jsonl(run_dir / "grounding_results.jsonl", grounding_rows)
    _write_jsonl(run_dir / "failures.jsonl", failures)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(_report(run_dir.name, run_config, metrics, failures), encoding="utf-8")
    print(run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def _trajectory(key: str, trial: int, case: dict[str, Any], response: Any, run: Any, grounding: Any) -> dict[str, Any]:
    state = run.state if run else None
    return {"case_id": case["id"], "case_key": key, "trial": trial, "question": case["question"], "category": case.get("category"), "initial_ranked_evidence": [asdict(item) for item in run.initial_ranked_evidence] if run else [], "tool_proposed": bool(state and state.tool_proposals_used), "policy_decision": state.policy_decision if state else None, "policy_reason_codes": list(state.policy_reason_codes) if state else [], "tool_executed": bool(state and state.tool_calls_used), "recovery_ranked_evidence": [asdict(item) for item in run.recovery_ranked_evidence] if run else [], "observed_evidence": [asdict(item) for item in run.observed_evidence] if run else [], "final_answer": run.draft.answer if run and run.draft else None, "final_claims": [asdict(item) for item in run.claims] if run else [], "citation_integrity_result": response.route.value == "answer" or "invalid_citation" not in response.safety_reasons, "grounding_coverage_ok": grounding.coverage_ok if grounding else None, "claim_verdicts": [item.verdict.value for item in grounding.claim_results] if grounding else [], "route": response.route.value, "stop_reason": run.stop_reason.value if run and run.stop_reason else None, "model_turns_used": state.model_turns_used if state else 0, "tool_proposals_used": state.tool_proposals_used if state else 0, "tool_calls_used": state.tool_calls_used if state else 0}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _report(name: str, config: dict[str, Any], metrics: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    return "\n".join([f"# M2 focused M1-vs-M2 run: `{name}`", "", "This is a focused diagnostic, not a generalization or medical-accuracy claim.", "", "## Configuration", "", *[f"- `{key}`: `{value}`" for key, value in config.items() if key != "live_api_key"], "", "## Metrics", "", "```json", json.dumps(metrics, ensure_ascii=False, indent=2), "```", "", f"- OOD answer failure records: `{len(failures)}`", "", "Same-model verification is not an independent judge. M3 was not started.", ""])


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
