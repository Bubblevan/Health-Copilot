"""Record and strictly replay a small, fixed M3-derived public diagnostic pack."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from health_ai_copilot.agent.model import AgentOutputMode, OpenAICompatibleAgentModel
from health_ai_copilot.config import OpenAIConfig, load_openai_config
from health_ai_copilot.eval.runner import load_cases
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.knowledge.scope import load_knowledge_scope
from health_ai_copilot.pipeline import HealthCopilotPipeline
from health_ai_copilot.policy.model import OpenAICompatibleEvidencePolicy
from health_ai_copilot.retrieval.bm25 import BM25Retriever
from health_ai_copilot.runtime import (
    OpenAICompatibleProviderExecutor,
    RecordingProviderExecutor,
    RecordingToolRunner,
    ReplayProviderExecutor,
    ReplayToolRunner,
    RunContext,
    read_provider_exchanges,
    read_tool_exchanges,
    write_provider_exchanges,
    write_tool_exchanges,
)
from health_ai_copilot.runtime.trace import RunTrace, TraceContentPolicy
from health_ai_copilot.verification.grounding import OpenAICompatibleClaimSupportVerifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("record", "replay"), required=True)
    parser.add_argument("--run-dir", help="existing run directory for replay")
    parser.add_argument("--dataset", default="evals/m4_replay.jsonl")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--knowledge-scope", default="data/knowledge_scope.json")
    parser.add_argument("--output-root", default="runs/m4")
    args = parser.parse_args(argv)
    run_dir = _run_dir(args)
    cards = load_knowledge_cards(args.knowledge_dir)
    scope = load_knowledge_scope(args.knowledge_scope, cards)
    cases = load_cases(args.dataset)
    retriever = BM25Retriever(cards)
    if args.mode == "record":
        config = load_openai_config()
        _write_json(run_dir / "config.json", _config(args, config, scope, cases))
        _write_text(run_dir / "cases.jsonl", Path(args.dataset).read_text(encoding="utf-8"))
        rows = _record(run_dir, cases, retriever, scope, config)
    else:
        config = _replay_config(run_dir)
        rows = _replay(run_dir, cases, retriever, scope, config)
    _write_jsonl(run_dir / f"{args.mode}_results.jsonl", rows)
    metrics = _metrics(rows)
    _write_json(run_dir / f"{args.mode}_metrics.json", metrics)
    _write_text(run_dir / f"{args.mode}_report.md", _report(args.mode, metrics))
    print(run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def _record(run_dir, cases, retriever, scope, config):
    rows = []
    for case in cases:
        trace = RunTrace(
            run_dir / "traces" / f"{case['id']}.jsonl", TraceContentPolicy.PUBLIC_EVAL_CONTENT
        )
        runtime = RunContext.create("m4_live_record", trace=trace)
        provider = RecordingProviderExecutor(OpenAICompatibleProviderExecutor(config))
        pipeline = _pipeline(retriever, scope, config, provider, runtime)
        assert pipeline.agent_loop is not None
        pipeline.agent_loop.tool_runner = RecordingToolRunner(pipeline.agent_loop.tool_runner)
        response = pipeline.answer(case["question"])
        write_provider_exchanges(run_dir / "provider_exchanges" / f"{case['id']}.jsonl", provider.exchanges)
        write_tool_exchanges(
            run_dir / "tool_exchanges" / f"{case['id']}.jsonl", pipeline.agent_loop.tool_runner.exchanges
        )
        rows.append(_row(case, response, pipeline, runtime, provider.exchanges, "live"))
    return rows


def _replay(run_dir, cases, retriever, scope, config):
    rows = []
    for case in cases:
        trace = RunTrace(
            run_dir / "replay_traces" / f"{case['id']}.jsonl", TraceContentPolicy.PUBLIC_EVAL_CONTENT
        )
        runtime = RunContext.create("m4_semantic_replay", trace=trace)
        provider = ReplayProviderExecutor(
            read_provider_exchanges(run_dir / "provider_exchanges" / f"{case['id']}.jsonl")
        )
        pipeline = _pipeline(retriever, scope, config, provider, runtime)
        assert pipeline.agent_loop is not None
        tool_runner = ReplayToolRunner(
            read_tool_exchanges(run_dir / "tool_exchanges" / f"{case['id']}.jsonl")
        )
        pipeline.agent_loop.tool_runner = tool_runner
        response = pipeline.answer(case["question"])
        rows.append(
            _row(case, response, pipeline, runtime, (), "replay")
            | {
                "provider_exchanges_remaining": provider.remaining_exchanges,
                "tool_exchanges_remaining": tool_runner.remaining_exchanges,
            }
        )
    return rows


def _pipeline(retriever, scope, config, provider, runtime):
    return HealthCopilotPipeline(
        retriever,
        agent_model=OpenAICompatibleAgentModel(
            output_mode=AgentOutputMode.M3_CLAIM_FIRST,
            provider_executor=provider,
            model=config.model,
            temperature=config.temperature,
            runtime=runtime,
        ),
        evidence_policy=OpenAICompatibleEvidencePolicy(
            knowledge_scope=scope,
            provider_executor=provider,
            model=config.model,
            runtime=runtime,
        ),
        knowledge_scope=scope,
        claim_support_verifier=OpenAICompatibleClaimSupportVerifier(
            provider_executor=provider,
            model=config.model,
            runtime=runtime,
        ),
        runtime=runtime,
    )


def _row(case, response, pipeline, runtime, exchanges, mode):
    run = pipeline.last_agent_run
    state = run.state if run else None
    return {
        "case_id": case["id"],
        "mode": mode,
        "route": response.route.value,
        "expected_route": case["expected_route"],
        "tool_executed": bool(state and state.tool_calls_used),
        "policy_decision": state.policy_decision if state else None,
        "harness_disposition": pipeline.last_harness_disposition,
        "agent_stop_reason": run.stop_reason.value if run and run.stop_reason else None,
        "provider_call_count": len(exchanges) if mode == "live" else runtime.budget.provider_calls_used,
        "tool_execution_count": runtime.budget.tool_executions_used,
        "budget": _budget_snapshot(runtime),
    }


def _metrics(rows):
    return {
        "case_count": len(rows),
        "route_match_rate": _ratio(sum(_match(row) for row in rows), len(rows)),
        "ood_tool_execution": sum(
            row["tool_executed"] and row["expected_route"] == "unanswerable" for row in rows
        ),
        "provider_call_count": sum(row["provider_call_count"] for row in rows),
        "tool_execution_count": sum(row["tool_execution_count"] for row in rows),
    }


def _report(mode, metrics):
    return "\n".join(
        [
            f"# M4 {mode} diagnostic",
            "",
            "This is a fixed, public M3-derived regression diagnostic; it is not a generalization estimate.",
            "",
            "```json",
            json.dumps(metrics, ensure_ascii=False, indent=2),
            "```",
            "",
            "Replay uses recorded provider and tool exchanges and does not create a live provider client.",
            "",
        ]
    )


def _match(row):
    expected = "answer" if row["expected_route"] == "answer" else "abstain"
    return row["route"] == expected


def _config(args, config, scope, cases):
    return {
        "commit_sha": _git_sha(),
        "model": config.model,
        "policy_model": config.model,
        "verifier_model": config.model,
        "same_model_verification": True,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "dataset_path": args.dataset,
        "dataset_sha256": _sha256(Path(args.dataset)),
        "knowledge_pack_version": scope.knowledge_pack_version,
        "knowledge_scope_id": scope.scope_id,
        "knowledge_scope_version": scope.version,
        "knowledge_scope_sha256": _sha256(Path(args.knowledge_scope)),
        "case_count": len(cases),
        "retries": 0,
        "max_model_turns": 2,
        "max_tool_calls": 1,
        "trace_content_policy": "public_eval_content",
    }


def _replay_config(run_dir):
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return OpenAIConfig(api_key="", model=raw["model"], base_url=raw.get("base_url"), temperature=raw["temperature"])


def _run_dir(args):
    if args.run_dir:
        return Path(args.run_dir)
    if args.mode == "replay":
        raise SystemExit("--run-dir is required for replay")
    path = Path(args.output_root) / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path, value):
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path, rows):
    _write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _write_text(path, value):
    path.write_text(value, encoding="utf-8", newline="\n")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _budget_snapshot(runtime):
    budget = runtime.budget
    return {
        "provider_calls_used": budget.provider_calls_used,
        "tool_executions_used": budget.tool_executions_used,
        "input_tokens_used": budget.input_tokens_used,
        "output_tokens_used": budget.output_tokens_used,
        "total_tokens_used": budget.total_tokens_used,
        "elapsed_ms": budget.elapsed_ms,
    }


if __name__ == "__main__":
    raise SystemExit(main())
