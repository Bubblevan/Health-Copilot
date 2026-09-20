"""Run a reviewed M3/M4 focused pack through four interchangeable retrievers.

This is deliberately an end-to-end diagnostic, not a replacement for the
retrieval-only ablation.  The agent, policy, verifier, budgets, and prompts are
identical across arms; only the Retriever implementation changes.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from health_ai_copilot.agent.messages import AssistantToolCallMessage
from health_ai_copilot.agent.model import AgentOutputMode, OpenAICompatibleAgentModel
from health_ai_copilot.config import load_openai_config
from health_ai_copilot.eval.runner import load_cases
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.knowledge.scope import load_knowledge_scope
from health_ai_copilot.pipeline import HealthCopilotPipeline
from health_ai_copilot.policy.model import OpenAICompatibleEvidencePolicy
from health_ai_copilot.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HashingEmbeddingBackend,
    HybridRetriever,
    RerankedRetriever,
    SentenceTransformerEmbeddingBackend,
    SentenceTransformerReranker,
    TokenOverlapReranker,
)
from health_ai_copilot.runtime import OpenAICompatibleProviderExecutor, RunContext
from health_ai_copilot.runtime.trace import RunTrace, TraceContentPolicy
from health_ai_copilot.verification.grounding import OpenAICompatibleClaimSupportVerifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/m4_replay.jsonl")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--knowledge-scope", default="data/knowledge_scope.json")
    parser.add_argument("--output-root", default="runs/m5")
    parser.add_argument("--initial-top-k", type=int, default=5)
    parser.add_argument("--recovery-top-k", type=int, default=3)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--embedding-backend", choices=("hashing", "sentence_transformers"), default="hashing")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--reranker-backend", choices=("token_overlap", "cross_encoder"), default="token_overlap")
    parser.add_argument("--reranker-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    args = parser.parse_args(argv)
    config = load_openai_config()
    cards = load_knowledge_cards(args.knowledge_dir)
    scope = load_knowledge_scope(args.knowledge_scope, cards)
    cases = load_cases(args.dataset)
    retrievers, backend, reranker = _retrievers(args, cards)
    run_dir = Path(args.output_root) / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", _config(args, config, scope, cases, backend, reranker))
    _write_text(run_dir / "cases.jsonl", Path(args.dataset).read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    for arm, retriever in retrievers.items():
        for case in cases:
            trace = RunTrace(
                run_dir / "traces" / arm / f"{case['id']}.jsonl",
                TraceContentPolicy.PUBLIC_EVAL_CONTENT,
            )
            runtime = RunContext.create("m5_live_e2e", trace=trace)
            provider = OpenAICompatibleProviderExecutor(config)
            pipeline = _pipeline(retriever, scope, config, provider, runtime, args)
            response = pipeline.answer(case["question"])
            rows.append(_row(arm, case, response, pipeline, runtime))

    metrics = {arm: _metrics([row for row in rows if row["arm"] == arm]) for arm in retrievers}
    failures = [row for row in rows if _is_failure(row)]
    _write_jsonl(run_dir / "e2e_results.jsonl", rows)
    _write_jsonl(run_dir / "failures.jsonl", failures)
    _write_json(run_dir / "e2e_metrics.json", metrics)
    _write_text(run_dir / "e2e_report.md", _report(metrics, failures))
    print(run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def _retrievers(args, cards):
    backend = (
        HashingEmbeddingBackend()
        if args.embedding_backend == "hashing"
        else SentenceTransformerEmbeddingBackend(args.embedding_model)
    )
    bm25 = BM25Retriever(cards)
    dense = DenseRetriever.from_knowledge_cards(
        cards, backend, knowledge_pack_version="m0.2-2026-09-15", build_commit=_git_sha()
    )
    hybrid = HybridRetriever(bm25, dense, rrf_k=args.rrf_k)
    reranker = (
        TokenOverlapReranker()
        if args.reranker_backend == "token_overlap"
        else SentenceTransformerReranker(args.reranker_model)
    )
    return (
        {
            "bm25": bm25,
            "dense": dense,
            "hybrid": hybrid,
            "hybrid_rerank": RerankedRetriever(
                hybrid, reranker, candidate_top_k=args.candidate_top_k
            ),
        },
        backend,
        reranker,
    )


def _pipeline(retriever, scope, config, provider, runtime, args):
    return HealthCopilotPipeline(
        retriever,
        top_k=args.initial_top_k,
        recovery_top_k=args.recovery_top_k,
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
            provider_executor=provider, model=config.model, runtime=runtime
        ),
        runtime=runtime,
    )


def _row(arm, case, response, pipeline, runtime):
    run = pipeline.last_agent_run
    state = run.state if run else None
    query = None
    if state:
        for message in state.session.messages:
            if isinstance(message, AssistantToolCallMessage) and message.tool_calls:
                arguments = message.tool_calls[0].arguments
                query = arguments.get("query") if isinstance(arguments, dict) else None
                break
    return {
        "arm": arm,
        "case_id": case["id"],
        "question": case["question"],
        "category": case.get("category"),
        "expected_route": case["expected_route"],
        "route": response.route.value,
        "safety_reasons": response.safety_reasons,
        "tool_proposed": bool(state and state.tool_proposals_used),
        "proposed_search_query": query,
        "tool_executed": bool(state and state.tool_calls_used),
        "policy_decision": state.policy_decision if state else None,
        "policy_matched_topic_ids": list(state.policy_matched_topic_ids) if state else [],
        "agent_stop_reason": run.stop_reason.value if run and run.stop_reason else None,
        "harness_disposition": pipeline.last_harness_disposition,
        "provider_calls_used": runtime.budget.provider_calls_used,
        "tool_calls_used": runtime.budget.tool_executions_used,
        "total_tokens_used": runtime.budget.total_tokens_used,
        "elapsed_ms": runtime.budget.elapsed_ms,
        "final_claims": [asdict(item) for item in run.claims] if run else [],
    }


def _metrics(rows):
    answerable = [row for row in rows if row["expected_route"] == "answer"]
    ood = [row for row in rows if row["category"] == "ood_false_retrieval"]
    return {
        "case_count": len(rows),
        "expected_answer_rate": _rate(answerable, lambda row: row["route"] == "answer"),
        "unexpected_abstain_rate": _rate(answerable, lambda row: row["route"] != "answer"),
        "ood_answer_rate": _rate(ood, lambda row: row["route"] == "answer"),
        "ood_tool_execution_rate": _rate(ood, lambda row: bool(row["tool_executed"])),
        "mean_tool_calls": _mean(rows, "tool_calls_used"),
        "mean_provider_calls": _mean(rows, "provider_calls_used"),
        "mean_latency_ms": _mean(rows, "elapsed_ms"),
        "mean_total_tokens": _mean(rows, "total_tokens_used"),
    }


def _is_failure(row):
    return (
        row["expected_route"] == "answer" and row["route"] != "answer"
    ) or (row["category"] == "ood_false_retrieval" and (row["route"] == "answer" or row["tool_executed"]))


def _config(args, config, scope, cases, backend, reranker):
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
        "bm25": {"k1": 1.5, "b": 0.75},
        "embedding_backend": backend.identity,
        "embedding_dimension": getattr(backend, "dimension", None),
        "normalization": "l2",
        "rrf_k": args.rrf_k,
        "reranker_backend": getattr(reranker, "identity", type(reranker).__name__),
        "candidate_top_k": args.candidate_top_k,
        "initial_top_k": args.initial_top_k,
        "recovery_top_k": args.recovery_top_k,
        "case_count": len(cases),
        "max_model_turns": 2,
        "max_tool_calls": 1,
        "timeout": 30.0,
        "retries": 0,
        "trace_content_policy": "public_eval_content",
    }


def _report(metrics, failures):
    return "\n".join(
        [
            "# M5 M3/M4 end-to-end retriever-only ablation",
            "",
            "All arms use the same M3 claim-first pipeline and M4 runtime context; only the retriever changes.",
            "This focused diagnostic is not a generalization or medical-correctness estimate.",
            "",
            "```json",
            json.dumps(metrics, ensure_ascii=False, indent=2),
            "```",
            "",
            f"- failure records: `{len(failures)}`",
            "- Same-model verification is not independent verification.",
            "",
        ]
    )


def _rate(rows, predicate):
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else None


def _mean(rows, key):
    return mean(float(row[key]) for row in rows) if rows else None


def _write_json(path, value):
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path, rows):
    _write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    raise SystemExit(main())
