"""Run deterministic BM25/dense/RRF/rerank retrieval-only ablations."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from health_ai_copilot.eval.retrieval import evaluate_retriever
from health_ai_copilot.eval.runner import load_cases
from health_ai_copilot.knowledge.loader import load_knowledge_cards
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/m0.jsonl")
    parser.add_argument("--knowledge-dir", default="data/knowledge_cards")
    parser.add_argument("--output-root", default="runs/m5")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--embedding-backend", choices=("hashing", "sentence_transformers"), default="hashing")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--reranker-backend", choices=("token_overlap", "cross_encoder"), default="token_overlap")
    parser.add_argument("--reranker-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    args = parser.parse_args(argv)
    cards = load_knowledge_cards(args.knowledge_dir)
    cases = load_cases(args.dataset)
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
    reranked = RerankedRetriever(hybrid, reranker, candidate_top_k=args.candidate_top_k)
    arms = {"bm25": bm25, "dense": dense, "hybrid": hybrid, "hybrid_rerank": reranked}
    run_dir = Path(args.output_root) / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    run_dir.mkdir(parents=True, exist_ok=False)
    dense.index.save(run_dir / "dense_index")
    metrics, all_rows = {}, []
    for name, retriever in arms.items():
        arm_metrics, rows = evaluate_retriever(cases, retriever, top_k=args.top_k)
        metrics[name] = arm_metrics
        all_rows.extend({"arm": name, **row} for row in rows)
    _write_json(run_dir / "config.json", _config(args, backend, reranker, len(cases)))
    _write_text(run_dir / "dataset.jsonl", Path(args.dataset).read_text(encoding="utf-8"))
    _write_json(run_dir / "retrieval_metrics.json", metrics)
    _write_jsonl(run_dir / "retrieval_results.jsonl", all_rows)
    _write_jsonl(run_dir / "failure_table.jsonl", _failure_table(cases, all_rows))
    _write_text(run_dir / "ablation.md", _ablation(metrics))
    print(run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def _config(args, backend, reranker, case_count):
    return {
        "commit_sha": _git_sha(),
        "dataset_path": args.dataset,
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "knowledge_pack_version": "m0.2-2026-09-15",
        "bm25": {"k1": 1.5, "b": 0.75},
        "embedding_backend": backend.identity,
        "embedding_dimension": _embedding_dimension(backend),
        "normalization": "l2",
        "rrf_k": args.rrf_k,
        "reranker_backend": getattr(reranker, "identity", type(reranker).__name__),
        "candidate_top_k": args.candidate_top_k,
        "final_top_k": args.top_k,
        "case_count": case_count,
    }


def _ablation(metrics):
    rows = ["# M5 retrieval-only ablation", "", "| Arm | Hit@1 | Hit@3 | Recall@5 | MRR | nDCG@5 | Mean latency ms |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, metric in metrics.items():
        rows.append(f"| {name} | {metric['hit_at_1']:.4f} | {metric['hit_at_3']:.4f} | {metric['recall_at_5']:.4f} | {metric['mrr']:.4f} | {metric['ndcg_at_5']:.4f} | {metric['mean_latency_ms']:.3f} |")
    return "\n".join(rows) + "\n"


def _failure_table(cases, rows):
    """Emit one deterministic comparison row per case, rather than one per arm."""

    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["arm"]] = row
    return [
        {
            "case_id": case["id"],
            "category": case.get("category"),
            "challenge_type": case.get("challenge_type"),
            "expected_source_ids": list(case.get("expected_source_ids", ())),
            "arms": {
                arm: {
                    "retrieved_source_ids": row["retrieved_source_ids"],
                    "first_expected_rank": row.get("first_expected_rank"),
                    "failure_type": row["failure_type"],
                }
                for arm, row in by_case[case["id"]].items()
            },
        }
        for case in cases
    ]


def _write_json(path, value):
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path, rows):
    _write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _write_text(path, value):
    path.write_text(value, encoding="utf-8", newline="\n")


def _git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _embedding_dimension(backend):
    return getattr(backend, "dimension", None)


if __name__ == "__main__":
    raise SystemExit(main())
