"""Run non-frozen BM25, dense, and RRF retrieval sanity checks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from _common import DATA_ROOT, REPO_ROOT, RUN_ROOT, load_jsonl, update_summary, write_json

sys.path.insert(0, str(REPO_ROOT / "src"))

from health_ai_copilot.huiyi.bm25 import HuiyiBM25
from health_ai_copilot.huiyi.embed import DEFAULT_MODEL_ROOT, Qwen3LocalEmbedder
from health_ai_copilot.huiyi.hybrid import RRF_K, hybrid_search
from health_ai_copilot.huiyi.milvus_store import COLLECTION_NAME, HuiyiMilvusStore
from health_ai_copilot.huiyi.validation import validate_corpus


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile + 0.5)))
    return round(ordered[index], 3)


def _metrics(results: list[dict[str, Any]], latencies: list[float]) -> dict[str, Any]:
    count = len(results)
    if not count:
        return {"query_count": 0, "hit_at_1": 0.0, "hit_at_3": 0.0, "hit_at_5": 0.0, "mrr_at_5": 0.0}
    hits: dict[int, int] = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks = 0.0
    for result in results:
        expected = set(result["expected_source_ids"])
        ranked = result["ranked_source_ids"][:5]
        for k in hits:
            if expected.intersection(result["ranked_source_ids"][:k]):
                hits[k] += 1
        first = next((rank for rank, source_id in enumerate(ranked, start=1) if source_id in expected), None)
        if first is not None:
            reciprocal_ranks += 1.0 / first
    return {
        "query_count": count,
        "hit_at_1": hits[1] / count,
        "hit_at_3": hits[3] / count,
        "hit_at_5": hits[5] / count,
        "mrr_at_5": reciprocal_ranks / count,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
    }


def _source_ranked(rows: list[dict[str, Any]]) -> list[str]:
    return [row["source_id"] for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    args = parser.parse_args()
    validate_corpus(DATA_ROOT)
    chunks = load_jsonl(DATA_ROOT / "chunks" / "chunks.jsonl")
    queries = load_jsonl(DATA_ROOT / "eval" / "smoke_queries.jsonl")
    if len(queries) < 30:
        raise SystemExit(f"Smoke set must contain at least 30 queries; found {len(queries)}")
    known_source_ids = {row["source_id"] for row in chunks}
    for query in queries:
        if not set(query["expected_source_ids"]).intersection(known_source_ids):
            raise SystemExit(f"Smoke query has no expected source in the corpus: {query['query_id']}")

    bm25 = HuiyiBM25(chunks)
    embedder = Qwen3LocalEmbedder(args.model_root)
    store = HuiyiMilvusStore(args.milvus_uri, collection_name=COLLECTION_NAME)
    # Exercise a real metadata filter against Milvus and verify every row.
    tagged_department = next(
        (row for row in chunks if row["document_type"] == "department" and row.get("topics")),
        None,
    )
    if tagged_department is None:
        raise SystemExit("No tagged department chunk available for metadata-filter smoke.")
    filter_vector = embedder.encode_query("医院有哪些眼科专科？")
    filter_topic = tagged_department["topics"][0]
    array_filter = f'ARRAY_CONTAINS(topics, "{filter_topic}")'
    filtered = store.search(filter_vector, top_k=5, filter_expression=array_filter)
    filter_ok = bool(filtered) and all(filter_topic in row["topics"] for row in filtered)
    if not filter_ok:
        raise SystemExit(f"Milvus topic-array filter smoke failed for {array_filter}")

    detailed: list[dict[str, Any]] = []
    bm25_latencies: list[float] = []
    dense_latencies: list[float] = []
    hybrid_latencies: list[float] = []
    by_retriever: dict[str, list[dict[str, Any]]] = {"bm25": [], "dense": [], "hybrid": []}
    for row in queries:
        expected = list(row["expected_source_ids"])
        hybrid_start = time.perf_counter()
        start = time.perf_counter()
        lexical = bm25.search(row["query"], args.candidate_k)
        bm25_latencies.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        query_vector = embedder.encode_query(row["query"])
        dense = store.search(query_vector, top_k=args.candidate_k)
        dense_latencies.append((time.perf_counter() - start) * 1000)

        hybrid_rows = hybrid_search(
            row["query"],
            lambda _query, k, hits=lexical: hits[:k],
            lambda _query, k, hits=dense: hits[:k],
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )
        hybrid_latencies.append((time.perf_counter() - hybrid_start) * 1000)

        for name, hits, latencies in (
            ("bm25", lexical[:args.top_k], bm25_latencies),
            ("dense", dense[:args.top_k], dense_latencies),
            ("hybrid", hybrid_rows, hybrid_latencies),
        ):
            by_retriever[name].append({
                "query_id": row["query_id"],
                "query": row["query"],
                "expected_source_ids": expected,
                "ranked_chunk_ids": [item["chunk_id"] for item in hits],
                "ranked_source_ids": _source_ranked(hits),
                "ranked_urls": [item["source_url"] for item in hits],
            })
        detailed.append({
            "query_id": row["query_id"],
            "category": row["category"],
            "bm25": by_retriever["bm25"][-1],
            "dense": by_retriever["dense"][-1],
            "hybrid": by_retriever["hybrid"][-1],
        })

    metrics = {
        "bm25": _metrics(by_retriever["bm25"], bm25_latencies),
        "dense": _metrics(by_retriever["dense"], dense_latencies),
        "hybrid": _metrics(by_retriever["hybrid"], hybrid_latencies),
    }
    report = {
        "label": "HY-DATA-0 SMOKE; NOT FROZEN BENCHMARK; NOT CLINICAL ACCURACY",
        "query_count": len(queries),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "rrf": {"k": RRF_K, "bm25_weight": 1.0, "dense_weight": 1.0},
        "corpus_identity_sha256": json.loads((DATA_ROOT / "index" / "corpus_manifest.json").read_text(encoding="utf-8")).get("corpus_identity_sha256"),
        "index_identity_sha256": json.loads((DATA_ROOT / "index" / "index_manifest.json").read_text(encoding="utf-8")).get("index_identity_sha256"),
        "metadata_filter_smoke": {
            "filter": array_filter,
            "passed": filter_ok,
            "result_count": len(filtered),
        },
        "metrics": metrics,
        "results": detailed,
    }
    write_json(RUN_ROOT / "retrieval_smoke.json", report)
    milvus_report_path = RUN_ROOT / "milvus_report.json"
    if milvus_report_path.is_file():
        milvus_report = json.loads(milvus_report_path.read_text(encoding="utf-8"))
        milvus_report["metadata_filter_smoke"] = "passed" if filter_ok else "failed"
        milvus_report["metadata_filter"] = report["metadata_filter_smoke"]
        write_json(milvus_report_path, milvus_report)
    update_summary()
    print(json.dumps({"label": report["label"], "query_count": report["query_count"], "metrics": metrics, "metadata_filter_smoke": report["metadata_filter_smoke"]}, ensure_ascii=False, indent=2, sort_keys=True))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
