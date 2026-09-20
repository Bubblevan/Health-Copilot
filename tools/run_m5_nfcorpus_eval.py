"""Run M5 retrieval arms on external NFCorpus without product provenance fabrication."""

import json

from health_ai_copilot.eval.nfcorpus import evaluate_nfcorpus, load_nfcorpus
from health_ai_copilot.retrieval import (
    BM25Retriever,
    DenseIndex,
    DenseRetriever,
    HashingEmbeddingBackend,
    HybridRetriever,
    RerankedRetriever,
    TokenOverlapReranker,
)


def main() -> int:
    dataset = load_nfcorpus("artifacts/benchmarks/nfcorpus")
    backend = HashingEmbeddingBackend()
    bm25 = BM25Retriever(dataset.documents)
    dense = DenseRetriever(
        DenseIndex.build(dataset.documents, backend, knowledge_pack_version=None, build_commit="external"),
        backend,
    )
    hybrid = HybridRetriever(bm25, dense)
    arms = {
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
        "hybrid_rerank": RerankedRetriever(hybrid, TokenOverlapReranker()),
    }
    metrics = {name: evaluate_nfcorpus(dataset, retriever=retriever) for name, retriever in arms.items()}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
