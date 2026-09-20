from health_ai_copilot.contracts import Evidence
from health_ai_copilot.eval.retrieval import evaluate_retriever
from health_ai_copilot.retrieval.dense import DenseIndex, DenseRetriever, FakeEmbeddingBackend
from health_ai_copilot.retrieval.documents import RetrievalDocument
from health_ai_copilot.retrieval.hybrid import FakeReranker, HybridRetriever, RerankedRetriever


class StaticRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5):
        return self.results[:top_k]


def _evidence(source_id, score=0.0):
    return Evidence(source_id, source_id, source_id, "https://example.test", score)


def test_dense_normalized_cosine_and_id_tie_break_are_deterministic() -> None:
    documents = [RetrievalDocument("b", "b"), RetrievalDocument("a", "a")]
    backend = FakeEmbeddingBackend({"a": (1, 0), "b": (1, 0), "query": (2, 0)})
    index = DenseIndex.build(documents, backend, knowledge_pack_version="test", build_commit="commit")

    results = DenseRetriever(index, backend).search("query", top_k=2)

    assert [item.source_id for item in results] == ["a", "b"]
    assert [item.score for item in results] == [1.0, 1.0]


def test_dense_index_manifest_mismatch_fails_closed(tmp_path) -> None:
    document = RetrievalDocument("a", "a")
    backend = FakeEmbeddingBackend({"a": (1, 0)})
    index = DenseIndex.build([document], backend, knowledge_pack_version="test", build_commit="commit")
    index.save(tmp_path)

    changed = type(index.manifest)(**{**index.manifest.__dict__, "build_commit": "other"})
    try:
        DenseIndex.load(tmp_path, expected=changed)
    except ValueError as error:
        assert "manifest" in str(error)
    else:
        raise AssertionError("stale dense index must not load")


def test_rrf_uses_one_based_rank_deduplicates_and_breaks_ties_by_id() -> None:
    bm25 = StaticRetriever([_evidence("b"), _evidence("a")])
    dense = StaticRetriever([_evidence("a"), _evidence("b")])

    results = HybridRetriever(bm25, dense, rrf_k=0).search("query", top_k=2)

    assert [item.source_id for item in results] == ["a", "b"]
    assert results[0].score == results[1].score == 1.5


def test_reranked_retriever_uses_candidate_pool_and_final_top_k() -> None:
    base = StaticRetriever([_evidence("a"), _evidence("b"), _evidence("c")])
    reranker = FakeReranker({"c": 3, "b": 2, "a": 1})

    results = RerankedRetriever(base, reranker, candidate_top_k=3).search("query", top_k=2)

    assert [item.source_id for item in results] == ["c", "b"]
    assert reranker.calls == [("query", ("a", "b", "c"), 2)]


def test_retrieval_failure_taxonomy_is_deterministic_and_not_model_generated() -> None:
    cases = [
        {
            "id": "semantic",
            "question": "q",
            "expected_source_ids": ["expected"],
            "challenge_type": "synonym_paraphrase",
        },
        {"id": "uncovered", "question": "q", "expected_source_ids": []},
    ]

    _, rows = evaluate_retriever(cases, StaticRetriever([_evidence("other")]))

    assert [row["failure_type"] for row in rows] == ["semantic_miss", "corpus_uncovered"]
