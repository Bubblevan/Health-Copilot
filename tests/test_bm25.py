from pathlib import Path

from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.retrieval.bm25 import BM25Retriever
from health_ai_copilot.retrieval.tokenizer import tokenize

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "knowledge_cards"


def make_retriever() -> BM25Retriever:
    return BM25Retriever(load_knowledge_cards(FIXTURE_DIR))


def test_chinese_queries_share_meaningful_tokens() -> None:
    left = set(tokenize("高血压患者低盐饮食"))
    right = set(tokenize("高血压 饮食原则"))

    assert {"高血压", "饮食"}.issubset(left | right)
    assert left.intersection(right)


def test_relevant_card_ranks_above_irrelevant_card() -> None:
    results = make_retriever().search("高血压患者低盐饮食生活方式", top_k=3)

    assert results[0].source_id == "fixture-hypertension"
    assert results[0].score > results[-1].score


def test_unrelated_query_returns_no_evidence() -> None:
    assert make_retriever().search("量子纠缠粒子加速器", top_k=3) == []


def test_top_k_and_tie_breaking_are_deterministic() -> None:
    retriever = make_retriever()

    first = retriever.search("lifestyle", top_k=2)
    second = retriever.search("lifestyle", top_k=2)

    assert len(first) == 2
    assert [(item.source_id, item.score) for item in first] == [
        (item.source_id, item.score) for item in second
    ]
