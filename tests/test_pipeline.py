from collections.abc import Sequence
from pathlib import Path

from health_ai_copilot.contracts import Evidence, GenerationDraft, Route
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.pipeline import HealthCopilotPipeline
from health_ai_copilot.retrieval.bm25 import BM25Retriever

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "knowledge_cards"


class SpyRetriever:
    def __init__(self, evidence: list[Evidence]):
        self.evidence = evidence
        self.calls = 0

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        self.calls += 1
        return self.evidence[:top_k]


class FakeGenerator:
    def __init__(self, draft: GenerationDraft):
        self.draft = draft
        self.calls = 0
        self.received_evidence: Sequence[Evidence] = []

    def generate(self, question: str, evidence: Sequence[Evidence]) -> GenerationDraft:
        self.calls += 1
        self.received_evidence = evidence
        return self.draft


class RaisingRetriever:
    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        raise RuntimeError("synthetic retrieval failure")


class RaisingGenerator:
    def generate(self, question: str, evidence: Sequence[Evidence]) -> GenerationDraft:
        raise RuntimeError("synthetic generation failure")


def make_evidence() -> list[Evidence]:
    return [
        Evidence(
            source_id="source-a",
            title="Stored source title",
            excerpt="Stored source excerpt",
            source_url="https://example.org/source-a",
            score=1.0,
        )
    ]


def test_urgent_request_never_invokes_retriever_or_generator() -> None:
    retriever = SpyRetriever(make_evidence())
    generator = FakeGenerator(GenerationDraft("unused", ["source-a"]))

    result = HealthCopilotPipeline(retriever, generator).answer("我持续胸痛，怎么办？")

    assert result.route == Route.URGENT_CARE
    assert retriever.calls == 0
    assert generator.calls == 0


def test_prescription_request_never_invokes_retriever_or_generator() -> None:
    retriever = SpyRetriever(make_evidence())
    generator = FakeGenerator(GenerationDraft("unused", ["source-a"]))

    result = HealthCopilotPipeline(retriever, generator).answer("请告诉我停药和剂量？")

    assert result.route == Route.HUMAN_REVIEW
    assert retriever.calls == 0
    assert generator.calls == 0


def test_no_retrieval_evidence_abstains_without_generator_call() -> None:
    retriever = SpyRetriever([])
    generator = FakeGenerator(GenerationDraft("unused", ["source-a"]))

    result = HealthCopilotPipeline(retriever, generator).answer("普通教育问题")

    assert result.route == Route.ABSTAIN
    assert result.safety_reasons == ["insufficient_evidence"]
    assert generator.calls == 0
    assert "审核资料不足" in result.message


def test_retrieval_error_has_system_failure_message() -> None:
    generator = FakeGenerator(GenerationDraft("unused", ["source-a"]))

    result = HealthCopilotPipeline(RaisingRetriever(), generator).answer("普通教育问题")

    assert result.route == Route.ABSTAIN
    assert result.safety_reasons == ["retrieval_error"]
    assert "检索服务" in result.message
    assert "审核资料不足" not in result.message


def test_generation_error_has_system_failure_message() -> None:
    retriever = SpyRetriever(make_evidence())

    result = HealthCopilotPipeline(retriever, RaisingGenerator()).answer("普通教育问题")

    assert result.route == Route.ABSTAIN
    assert result.safety_reasons == ["generation_error"]
    assert "回答生成服务" in result.message
    assert "审核资料不足" not in result.message


def test_generator_abstention_returns_abstain() -> None:
    retriever = SpyRetriever(make_evidence())
    generator = FakeGenerator(GenerationDraft("", [], abstain=True))

    result = HealthCopilotPipeline(retriever, generator).answer("普通教育问题")

    assert result.route == Route.ABSTAIN
    assert result.safety_reasons == ["generator_abstained"]


def test_fabricated_source_id_returns_abstain() -> None:
    retriever = SpyRetriever(make_evidence())
    generator = FakeGenerator(GenerationDraft("answer", ["not-retrieved"]))

    result = HealthCopilotPipeline(retriever, generator).answer("普通教育问题")

    assert result.route == Route.ABSTAIN
    assert result.safety_reasons == ["invalid_citation"]
    assert "来源校验" in result.message
    assert "审核资料不足" not in result.message


def test_normal_path_returns_citations_from_stored_evidence() -> None:
    retriever = BM25Retriever(load_knowledge_cards(FIXTURE_DIR))
    generator = FakeGenerator(GenerationDraft("grounded answer", ["fixture-hypertension"]))

    result = HealthCopilotPipeline(retriever, generator).answer("高血压患者低盐饮食")

    assert result.route == Route.ANSWER
    assert result.message == "grounded answer"
    assert result.citations[0].source_id == "fixture-hypertension"
    assert result.citations[0].title == "Synthetic hypertension lifestyle card"
    assert result.citations[0].source_url == "https://example.org/fixtures/hypertension"
