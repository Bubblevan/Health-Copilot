from health_ai_copilot.contracts import KnowledgeCard
from health_ai_copilot.retrieval.documents import RetrievalDocument, document_from_knowledge_card


def test_generic_retrieval_document_needs_no_review_provenance() -> None:
    document = RetrievalDocument(id="external-1", title="External", text="benchmark text")

    assert document.id == "external-1"
    assert document.metadata == {}


def test_product_card_maps_one_way_to_retrieval_document() -> None:
    card = KnowledgeCard(
        id="reviewed-1",
        title="Reviewed",
        content="reviewed text",
        source_url="https://example.test/source",
        publisher="Publisher",
        published_at=None,
        collected_at="2026-01-01",
        reviewed_at="2026-01-02",
        reviewer="reviewer",
        version="1",
        expires_at=None,
        audience=["adult"],
        tags=["tag"],
    )

    document = document_from_knowledge_card(card)

    assert document.id == card.id
    assert document.text == card.content
    assert document.metadata["source_url"] == card.source_url
