from health_ai_copilot.contracts import Evidence, GenerationDraft
from health_ai_copilot.verification.citations import verify_citations


def evidence() -> list[Evidence]:
    return [
        Evidence(
            source_id="source-a",
            title="A",
            excerpt="evidence",
            source_url="https://example.org/a",
            score=1.0,
        )
    ]


def test_valid_citation_is_accepted() -> None:
    result = verify_citations(GenerationDraft("answer", ["source-a"]), evidence())

    assert result.valid
    assert result.citation_ids == ["source-a"]


def test_fabricated_citation_is_rejected() -> None:
    result = verify_citations(GenerationDraft("answer", ["made-up"]), evidence())

    assert not result.valid
    assert result.reasons == ["invalid_citation"]


def test_duplicate_citation_ids_are_deduplicated() -> None:
    result = verify_citations(
        GenerationDraft("answer", ["source-a", "source-a"]), evidence()
    )

    assert result.valid
    assert result.citation_ids == ["source-a"]


def test_non_abstaining_answer_requires_a_citation() -> None:
    result = verify_citations(GenerationDraft("answer", []), evidence())

    assert not result.valid
    assert result.reasons == ["missing_citation"]
