import pytest

from eval.e1_2_answer_provider import (
    build_answer_messages,
    parse_answer,
    verify_served_model,
)


def test_all_answer_arms_share_a_prompt_template_with_evidence_as_only_context_change():
    case = {"question": "Which option is correct?", "options": {"A": "first", "B": "second"}}
    closed_book = build_answer_messages(case, [])
    rag = build_answer_messages(case, [{"id": "doc-1", "title": "Text", "content": "Fact."}])
    assert closed_book[0] == rag[0]
    assert "Which option is correct?" in closed_book[1]["content"]
    assert "No retrieved evidence" in closed_book[1]["content"]
    assert "doc-1" in rag[1]["content"]


def test_answer_parser_accepts_only_a_supplied_option_label():
    assert parse_answer('{"answer":"b"}', {"A", "B"}) == ("B", False)
    assert parse_answer("answer: A", {"A", "B"}) == ("A", False)
    assert parse_answer('{"answer":"C"}', {"A", "B"}) == ("C", True)
    assert parse_answer("not a selection", {"A", "B"}) == (None, True)


def test_served_model_identity_must_match_the_frozen_path():
    verify_served_model(
        {"served_model": r"F:\Health-Copilot-E1.2\models\Qwen3-8B-Q4_K_M.gguf"},
        expected_model_id=r"F:\Health-Copilot-E1.2\models\Qwen3-8B-Q4_K_M.gguf",
    )


def test_served_model_identity_rejects_unexpected_model():
    with pytest.raises(RuntimeError, match="outside the frozen E1.2 identity"):
        verify_served_model(
            {"served_model": "some-other-model"},
            expected_model_id=r"F:\Health-Copilot-E1.2\models\Qwen3-8B-Q4_K_M.gguf",
        )
