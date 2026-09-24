import pytest

from eval.e1_2_capability_router import (
    JEV_QUESTIONS,
    ROUTER_VERSION,
    RetrievalAction,
    route_cheap,
    route_from_jev_probabilities,
)


@pytest.mark.parametrize(
    ("help_probability", "specialized_probability", "expected"),
    [
        (0.49, 0.99, RetrievalAction.CLOSED_BOOK),
        (0.50, 0.49, RetrievalAction.RAG_BM25),
        (0.90, 0.50, RetrievalAction.RAG_MEDCPT),
    ],
)
def test_jev_policy_maps_typed_probabilities_to_fixed_actions(
    help_probability, specialized_probability, expected
):
    decision = route_from_jev_probabilities(
        {
            "retrieval_likely_to_help": help_probability,
            "requires_specialized_detail": specialized_probability,
        }
    )
    assert decision.action is expected
    assert decision.policy_version == ROUTER_VERSION
    assert decision.fallback is False
    assert decision.to_dict()["action"] == expected.value


@pytest.mark.parametrize(
    "probabilities",
    [
        {"retrieval_likely_to_help": 1.1, "requires_specialized_detail": 0.0},
        {"retrieval_likely_to_help": True, "requires_specialized_detail": 0.0},
        {"retrieval_likely_to_help": 0.5},
    ],
)
def test_jev_policy_rejects_invalid_or_incomplete_probabilities(probabilities):
    with pytest.raises((TypeError, ValueError)):
        route_from_jev_probabilities(probabilities)


def test_cheap_policy_is_fixed_and_records_safe_fallback():
    assert route_cheap("What is the mechanism of receptor X?").action is RetrievalAction.RAG_MEDCPT
    assert route_cheap("Which treatment is recommended?").action is RetrievalAction.RAG_BM25
    fallback = route_cheap("A short generic question", fallback_reason="jev_unavailable")
    assert fallback.action is RetrievalAction.CLOSED_BOOK
    assert fallback.fallback is True
    assert fallback.fallback_reason == "jev_unavailable"


def test_jev_classifier_asks_only_about_question_capability_not_answer_content():
    serialized = str(JEV_QUESTIONS).lower()
    assert set(JEV_QUESTIONS) == {"retrieval_likely_to_help", "requires_specialized_detail"}
    assert "answer option" in serialized
    assert "correct answer" not in serialized
