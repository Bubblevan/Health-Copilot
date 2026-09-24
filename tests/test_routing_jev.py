"""Offline contract tests for the optional Jev adapters."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from typing import Any

import pytest

from health_ai_copilot.routing import (
    Architecture,
    JevAPIError,
    JevClient,
    JevConfig,
    JevContextSelector,
    JevResult,
    JevTaskIntentRouter,
    PrimaryTaskIntent,
)
from health_ai_copilot.routing.task_intent import TASK_INTENT_FACETS, parse_task_intent
from health_ai_copilot.runtime.context_manager import (
    ContextItem,
    ContextItemCategory,
    ContextPriority,
)


class StubJevClient:
    def __init__(
        self,
        answers: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.answers = answers
        self.error = error
        self.calls = 0
        self.state: Any = None
        self.questions: Mapping[str, Mapping[str, Any]] = {}

    async def evaluate(self, *, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> JevResult:
        self.calls += 1
        self.state = state
        self.questions = questions
        if self.error is not None:
            raise self.error
        answers = self.answers or {
            name: {"type": "noul", "noul": 0.75} for name in questions
        }
        return JevResult("jev-test-pinned", answers, 11, 3, 17)


def valid_task_answers(
    *,
    team_probability: float = 0.1,
    facets: Mapping[str, float] | None = None,
    primary_intent: str = PrimaryTaskIntent.DIRECT_LOOKUP.value,
) -> dict[str, dict[str, Any]]:
    facet_values = {name: 0.1 for name in TASK_INTENT_FACETS}
    facet_values.update(facets or {})
    answers: dict[str, dict[str, Any]] = {
        "architecture": {
            "type": "choice",
            "choice": "single" if team_probability < 0.5 else "parallel_team",
            "probabilities": {"single": 1 - team_probability, "parallel_team": team_probability},
            "confidence": max(team_probability, 1 - team_probability),
        },
        "who_global": {"type": "noul", "noul": 0.9},
        "cdc_us": {"type": "noul", "noul": 0.8},
        "nhc_china": {"type": "noul", "noul": 0.2},
        "literature": {"type": "noul", "noul": 0.7},
        "primary_task_intent": {
            "type": "choice",
            "choice": primary_intent,
            "probabilities": {
                intent.value: float(intent.value == primary_intent)
                for intent in PrimaryTaskIntent
            },
            "confidence": 1.0,
        },
    }
    answers.update({name: {"type": "noul", "noul": value} for name, value in facet_values.items()})
    return answers


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_jev_client_validates_and_normalizes_response(monkeypatch: pytest.MonkeyPatch) -> None:
    client = JevClient(JevConfig(api_key="test-only", model="jev-pinned-test"))
    received: dict[str, Any] = {}

    def fake_post(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        received.update(payload)
        return {
            "model": "jev-pinned-test",
            "answers": {"route": {"type": "choice", "choice": "closed_book"}},
            "usage": {"input_tokens": 11, "output_tokens": 3},
        }

    monkeypatch.setattr(client, "_post", fake_post)
    result = run(
        client.evaluate(
            state={"question": "public fixture"},
            questions={"route": {"type": "choice"}},
        )
    )

    assert received["model"] == "jev-pinned-test"
    assert received["state"] == {"question": "public fixture"}
    assert result.model == "jev-pinned-test"
    assert result.input_tokens == 11
    assert result.output_tokens == 3
    assert result.answers["route"]["choice"] == "closed_book"


@pytest.mark.parametrize(
    "response",
    [
        {"usage": {"input_tokens": 1, "output_tokens": 1}},
        {"answers": [], "usage": {"input_tokens": 1, "output_tokens": 1}},
        {"answers": {"route": "closed_book"}, "usage": {"input_tokens": 1, "output_tokens": 1}},
        {"answers": {"route": {"type": "choice"}}, "usage": {"input_tokens": True, "output_tokens": 1}},
        {"answers": {"route": {"type": "choice"}}, "usage": {"input_tokens": -1, "output_tokens": 1}},
    ],
)
def test_jev_client_rejects_malformed_response(response: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    client = JevClient(JevConfig(api_key="test-only"))
    monkeypatch.setattr(client, "_post", lambda _payload: response)
    with pytest.raises(JevAPIError):
        run(client.evaluate(state={}, questions={"route": {"type": "choice"}}))


def test_parse_task_intent_reads_primary_choice_and_all_facets() -> None:
    parsed = parse_task_intent(JevResult("jev-test", valid_task_answers(), 0, 0, 0))

    assert parsed.primary_intent == PrimaryTaskIntent.DIRECT_LOOKUP
    assert parsed.facet_probabilities == {name: 0.1 for name in TASK_INTENT_FACETS}


def test_parse_task_intent_rejects_unknown_choice() -> None:
    with pytest.raises(ValueError, match="unknown primary task intent"):
        parse_task_intent(
            JevResult(
                "jev-test",
                valid_task_answers(primary_intent="made_up_intent"),
                0,
                0,
                0,
            )
        )


def test_parse_task_intent_rejects_missing_probability() -> None:
    answers = valid_task_answers()
    del answers["primary_task_intent"]["probabilities"][PrimaryTaskIntent.DIRECT_LOOKUP.value]
    with pytest.raises(TypeError, match="numeric values"):
        parse_task_intent(JevResult("jev-test", answers, 0, 0, 0))


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_parse_task_intent_rejects_out_of_range_probability(value: float) -> None:
    answers = valid_task_answers()
    answers["needs_current_guidance"]["noul"] = value
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_task_intent(JevResult("jev-test", answers, 0, 0, 0))


def test_jev_task_router_routes_direct_question_to_single() -> None:
    client = StubJevClient(valid_task_answers())
    router = JevTaskIntentRouter(client=client)  # type: ignore[arg-type]

    decision = run(router.route(question="What is a common definition?", data_classification="public"))

    assert decision.architecture == Architecture.SINGLE
    assert decision.worker_roles == ()
    assert client.state == {"question": "What is a common definition?"}


@pytest.mark.parametrize(
    "facet",
    [
        "requires_cross_source_comparison",
        "needs_independent_sources",
        "requires_conflict_review",
    ],
)
def test_jev_task_router_routes_team_facets_to_team(facet: str) -> None:
    client = StubJevClient(valid_task_answers(facets={facet: 0.9}))
    router = JevTaskIntentRouter(client=client)  # type: ignore[arg-type]

    decision = run(router.route(question="Compare these public sources", data_classification="synthetic"))

    assert decision.architecture == Architecture.PARALLEL_TEAM
    assert len(decision.worker_roles) >= 2
    assert facet in decision.route_reasons


def test_serial_dependency_alone_does_not_route_to_parallel_team() -> None:
    client = StubJevClient(valid_task_answers(facets={"requires_serial_dependency": 0.99}))
    router = JevTaskIntentRouter(client=client)  # type: ignore[arg-type]

    decision = run(router.route(question="Do these steps in sequence", data_classification="public"))

    assert decision.architecture == Architecture.SINGLE
    assert decision.route_reasons == ()


def test_jev_task_router_requires_explicit_safe_data_classification() -> None:
    client = StubJevClient(valid_task_answers())
    router = JevTaskIntentRouter(client=client)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        router.route(question="A public question")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="public or synthetic"):
        run(router.route(question="A private patient question", data_classification="private"))
    assert client.calls == 0


def test_jev_task_router_error_fallback_is_stable_and_recorded() -> None:
    router = JevTaskIntentRouter(client=StubJevClient(error=JevAPIError("offline")))  # type: ignore[arg-type]

    first = run(router.route(question="Public question", data_classification="public"))
    second = run(router.route(question="Public question", data_classification="public"))

    assert first.metadata() == second.metadata()
    assert first.architecture == Architecture.PARALLEL_TEAM
    assert first.fallback_reason == "JevAPIError"
    assert first.route_reasons == ("jev_error_team_fallback",)


def _context_item(
    item_id: str,
    content: str,
    *,
    protected: bool = False,
    group_id: str | None = None,
    classification: str = "public",
) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        category=ContextItemCategory.MEMORY if not protected else ContextItemCategory.CURRENT_EVIDENCE,
        content=content,
        estimated_tokens=5,
        priority=ContextPriority.PROTECTED if protected else ContextPriority.NORMAL,
        protected=protected,
        provenance={"research_data_classification": classification},
        group_id=group_id,
    )


def test_context_selector_never_sends_or_mutates_protected_items() -> None:
    protected = _context_item("protected", "do not send", protected=True, classification="private")
    ordinary = _context_item("ordinary", "eligible memory")
    client = StubJevClient({})
    selector = JevContextSelector(client=client)  # type: ignore[arg-type]

    decision = run(
        selector.advise(
            current_task="public task",
            candidates=(protected, ordinary),
            data_classification="public",
        )
    )

    sent = repr(client.state) + repr(client.questions)
    assert "do not send" not in sent
    assert all(
        candidate["candidate_key"] != "item:protected"
        for candidate in client.state["candidates"]
    )
    assert protected.priority == ContextPriority.PROTECTED
    assert protected.protected is True
    assert protected.item_id not in decision.priority_hints
    assert ordinary.priority == ContextPriority.NORMAL


def test_context_selector_keeps_atomic_group_together() -> None:
    first = _context_item("first", "first half", group_id="turn-pair")
    second = _context_item("second", "second half", group_id="turn-pair")
    client = StubJevClient({})
    selector = JevContextSelector(client=client)  # type: ignore[arg-type]

    decision = run(
        selector.advise(
            current_task="use this context",
            candidates=(first, second),
            data_classification="public",
        )
    )

    assert len(client.questions) == 1
    assert len(client.state["candidates"]) == 1
    assert {item["item_id"] for item in client.state["candidates"][0]["items"]} == {
        "first",
        "second",
    }
    assert decision.priority_hints == {"first": "high", "second": "high"}
    assert decision.keep_probabilities["first"] == decision.keep_probabilities["second"]


def test_context_selector_rejects_mismatched_classification_before_call() -> None:
    client = StubJevClient({})
    selector = JevContextSelector(client=client)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="matching research_data_classification"):
        run(
            selector.advise(
                current_task="public task",
                candidates=(_context_item("private", "not public", classification="synthetic"),),
                data_classification="public",
            )
        )
    assert client.calls == 0


def test_context_selector_enforces_candidate_cap_before_call() -> None:
    client = StubJevClient({})
    selector = JevContextSelector(client=client, max_candidates=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="limited to 1 candidates"):
        run(
            selector.advise(
                current_task="public task",
                candidates=(_context_item("one", "1"), _context_item("two", "2")),
                data_classification="public",
            )
        )
    assert client.calls == 0


def test_context_selector_question_key_uses_stable_group_hash() -> None:
    client = StubJevClient({})
    selector = JevContextSelector(client=client)  # type: ignore[arg-type]

    run(
        selector.advise(
            current_task="public task",
            candidates=(_context_item("one", "one", group_id="pair"), _context_item("two", "two", group_id="pair")),
            data_classification="public",
        )
    )

    expected = "keep_" + hashlib.sha256(b"group:pair").hexdigest()[:16]
    assert list(client.questions) == [expected]
