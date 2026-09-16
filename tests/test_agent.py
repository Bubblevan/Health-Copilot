from collections.abc import Sequence

import pytest

from health_ai_copilot.agent import (
    AgentLoop,
    AgentLoopConfig,
    AgentModel,
    AgentSession,
    FinalTurn,
    StopReason,
    ToolCall,
    ToolCallTurn,
    ToolRegistry,
    ToolResultMessage,
    UserMessage,
)
from health_ai_copilot.agent.events import AgentEventType
from health_ai_copilot.contracts import Evidence, Route
from health_ai_copilot.pipeline import HealthCopilotPipeline
from health_ai_copilot.tools.search_knowledge import SearchKnowledgeTool


def make_evidence(source_id: str, *, score: float = 1.0) -> Evidence:
    return Evidence(
        source_id=source_id,
        title=f"Title {source_id}",
        excerpt=f"Excerpt {source_id}",
        source_url=f"https://example.org/{source_id}",
        score=score,
    )


class SpyRetriever:
    def __init__(self, initial: list[Evidence], recovered: list[Evidence] | None = None):
        self.initial = initial
        self.recovered = recovered if recovered is not None else initial
        self.calls: list[str] = []

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        self.calls.append(query)
        return (self.initial if len(self.calls) == 1 else self.recovered)[:top_k]


class FakeAgentModel:
    def __init__(self, responses: Sequence[object]):
        self.responses = list(responses)
        self.calls = 0
        self.received_messages = []
        self.received_tools = []

    def respond(self, messages, tools):
        self.received_messages.append(tuple(messages))
        self.received_tools.append(tuple(tools))
        response = self.responses[self.calls]
        self.calls += 1
        return response


class RaisingRetriever(SpyRetriever):
    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        self.calls.append(query)
        if len(self.calls) > 1:
            raise RuntimeError("synthetic search failure")
        return self.initial[:top_k]


def make_pipeline(model: AgentModel, retriever: SpyRetriever) -> HealthCopilotPipeline:
    return HealthCopilotPipeline(retriever, agent_model=model)


def test_immediate_final_uses_zero_tool_calls_and_preserves_citation_behavior() -> None:
    model = FakeAgentModel([FinalTurn("grounded", ["source-a"])])
    retriever = SpyRetriever([make_evidence("source-a")])

    result = make_pipeline(model, retriever).answer("患者教育问题")

    assert result.route == Route.ANSWER
    assert result.message == "grounded"
    assert [citation.source_id for citation in result.citations] == ["source-a"]
    assert model.calls == 1
    assert len(model.received_tools[0]) == 1
    assert model.received_tools[0][0].name == "search_knowledge"
    assert retriever.calls == ["患者教育问题"]


def test_one_recovery_executes_tool_then_calls_model_again() -> None:
    model = FakeAgentModel(
        [
            ToolCallTurn([ToolCall("call-1", "search_knowledge", {"query": "更合适的检索词"})]),
            FinalTurn("recovered", ["source-d"]),
        ]
    )
    retriever = SpyRetriever([make_evidence("source-a")], [make_evidence("source-d")])
    events = []

    result = HealthCopilotPipeline(retriever, agent_model=model, event_sink=events.append).answer(
        "表达不匹配的问题"
    )

    assert result.route == Route.ANSWER
    assert [citation.source_id for citation in result.citations] == ["source-d"]
    assert model.calls == 2
    assert retriever.calls == ["表达不匹配的问题", "更合适的检索词"]
    assert [event.event_type for event in events] == [
        AgentEventType.AGENT_START,
        AgentEventType.TURN_START,
        AgentEventType.MODEL_RESPONSE,
        AgentEventType.TOOL_START,
        AgentEventType.TOOL_END,
        AgentEventType.TURN_END,
        AgentEventType.TURN_START,
        AgentEventType.MODEL_RESPONSE,
        AgentEventType.TURN_END,
        AgentEventType.AGENT_END,
    ]


def test_recovered_evidence_is_used_for_final_citation_verification() -> None:
    model = FakeAgentModel(
        [
            ToolCallTurn([ToolCall("call-1", "search_knowledge", {"query": "recovery"})]),
            FinalTurn("uses recovery evidence", ["source-recovered"]),
        ]
    )
    retriever = SpyRetriever([make_evidence("source-initial")], [make_evidence("source-recovered")])

    result = make_pipeline(model, retriever).answer("问题")

    assert result.route == Route.ANSWER
    assert result.citations[0].source_id == "source-recovered"
    assert result.citations[0].source_url == "https://example.org/source-recovered"


def test_unobserved_source_is_rejected_after_recovery() -> None:
    model = FakeAgentModel(
        [
            ToolCallTurn([ToolCall("call-1", "search_knowledge", {"query": "recovery"})]),
            FinalTurn("fabricated", ["source-not-observed"]),
        ]
    )
    retriever = SpyRetriever([make_evidence("source-initial")], [make_evidence("source-recovered")])

    result = make_pipeline(model, retriever).answer("问题")

    assert result.route == Route.ABSTAIN
    assert result.safety_reasons == ["invalid_citation"]


def test_observed_evidence_is_deduplicated_by_source_id() -> None:
    initial = [make_evidence("source-a"), make_evidence("source-a", score=0.5)]
    recovered = [make_evidence("source-b"), make_evidence("source-a", score=0.2)]
    model = FakeAgentModel(
        [
            ToolCallTurn([ToolCall("call-1", "search_knowledge", {"query": "recovery"})]),
            FinalTurn("deduplicated", ["source-a", "source-b"]),
        ]
    )
    retriever = SpyRetriever(initial, recovered)
    retriever.calls.append("initial")
    loop = AgentLoop(
        model,
        ToolRegistry([SearchKnowledgeTool(retriever)]),
    )

    run = loop.run("问题", initial)

    assert [item.source_id for item in run.observed_evidence] == ["source-a", "source-b"]


def test_registry_returns_structured_errors_for_unknown_and_malformed_calls() -> None:
    retriever = SpyRetriever([make_evidence("source-a")])
    registry = ToolRegistry([SearchKnowledgeTool(retriever)])

    unknown = registry.execute(ToolCall("unknown", "not_registered", {}))
    malformed = registry.execute(ToolCall("malformed", "search_knowledge", {"query": ""}))

    assert not unknown.ok
    assert unknown.error is not None
    assert unknown.error.code == "unknown_tool"
    assert not malformed.ok
    assert malformed.error is not None
    assert malformed.error.code == "invalid_arguments"
    assert retriever.calls == []


def test_registry_rejects_duplicate_tool_names() -> None:
    tool = SearchKnowledgeTool(SpyRetriever([make_evidence("source-a")]))
    registry = ToolRegistry([tool])

    try:
        registry.register(tool)
    except ValueError as exc:
        assert str(exc) == "duplicate tool name: search_knowledge"
    else:
        raise AssertionError("duplicate tool registration should fail")


def test_model_failure_becomes_controlled_stop() -> None:
    class RaisingModel:
        def respond(self, messages, tools):
            raise RuntimeError("synthetic provider failure")

    loop = AgentLoop(
        RaisingModel(),
        ToolRegistry(),
    )

    run = loop.run("问题", [make_evidence("source-a")])

    assert run.stop_reason == StopReason.MODEL_ERROR
    assert run.draft is None


def test_tool_exception_becomes_observation_and_second_turn_can_abstain() -> None:
    model = FakeAgentModel(
        [
            ToolCallTurn([ToolCall("call-1", "search_knowledge", {"query": "boom"})]),
            FinalTurn("", [], abstain=True),
        ]
    )
    retriever = RaisingRetriever([make_evidence("source-a")])
    retriever.calls.append("initial")
    loop = AgentLoop(model, ToolRegistry([SearchKnowledgeTool(retriever)]))

    run = loop.run("问题", [make_evidence("source-a")])

    assert run.draft is not None and run.draft.abstain
    assert run.stop_reason == StopReason.ABSTAIN
    tool_messages = [message for message in run.state.session.messages if isinstance(message, ToolResultMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].result.error is not None
    assert tool_messages[0].result.error.code == "tool_execution_error"


def test_second_tool_call_is_not_executed_and_fails_closed() -> None:
    model = FakeAgentModel(
        [
            ToolCallTurn([ToolCall("call-1", "search_knowledge", {"query": "one"})]),
            ToolCallTurn([ToolCall("call-2", "search_knowledge", {"query": "two"})]),
        ]
    )
    retriever = SpyRetriever([make_evidence("source-a")], [make_evidence("source-b")])
    pipeline = make_pipeline(model, retriever)

    result = pipeline.answer("问题")

    assert result.route == Route.ABSTAIN
    assert result.safety_reasons == ["max_tool_calls"]
    assert model.calls == 2
    assert retriever.calls == ["问题", "one"]


def test_no_final_answer_before_model_budget_fails_closed() -> None:
    model = FakeAgentModel([ToolCallTurn([]), ToolCallTurn([])])
    retriever = SpyRetriever([make_evidence("source-a")])
    loop = AgentLoop(
        model,
        ToolRegistry([SearchKnowledgeTool(retriever)]),
        AgentLoopConfig(max_model_turns=2, max_tool_calls=1),
    )

    run = loop.run("问题", [make_evidence("source-a")])

    assert run.draft is None
    assert run.stop_reason == StopReason.MAX_MODEL_TURNS
    assert model.calls == 2


def test_safety_short_circuits_agent_and_tool_calls() -> None:
    model = FakeAgentModel([FinalTurn("should not run", ["source-a"])])
    retriever = SpyRetriever([make_evidence("source-a")])

    urgent = make_pipeline(model, retriever).answer("我现在胸痛，怎么办？")
    prescription = make_pipeline(model, retriever).answer("请告诉我降压药剂量")

    assert urgent.route == Route.URGENT_CARE
    assert prescription.route == Route.HUMAN_REVIEW
    assert model.calls == 0
    assert retriever.calls == []


def test_no_initial_evidence_preserves_fail_closed_before_agent() -> None:
    model = FakeAgentModel([FinalTurn("should not run", ["source-a"])])
    retriever = SpyRetriever([])

    result = make_pipeline(model, retriever).answer("没有证据的问题")

    assert result.route == Route.ABSTAIN
    assert result.safety_reasons == ["insufficient_evidence"]
    assert model.calls == 0


def test_session_distinguishes_transcript_from_execution_state() -> None:
    model = FakeAgentModel([FinalTurn("answer", ["source-a"])])
    session = AgentSession(session_id="session-test")
    loop = AgentLoop(model, ToolRegistry())

    run = loop.run("问题", [make_evidence("source-a")], session=session)

    assert run.state.session is session
    assert run.state.model_turns_used == 1
    assert isinstance(session.messages[0], UserMessage)
    assert session.messages[0].content == "问题"


def test_pipeline_rejects_generator_and_agent_model_together() -> None:
    retriever = SpyRetriever([make_evidence("source-a")])
    generator = lambda question, evidence: None

    with pytest.raises(ValueError, match="mutually exclusive"):
        HealthCopilotPipeline(
            retriever,
            generator,
            agent_model=FakeAgentModel([FinalTurn("answer", ["source-a"])]),
        )
