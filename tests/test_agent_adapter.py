import json

import pytest

from health_ai_copilot.agent.messages import (
    AssistantToolCallMessage,
    FinalTurn,
    ToolCall,
    ToolCallTurn,
    ToolResultMessage,
    UserMessage,
)
from health_ai_copilot.agent.model import (
    AgentModelError,
    OpenAICompatibleAgentModel,
)
from health_ai_copilot.agent.tools import ToolRegistry, ToolResult, ToolSpec
from health_ai_copilot.contracts import Evidence
from health_ai_copilot.runtime import (
    FakeProviderExecutor,
    ProviderCallKind,
    ProviderResponse,
)
from health_ai_copilot.tools.search_knowledge import SearchKnowledgeTool


def _tool_spec() -> ToolSpec:
    return ToolSpec(
        name="search_knowledge",
        description="search reviewed knowledge",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def _adapter(*, content=None, tool_calls=()):
    executor = FakeProviderExecutor(
        [
            ProviderResponse(
                call_id="recorded",
                kind=ProviderCallKind.AGENT,
                model="fake-model",
                content=content,
                tool_calls=tool_calls,
            )
        ]
    )
    return OpenAICompatibleAgentModel(provider_executor=executor, model="fake-model"), executor


def _evidence() -> Evidence:
    return Evidence(
        source_id="source-a",
        title="Title A",
        excerpt="Excerpt A",
        source_url="https://example.org/source-a",
        score=1.0,
    )


def test_openai_adapter_maps_native_tool_call_to_tool_call_turn() -> None:
    adapter, executor = _adapter(
        tool_calls=(
            {"id": "call-1", "function": {"name": "search_knowledge", "arguments": '{"query":"low sodium"}'}},
        )
    )

    turn = adapter.respond([UserMessage("question")], [_tool_spec()])

    assert isinstance(turn, ToolCallTurn)
    assert turn.tool_calls[0] == ToolCall(
        "call-1", "search_knowledge", {"query": "low sodium"}
    )
    assert executor.requests[0].tools[0]["function"]["name"] == (
        "search_knowledge"
    )


def test_openai_adapter_maps_tool_result_transcript_to_provider_messages() -> None:
    messages = [
        UserMessage("question", evidence=(_evidence(),)),
        AssistantToolCallMessage(
            [ToolCall("call-1", "search_knowledge", {"query": "rewrite"})]
        ),
        ToolResultMessage(
            "call-1",
            "search_knowledge",
            ToolResult.failure("invalid_arguments", "query is invalid"),
        ),
    ]

    provider_messages = OpenAICompatibleAgentModel._provider_messages(messages)

    assert provider_messages[0]["role"] == "system"
    user_payload = json.loads(provider_messages[1]["content"])
    assert user_payload["question"] == "question"
    assert user_payload["observed_evidence"][0]["source_id"] == "source-a"
    assert "source_url" not in user_payload["observed_evidence"][0]
    assert provider_messages[2]["tool_calls"][0]["function"]["name"] == (
        "search_knowledge"
    )
    tool_payload = json.loads(provider_messages[3]["content"])
    assert tool_payload == {
        "ok": False,
        "error": {"code": "invalid_arguments", "message": "query is invalid"},
    }


def test_openai_adapter_parses_structured_final_json() -> None:
    adapter, _ = _adapter(content='{"answer":"grounded","citation_ids":["source-a"],"abstain":false}')

    turn = adapter.respond([UserMessage("question")], [_tool_spec()])

    assert isinstance(turn, FinalTurn)
    assert turn.answer == "grounded"
    assert turn.citation_ids == ["source-a"]
    assert turn.abstain is False


def test_openai_adapter_preserves_malformed_tool_arguments_for_structured_validation() -> None:
    adapter, _ = _adapter(tool_calls=({"id": "call-bad", "function": {"name": "search_knowledge", "arguments": "{not-json"}},))
    turn = adapter.respond([UserMessage("question")], [_tool_spec()])

    assert isinstance(turn, ToolCallTurn)
    retriever = lambda query, top_k=3: [_evidence()]
    result = ToolRegistry([SearchKnowledgeTool(retriever)]).execute(turn.tool_calls[0])

    assert not result.ok
    assert result.error is not None
    assert result.error.code == "invalid_arguments"


def test_openai_adapter_rejects_malformed_final_json() -> None:
    adapter, _ = _adapter(content="not-json")

    with pytest.raises(AgentModelError, match="invalid final response"):
        adapter.respond([UserMessage("question")], [_tool_spec()])
