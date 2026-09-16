import json
from types import SimpleNamespace

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


class _FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))


def _adapter(response):
    adapter = OpenAICompatibleAgentModel.__new__(OpenAICompatibleAgentModel)
    client = _FakeClient(response)
    adapter._client = client
    adapter._model = "fake-model"
    adapter._temperature = 0.1
    return adapter, client


def _response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _evidence() -> Evidence:
    return Evidence(
        source_id="source-a",
        title="Title A",
        excerpt="Excerpt A",
        source_url="https://example.org/source-a",
        score=1.0,
    )


def test_openai_adapter_maps_native_tool_call_to_tool_call_turn() -> None:
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name="search_knowledge",
                    arguments='{"query":"low sodium"}',
                ),
            )
        ],
    )
    adapter, client = _adapter(_response(message))

    turn = adapter.respond([UserMessage("question")], [_tool_spec()])

    assert isinstance(turn, ToolCallTurn)
    assert turn.tool_calls[0] == ToolCall(
        "call-1", "search_knowledge", {"query": "low sodium"}
    )
    assert client.chat.completions.kwargs["tool_choice"] == "auto"
    assert client.chat.completions.kwargs["tools"][0]["function"]["name"] == (
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
    message = SimpleNamespace(
        content='{"answer":"grounded","citation_ids":["source-a"],"abstain":false}',
        tool_calls=None,
    )
    adapter, _ = _adapter(_response(message))

    turn = adapter.respond([UserMessage("question")], [_tool_spec()])

    assert isinstance(turn, FinalTurn)
    assert turn.answer == "grounded"
    assert turn.citation_ids == ["source-a"]
    assert turn.abstain is False


def test_openai_adapter_preserves_malformed_tool_arguments_for_structured_validation() -> None:
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call-bad",
                function=SimpleNamespace(
                    name="search_knowledge",
                    arguments="{not-json",
                ),
            )
        ],
    )
    adapter, _ = _adapter(_response(message))
    turn = adapter.respond([UserMessage("question")], [_tool_spec()])

    assert isinstance(turn, ToolCallTurn)
    retriever = lambda query, top_k=3: [_evidence()]
    result = ToolRegistry([SearchKnowledgeTool(retriever)]).execute(turn.tool_calls[0])

    assert not result.ok
    assert result.error is not None
    assert result.error.code == "invalid_arguments"


def test_openai_adapter_rejects_malformed_final_json() -> None:
    message = SimpleNamespace(content="not-json", tool_calls=None)
    adapter, _ = _adapter(_response(message))

    with pytest.raises(AgentModelError, match="invalid final response"):
        adapter.respond([UserMessage("question")], [_tool_spec()])
