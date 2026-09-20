import pytest

from health_ai_copilot.agent.messages import ToolCall
from health_ai_copilot.agent.tools import ToolResult
from health_ai_copilot.runtime import (
    FakeProviderExecutor,
    ProviderCallKind,
    ProviderFailure,
    ProviderFailureKind,
    ProviderRequest,
    ProviderResponse,
    RecordedToolExchange,
    RecordingProviderExecutor,
    ReplayProviderExecutor,
    ReplayToolRunner,
    RunContext,
    read_provider_exchanges,
    read_tool_exchanges,
    write_provider_exchanges,
    write_tool_exchanges,
)


def _request(*, model: str = "fixture") -> ProviderRequest:
    return ProviderRequest.create(
        kind=ProviderCallKind.GENERATOR,
        model=model,
        messages=({"role": "user", "content": "reviewed fixture"},),
        temperature=0.1,
    )


def test_recorded_provider_exchange_replays_without_live_delegate(tmp_path) -> None:
    recording = RecordingProviderExecutor(
        FakeProviderExecutor(
                [ProviderResponse("", ProviderCallKind.GENERATOR, "fixture", "recorded")]
        )
    )
    recording.execute(_request(), RunContext.create("live"))
    path = tmp_path / "provider_exchanges.jsonl"
    write_provider_exchanges(path, recording.exchanges)

    replay = ReplayProviderExecutor(read_provider_exchanges(path))
    response = replay.execute(_request(), RunContext.create("replay"))

    assert response.content == "recorded"
    assert replay.remaining_exchanges == 0


def test_replay_provider_rejects_request_contract_mismatch() -> None:
    recorded = RecordingProviderExecutor(
        FakeProviderExecutor(
                [ProviderResponse("", ProviderCallKind.GENERATOR, "fixture", "recorded")]
        )
    )
    recorded.execute(_request(), RunContext.create("live"))
    replay = ReplayProviderExecutor(recorded.exchanges)

    with pytest.raises(ProviderFailure) as error:
        replay.execute(_request(model="other-model"), RunContext.create("replay"))

    assert error.value.kind == ProviderFailureKind.REPLAY_MISMATCH


def test_replay_tool_runner_returns_recorded_result_without_live_registry(tmp_path) -> None:
    exchange = RecordedToolExchange(
        "search_knowledge",
        {"query": "reviewed fixture"},
        ToolResult.success("recorded result"),
    )
    path = tmp_path / "tool_exchanges.jsonl"
    write_tool_exchanges(path, [exchange])
    replay = ReplayToolRunner(read_tool_exchanges(path))

    result = replay.execute(
        ToolCall("call-1", "search_knowledge", {"query": "reviewed fixture"}),
        RunContext.create("replay"),
    )

    assert result.ok is True
    assert result.data == "recorded result"
    assert replay.remaining_exchanges == 0
