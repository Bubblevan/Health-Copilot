import pytest

from health_ai_copilot.agent.messages import ToolCall
from health_ai_copilot.agent.tools import ToolResult
from health_ai_copilot.runtime import (
    FailureInjectingProviderExecutor,
    FailureInjectingToolRunner,
    FailureInjectionPlan,
    FakeProviderExecutor,
    ProviderCallKind,
    ProviderFailure,
    ProviderFailureKind,
    ProviderRequest,
    ProviderResponse,
    RunContext,
)


def test_provider_failure_is_injected_before_delegate_side_effect() -> None:
    delegate = FakeProviderExecutor(
        [ProviderResponse("", ProviderCallKind.AGENT, "fixture", "{}")]
    )
    injected = FailureInjectingProviderExecutor(
        delegate,
        FailureInjectionPlan({ProviderCallKind.AGENT: ProviderFailureKind.TIMEOUT}),
    )
    request = ProviderRequest.create(
        kind=ProviderCallKind.AGENT,
        model="fixture",
        messages=({"role": "user", "content": "fixture"},),
    )

    with pytest.raises(ProviderFailure) as error:
        injected.execute(request, RunContext.create("counterfactual"))

    assert error.value.kind == ProviderFailureKind.TIMEOUT
    assert delegate.requests == []


def test_tool_failure_is_injected_before_delegate_side_effect() -> None:
    calls: list[ToolCall] = []

    class Delegate:
        def execute(self, call: ToolCall, runtime: RunContext) -> ToolResult:
            calls.append(call)
            return ToolResult.success("live result")

    injected = FailureInjectingToolRunner(
        Delegate(), FailureInjectionPlan({}, tool_failure_code="tool_timeout")
    )
    result = injected.execute(
        ToolCall("call-1", "search_knowledge", {"query": "fixture"}),
        RunContext.create("counterfactual"),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "tool_timeout"
    assert calls == []
