from health_ai_copilot.agent.messages import ToolCall
from health_ai_copilot.agent.tools import FunctionTool, ToolRegistry, ToolResult, ToolSpec
from health_ai_copilot.runtime import (
    FakeProviderExecutor,
    ProviderCallKind,
    ProviderFailure,
    ProviderFailureKind,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    RunContext,
)
from health_ai_copilot.runtime.budget import BudgetDenied, RunBudgetConfig, RunBudgetState
from health_ai_copilot.runtime.tools import LiveToolRunner


def _request() -> ProviderRequest:
    return ProviderRequest.create(
        kind=ProviderCallKind.GENERATOR,
        model="fixture",
        messages=({"role": "user", "content": "fixture"},),
    )


def test_provider_budget_denies_before_fake_side_effect() -> None:
    runtime = RunContext.create("test", RunBudgetConfig(max_provider_calls=0))
    executor = FakeProviderExecutor(
        [ProviderResponse("recorded", ProviderCallKind.GENERATOR, "fixture", "{}")]
    )

    try:
        executor.execute(_request(), runtime)
    except ProviderFailure as error:
        assert error.kind == ProviderFailureKind.BUDGET_DENIED
    else:
        raise AssertionError("provider side effect should be denied")
    assert executor.requests == []


def test_deadline_uses_injected_monotonic_clock() -> None:
    now = [0.0]
    budget = RunBudgetState(RunBudgetConfig(deadline_ms=5), clock=lambda: now[0])
    now[0] = 0.006

    try:
        budget.guard_tool()
    except BudgetDenied as error:
        assert str(error) == "runtime_deadline_exceeded"
    else:
        raise AssertionError("expired deadline should deny tool")


def test_token_budget_denies_the_next_provider_call_before_side_effect() -> None:
    runtime = RunContext.create("test", RunBudgetConfig(max_total_tokens=2))
    executor = FakeProviderExecutor(
        [
            ProviderResponse(
                "first",
                ProviderCallKind.GENERATOR,
                "fixture",
                "{}",
                usage=ProviderUsage(total_tokens=2),
            ),
            ProviderResponse("second", ProviderCallKind.GENERATOR, "fixture", "{}"),
        ]
    )

    executor.execute(_request(), runtime)
    try:
        executor.execute(_request(), runtime)
    except ProviderFailure as error:
        assert error.kind == ProviderFailureKind.BUDGET_DENIED
    else:
        raise AssertionError("token-budget-exhausted call should be denied")
    assert len(executor.requests) == 1


def test_missing_usage_fails_closed_when_hard_token_budget_is_enabled() -> None:
    runtime = RunContext.create("test", RunBudgetConfig(max_total_tokens=10))
    executor = FakeProviderExecutor(
        [
            ProviderResponse("first", ProviderCallKind.GENERATOR, "fixture", "{}"),
            ProviderResponse("second", ProviderCallKind.GENERATOR, "fixture", "{}"),
        ]
    )

    executor.execute(_request(), runtime)
    assert runtime.budget.total_tokens_used is None
    try:
        executor.execute(_request(), runtime)
    except ProviderFailure as error:
        assert error.kind == ProviderFailureKind.BUDGET_DENIED
    else:
        raise AssertionError("unknown usage should fail closed under a hard token cap")
    assert len(executor.requests) == 1


def test_tool_budget_denies_before_registry_handler_runs() -> None:
    calls: list[ToolCall] = []

    def handler(arguments: object) -> ToolResult:
        calls.append(ToolCall("tool-call-1", "search_knowledge", arguments))
        return ToolResult.success("tool")

    tool = FunctionTool(
        ToolSpec(
            "search_knowledge",
            "fixture",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        ),
        handler,
        lambda arguments: arguments,
    )
    runner = LiveToolRunner(ToolRegistry([tool]))
    runtime = RunContext.create("test", RunBudgetConfig(max_tool_executions=0))
    result = runner.execute(
        ToolCall("tool-call-1", "search_knowledge", {"query": "fixture"}), runtime
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "runtime_budget_denied"
    assert calls == []
