from health_ai_copilot.runtime import (
    FakeProviderExecutor,
    ProviderCallKind,
    ProviderFailure,
    ProviderFailureKind,
    ProviderRequest,
    ProviderResponse,
    RunContext,
)
from health_ai_copilot.runtime.budget import BudgetDenied, RunBudgetConfig, RunBudgetState


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
