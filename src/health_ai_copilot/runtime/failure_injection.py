"""Narrow offline failure injection for M4 replay/counterfactual diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .context import RunContext
from .provider import (
    ProviderCallKind,
    ProviderExecutor,
    ProviderFailure,
    ProviderFailureKind,
    ProviderRequest,
    ProviderResponse,
)

if TYPE_CHECKING:
    from ..agent.messages import ToolCall
    from ..agent.tools import ToolResult


@dataclass(frozen=True)
class FailureInjectionPlan:
    """A deliberately small, deterministic injection plan for one replay run."""

    provider_failures: Mapping[ProviderCallKind, ProviderFailureKind]
    tool_failure_code: str | None = None


class FailureInjectingProviderExecutor:
    """Fails selected provider kinds before delegating; no live fallback is possible."""

    def __init__(self, delegate: ProviderExecutor, plan: FailureInjectionPlan) -> None:
        self.delegate = delegate
        self.plan = plan
        self.injected: list[ProviderCallKind] = []

    def execute(self, request: ProviderRequest, runtime: RunContext) -> ProviderResponse:
        failure = self.plan.provider_failures.get(request.kind)
        if failure is not None:
            self.injected.append(request.kind)
            raise ProviderFailure(failure)
        return self.delegate.execute(request, runtime)


class FailureInjectingToolRunner:
    """Returns a controlled observation before tool dispatch when configured."""

    def __init__(self, delegate: Any, plan: FailureInjectionPlan) -> None:
        self.delegate = delegate
        self.plan = plan
        self.injected_call_ids: list[str] = []

    def execute(self, call: ToolCall, runtime: RunContext) -> ToolResult:
        if self.plan.tool_failure_code is not None:
            from ..agent.tools import ToolResult

            self.injected_call_ids.append(call.id)
            return ToolResult.failure(self.plan.tool_failure_code, "injected runtime failure")
        return self.delegate.execute(call, runtime)
