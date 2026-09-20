"""Monotonic runtime budgets enforced before provider or tool side effects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .provider import ProviderUsage


class BudgetDenied(RuntimeError):
    """A guard denied a side effect before it occurred."""


@dataclass(frozen=True)
class RunBudgetConfig:
    max_provider_calls: int | None = None
    max_tool_executions: int | None = None
    deadline_ms: float | None = None
    max_total_tokens: int | None = None


@dataclass
class RunBudgetState:
    config: RunBudgetConfig
    clock: Callable[[], float] = monotonic
    started_at: float = field(init=False)
    provider_calls_used: int = 0
    tool_executions_used: int = 0
    input_tokens_used: int | None = 0
    output_tokens_used: int | None = 0
    total_tokens_used: int | None = 0

    def __post_init__(self) -> None:
        self.started_at = self.clock()

    @property
    def elapsed_ms(self) -> float:
        return (self.clock() - self.started_at) * 1000

    def guard_provider(self) -> float | None:
        self._guard_deadline()
        if self.config.max_provider_calls is not None and self.provider_calls_used >= self.config.max_provider_calls:
            raise BudgetDenied("provider_call_budget_exceeded")
        if self.config.max_total_tokens is not None:
            if self.total_tokens_used is None:
                raise BudgetDenied("token_usage_unknown")
            if self.total_tokens_used >= self.config.max_total_tokens:
                raise BudgetDenied("token_budget_exceeded")
        self.provider_calls_used += 1
        return self.remaining_seconds()

    def guard_tool(self) -> None:
        self._guard_deadline()
        if self.config.max_tool_executions is not None and self.tool_executions_used >= self.config.max_tool_executions:
            raise BudgetDenied("tool_execution_budget_exceeded")
        self.tool_executions_used += 1

    def record_usage(self, usage: ProviderUsage | None) -> None:
        if usage is None:
            if self.config.max_total_tokens is not None:
                self.input_tokens_used = self.output_tokens_used = self.total_tokens_used = None
            return
        self.input_tokens_used = _add(self.input_tokens_used, usage.input_tokens)
        self.output_tokens_used = _add(self.output_tokens_used, usage.output_tokens)
        self.total_tokens_used = _add(self.total_tokens_used, usage.total_tokens)

    def remaining_seconds(self) -> float | None:
        if self.config.deadline_ms is None:
            return None
        return max(0.0, (self.config.deadline_ms - self.elapsed_ms) / 1000)

    def _guard_deadline(self) -> None:
        if self.config.deadline_ms is not None and self.elapsed_ms >= self.config.deadline_ms:
            raise BudgetDenied("runtime_deadline_exceeded")


def _add(current: int | None, incoming: int | None) -> int | None:
    return None if current is None or incoming is None else current + incoming
