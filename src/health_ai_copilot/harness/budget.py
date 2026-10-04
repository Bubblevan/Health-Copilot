"""One per-request budget ledger shared by providers, tools, and reasoners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from time import monotonic


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetLimits:
    max_provider_calls: int = 32
    max_tool_calls: int = 16
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    deadline_ms: float = 120_000

    def __post_init__(self) -> None:
        for name in ("max_provider_calls", "max_tool_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("max_input_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if not isfinite(self.deadline_ms) or self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")


class BudgetLedger:
    def __init__(self, limits: BudgetLimits, *, clock: Callable[[], float] = monotonic) -> None:
        self.limits = limits
        self._clock = clock
        self._started = clock()
        self.provider_calls = 0
        self.tool_calls = 0
        self.input_tokens: int | None = 0
        self.output_tokens: int | None = 0
        self._reserved_output_tokens = 0

    @property
    def elapsed_ms(self) -> float:
        return (self._clock() - self._started) * 1000

    def reserve_provider(self, requested_output_tokens: int) -> tuple[float, int]:
        self._check_deadline()
        if self.provider_calls >= self.limits.max_provider_calls:
            raise BudgetExceeded("provider_call_budget_exceeded")
        allowed_output = requested_output_tokens
        if self.limits.max_output_tokens is not None:
            if self.output_tokens is None:
                raise BudgetExceeded("output_token_usage_unknown")
            remaining = (
                self.limits.max_output_tokens
                - self.output_tokens
                - self._reserved_output_tokens
            )
            if remaining <= 0:
                raise BudgetExceeded("output_token_budget_exceeded")
            allowed_output = min(requested_output_tokens, remaining)
            self._reserved_output_tokens += allowed_output
        self.provider_calls += 1
        timeout = max(0.001, (self.limits.deadline_ms - self.elapsed_ms) / 1000)
        return timeout, allowed_output

    def reserve_tool(self) -> float:
        self._check_deadline()
        if self.tool_calls >= self.limits.max_tool_calls:
            raise BudgetExceeded("tool_call_budget_exceeded")
        self.tool_calls += 1
        return max(0.001, (self.limits.deadline_ms - self.elapsed_ms) / 1000)

    def release_output_reservation(self, reserved_tokens: int) -> None:
        self._reserved_output_tokens = max(0, self._reserved_output_tokens - reserved_tokens)

    def record_usage(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
        *,
        reserved_output_tokens: int = 0,
    ) -> None:
        self.release_output_reservation(reserved_output_tokens)
        if input_tokens is None:
            self.input_tokens = None
            if self.limits.max_input_tokens is not None:
                raise BudgetExceeded("input_token_usage_unknown")
        elif self.input_tokens is not None:
            self.input_tokens += input_tokens
        if output_tokens is None:
            self.output_tokens = None
            if self.limits.max_output_tokens is not None:
                raise BudgetExceeded("output_token_usage_unknown")
        elif self.output_tokens is not None:
            self.output_tokens += output_tokens
        if (
            self.limits.max_input_tokens is not None
            and self.input_tokens is not None
            and self.input_tokens > self.limits.max_input_tokens
        ):
            raise BudgetExceeded("input_token_budget_exceeded")
        if (
            self.limits.max_output_tokens is not None
            and self.output_tokens is not None
            and self.output_tokens > self.limits.max_output_tokens
        ):
            raise BudgetExceeded("output_token_budget_exceeded")

    def _check_deadline(self) -> None:
        if self.elapsed_ms >= self.limits.deadline_ms:
            raise BudgetExceeded("deadline_exceeded")
