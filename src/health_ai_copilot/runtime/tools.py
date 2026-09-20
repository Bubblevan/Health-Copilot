"""Runtime tool boundary: guard before dispatching to the existing registry."""

from typing import Protocol

from ..agent.messages import ToolCall
from ..agent.tools import ToolRegistry, ToolResult
from .budget import BudgetDenied
from .context import RunContext


class ToolRunner(Protocol):
    def execute(self, call: ToolCall, runtime: RunContext) -> ToolResult:
        ...


class LiveToolRunner:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(self, call: ToolCall, runtime: RunContext) -> ToolResult:
        try:
            runtime.budget.guard_tool()
        except BudgetDenied as exc:
            return ToolResult.failure("runtime_budget_denied", str(exc))
        return self.registry.execute(call)
