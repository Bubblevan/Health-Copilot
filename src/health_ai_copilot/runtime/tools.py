"""Runtime tool boundary: guard before dispatching to the existing registry."""

from typing import TYPE_CHECKING, Protocol

from .budget import BudgetDenied
from .context import RunContext

if TYPE_CHECKING:
    from ..agent.messages import ToolCall
    from ..agent.tools import ToolRegistry, ToolResult


class ToolRunner(Protocol):
    def execute(self, call: "ToolCall", runtime: RunContext) -> "ToolResult":
        ...


class LiveToolRunner:
    def __init__(self, registry: "ToolRegistry") -> None:
        self.registry = registry

    def execute(self, call: "ToolCall", runtime: RunContext) -> "ToolResult":
        try:
            runtime.budget.guard_tool()
        except BudgetDenied as exc:
            # Deferred import avoids a package-import cycle with AgentLoop.
            from ..agent.tools import ToolResult

            return ToolResult.failure("runtime_budget_denied", str(exc))
        return self.registry.execute(call)
