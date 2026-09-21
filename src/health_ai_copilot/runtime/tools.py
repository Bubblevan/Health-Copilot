"""Runtime tool boundary: guard before dispatching to the existing registry."""

from typing import TYPE_CHECKING, Protocol

from .budget import BudgetDenied
from .context import RunContext
from .trace import TraceEventType, canonical_json_sha256

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
            if runtime.trace is not None:
                runtime.trace.emit(
                    TraceEventType.BUDGET_DENIED,
                    side_effect="tool",
                    reason=str(exc),
                )
            # Deferred import avoids a package-import cycle with AgentLoop.
            from ..agent.tools import ToolResult

            return ToolResult.failure("runtime_budget_denied", str(exc))
        if runtime.trace is not None:
            runtime.trace.emit(
                TraceEventType.TOOL_START,
                tool_name=call.name,
                tool_call_id=call.id,
                arguments_fingerprint=canonical_json_sha256(call.arguments),
            )
        result = self.registry.execute(call, runtime=runtime)
        if runtime.trace is not None:
            runtime.trace.emit(
                TraceEventType.TOOL_END,
                tool_name=call.name,
                tool_call_id=call.id,
                success=result.ok,
                error_code=result.error.code if result.error else None,
                observed_evidence_count=len(result.observed_evidence),
            )
        return result
