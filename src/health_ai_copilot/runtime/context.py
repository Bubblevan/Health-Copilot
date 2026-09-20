"""Narrow execution identity/context; this is not conversation memory."""

from dataclasses import dataclass, field
from uuid import uuid4

from .budget import RunBudgetConfig, RunBudgetState
from .trace import RunTrace, TraceEventType


@dataclass(frozen=True)
class RunIdentity:
    """Non-content identity for one harness execution."""

    run_id: str
    runtime_mode: str
    code_commit: str | None = None
    config_hash: str | None = None

    @classmethod
    def create(cls, runtime_mode: str) -> "RunIdentity":
        return cls(run_id=f"run-{uuid4().hex}", runtime_mode=runtime_mode)


@dataclass
class RunContext:
    """Execution control-plane state, extended with budgets/traces in later M4 phases."""

    identity: RunIdentity
    budget: RunBudgetState = field(default_factory=lambda: RunBudgetState(RunBudgetConfig()))
    metadata: dict[str, str] = field(default_factory=dict)
    trace: RunTrace | None = None

    @classmethod
    def create(
        cls,
        runtime_mode: str,
        budget: RunBudgetConfig | None = None,
        trace: RunTrace | None = None,
    ) -> "RunContext":
        context = cls(
            identity=RunIdentity.create(runtime_mode),
            budget=RunBudgetState(budget or RunBudgetConfig()),
            trace=trace,
        )
        if trace is not None:
            trace.emit(TraceEventType.RUN_START, run_id=context.identity.run_id, runtime_mode=runtime_mode)
        return context
