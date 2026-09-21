"""Narrow execution identity/context; this is not conversation memory."""

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
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
    profile_id: str | None = None
    component_manifest_hash: str | None = None

    @classmethod
    def create(
        cls,
        runtime_mode: str,
        *,
        profile_id: str | None = None,
        component_manifest_hash: str | None = None,
        config_hash: str | None = None,
        code_commit: str | None = None,
    ) -> "RunIdentity":
        return cls(
            run_id=f"run-{uuid4().hex}",
            runtime_mode=runtime_mode,
            code_commit=code_commit,
            config_hash=config_hash or component_manifest_hash,
            profile_id=profile_id,
            component_manifest_hash=component_manifest_hash,
        )


def call_with_optional_runtime(
    operation: Callable[..., Any], *args: Any, runtime: "RunContext"
) -> Any:
    """Call old test fakes and new adapters without storing run state on either."""

    try:
        parameters = inspect.signature(operation).parameters.values()
        supports_runtime = any(
            parameter.name == "runtime" or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        supports_runtime = True
    if supports_runtime:
        return operation(*args, runtime=runtime)
    return operation(*args)


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
        profile_id: str | None = None,
        component_manifest_hash: str | None = None,
        config_hash: str | None = None,
        code_commit: str | None = None,
        session_revision: int | None = None,
        memory_snapshot_hash: str | None = None,
        initial_context_plan_hash: str | None = None,
        current_context_plan_hash: str | None = None,
        context_plan_hashes: tuple[str, ...] | None = None,
    ) -> "RunContext":
        context = cls(
            identity=RunIdentity.create(
                runtime_mode,
                profile_id=profile_id,
                component_manifest_hash=component_manifest_hash,
                config_hash=config_hash,
                code_commit=code_commit,
            ),
            budget=RunBudgetState(budget or RunBudgetConfig()),
            trace=trace,
        )
        if session_revision is not None:
            context.metadata["session_revision"] = str(session_revision)
        if memory_snapshot_hash is not None:
            context.metadata["memory_snapshot_hash"] = memory_snapshot_hash
        if initial_context_plan_hash is not None:
            context.metadata["initial_context_plan_hash"] = initial_context_plan_hash
        if current_context_plan_hash is not None:
            context.metadata["current_context_plan_hash"] = current_context_plan_hash
        if context_plan_hashes is not None:
            context.metadata["context_plan_hashes"] = json.dumps(
                list(context_plan_hashes), ensure_ascii=False, separators=(",", ":")
            )
        if trace is not None:
            trace.emit(
                TraceEventType.RUN_START,
                run_id=context.identity.run_id,
                runtime_mode=runtime_mode,
                profile_id=profile_id,
                component_manifest_hash=component_manifest_hash,
                code_commit=code_commit,
            )
        return context


# M10 keeps RunContext and model-call Context distinct, while exposing the
# context selection API from the legacy runtime namespace for callers that
# already import runtime.context.
from .context_manager import ContextBudget, ContextItem, ContextManager, ContextPlan  # noqa: F401
