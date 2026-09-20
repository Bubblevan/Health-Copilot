"""Narrow execution identity/context; this is not conversation memory."""

from dataclasses import dataclass, field
from uuid import uuid4


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
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(cls, runtime_mode: str) -> "RunContext":
        return cls(identity=RunIdentity.create(runtime_mode))
