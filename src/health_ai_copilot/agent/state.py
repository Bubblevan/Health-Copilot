"""Mutable execution state and explicit stop reasons for M1."""

from dataclasses import dataclass, field
from enum import StrEnum

from ..contracts import Evidence, GenerationDraft
from .session import AgentSession


class AgentStatus(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"


class StopReason(StrEnum):
    FINAL = "final"
    ABSTAIN = "abstain"
    MAX_MODEL_TURNS = "max_model_turns"
    MAX_TOOL_CALLS = "max_tool_calls"
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"


@dataclass
class AgentState:
    """Current bounded-run state; product configuration stays outside this object."""

    session: AgentSession
    model_turns_used: int = 0
    tool_calls_used: int = 0
    status: AgentStatus = AgentStatus.RUNNING
    stop_reason: StopReason | None = None
    final_draft: GenerationDraft | None = None
    observed_evidence: list[Evidence] = field(default_factory=list)

    def add_evidence(self, evidence: list[Evidence] | tuple[Evidence, ...]) -> None:
        """Accumulate evidence by source ID while preserving first-seen order."""
        seen = {item.source_id for item in self.observed_evidence}
        for item in evidence:
            if item.source_id not in seen:
                self.observed_evidence.append(item)
                seen.add(item.source_id)

    def stop(self, reason: StopReason, draft: GenerationDraft | None = None) -> None:
        self.status = AgentStatus.STOPPED
        self.stop_reason = reason
        self.final_draft = draft
