"""Mutable execution state and explicit stop reasons for M1."""

from dataclasses import dataclass, field
from enum import StrEnum

from ..contracts import Evidence, GenerationDraft
from ..verification.grounding import GroundedClaim
from .session import AgentSession


class AgentStatus(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"


class StopReason(StrEnum):
    """Terminal runtime outcomes; tool errors remain non-terminal observations."""

    FINAL = "final"
    ABSTAIN = "abstain"
    MAX_MODEL_TURNS = "max_model_turns"
    MAX_TOOL_CALLS = "max_tool_calls"
    MODEL_ERROR = "model_error"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    EVIDENCE_CONFLICTING = "evidence_conflicting"
    POLICY_ERROR = "policy_error"
    GROUNDING_FAILED = "grounding_failed"
    VERIFIER_ERROR = "verifier_error"
    CONTEXT_BUDGET_EXCEEDED = "context_budget_exhausted"
    CONTEXT_PROJECTION_ERROR = "context_projection_error"


@dataclass
class AgentState:
    """Current bounded-run state; product configuration stays outside this object."""

    session: AgentSession
    model_turns_used: int = 0
    tool_calls_used: int = 0
    tool_proposals_used: int = 0
    policy_calls_used: int = 0
    verifier_calls_used: int = 0
    policy_decision: str | None = None
    policy_reason_codes: tuple[str, ...] = ()
    policy_supporting_source_ids: tuple[str, ...] = ()
    policy_matched_topic_ids: tuple[str, ...] = ()
    status: AgentStatus = AgentStatus.RUNNING
    stop_reason: StopReason | None = None
    final_draft: GenerationDraft | None = None
    final_claims: tuple[GroundedClaim, ...] = ()
    initial_ranked_evidence: list[Evidence] = field(default_factory=list)
    recovery_ranked_evidence: list[Evidence] = field(default_factory=list)
    observed_evidence: list[Evidence] = field(default_factory=list)

    def add_evidence(self, evidence: list[Evidence] | tuple[Evidence, ...]) -> None:
        """Accumulate evidence by source ID while preserving first-seen order."""
        seen = {item.source_id for item in self.observed_evidence}
        for item in evidence:
            if item.source_id not in seen:
                self.observed_evidence.append(item)
                seen.add(item.source_id)

    def add_recovery_evidence(self, evidence: list[Evidence] | tuple[Evidence, ...]) -> None:
        """Keep recovery ranking separate from the evidence union."""
        self.recovery_ranked_evidence.extend(evidence)
        self.add_evidence(evidence)

    def stop(self, reason: StopReason, draft: GenerationDraft | None = None) -> None:
        self.status = AgentStatus.STOPPED
        self.stop_reason = reason
        self.final_draft = draft
