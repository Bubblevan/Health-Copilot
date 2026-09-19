"""Small in-memory lifecycle events for observable agent runs."""

from dataclasses import dataclass
from enum import StrEnum


class AgentEventType(StrEnum):
    AGENT_START = "agent_start"
    TURN_START = "turn_start"
    MODEL_RESPONSE = "model_response"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    POLICY_START = "policy_start"
    POLICY_END = "policy_end"
    GROUNDING_START = "grounding_start"
    GROUNDING_END = "grounding_end"
    TURN_END = "turn_end"
    AGENT_END = "agent_end"


@dataclass(frozen=True)
class AgentEvent:
    """Metadata only: event payloads intentionally exclude questions, arguments and answers."""

    event_type: AgentEventType
    session_id: str
    turn: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    response_kind: str | None = None
    success: bool | None = None
    stop_reason: str | None = None
    error_code: str | None = None

    @property
    def type(self) -> AgentEventType:
        """Convenience alias matching common event-stream terminology."""
        return self.event_type
