"""Minimal in-memory conversation state for one agent run."""

from dataclasses import dataclass, field
from uuid import uuid4

from .messages import AgentMessage


@dataclass
class AgentSession:
    """A run transcript, not long-term memory or cross-process persistence."""

    session_id: str = field(default_factory=lambda: uuid4().hex)
    messages: list[AgentMessage] = field(default_factory=list)

    def append(self, message: AgentMessage) -> None:
        self.messages.append(message)
