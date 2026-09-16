"""Typed, observable messages exchanged by the bounded agent runtime."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from ..contracts import Evidence, GenerationDraft

if TYPE_CHECKING:
    from .tools import ToolResult


@dataclass(frozen=True)
class ToolCall:
    """One provider-native function/tool call."""

    id: str
    name: str
    arguments: object


@dataclass(frozen=True)
class UserMessage:
    """The user question plus the evidence supplied to the first model turn."""

    content: str
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class AssistantFinalMessage:
    """A structured final answer; no hidden reasoning is represented here."""

    answer: str
    citation_ids: list[str] = field(default_factory=list)
    abstain: bool = False

    def to_draft(self) -> GenerationDraft:
        return GenerationDraft(
            answer=self.answer,
            citation_ids=list(self.citation_ids),
            abstain=self.abstain,
        )


@dataclass(frozen=True)
class AssistantToolCallMessage:
    """An assistant turn containing explicit provider-native tool calls."""

    tool_calls: list[ToolCall]


@dataclass(frozen=True)
class ToolResultMessage:
    """A tool observation appended to the session transcript."""

    tool_call_id: str
    tool_name: str
    result: "ToolResult"


AgentMessage: TypeAlias = (
    UserMessage | AssistantFinalMessage | AssistantToolCallMessage | ToolResultMessage
)


@dataclass(frozen=True)
class FinalTurn:
    """Provider response variant for a structured final answer."""

    answer: str
    citation_ids: list[str] = field(default_factory=list)
    abstain: bool = False

    @classmethod
    def from_draft(cls, draft: GenerationDraft) -> "FinalTurn":
        return cls(draft.answer, list(draft.citation_ids), draft.abstain)

    def as_message(self) -> AssistantFinalMessage:
        return AssistantFinalMessage(
            answer=self.answer,
            citation_ids=list(self.citation_ids),
            abstain=self.abstain,
        )

    def to_draft(self) -> GenerationDraft:
        return self.as_message().to_draft()


@dataclass(frozen=True)
class ToolCallTurn:
    """Provider response variant for explicit tool calls."""

    tool_calls: list[ToolCall]

    def as_message(self) -> AssistantToolCallMessage:
        return AssistantToolCallMessage(tool_calls=list(self.tool_calls))


AssistantTurn: TypeAlias = FinalTurn | ToolCallTurn
