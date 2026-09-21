"""Executable projection from a ContextPlan to provider-facing messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..contracts import Evidence
from .context import RunContext
from .context_manager import (
    ContextAtomicityViolation,
    ContextItem,
    ContextItemCategory,
    ContextManager,
    ContextPlan,
    ContextPriority,
)
from .memory import MemoryRecord, MemorySnapshotIdentity
from .session import SessionEvent, SessionStore

if TYPE_CHECKING:
    from ..agent.messages import AgentMessage, ToolResultMessage, UserMessage
    from ..agent.session import AgentSession
    from .memory import MemoryStore


class ContextProjectionError(RuntimeError):
    """Raised when a selected plan cannot be projected atomically."""


@dataclass(frozen=True)
class ProjectedContext:
    """The only provider-visible context produced by an M10 projector."""

    messages: tuple[AgentMessage, ...]
    plan: ContextPlan
    memory_snapshot: MemorySnapshotIdentity | None = None

    @property
    def memory_snapshot_hash(self) -> str | None:
        return self.memory_snapshot.snapshot_sha256 if self.memory_snapshot else None


@runtime_checkable
class ContextProjector(Protocol):
    def project(
        self,
        *,
        session: AgentSession,
        current_evidence: Sequence[Evidence],
        model_turn: int,
        runtime: RunContext,
    ) -> ProjectedContext: ...


class SessionContextProjector:
    """Project persistent history and selected memory for every model turn."""

    version = "m10-projector-v1"

    def __init__(
        self,
        *,
        context_manager: ContextManager,
        session_store: SessionStore,
        session_id: str,
        question: str,
        memory_records: Sequence[MemoryRecord] = (),
        memory_store: MemoryStore | None = None,
        retrieval_query: str | None = None,
    ) -> None:
        self.context_manager = context_manager
        self.session_store = session_store
        self.session_id = session_id
        self.question = question
        self.memory_records = tuple(memory_records)
        self.memory_store = memory_store
        self.retrieval_query = retrieval_query

    def project(
        self,
        *,
        session: AgentSession,
        current_evidence: Sequence[Evidence],
        model_turn: int,
        runtime: RunContext,
    ) -> ProjectedContext:
        if model_turn <= 0:
            raise ContextProjectionError("model_turn must be positive")
        persistent_events = self.session_store.list_events(self.session_id)
        persistent_session = self.session_store.resume_session(self.session_id)
        current_items, current_messages = self._current_items(session)
        snapshot = (
            self.memory_store.snapshot(
                self.session_id, session_revision=persistent_session.current_revision
            )
            if self.memory_store is not None
            else None
        )
        plan = self.context_manager.build_plan(
            session_id=self.session_id,
            session_revision=persistent_session.current_revision,
            current_user=self.question,
            current_evidence=current_evidence,
            memory_records=self.memory_records,
            history=persistent_events,
            retrieval_query=self.retrieval_query,
            extra_items=current_items,
        )
        messages = self._project_messages(plan, persistent_events, current_messages)
        self._validate_projection(plan, messages, current_messages)
        runtime.metadata.update(
            {
                "session_revision": str(persistent_session.current_revision),
                "memory_snapshot_hash": snapshot.snapshot_sha256 if snapshot else "",
                "current_context_plan_hash": plan.plan_hash,
            }
        )
        if model_turn == 1:
            runtime.metadata["initial_context_plan_hash"] = plan.plan_hash
        return ProjectedContext(messages, plan, snapshot)

    def _current_items(
        self, session: AgentSession
    ) -> tuple[list[ContextItem], dict[str, AgentMessage]]:
        from ..agent.messages import (
            AssistantFinalMessage,
            AssistantToolCallMessage,
            MemoryContextMessage,
            ToolResultMessage,
            UserMessage,
        )

        items: list[ContextItem] = []
        messages: dict[str, AgentMessage] = {}
        for index, message in enumerate(session.messages):
            if isinstance(message, (UserMessage, MemoryContextMessage)):
                continue
            if isinstance(message, AssistantToolCallMessage):
                call_ids = [call.id for call in message.tool_calls]
                if not call_ids:
                    continue
                item_id = f"agent-tool-call-{index}"
                items.append(
                    ContextItem(
                        item_id=item_id,
                        category=ContextItemCategory.TOOL_EXCHANGE,
                        content={"tool_calls": [_tool_call_dict(call) for call in message.tool_calls]},
                        estimated_tokens=self.context_manager.estimator.estimate(
                            {"tool_calls": [_tool_call_dict(call) for call in message.tool_calls]}
                        ),
                        priority=ContextPriority.PROTECTED,
                        protected=True,
                        provenance={"current_message_id": item_id, "message_type": "tool_call"},
                        group_id=call_ids[0],
                    )
                )
                messages[item_id] = message
            elif isinstance(message, ToolResultMessage):
                item_id = f"agent-tool-result-{index}"
                content = _tool_result_dict(message)
                items.append(
                    ContextItem(
                        item_id=item_id,
                        category=ContextItemCategory.TOOL_EXCHANGE,
                        content=content,
                        estimated_tokens=self.context_manager.estimator.estimate(content),
                        priority=ContextPriority.PROTECTED,
                        protected=True,
                        provenance={"current_message_id": item_id, "message_type": "tool_result"},
                        group_id=message.tool_call_id,
                    )
                )
                messages[item_id] = message
            elif isinstance(message, AssistantFinalMessage):
                item_id = f"agent-final-{index}"
                content = {
                    "answer": message.answer,
                    "citation_ids": list(message.citation_ids),
                    "abstain": message.abstain,
                }
                items.append(
                    ContextItem(
                        item_id=item_id,
                        category=ContextItemCategory.RECENT_HISTORY,
                        content=content,
                        estimated_tokens=self.context_manager.estimator.estimate(content),
                        priority=ContextPriority.NORMAL,
                        provenance={"current_message_id": item_id, "message_type": "assistant_final"},
                    )
                )
                messages[item_id] = message
        return items, messages

    def _project_messages(
        self,
        plan: ContextPlan,
        persistent_events: Sequence[SessionEvent],
        current_messages: Mapping[str, AgentMessage],
    ) -> tuple[AgentMessage, ...]:
        from ..agent.messages import (
            MemoryContextMessage,
            SessionContextItem,
            SessionContextMessage,
        )

        selected_ids = {item.item_id for item in plan.items}
        selected_memory_ids = set(plan.selected_memory_ids)
        selected_records = tuple(
            record for record in self.memory_records if record.memory_id in selected_memory_ids
        )
        output: list[AgentMessage] = []
        if selected_records:
            output.append(MemoryContextMessage(selected_records))

        persistent_by_event = {event.event_id: event for event in persistent_events}
        history_items: list[SessionContextItem] = []
        for item in plan.items:
            if item.item_id not in selected_ids:
                continue
            if item.item_id in current_messages:
                continue
            if item.category not in {
                ContextItemCategory.RECENT_HISTORY,
                ContextItemCategory.TOOL_EXCHANGE,
                ContextItemCategory.SESSION_SUMMARY,
            }:
                continue
            event_id = item.provenance.get("event_id")
            event = persistent_by_event.get(str(event_id)) if event_id else None
            history_items.append(
                SessionContextItem(
                    context_id=item.item_id,
                    event_id=event.event_id if event else (str(event_id) if event_id else None),
                    event_type=(
                        event.event_type.value
                        if event
                        else str(item.provenance.get("event_type", "session_summary"))
                    ),
                    content=item.content,
                    compacted=item.category == ContextItemCategory.SESSION_SUMMARY,
                )
            )
        if history_items:
            output.append(SessionContextMessage(tuple(history_items)))

        # Current user/evidence are protected plan items and are projected as
        # the ordinary user message. The plan contains the evidence items too,
        # so a missing protected item is caught before the provider is called.
        output.append(self._user_message_from_plan(plan))

        for item_id, message in current_messages.items():
            if item_id in selected_ids:
                output.append(message)
        return tuple(output)

    def _user_message_from_plan(self, plan: ContextPlan) -> UserMessage:
        from ..agent.messages import UserMessage

        evidence: list[Evidence] = []
        for item in plan.items:
            if item.category != ContextItemCategory.CURRENT_EVIDENCE:
                continue
            if isinstance(item.content, Evidence):
                evidence.append(item.content)
            elif isinstance(item.content, Mapping):
                try:
                    evidence.append(Evidence(**dict(item.content)))
                except (TypeError, ValueError):
                    raise ContextProjectionError("current Evidence cannot be projected") from None
        if not any(item.item_id == "current-user" for item in plan.items):
            raise ContextProjectionError("current user is absent from ContextPlan")
        return UserMessage(self.question, tuple(evidence))

    @staticmethod
    def _validate_projection(
        plan: ContextPlan,
        messages: Sequence[AgentMessage],
        current_messages: Mapping[str, AgentMessage],
    ) -> None:
        from ..agent.messages import AssistantToolCallMessage, ToolResultMessage, UserMessage

        if not any(isinstance(message, UserMessage) for message in messages):
            raise ContextProjectionError("projected context has no current user")
        selected_ids = {item.item_id for item in plan.items}
        projected_current_ids = {
            item_id for item_id in current_messages if item_id in selected_ids
        }
        projected_tool_ids = {
            item_id
            for item_id in projected_current_ids
            if isinstance(current_messages[item_id], (AssistantToolCallMessage, ToolResultMessage))
        }
        for item_id in projected_tool_ids:
            message = current_messages[item_id]
            if isinstance(message, AssistantToolCallMessage):
                call_ids = {call.id for call in message.tool_calls}
                for other_id in projected_tool_ids:
                    other = current_messages[other_id]
                    if isinstance(other, ToolResultMessage) and other.tool_call_id in call_ids:
                        break
                else:
                    raise ContextAtomicityViolation("projected tool call has no result")
            elif isinstance(message, ToolResultMessage):
                if not any(
                    isinstance(other, AssistantToolCallMessage)
                    and any(call.id == message.tool_call_id for call in other.tool_calls)
                    for other in (current_messages[other_id] for other_id in projected_tool_ids)
                ):
                    raise ContextAtomicityViolation("projected tool result has no call")


def _tool_call_dict(call: Any) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def _tool_result_dict(message: ToolResultMessage) -> dict[str, Any]:
    result = message.result
    return {
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "ok": result.ok,
        "data": result.data,
        "error": (
            {"code": result.error.code, "message": result.error.message}
            if result.error
            else None
        ),
        "observed_evidence": [
            {
                "source_id": evidence.source_id,
                "title": evidence.title,
                "excerpt": evidence.excerpt,
                "source_url": evidence.source_url,
                "score": evidence.score,
            }
            for evidence in result.observed_evidence
        ],
    }


__all__ = [
    "ContextProjectionError",
    "ContextProjector",
    "ProjectedContext",
    "SessionContextProjector",
]
