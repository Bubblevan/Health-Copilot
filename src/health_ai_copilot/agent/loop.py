"""Readable, deterministic model -> tool -> observation -> model loop."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..contracts import Evidence, GenerationDraft
from ..knowledge.scope import KnowledgeScope
from ..policy.evidence import EvidenceDecision, EvidencePolicy, validate_assessment
from ..runtime.context import RunContext, call_with_optional_runtime
from ..runtime.context_manager import ContextAtomicityViolation, ContextBudgetExceeded
from ..runtime.projector import ContextProjector
from ..runtime.tools import LiveToolRunner, ToolRunner
from ..runtime.trace import TraceEventType
from .events import AgentEvent, AgentEventType
from .messages import (
    FinalTurn,
    ToolCallTurn,
    ToolResultMessage,
    UserMessage,
)
from .model import AgentModel
from .session import AgentSession
from .state import AgentState, StopReason
from .tools import ToolRegistry, ToolResult

EventSink = Callable[[AgentEvent], None]


@dataclass(frozen=True)
class AgentLoopConfig:
    max_model_turns: int = 2
    max_tool_calls: int = 1

    def __post_init__(self) -> None:
        if self.max_model_turns <= 0:
            raise ValueError("max_model_turns must be greater than zero")
        if self.max_model_turns > 2:
            raise ValueError("M1 max_model_turns cannot exceed two")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must not be negative")
        if self.max_tool_calls > 1:
            raise ValueError("M1 max_tool_calls cannot exceed one")


@dataclass(frozen=True)
class AgentRunResult:
    state: AgentState
    events: tuple[AgentEvent, ...]
    context_plans: tuple[object, ...] = ()

    @property
    def draft(self) -> GenerationDraft | None:
        return self.state.final_draft

    @property
    def observed_evidence(self) -> list[Evidence]:
        return list(self.state.observed_evidence)

    @property
    def initial_ranked_evidence(self) -> list[Evidence]:
        return list(self.state.initial_ranked_evidence)

    @property
    def recovery_ranked_evidence(self) -> list[Evidence]:
        return list(self.state.recovery_ranked_evidence)

    @property
    def stop_reason(self) -> StopReason | None:
        return self.state.stop_reason

    @property
    def claims(self):
        return self.state.final_claims


class AgentLoop:
    """M1's only runtime: bounded, sequential, one-agent execution."""

    def __init__(
        self,
        model: AgentModel,
        registry: ToolRegistry,
        config: AgentLoopConfig | None = None,
        event_sink: EventSink | None = None,
        evidence_policy: EvidencePolicy | None = None,
        knowledge_scope: KnowledgeScope | None = None,
        tool_runner: ToolRunner | None = None,
        runtime: RunContext | None = None,
        context_projector: ContextProjector | None = None,
    ) -> None:
        self.model = model
        self.registry = registry
        self.config = config or AgentLoopConfig()
        self.event_sink = event_sink
        self.evidence_policy = evidence_policy
        self.knowledge_scope = knowledge_scope
        self.tool_runner = tool_runner or LiveToolRunner(registry)
        self.runtime = runtime
        self.context_projector = context_projector

    def run(
        self,
        question: str,
        initial_evidence: Sequence[Evidence],
        session: AgentSession | None = None,
        runtime: RunContext | None = None,
    ) -> AgentRunResult:
        active_session = session or AgentSession()
        active_runtime = runtime or self.runtime or RunContext.create("agent")
        ranked_initial_evidence = list(initial_evidence)
        state = AgentState(
            session=active_session,
            initial_ranked_evidence=list(ranked_initial_evidence),
        )
        state.add_evidence(ranked_initial_evidence)
        events: list[AgentEvent] = []
        context_plans: list[object] = []
        projector = self.context_projector

        def emit(event: AgentEvent) -> None:
            events.append(event)
            if self.event_sink is not None:
                try:
                    self.event_sink(event)
                except Exception:  # noqa: BLE001 - observers cannot break safety flow
                    return

        emit(AgentEvent(AgentEventType.AGENT_START, active_session.session_id))
        active_session.append(
            UserMessage(content=question, evidence=tuple(state.observed_evidence))
        )

        while state.model_turns_used < self.config.max_model_turns:
            turn = state.model_turns_used + 1
            emit(AgentEvent(AgentEventType.TURN_START, active_session.session_id, turn=turn))
            state.model_turns_used += 1

            provider_messages = tuple(active_session.messages)
            if projector is not None:
                try:
                    projected = projector.project(
                        session=active_session,
                        current_evidence=tuple(state.observed_evidence),
                        model_turn=turn,
                        runtime=active_runtime,
                    )
                    provider_messages = projected.messages
                    context_plans.append(projected.plan)
                    if active_runtime.trace is not None:
                        active_runtime.trace.emit(
                            TraceEventType.CONTEXT_PLAN,
                            model_turn=turn,
                            context_plan_hash=projected.plan.plan_hash,
                            session_revision=projected.plan.session_revision,
                            memory_snapshot_hash=projected.memory_snapshot_hash,
                            selected_memory_count=len(projected.plan.selected_memory_ids),
                            selected_history_count=(
                                len(projected.plan.selected_event_ids)
                                + len(projected.plan.compacted_event_ids)
                            ),
                            compaction_count=projected.plan.compaction_count,
                            estimated_tokens=projected.plan.estimated_tokens,
                        )
                        if projected.plan.compaction_count:
                            active_runtime.trace.emit(
                                TraceEventType.CONTEXT_COMPACTED,
                                model_turn=turn,
                                compaction_count=projected.plan.compaction_count,
                                compacted_event_count=len(projected.plan.compacted_event_ids),
                            )
                except ContextBudgetExceeded:
                    emit(
                        AgentEvent(
                            AgentEventType.MODEL_RESPONSE,
                            active_session.session_id,
                            turn=turn,
                            response_kind="error",
                            success=False,
                            error_code="context_budget_exhausted",
                        )
                    )
                    emit(
                        AgentEvent(
                            AgentEventType.TURN_END,
                            active_session.session_id,
                            turn=turn,
                            success=False,
                            error_code="context_budget_exhausted",
                        )
                    )
                    state.stop(StopReason.CONTEXT_BUDGET_EXCEEDED)
                    break
                except ContextAtomicityViolation:
                    emit(
                        AgentEvent(
                            AgentEventType.MODEL_RESPONSE,
                            active_session.session_id,
                            turn=turn,
                            response_kind="error",
                            success=False,
                            error_code="context_atomicity_violation",
                        )
                    )
                    emit(
                        AgentEvent(
                            AgentEventType.TURN_END,
                            active_session.session_id,
                            turn=turn,
                            success=False,
                            error_code="context_atomicity_violation",
                        )
                    )
                    state.stop(StopReason.CONTEXT_PROJECTION_ERROR)
                    break
                except Exception:  # noqa: BLE001 - projection is fail-closed
                    emit(
                        AgentEvent(
                            AgentEventType.MODEL_RESPONSE,
                            active_session.session_id,
                            turn=turn,
                            response_kind="error",
                            success=False,
                            error_code="context_projection_error",
                        )
                    )
                    emit(
                        AgentEvent(
                            AgentEventType.TURN_END,
                            active_session.session_id,
                            turn=turn,
                            success=False,
                            error_code="context_projection_error",
                        )
                    )
                    state.stop(StopReason.CONTEXT_PROJECTION_ERROR)
                    break

            try:
                response = call_with_optional_runtime(
                    self.model.respond,
                    provider_messages,
                    tuple(self.registry.list_model_tool_specs()),
                    runtime=active_runtime,
                )
            except Exception:  # noqa: BLE001 - model failures become controlled stops
                emit(
                    AgentEvent(
                        AgentEventType.MODEL_RESPONSE,
                        active_session.session_id,
                        turn=turn,
                        response_kind="error",
                        success=False,
                        error_code="model_error",
                    )
                )
                emit(
                    AgentEvent(
                        AgentEventType.TURN_END,
                        active_session.session_id,
                        turn=turn,
                        success=False,
                    )
                )
                state.stop(StopReason.MODEL_ERROR)
                break

            if isinstance(response, FinalTurn):
                final_message = response.as_message()
                active_session.append(final_message)
                state.final_draft = response.to_draft()
                state.final_claims = response.claims
                emit(
                    AgentEvent(
                        AgentEventType.MODEL_RESPONSE,
                        active_session.session_id,
                        turn=turn,
                        response_kind="final",
                        success=True,
                    )
                )
                reason = StopReason.ABSTAIN if response.abstain else StopReason.FINAL
                emit(
                    AgentEvent(
                        AgentEventType.TURN_END,
                        active_session.session_id,
                        turn=turn,
                        success=True,
                    )
                )
                state.stop(reason, response.to_draft())
                break

            if not isinstance(response, ToolCallTurn):
                emit(
                    AgentEvent(
                        AgentEventType.MODEL_RESPONSE,
                        active_session.session_id,
                        turn=turn,
                        response_kind="invalid",
                        success=False,
                        error_code="invalid_model_response",
                    )
                )
                emit(
                    AgentEvent(
                        AgentEventType.TURN_END,
                        active_session.session_id,
                        turn=turn,
                        success=False,
                    )
                )
                state.stop(StopReason.MODEL_ERROR)
                break

            active_session.append(response.as_message())
            emit(
                AgentEvent(
                    AgentEventType.MODEL_RESPONSE,
                    active_session.session_id,
                    turn=turn,
                    response_kind="tool_call",
                    success=True,
                )
            )

            if not response.tool_calls:
                # A provider should normally return either a final turn or a
                # non-empty tool-call turn. Treat an empty action as a
                # non-final observation so the hard model-turn budget still
                # controls termination and fails closed.
                emit(
                    AgentEvent(
                        AgentEventType.TURN_END,
                        active_session.session_id,
                        turn=turn,
                        success=False,
                    )
                )
                continue

            # A policy-vetoed proposal does not execute a tool, but it still consumes
            # the single M1 action opportunity. This prevents a second proposal from
            # bypassing the original one-step interaction boundary.
            if state.tool_proposals_used + len(response.tool_calls) > self.config.max_tool_calls:
                emit(
                    AgentEvent(
                        AgentEventType.TURN_END,
                        active_session.session_id,
                        turn=turn,
                        success=False,
                    )
                )
                state.stop(StopReason.MAX_TOOL_CALLS)
                break

            # M1 is sequential and configured for one call, even though the
            # structural turn type can represent a list of provider calls.
            call = response.tool_calls[0]
            state.tool_proposals_used += 1
            if self.evidence_policy is not None and call.name == "search_knowledge":
                proposed_query = (
                    call.arguments.get("query", "")
                    if isinstance(call.arguments, dict)
                    else ""
                )
                state.policy_calls_used += 1
                emit(AgentEvent(AgentEventType.POLICY_START, active_session.session_id, turn=turn, tool_call_id=call.id, tool_name=call.name))
                try:
                    assessment = validate_assessment(
                        call_with_optional_runtime(
                            self.evidence_policy.assess,
                            question,
                            tuple(state.observed_evidence),
                            proposed_query,
                            runtime=active_runtime,
                        ),
                        tuple(state.observed_evidence),
                        self.knowledge_scope,
                    )
                except Exception:  # noqa: BLE001 - a policy failure is terminal and closed
                    emit(AgentEvent(AgentEventType.POLICY_END, active_session.session_id, turn=turn, tool_call_id=call.id, tool_name=call.name, success=False, error_code="policy_error"))
                    emit(AgentEvent(AgentEventType.TURN_END, active_session.session_id, turn=turn, success=False))
                    state.stop(StopReason.POLICY_ERROR)
                    break
                emit(AgentEvent(AgentEventType.POLICY_END, active_session.session_id, turn=turn, tool_call_id=call.id, tool_name=call.name, success=True))
                state.policy_decision = assessment.decision.value
                state.policy_reason_codes = assessment.reason_codes
                state.policy_supporting_source_ids = assessment.supporting_source_ids
                state.policy_matched_topic_ids = assessment.matched_topic_ids
                if active_runtime.trace is not None:
                    active_runtime.trace.emit(
                        TraceEventType.POLICY_DECISION,
                        decision=assessment.decision.value,
                        reason_codes=list(assessment.reason_codes),
                        matched_topic_ids=list(assessment.matched_topic_ids),
                        supporting_source_count=len(assessment.supporting_source_ids),
                    )
                if assessment.decision == EvidenceDecision.SUFFICIENT:
                    active_session.append(ToolResultMessage(call.id, call.name, ToolResult.failure("policy_denied", "current evidence is sufficient; recovery search was not executed")))
                    emit(AgentEvent(AgentEventType.TURN_END, active_session.session_id, turn=turn, success=False, error_code="policy_denied"))
                    continue
                if assessment.decision == EvidenceDecision.INSUFFICIENT:
                    emit(AgentEvent(AgentEventType.TURN_END, active_session.session_id, turn=turn, success=False))
                    state.stop(StopReason.EVIDENCE_INSUFFICIENT)
                    break
                if assessment.decision == EvidenceDecision.CONFLICTING:
                    emit(AgentEvent(AgentEventType.TURN_END, active_session.session_id, turn=turn, success=False))
                    state.stop(StopReason.EVIDENCE_CONFLICTING)
                    break
            state.tool_calls_used += 1
            emit(
                AgentEvent(
                    AgentEventType.TOOL_START,
                    active_session.session_id,
                    turn=turn,
                    tool_call_id=call.id,
                    tool_name=call.name,
                )
            )
            result = self.tool_runner.execute(call, active_runtime)
            emit(
                AgentEvent(
                    AgentEventType.TOOL_END,
                    active_session.session_id,
                    turn=turn,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    success=result.ok,
                    error_code=result.error.code if result.error else None,
                )
            )
            if result.ok and result.observed_evidence:
                state.add_recovery_evidence(list(result.observed_evidence))
            active_session.append(
                ToolResultMessage(tool_call_id=call.id, tool_name=call.name, result=result)
            )
            emit(
                AgentEvent(
                    AgentEventType.TURN_END,
                    active_session.session_id,
                    turn=turn,
                    success=result.ok,
                )
            )

        if state.status.value == "running":
            state.stop(StopReason.MAX_MODEL_TURNS)

        emit(
            AgentEvent(
                AgentEventType.AGENT_END,
                active_session.session_id,
                stop_reason=state.stop_reason.value if state.stop_reason else None,
                success=state.stop_reason in {StopReason.FINAL, StopReason.ABSTAIN},
            )
        )
        return AgentRunResult(
            state=state,
            events=tuple(events),
            context_plans=tuple(context_plans),
        )
