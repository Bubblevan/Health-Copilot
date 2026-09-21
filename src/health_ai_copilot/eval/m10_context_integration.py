"""Deterministic M10.1 context-projection integration fixtures.

These cases exercise the executable boundary between a ContextPlan and the
provider-facing transcript.  They intentionally use synthetic data only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..agent.loop import AgentLoop, AgentLoopConfig
from ..agent.messages import (
    AssistantToolCallMessage,
    FinalTurn,
    MemoryContextMessage,
    SessionContextMessage,
    ToolCall,
    ToolCallTurn,
    ToolResultMessage,
    UserMessage,
)
from ..agent.session import AgentSession
from ..agent.tools import ToolRegistry, ToolResult
from ..contracts import Evidence
from ..runtime.context import RunContext
from ..runtime.context_manager import ContextBudget, ContextManager
from ..runtime.memory import (
    InMemoryMemoryStore,
    MemoryKind,
    MemoryOperation,
    MemoryQuery,
)
from ..runtime.projector import SessionContextProjector
from ..runtime.replay import RecordedToolExchange, ReplayMetadata, ReplayToolRunner
from ..runtime.session import (
    InMemorySessionStore,
    SessionEventType,
    SessionRevisionConflictError,
)
from ..safety import route_question


class _CapturingModel:
    def __init__(self, responses: Sequence[object]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[object, ...]] = []

    def respond(self, messages, tools, *, runtime=None):
        self.requests.append(tuple(messages))
        return self.responses.pop(0)


class _FixedToolRunner:
    def execute(self, call, runtime):
        return ToolResult.success({"observation": "synthetic tool observation"})


def _fixture() -> tuple[InMemorySessionStore, InMemoryMemoryStore, ContextManager, str]:
    session_store = InMemorySessionStore()
    session_store.create_session(session_id="m101-session")
    memory_store = InMemoryMemoryStore()
    manager = ContextManager()
    return session_store, memory_store, manager, "m101-session"


def _project(
    *,
    session_store: InMemorySessionStore,
    memory_store: InMemoryMemoryStore,
    manager: ContextManager,
    session_id: str,
    question: str = "What should I know?",
    evidence: Sequence[Evidence] = (),
    records=(),
    session: AgentSession | None = None,
    retrieval_query: str | None = None,
):
    runtime = RunContext.create("m10-context-integration")
    projected = SessionContextProjector(
        context_manager=manager,
        session_store=session_store,
        session_id=session_id,
        question=question,
        memory_records=records,
        memory_store=memory_store,
        retrieval_query=retrieval_query,
    ).project(
        session=session or AgentSession(session_id="m101-agent"),
        current_evidence=evidence,
        model_turn=1,
        runtime=runtime,
    )
    return projected


def _record_ids(messages) -> set[str]:
    for message in messages:
        if isinstance(message, SessionContextMessage):
            return {item.event_id for item in message.items if item.event_id}
    return set()


def _memory_ids(messages) -> set[str]:
    for message in messages:
        if isinstance(message, MemoryContextMessage):
            return {record.memory_id for record in message.records}
    return set()


def _tool_exchange_is_atomic(messages) -> bool:
    calls = {
        call.id
        for message in messages
        if isinstance(message, AssistantToolCallMessage)
        for call in message.tool_calls
    }
    results = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolResultMessage)
    }
    return bool(calls) and calls == results


def _run_agent_with_projector(*, budget: ContextBudget | None = None):
    session_store, memory_store, manager, session_id = _fixture()
    if budget is not None:
        manager = ContextManager(budget=budget)
    model = _CapturingModel(
        [
            ToolCallTurn([ToolCall("tool-1", "synthetic_tool", {})]),
            FinalTurn("done", [], False),
        ]
    )
    projector = SessionContextProjector(
        context_manager=manager,
        session_store=session_store,
        session_id=session_id,
        question="What should I know?",
        memory_store=memory_store,
    )
    result = AgentLoop(
        model,
        ToolRegistry(),
        AgentLoopConfig(max_model_turns=2, max_tool_calls=1),
        tool_runner=_FixedToolRunner(),
        context_projector=projector,
    ).run(
        "What should I know?",
        (),
        session=AgentSession(session_id="m101-agent"),
        runtime=RunContext.create("m10-context-integration"),
    )
    return result, model


def run_m10_context_case(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one JSONL fixture and return only deterministic metadata/booleans."""

    scenario = payload.get("scenario")
    observed: dict[str, Any] = {"scenario": scenario}
    try:
        session_store, memory_store, manager, session_id = _fixture()
        if scenario == "history_reaches_provider":
            first = session_store.append(session_id, SessionEventType.USER_INPUT, {"text": "earlier question"})
            second = session_store.append(session_id, SessionEventType.ASSISTANT_OUTPUT, {"text": "earlier answer"})
            projected = _project(
                session_store=session_store,
                memory_store=memory_store,
                manager=manager,
                session_id=session_id,
            )
            provider_ids = _record_ids(projected.messages)
            observed.update(
                selected_event_ids=list(projected.plan.selected_event_ids),
                provider_event_ids=sorted(provider_ids),
                history_reaches_provider=provider_ids == {first.event_id, second.event_id},
            )
        elif scenario == "dropped_history_absent":
            event = session_store.append(
                session_id,
                SessionEventType.USER_INPUT,
                {"text": "DROP-ME-RAW " + ("x" * 1200)},
            )
            manager = ContextManager(
                budget=ContextBudget(
                    max_estimated_input_tokens=180,
                    reserved_system_tokens=8,
                    reserved_current_turn_tokens=8,
                    max_memory_tokens=16,
                    max_history_tokens=8,
                    max_tool_observation_tokens=8,
                )
            )
            projected = _project(
                session_store=session_store,
                memory_store=memory_store,
                manager=manager,
                session_id=session_id,
            )
            provider_text = repr(projected.messages)
            observed.update(
                dropped_event_ids=list(projected.plan.dropped_event_ids),
                dropped_raw_absent="DROP-ME-RAW" not in provider_text,
                dropped_history_absent=event.event_id not in _record_ids(projected.messages),
            )
        elif scenario == "active_memory_reaches_provider":
            record = memory_store.apply(
                MemoryOperation.add(
                    scope_id=session_id,
                    key="care_goal",
                    kind=MemoryKind.TASK_STATE,
                    value="track morning readings",
                )
            )
            selected = memory_store.query(MemoryQuery(scope_id=session_id, text="care_goal", top_k=5))
            projected = _project(
                session_store=session_store,
                memory_store=memory_store,
                manager=manager,
                session_id=session_id,
                records=selected,
            )
            observed.update(
                selected_memory_ids=list(projected.plan.selected_memory_ids),
                provider_memory_ids=sorted(_memory_ids(projected.messages)),
                active_memory_reaches_provider=record is not None
                and _memory_ids(projected.messages) == {record.memory_id},
            )
        elif scenario == "superseded_memory_absent":
            old = memory_store.apply(
                MemoryOperation.add(
                    scope_id=session_id,
                    key="care_goal",
                    kind=MemoryKind.USER_ASSERTED_CONTEXT,
                    value="old goal",
                )
            )
            new = memory_store.apply(
                MemoryOperation.update(
                    scope_id=session_id,
                    key="care_goal",
                    kind=MemoryKind.USER_ASSERTED_CONTEXT,
                    value="current goal",
                    supersedes_id=old.memory_id if old else None,
                )
            )
            selected = memory_store.query(MemoryQuery(scope_id=session_id, text="care_goal", top_k=5))
            projected = _project(
                session_store=session_store,
                memory_store=memory_store,
                manager=manager,
                session_id=session_id,
                records=selected,
            )
            observed.update(
                superseded_memory_absent=old is not None
                and new is not None
                and old.memory_id not in _memory_ids(projected.messages)
                and new.memory_id in _memory_ids(projected.messages),
            )
        elif scenario == "current_evidence_protected":
            evidence = (Evidence("card-1", "Synthetic card", "observed fact", "synthetic://card-1", 0.9),)
            projected = _project(
                session_store=session_store,
                memory_store=memory_store,
                manager=manager,
                session_id=session_id,
                evidence=evidence,
            )
            user_messages = [message for message in projected.messages if isinstance(message, UserMessage)]
            observed.update(
                current_evidence_protected=bool(user_messages)
                and user_messages[-1].evidence == evidence
                and any(item.item_id == "evidence-card-1" for item in projected.plan.items),
            )
        elif scenario == "tool_atomicity_turn2":
            result, model = _run_agent_with_projector()
            second_request = model.requests[1] if len(model.requests) > 1 else ()
            observed.update(
                model_turns=len(model.requests),
                stop_reason=result.stop_reason.value if result.stop_reason else None,
                tool_atomicity_pass=_tool_exchange_is_atomic(second_request),
                provider_tool_message_count=sum(
                    isinstance(message, (AssistantToolCallMessage, ToolResultMessage))
                    for message in second_request
                ),
            )
        elif scenario == "safety_precedes_memory":
            calls = {"memory": 0, "retriever": 0, "provider": 0}

            class _FailingMemory:
                def query(self, query):
                    calls["memory"] += 1
                    raise AssertionError("memory must not be queried for terminal safety input")

            terminal_response = route_question(payload.get("question", ""))
            route = terminal_response.route if terminal_response else None
            if route is not None:
                observed.update(safety_precedes_memory=True, route=route.value, **calls)
            else:
                _FailingMemory().query(MemoryQuery(scope_id=session_id, text="unused"))
        elif scenario == "context_budget_fail_closed":
            result, model = _run_agent_with_projector(
                budget=ContextBudget(
                    max_estimated_input_tokens=0,
                    reserved_system_tokens=0,
                    reserved_current_turn_tokens=0,
                    max_memory_tokens=0,
                    max_history_tokens=0,
                    max_tool_observation_tokens=0,
                )
            )
            observed.update(
                provider_calls=len(model.requests),
                stop_reason=result.stop_reason.value if result.stop_reason else None,
                context_budget_fail_closed=(
                    len(model.requests) == 0
                    and result.stop_reason is not None
                    and result.stop_reason.value == "context_budget_exhausted"
                ),
            )
        elif scenario == "atomic_session_commit":
            before = session_store.resume_session(session_id).current_revision
            try:
                session_store.append_batch(
                    session_id,
                    [
                        (SessionEventType.USER_INPUT, {"text": "valid"}),
                        {"payload": {"text": "invalid second event"}},
                    ],
                    expected_revision=before,
                )
            except (TypeError, ValueError):
                pass
            after_failed_batch = session_store.resume_session(session_id).current_revision
            try:
                session_store.commit_turn(
                    session_id,
                    (SessionEventType.USER_INPUT, {"text": "stale"}),
                    expected_revision=before + 1,
                )
            except SessionRevisionConflictError:
                pass
            after_stale = session_store.resume_session(session_id).current_revision
            observed.update(
                atomic_turn_commit=(before == after_failed_batch == after_stale == 0),
                persisted_event_count=len(session_store.list_events(session_id)),
            )
        elif scenario == "memory_replay_identity":
            metadata = ReplayMetadata(
                session_revision=4,
                memory_snapshot_hash="memory-hash",
                context_plan_hash="plan-hash",
                context_plan_hashes=("plan-hash", "plan-hash-2"),
            )
            exchange = RecordedToolExchange("synthetic_tool", {}, ToolResult.success({"ok": True}))
            matching = ReplayToolRunner(
                [exchange],
                recorded_metadata=metadata,
                expected_session_revision=4,
                expected_memory_snapshot_hash="memory-hash",
                expected_context_plan_hash="plan-hash",
                expected_context_plan_hashes=("plan-hash", "plan-hash-2"),
            )
            matching_runtime = RunContext.create("replay", session_revision=4, memory_snapshot_hash="memory-hash", context_plan_hashes=("plan-hash", "plan-hash-2"))
            matching_result = matching.execute(ToolCall("call-1", "synthetic_tool", {}), matching_runtime)
            mismatch = ReplayToolRunner(
                [exchange],
                recorded_metadata=metadata,
                expected_session_revision=5,
                expected_memory_snapshot_hash="memory-hash",
                expected_context_plan_hash="plan-hash",
                expected_context_plan_hashes=("plan-hash", "plan-hash-2"),
            )
            mismatch_runtime = RunContext.create("replay", session_revision=5, memory_snapshot_hash="memory-hash", context_plan_hashes=("plan-hash", "plan-hash-2"))
            mismatch_result = mismatch.execute(ToolCall("call-1", "synthetic_tool", {}), mismatch_runtime)
            observed.update(
                matching_result_ok=matching_result.ok,
                mismatch_code=mismatch_result.error.code if mismatch_result.error else None,
                memory_replay_identity=(matching_result.ok and mismatch_result.error is not None and mismatch_result.error.code == "replay_mismatch"),
            )
        else:
            observed["failure_code"] = "unknown_m10_context_scenario"
        checks = [value for key, value in observed.items() if key.endswith(("_pass", "_passed", "_protected", "_absent", "_provider", "_commit", "_identity")) or key in {"history_reaches_provider", "active_memory_reaches_provider", "tool_atomicity_pass", "context_budget_fail_closed", "safety_precedes_memory", "atomic_turn_commit", "memory_replay_identity"}]
        observed["case_passed"] = bool(checks) and all(bool(value) for value in checks)
    except (AssertionError, AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        observed.update(case_passed=False, failure_code="m10_context_case_failed", error_type=type(exc).__name__)
    return observed
