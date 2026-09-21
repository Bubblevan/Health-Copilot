"""Offline M10 memory/context fixtures and the eval-only action tool."""

from __future__ import annotations

from collections.abc import Mapping

from ..runtime.context_manager import ContextBudget, ContextManager
from ..runtime.memory import (
    ContextIntent,
    FakeClock,
    InMemoryMemoryStore,
    MemoryOperation,
    MemoryQuery,
)
from ..runtime.session import SessionEvent, SessionEventType


def prepare_education_output(topic: str, language: str, format: str, unit: str) -> dict[str, str]:
    """Eval-only action fixture; it has no product side effect."""

    return {"topic": topic, "language": language, "format": format, "unit": unit}


def run_m10_case(payload: Mapping[str, object]) -> dict[str, object]:
    scenario = str(payload.get("scenario", ""))
    scope = "synthetic-scope"
    store = InMemoryMemoryStore()
    observed: dict[str, object] = {"category": scenario, "memory_trajectory_schema": "memory_trajectory_v1"}
    try:
        if scenario == "basic_retrieval":
            store.apply(
                MemoryOperation.add(
                    scope_id=scope,
                    key=str(payload["key"]),
                    kind="preference",
                    value=payload["value"],
                )
            )
            records = store.query(MemoryQuery(scope_id=scope, text=str(payload["query"])))
            observed.update(
                retrieved_memory_ids=[record.memory_id for record in records],
                retrieved_values=[record.value for record in records],
                memory_retrieval_hit=bool(records and records[0].value == payload["expected_value"]),
            )
        elif scenario == "supersession":
            old = store.apply(
                MemoryOperation.add(
                    scope_id=scope, key=str(payload["key"]), kind="preference", value=payload["old_value"]
                )
            )
            new = store.apply(
                MemoryOperation.update(
                    scope_id=scope,
                    key=str(payload["key"]),
                    kind="preference",
                    value=payload["new_value"],
                    expected_version=old.version if old else 1,
                )
            )
            records = store.query(MemoryQuery(scope_id=scope, text=str(payload["query"])))
            observed.update(
                retrieved_values=[record.value for record in records],
                superseded_id=new.supersedes_id if new else None,
                supersession_ok=bool(new and records and records[0].value == payload["new_value"] and payload["old_value"] not in [r.value for r in records]),
            )
        elif scenario == "expiry":
            clock = FakeClock(str(payload["now"]))
            store = InMemoryMemoryStore(clock=clock)
            store.apply(
                MemoryOperation.add(
                    scope_id=scope, key=str(payload["key"]), kind="task_state", value=payload["value"],
                    valid_from=payload.get("valid_from"), valid_until=payload.get("valid_until"), expires_at=payload.get("expires_at"),
                )
            )
            records = store.query(MemoryQuery(scope_id=scope, text=str(payload["key"]), now=str(payload["now"])))
            observed.update(retrieved_count=len(records), expiry_ok=not records)
        elif scenario == "intent_mismatch":
            query_intent = ContextIntent(
                goal=str(payload["goal"]), action_type=str(payload["action_type"]), entity_types=tuple(payload["entity_types"])
            )
            store.apply(
                MemoryOperation.add(
                    scope_id=scope, key="same_entity", kind="task_state", value=payload["good_value"],
                    intent=query_intent,
                )
            )
            store.apply(
                MemoryOperation.add(
                    scope_id=scope, key="same_entity_other_goal", kind="task_state", value=payload["bad_value"],
                    intent=ContextIntent(goal="other_goal", action_type="other_action", entity_types=("other",)),
                )
            )
            records = store.query(MemoryQuery(scope_id=scope, text=str(payload["query"]), intent=query_intent))
            values = [record.value for record in records]
            observed.update(retrieved_values=values, intent_mismatch_ok=payload["good_value"] in values and payload["bad_value"] not in values)
        elif scenario == "delete":
            store.apply(MemoryOperation.add(scope_id=scope, key=str(payload["key"]), kind="session_note", value=payload["value"]))
            store.apply(MemoryOperation.delete(scope_id=scope, key=str(payload["key"])))
            records = store.query(MemoryQuery(scope_id=scope, text=str(payload["key"])))
            observed.update(retrieved_count=len(records), tombstone_count=len(store.history(scope)), delete_ok=not records and store.history(scope)[-1].operation.value == "delete")
        elif scenario == "scope_isolation":
            store.apply(MemoryOperation.add(scope_id="scope-a", key=str(payload["key"]), kind="preference", value=payload["scope_a_value"]))
            store.apply(MemoryOperation.add(scope_id="scope-b", key=str(payload["key"]), kind="preference", value=payload["scope_b_value"]))
            records = store.query(MemoryQuery(scope_id="scope-a", text=str(payload["key"])))
            observed.update(retrieved_values=[record.value for record in records], scope_isolation_ok=bool(records and all(record.value == payload["scope_a_value"] for record in records)))
        elif scenario == "compaction":
            budget = int(payload["budget"])
            manager = ContextManager(
                budget=ContextBudget(
                    max_estimated_input_tokens=budget,
                    reserved_system_tokens=8,
                    reserved_current_turn_tokens=8,
                    max_memory_tokens=8,
                    max_history_tokens=budget,
                    max_tool_observation_tokens=budget,
                ),
                history_window=100,
            )
            history = [
                SessionEvent.create("session", index + 1, SessionEventType.USER_INPUT, {"text": "old synthetic turn " + str(index)})
                for index in range(int(payload["history_count"]))
            ]
            plan = manager.build_plan(session_id="session", session_revision=len(history), current_user="current protected question", history=history)
            observed.update(
                estimated_context_tokens=plan.estimated_tokens,
                context_budget_pass=plan.estimated_tokens <= budget,
                protected_current_retained=any(item.item_id == "current-user" for item in plan.items),
                compaction_count=plan.compaction_count,
                compaction_provenance=bool(plan.compacted_event_ids or not plan.dropped_event_ids),
            )
        elif scenario == "action_grounding":
            for key in ("language", "format", "unit"):
                store.apply(MemoryOperation.add(scope_id=scope, key=key, kind="preference", value=payload[key]))
            records = store.query(MemoryQuery(scope_id=scope, text="", top_k=8))
            values = {record.key: record.value for record in records}
            arguments = prepare_education_output("blood pressure education", values["language"], values["format"], values["unit"])
            observed.update(
                action_tool="prepare_education_output",
                action_arguments=arguments,
                action_grounding_ok=arguments == {"topic": "blood pressure education", "language": payload["language"], "format": payload["format"], "unit": payload["unit"]},
            )
        else:
            observed["failure_code"] = "m10_case_failed"
        checks = [
            value
            for key, value in observed.items()
            if key.endswith(("_ok", "_pass", "_passed"))
            or key in {"memory_retrieval_hit", "action_grounding_ok"}
        ]
        observed["case_passed"] = bool(checks) and all(bool(value) for value in checks)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:  # record the deterministic stage
        observed.update(case_passed=False, failure_code="m10_case_failed", error_type=type(exc).__name__)
    return observed
