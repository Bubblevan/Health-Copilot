from __future__ import annotations

import pytest

from health_ai_copilot.eval.registry import default_eval_suite_registry
from health_ai_copilot.eval.system import EvaluationRunner
from health_ai_copilot.runtime.builder import default_runtime_profiles
from health_ai_copilot.runtime.context_manager import (
    ContextAtomicityViolation,
    ContextBudget,
    ContextBudgetExceeded,
    ContextManager,
)
from health_ai_copilot.runtime.memory import (
    ContextIntent,
    FakeClock,
    InMemoryMemoryStore,
    MemoryOperation,
    MemoryQuery,
    MemorySensitivity,
    MemorySourceType,
    MemoryWriteDenied,
)
from health_ai_copilot.runtime.session import (
    SessionEvent,
    SessionEventType,
    SessionRevisionConflictError,
    SQLiteSessionStore,
)


def test_old_profile_hashes_and_m10_fields_are_compatible() -> None:
    profiles = default_runtime_profiles()
    assert profiles["m3-bm25-default"].config_hash == "06ca629157cea96034f0a1d7ec0dbb7d32797d3543ebd40a508395605aaad2d7"
    assert profiles["m8-team-bm25-v1"].config_hash == "66a618302fd3f9fad02b90e3486e464bee10a183a5f9f40f5195b2731156da86"
    assert profiles["m9-mcp-search-bm25-v1"].config_hash == "7ffc0face2182618ea3ffc6c3a8f50b8909b8caeb94a6e3389dec69b65b8c194"
    assert "session_store" not in profiles["m3-bm25-default"].to_dict()
    assert profiles["m10-memory-bm25-v1"].session_store == "sqlite-session-v1"
    assert profiles["m10-memory-bm25-v1"].memory_store == "sqlite-memory-v1"


def test_session_revision_resume_and_fork(tmp_path) -> None:
    path = tmp_path / "session.sqlite"
    store = SQLiteSessionStore(path)
    record = store.create_session(session_id="opaque-a")
    store.append(record.session_id, SessionEventType.USER_INPUT, {"text": "synthetic"}, expected_revision=0)
    with pytest.raises(SessionRevisionConflictError):
        store.append(record.session_id, SessionEventType.USER_INPUT, {"text": "stale"}, expected_revision=0)
    child = store.fork_session(record.session_id, 1)
    store.close()
    reopened = SQLiteSessionStore(path)
    assert reopened.resume_session(record.session_id).current_revision == 1
    assert reopened.resume_session(child.session_id).parent_session_id == record.session_id
    assert len(reopened.list_events(child.session_id)) == 1
    reopened.close()


def test_memory_update_delete_expiry_scope_and_explicit_override(tmp_path) -> None:
    clock = FakeClock("2026-01-01T00:00:00Z")
    store = InMemoryMemoryStore(clock=clock)
    old = store.apply(MemoryOperation.add(scope_id="a", key="format", kind="preference", value="table"))
    new = store.apply(MemoryOperation.update(scope_id="a", key="format", kind="preference", value="list", expected_version=old.version))
    assert new.supersedes_id == old.memory_id
    assert [item.value for item in store.query(MemoryQuery(scope_id="a", text="format"))] == ["list"]
    assert store.query(MemoryQuery(scope_id="a", text="format", explicit_values={"format": "list"}))
    store.apply(MemoryOperation.delete(scope_id="a", key="format"))
    assert store.query(MemoryQuery(scope_id="a", text="format")) == []
    store.apply(MemoryOperation.add(scope_id="a", key="temporary", kind="task_state", value="expired", expires_at="2026-01-02T00:00:00Z"))
    clock.advance(days=2)
    assert store.query(MemoryQuery(scope_id="a", text="temporary")) == []
    store.apply(MemoryOperation.add(scope_id="a", key="language", kind="preference", value="Chinese"))
    store.apply(MemoryOperation.add(scope_id="b", key="language", kind="preference", value="English"))
    assert [item.value for item in store.query(MemoryQuery(scope_id="a", text="language"))] == ["Chinese"]


def test_memory_policy_rejects_untrusted_and_sensitive_sources() -> None:
    store = InMemoryMemoryStore()
    with pytest.raises(MemoryWriteDenied):
        store.apply(MemoryOperation.add(scope_id="a", key="x", kind="preference", value="tool", source_type=MemorySourceType.TOOL_OUTPUT))
    with pytest.raises(MemoryWriteDenied):
        store.apply(MemoryOperation.add(scope_id="a", key="x", kind="user_asserted_context", value="synthetic", sensitivity=MemorySensitivity.SENSITIVE_HEALTH))


def test_intent_filter_and_memory_is_data_not_evidence() -> None:
    store = InMemoryMemoryStore()
    intent = ContextIntent("measurement", "teach", ("reading",))
    store.apply(MemoryOperation.add(scope_id="a", key="blood_pressure", kind="task_state", value="measurement", intent=intent))
    store.apply(MemoryOperation.add(scope_id="a", key="blood_pressure_summary", kind="task_state", value="formatting", intent=ContextIntent("formatting", "summarize", ("log",))))
    result = store.query(MemoryQuery(scope_id="a", text="blood pressure", intent=intent))
    assert [item.value for item in result] == ["measurement"]


def test_context_budget_and_tool_exchange_atomicity() -> None:
    manager = ContextManager(budget=ContextBudget(max_estimated_input_tokens=256, reserved_system_tokens=8, reserved_current_turn_tokens=8, max_history_tokens=128, max_tool_observation_tokens=128))
    call = SessionEvent.create("s", 1, SessionEventType.TOOL_CALL, {"call_id": "c", "query": "synthetic"})
    result = SessionEvent.create("s", 2, SessionEventType.TOOL_RESULT, {"call_id": "c", "result": "synthetic"})
    plan = manager.build_plan(session_id="s", session_revision=2, current_user="current", history=[call, result])
    selected = {item.group_id for item in plan.items if item.group_id}
    assert selected == {"c"} or selected == set()
    unresolved_call = SessionEvent.create("s", 3, SessionEventType.TOOL_CALL, {"call_id": "u", "unresolved": True})
    unresolved_result = SessionEvent.create("s", 4, SessionEventType.TOOL_RESULT, {"call_id": "u", "unresolved": True})
    protected_plan = manager.build_plan(
        session_id="s", session_revision=4, current_user="current", history=[unresolved_call, unresolved_result]
    )
    assert {item.group_id for item in protected_plan.items if item.group_id} == {"u"}
    with pytest.raises(ContextBudgetExceeded):
        ContextManager(budget=ContextBudget(max_estimated_input_tokens=1, reserved_system_tokens=1, reserved_current_turn_tokens=1)).build_plan(session_id="s", session_revision=0, current_user="too large")
    with pytest.raises(ContextAtomicityViolation):
        manager.build_plan(session_id="s", session_revision=1, current_user="current", history=[call])


def test_m10_offline_suite_produces_standard_bundle(tmp_path) -> None:
    suite = default_eval_suite_registry().get("m10-memory-v1")
    assert suite.target_kind.value == "memory"
    runner = EvaluationRunner()
    spec = runner.prepare_run_spec("m10-memory-v1", execution_mode="offline", output_root=tmp_path / "runs")
    bundle = runner.run(spec, public_eval_content=True)
    assert {"run_spec.json", "run_manifest.json", "case_results.jsonl", "grader_results.jsonl", "failures.jsonl", "trajectories.jsonl", "metrics.json", "report.md"} <= {path.name for path in bundle.iterdir() if path.is_file()}
    assert (bundle / "failures.jsonl").read_text(encoding="utf-8") == ""
    metrics = __import__("json").loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["grader.m10_memory.pass_rate"]["numerator"] == 24
