import pytest

from health_ai_copilot.multi_agent.contracts import WorkerRole
from health_ai_copilot.multi_agent.shared_context import SharedContext


def test_shared_context_records_harness_owned_task_ids_and_status() -> None:
    context = SharedContext("trace-1", "question")
    task = context._harness_register_task(WorkerRole.CARE.value, "specialist analysis")
    context._harness_worker_status(task.worker_id, "running", phase="analysis")
    context._harness_timeline("complexity_decided", complexity="intermediate")

    snapshot = context.to_dict()
    assert task.task_id.startswith("task-")
    assert task.worker_id.startswith("worker-care-")
    assert snapshot["worker_status"][task.worker_id] == "running"
    assert snapshot["timeline"][-1]["complexity"] == "intermediate"


def test_shared_context_snapshots_are_copies_and_have_no_evaluator_plane() -> None:
    context = SharedContext("trace-2", "question")
    task = context._harness_register_task(WorkerRole.CARE.value, "specialist analysis")
    snapshot = context.to_dict()
    snapshot["worker_status"][task.worker_id] = "complete"
    snapshot["tasks"].clear()

    assert context.to_dict()["worker_status"][task.worker_id] == "pending"
    assert len(context.to_dict()["tasks"]) == 1
    assert "gold_answer" not in context.to_dict()


def test_shared_context_rejects_unknown_worker_status_updates() -> None:
    context = SharedContext("trace-3", "question")
    with pytest.raises(ValueError, match="unknown worker"):
        context._harness_worker_status("model-made-worker", "complete")


def test_shared_context_allows_dynamic_same_role_specialists() -> None:
    context = SharedContext("trace-4", "question")
    tasks = [
        context._harness_register_task(WorkerRole.CARE.value, f"specialist {index}")
        for index in range(4)
    ]
    assert len({task.worker_id for task in tasks}) == 4
