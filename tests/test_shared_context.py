import pytest

from health_ai_copilot.multi_agent.contracts import LeadPlan, RouteMode, WorkerRole
from health_ai_copilot.multi_agent.shared_context import SharedContext


def test_shared_context_creates_ids_and_keeps_evidence_in_ledger() -> None:
    context = SharedContext("trace-1", "question")
    task = context._harness_register_task(WorkerRole.EVIDENCE, "find sources")
    citation = context._harness_add_evidence(
        source_id="source-1", worker_id=task.worker_id, role="evidence",
        tool_call_id="tool-1", tool_id="external_retrieval", excerpt="observed",
        input_hash="in", output_hash="out", resource_versions=(("world", "v1"),),
    )
    assert task.task_id.startswith("task-")
    assert task.worker_id.startswith("worker-evidence-")
    assert citation.evidence_id.startswith("evidence-")
    assert context.lead_view()["evidence_ledger"][0]["source_id"] == "source-1"


def test_shared_context_does_not_expose_evaluator_truth_and_snapshots_are_copies() -> None:
    context = SharedContext("trace-2", "question")
    view = context.lead_view()
    view["worker_status"]["fake"] = "complete"
    assert "fake" not in context.lead_view()["worker_status"]
    assert "gold_answer" not in context.lead_view()


def test_context_rejects_unknown_worker_evidence_attribution() -> None:
    context = SharedContext("trace-3", "question")
    with pytest.raises(ValueError, match="unknown worker"):
        context._harness_add_evidence(
            source_id="source-1", worker_id="model-made-worker", role="evidence",
            tool_call_id="tool-1", tool_id="external_retrieval", excerpt="",
            input_hash="in", output_hash="out", resource_versions=(),
        )


def test_plan_is_serialized_with_only_semantic_task_fields() -> None:
    context = SharedContext("trace-4", "question")
    context._harness_set_plan(LeadPlan(
        RouteMode.TEAM, ((WorkerRole.EVIDENCE, "find evidence"),),
    ))
    assert set(context.lead_view()["plan"]) == {"mode", "tasks", "source"}
