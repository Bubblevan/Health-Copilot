from dataclasses import replace
from datetime import UTC, datetime

from health_ai_copilot.research.integration.counterfactual import CounterfactualRunner
from health_ai_copilot.research.integration.evidence_world import ExternalEvidenceRecord
from health_ai_copilot.research.integration.fixtures import build_synthetic_cases


def test_patient_snapshot_excludes_decision_time_plus_one_record() -> None:
    case = next(item for item in build_synthetic_cases() if item.case_id == "U1-TEMPORAL-LEAKAGE")
    rows = case.resources.patient_state_store.snapshot(case.episode.subject_id, case.episode.decision_time)
    assert all(row.timestamp <= case.episode.decision_time for row in rows)
    assert "mem-future-1" not in {row.record_id for row in rows}


def test_memory_read_at_t_cannot_return_t_plus_one_event() -> None:
    case = next(item for item in build_synthetic_cases() if item.case_id == "U1-TEMPORAL-LEAKAGE")
    bundle = CounterfactualRunner().run(case.episode, case.evaluation, case.resources)
    memory = next(item for item in bundle.arms if item.action_key.value == "MEMORY")
    assert "mem-future-1" not in memory.execution.observed_evidence_ids
    assert memory.execution.observed_evidence_ids == ()
    assert not memory.evaluation.outcome.task_success
    assert memory.evaluation.outcome.failure_category.value == "TEMPORAL_LEAKAGE"


def test_external_retrieval_at_t_excludes_not_yet_published_evidence() -> None:
    rag = next(item for item in build_synthetic_cases() if item.case_id == "U1-RAG")
    world = rag.resources.external_evidence_world
    future = ExternalEvidenceRecord(
        "future-guide", "GUIDELINE", datetime(2026, 9, 2, tzinfo=UTC),
        datetime(2026, 9, 2, tzinfo=UTC), (("authority", "synthetic"),),
        "Future evidence marker.", ("synthetic_guideline",),
    )
    changed_world = replace(world, records=(*world.records, future))
    hits = changed_world.retrieve(
        rag.episode.query, ("GUIDELINE",), as_of_time=rag.episode.decision_time
    )
    assert [item.source_id for item in hits] == ["guide-u1"]
