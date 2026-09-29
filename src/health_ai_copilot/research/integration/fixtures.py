"""Eight tiny synthetic contract episodes. They are not medical benchmark data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .contracts import (
    EpisodeBudget,
    EvaluationPlane,
    EvaluatorRef,
    ExternalEvidenceWorldRef,
    IntegrationEpisode,
    ObservableState,
    PatientRecordType,
    PatientStateRef,
    ToolSurfaceRef,
    WorkerManifest,
)
from .evidence_world import ExternalEvidenceRecord, ExternalEvidenceWorld
from .executor import ExecutionResources
from .state import PatientStateRecord, PatientStateStore

FIXTURE_STATUS = "SYNTHETIC_CONTRACT_FIXTURE; NOT_MEDICAL_BENCHMARK"
DECISION_TIME = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
EVALUATOR_REF = EvaluatorRef("u1-fixture-evaluator", "v1")


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    episode: IntegrationEpisode
    evaluation: EvaluationPlane
    resources: ExecutionResources
    contract_expectation: SyntheticContractExpectation


@dataclass(frozen=True)
class SyntheticContractExpectation:
    """Test-only fixture checks; never passed to evaluator or training views."""

    expected_team_tools: tuple[str, ...] = ()


def _budget() -> EpisodeBudget:
    return EpisodeBudget(
        24, 4, 4, 4, 12, "SYNTHETIC_STANDARD", "NO_DEADLINE",
        "u1.1-activation-observed-v1",
    )


def _team_workers(
    record_types: tuple[str, ...], families: tuple[str, ...], tool_ids: tuple[str, ...]
) -> tuple[WorkerManifest, ...]:
    scopes = [[], []]
    source_rows = [[], []]
    tools = [[], []]
    for index, value in enumerate(record_types):
        scopes[index % 2].append(value)
    for index, value in enumerate(families):
        source_rows[index % 2].append(value)
    for index in range(2):
        if scopes[index]:
            tools[index].append("memory_read")
        if source_rows[index]:
            tools[index].append("external_retrieval")
    for value in tool_ids:
        if value in {"memory_read", "external_retrieval"}:
            continue
        target = 0 if len(tools[0]) <= len(tools[1]) else 1
        tools[target].append(value)
    return (
        WorkerManifest("worker-a", tuple(scopes[0]), tuple(source_rows[0]), tuple(tools[0]), ("synthetic_partition_a",)),
        WorkerManifest("worker-b", tuple(scopes[1]), tuple(source_rows[1]), tuple(tools[1]), ("synthetic_partition_b",)),
    )


def _case(
    case_id: str,
    query: str,
    *,
    records: tuple[PatientStateRecord, ...] = (),
    evidence: tuple[ExternalEvidenceRecord, ...] = (),
    required_facts: tuple[str, ...],
    gold_answer: str,
    required_evidence_ids: tuple[str, ...] = (),
    required_memory_facts: tuple[str, ...] = (),
    required_memory_record_ids: tuple[str, ...] = (),
    safe_abstention: bool = False,
    tools: tuple[str, ...] = (),
) -> SyntheticCase:
    record_types = tuple(sorted({row.record_type.value for row in records}))
    families = tuple(sorted({row.source_family for row in evidence}))
    subject_id = "synthetic-subject-001" if records else None
    patient_ref = (PatientStateRef(subject_id, f"snapshot-{case_id}-t0", record_types)
                   if subject_id else None)
    world = (ExternalEvidenceWorld(f"world-{case_id}", "synthetic-v1", evidence)
             if evidence else None)
    world_ref = (ExternalEvidenceWorldRef(world.world_id, world.version, families)
                 if world else None)
    all_tools = list(tools)
    if records:
        all_tools.append("memory_read")
    if evidence:
        all_tools.append("external_retrieval")
    all_tools = sorted(set(all_tools))
    workers = _team_workers(record_types, families, tuple(all_tools))
    surface = ToolSurfaceRef(f"surface-{case_id}", "synthetic-v1", tuple(all_tools), workers)
    budget = _budget()
    visible_records = tuple(row for row in records if row.timestamp <= DECISION_TIME)
    visible_types = tuple(sorted({row.record_type.value for row in visible_records}))
    observable = ObservableState(
        history_exists=bool(visible_records),
        history_length_bucket="1-4" if visible_records else "0",
        history_time_span="synthetic-6-months" if visible_records else None,
        available_personal_state_types=visible_types,
        available_external_source_families=families,
        available_tool_ids=tuple(all_tools),
        available_worker_capabilities=("synthetic_partition_a", "synthetic_partition_b"),
        budget_class=budget.budget_class,
        deadline_class=budget.deadline_class,
        task_intent_metadata=(("intent", "synthetic_recall_or_lookup"),),
    )
    episode = IntegrationEpisode(
        episode_id=case_id, environment_version="u1-synthetic-env-v1",
        source_provenance=FIXTURE_STATUS, decision_time=DECISION_TIME,
        subject_id=subject_id, query=query, observable_state=observable,
        patient_state_ref=patient_ref, external_world_ref=world_ref,
        tool_surface_ref=surface, budget=budget, evaluator_ref=EVALUATOR_REF,
    )
    evaluation = EvaluationPlane(
        episode_id=case_id, gold_answer=gold_answer, required_facts=required_facts,
        required_evidence_ids=required_evidence_ids,
        task_success_predicate="safe_abstention" if safe_abstention else "all_required_facts_and_evidence",
        required_memory_facts=required_memory_facts,
        required_memory_record_ids=required_memory_record_ids,
        required_external_evidence_ids=required_evidence_ids,
        failure_labels=("TEMPORAL_LEAKAGE",) if case_id == "U1-TEMPORAL-LEAKAGE" else (),
    )
    return SyntheticCase(case_id, episode, evaluation,
                         ExecutionResources(PatientStateStore(records), world),
                         SyntheticContractExpectation(tools if case_id == "U1-TEAM" else ()))


def build_synthetic_cases() -> tuple[SyntheticCase, ...]:
    event_time = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    conversation_time = datetime(2026, 4, 11, 10, 0, tzinfo=UTC)
    future_time = datetime(2026, 9, 2, 9, 0, tzinfo=UTC)
    synthetic_date = datetime(2026, 8, 1, tzinfo=UTC)
    none = _case(
        "U1-NONE", "Current visit note: duration is 3 days. State duration.",
        required_facts=("3 days",), gold_answer="3 days",
    )
    memory = _case(
        "U1-MEM", "Using saved history marker history_key, report the care code.",
        records=(PatientStateRecord("mem-event-1", "synthetic-subject-001", event_time,
                                    PatientRecordType.EVENT, "synthetic-fixture", "Care code alpha-17.",
                                    ("history_key",)),),
        required_facts=("alpha-17",), gold_answer="alpha-17",
        required_memory_facts=("alpha-17",), required_memory_record_ids=("mem-event-1",),
    )
    rag = _case(
        "U1-RAG", "For synthetic_guideline, report the interval.",
        evidence=(ExternalEvidenceRecord("guide-u1", "GUIDELINE", synthetic_date, synthetic_date,
                                         (("authority", "synthetic publisher"),),
                                         "Synthetic guideline interval: 4 weeks.", ("synthetic_guideline",)),),
        required_facts=("4 weeks",), gold_answer="4 weeks",
        required_evidence_ids=("guide-u1",),
    )
    mem_rag = _case(
        "U1-MEM-RAG", "Combine history_marker and synthetic_public note.",
        records=(PatientStateRecord("mem-visit-1", "synthetic-subject-001", event_time,
                                    PatientRecordType.EVENT, "synthetic-fixture",
                                    "Recorded baseline is 12 minutes.", ("history_marker",)),),
        evidence=(ExternalEvidenceRecord("pub-u1", "PUBLIC_HEALTH", synthetic_date, synthetic_date,
                                         (("authority", "synthetic public notice"),),
                                         "Synthetic public note sets a 2 day window.", ("synthetic_public",)),),
        required_facts=("12 minutes", "2 day"), gold_answer="12 minutes; 2 day",
        required_evidence_ids=("pub-u1",), required_memory_facts=("12 minutes",),
        required_memory_record_ids=("mem-visit-1",),
    )
    team = _case(
        "U1-TEAM", "Synthetic arithmetic part_a=3; part_b=4; compute sum.",
        required_facts=("7",), gold_answer="7",
        tools=("query_part_a", "query_part_b"),
    )
    mem_team = _case(
        "U1-MEM-TEAM", "Use event_marker and conversation_marker; report both values.",
        records=(
            PatientStateRecord("mem-event-2", "synthetic-subject-001", event_time,
                               PatientRecordType.EVENT, "synthetic-fixture",
                               "Appointment date is 2026-08-04.", ("event_marker",)),
            PatientStateRecord("mem-conversation-2", "synthetic-subject-001", conversation_time,
                               PatientRecordType.CONVERSATION, "synthetic-fixture",
                               "Reported duration is 2 days.", ("conversation_marker",)),
        ),
        required_facts=("2026-08-04", "2 days"), gold_answer="2026-08-04; 2 days",
        required_memory_facts=("2026-08-04", "2 days"),
        required_memory_record_ids=("mem-event-2", "mem-conversation-2"),
    )
    all_caps = _case(
        "U1-ALL", "Use all_marker and synthetic_literature marker; include both results.",
        records=(PatientStateRecord("mem-event-3", "synthetic-subject-001", event_time,
                                    PatientRecordType.MEASUREMENT, "synthetic-fixture",
                                    "Recorded measurement is 120 units.", ("all_marker",)),),
        evidence=(ExternalEvidenceRecord("lit-u1", "LITERATURE", synthetic_date, synthetic_date,
                                         (("authority", "synthetic journal"),),
                                         "Synthetic literature example describes 2 groups.",
                                         ("synthetic_literature",)),),
        required_facts=("120 units", "2 groups"), gold_answer="120 units; 2 groups",
        required_evidence_ids=("lit-u1",), required_memory_facts=("120 units",),
        required_memory_record_ids=("mem-event-3",),
    )
    insufficient = _case(
        "U1-OOD-INSUFFICIENT", "Can you give a supported answer about an unspecified item?",
        required_facts=(), gold_answer="INSUFFICIENT_EVIDENCE", safe_abstention=True,
    )
    # Future-state record is present in the backing store but deliberately invisible at T.
    temporal = _case(
        "U1-TEMPORAL-LEAKAGE", "Using future_marker, report the recorded event.",
        records=(
            PatientStateRecord("mem-profile-1", "synthetic-subject-001", event_time,
                               PatientRecordType.PROFILE, "synthetic-fixture",
                               "Synthetic profile exists.", ("profile_only",)),
            PatientStateRecord("mem-future-1", "synthetic-subject-001", future_time,
                               PatientRecordType.EVENT, "synthetic-fixture",
                               "Future event is 2026-09-02.", ("future_marker",)),
        ),
        required_facts=("2026-09-02",), gold_answer="2026-09-02",
        required_memory_facts=("2026-09-02",), required_memory_record_ids=("mem-future-1",),
    )
    return none, memory, rag, mem_rag, team, mem_team, all_caps, insufficient, temporal
