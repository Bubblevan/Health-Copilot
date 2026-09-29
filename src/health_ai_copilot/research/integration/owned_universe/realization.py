"""Adapter from latent owned worlds to frozen U1.1 runtime/evaluator contracts."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import (
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
    stable_hash,
)
from ..evidence_world import ExternalEvidenceRecord, ExternalEvidenceWorld
from ..executor import ExecutionResources
from ..state import PatientStateRecord, PatientStateStore
from .schema import (
    WORLD_NOTICE,
    FactLocation,
    LatentWorld,
    OwnedScenario,
    derive_capability_requirement,
)


@dataclass(frozen=True)
class MaterializedCase:
    scenario: OwnedScenario
    episode: IntegrationEpisode
    evaluation: EvaluationPlane
    resources: ExecutionResources


def _budget(budget_class: str, deadline_class: str) -> EpisodeBudget:
    global_units = {"CHEAP": 16, "NORMAL": 24, "RELAXED": 36}[budget_class]
    return EpisodeBudget(
        global_units=global_units, memory_read_units=4, external_retrieval_units=4,
        team_worker_units=8, tool_units=12, budget_class=budget_class,
        deadline_class=deadline_class, cost_model_version="u1.1-activation-observed-v1",
    )


def _workers(
    record_types: tuple[str, ...], families: tuple[str, ...], tool_ids: tuple[str, ...]
) -> tuple[WorkerManifest, ...]:
    scopes: list[list[str]] = [[], []]
    source_rows: list[list[str]] = [[], []]
    tools: list[list[str]] = [[], []]
    for index, item in enumerate(record_types):
        scopes[index % 2].append(item)
    for index, item in enumerate(families):
        source_rows[index % 2].append(item)
    for index in range(2):
        if scopes[index]:
            tools[index].append("memory_read")
        if source_rows[index]:
            tools[index].append("external_retrieval")
    for tool_id in tool_ids:
        if tool_id in {"memory_read", "external_retrieval"}:
            continue
        target = 0 if len(tools[0]) <= len(tools[1]) else 1
        tools[target].append(tool_id)
    return tuple(
        WorkerManifest(
            worker_id=f"u2e-worker-{index + 1}",
            personal_state_scopes=tuple(scopes[index]),
            external_source_families=tuple(source_rows[index]),
            tool_ids=tuple(tools[index]),
            capability_domains=(f"synthetic-partition-{index + 1}",),
        )
        for index in range(2)
    )


def materialize(world: LatentWorld) -> MaterializedCase:
    """Realize one latent case while keeping evaluator-only truth off runtime."""
    oracle = derive_capability_requirement(world)
    patient_rows = tuple(PatientStateRecord(
        record_id=row.record_id, subject_id=row.subject_id, timestamp=row.timestamp,
        record_type=PatientRecordType(row.record_type), source_provenance=WORLD_NOTICE,
        content=row.natural_language_content, retrieval_terms=row.retrieval_terms,
    ) for row in world.patient_records)
    patient_store = PatientStateStore(patient_rows)
    evidence_rows = tuple(ExternalEvidenceRecord(
        source_id=row.source_id, source_family=row.source_family,
        publication_time=row.publication_time, effective_time=row.effective_time,
        authority_metadata=(("authority", "project-owned synthetic namespace"),),
        content=row.natural_language_content, retrieval_terms=row.retrieval_terms,
        effective_until=row.effective_until,
    ) for row in world.evidence_records)
    evidence_world = ExternalEvidenceWorld.from_records(
        f"EXT-{stable_hash([row.to_dict() for row in evidence_rows])[:20]}",
        "owned-evidence-v1", evidence_rows
    )

    all_record_types = tuple(sorted({row.record_type for row in world.patient_records}))
    visible_rows = patient_store.snapshot(world.subject_id, world.decision_time)
    visible_types = tuple(sorted({row.record_type.value for row in visible_rows}))
    families = tuple(sorted({row.source_family for row in world.evidence_records}))
    tool_ids = tuple(sorted({"memory_read", "external_retrieval", *world.tool_ids}))
    workers = _workers(all_record_types, families, tool_ids)
    budget = _budget(world.budget_class, world.deadline_class)
    patient_ref = PatientStateRef(
        world.subject_id, f"snapshot-{stable_hash([row.to_dict() for row in patient_rows])[:16]}",
        all_record_types,
    )
    world_ref = ExternalEvidenceWorldRef(evidence_world.world_id, evidence_world.version, families)
    tool_surface = ToolSurfaceRef(
        "u2f-shared-tools-v1", "u2f-shared-tools-v1",
        tool_ids, workers,
    )
    observable = ObservableState(
        history_exists=bool(visible_rows),
        history_length_bucket=(
            "0" if not visible_rows else
            "1-4" if len(visible_rows) <= 4 else
            "5-7" if len(visible_rows) <= 7 else
            "8-23" if len(visible_rows) <= 23 else
            "24-63" if len(visible_rows) <= 63 else "64+"
        ),
        history_time_span=(None if len(visible_rows) < 2 else
                           _history_span(visible_rows[0].timestamp, visible_rows[-1].timestamp)),
        available_personal_state_types=visible_types,
        available_external_source_families=families,
        available_tool_ids=tool_ids,
        available_worker_capabilities=tuple(sorted({
            capability for worker in workers for capability in worker.capability_domains
        })),
        budget_class=world.budget_class, deadline_class=world.deadline_class,
        task_intent_metadata=(),
    )
    episode = IntegrationEpisode(
        episode_id=f"{world.world_id.removeprefix('LW-')}",
        environment_version="u2f-owned-environment-v1",
        source_provenance=WORLD_NOTICE, decision_time=world.decision_time,
        subject_id=world.subject_id, query=world.query, observable_state=observable,
        patient_state_ref=patient_ref, external_world_ref=world_ref,
        tool_surface_ref=tool_surface, budget=budget,
        evaluator_ref=EvaluatorRef("u2e-structured-evaluator", "v1"),
    )

    required_fact_ids = set(world.graph.required_fact_ids)
    fact_index = world.graph.fact_index
    answerable = oracle.answerability
    required_memory_facts = tuple(sorted(
        fact_index[fact_id].value for fact_id in required_fact_ids
        if fact_index[fact_id].location == FactLocation.PATIENT_STATE
    )) if answerable else ()
    required_memory_records = tuple(sorted({
        row.record_id for row in world.patient_records
        if set(row.latent_fact_ids) & required_fact_ids
    })) if oracle.memory_required else ()
    required_external_records = tuple(sorted({
        row.source_id for row in world.evidence_records
        if set(row.latent_fact_ids) & required_fact_ids
    })) if oracle.external_retrieval_required else ()
    answer_values = tuple(
        fact_index[fact_id].value
        for component in sorted(world.graph.answer_components, key=lambda row: row.order)
        for fact_id in component.required_fact_ids
    )
    evaluation = EvaluationPlane(
        episode_id=episode.episode_id,
        gold_answer="; ".join(answer_values) if answerable else "INSUFFICIENT_EVIDENCE",
        required_facts=answer_values if answerable else (),
        required_evidence_ids=required_external_records,
        task_success_predicate=("all_required_facts_and_evidence" if answerable else "safe_abstention"),
        required_memory_facts=required_memory_facts,
        required_memory_record_ids=required_memory_records,
        required_external_evidence_ids=required_external_records,
    )
    scenario = OwnedScenario(world=world, episode_id=episode.episode_id, oracle=oracle)
    return MaterializedCase(scenario, episode, evaluation,
                            ExecutionResources(patient_store, evidence_world))


def _history_span(start, end) -> str:
    days = max(0, (end - start).days)
    if days <= 14:
        return "synthetic-1-to-14-days"
    if days <= 29:
        return "synthetic-15-to-29-days"
    if days <= 119:
        return "synthetic-30-to-119-days"
    if days <= 364:
        return "synthetic-120-to-364-days"
    return "synthetic-365-plus-days"
