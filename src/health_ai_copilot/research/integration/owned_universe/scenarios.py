"""Deterministic authoring of latent longitudinal worlds and surface queries."""

from __future__ import annotations

from datetime import timedelta

from .facts import key_token, make_fact, stable_code, value_token
from .grammar import render_current_cue, render_query
from .realization_text import realize_evidence_record, realize_patient_record
from .schema import (
    AnswerComponent,
    DependencyGraph,
    FactLocation,
    LatentFact,
    LatentWorld,
    OwnedEvidenceRecord,
    OwnedPatientRecord,
    StructuredAnswerType,
    derive_capability_requirement,
)
from .timeline import event_time

SCENARIO_FAMILIES = (
    "CURRENT_ONLY", "MEMORY_LOOKUP", "MEMORY_REVISION", "MEMORY_TEMPORAL_COMPARE",
    "MEMORY_MULTI_RECORD", "EXTERNAL_LOOKUP", "EXTERNAL_VERSIONED",
    "EXTERNAL_MULTI_SOURCE", "MEMORY_EXTERNAL_JOIN", "MEMORY_EXTERNAL_CONFLICT",
    "INSUFFICIENT_EVIDENCE", "DISTRACTOR_HEAVY", "TEMPORAL_BOUNDARY",
    "COMPOSITIONAL_MULTI_FACT",
)


def _new_fact(
    facts: list[LatentFact], *, seed: int, namespace: str,
    location: FactLocation, artifact: str, when, valid_until=None, revision_of=None,
) -> LatentFact:
    row = make_fact(
        fact_id=f"LF-{stable_code(seed, namespace, 12)}",
        value=value_token(seed, namespace), location=location,
        source_artifact_id=artifact, valid_from=when,
        valid_until=valid_until, revision_of=revision_of,
    )
    facts.append(row)
    return row


def _append_patient(
    records: list[OwnedPatientRecord], *, fact: LatentFact, subject_id: str,
    record_id: str, key: str, timestamp,
) -> None:
    records.append(realize_patient_record(
        record_id=record_id, subject_id=subject_id, timestamp=timestamp,
        fact_id=fact.fact_id, value=fact.value, key=key,
    ))


def _append_evidence(
    records: list[OwnedEvidenceRecord], *, fact: LatentFact,
    source_id: str, family: str, key: str, publication_time,
) -> None:
    records.append(realize_evidence_record(
        source_id=source_id, source_family=family,
        publication_time=publication_time, effective_time=publication_time,
        fact_id=fact.fact_id, value=fact.value, key=key,
    ))


def build_world(
    *, episode_id: str, split_role: str, subject_id: str,
    persona_seed: int, scenario_seed: int, scenario_family: str,
    template_family: str, surface_variant_seed: int,
    counterfactual_family_id: str, budget_class: str, deadline_class: str,
    template_pool: tuple[str, ...], diagnostic_mode: str = "standard",
    diagnostic_slot: int = 0,
) -> LatentWorld:
    """Construct one world without reading files, environment data, or external corpora."""
    if scenario_family not in SCENARIO_FAMILIES:
        raise ValueError(f"unregistered owned scenario family: {scenario_family}")
    facts: list[LatentFact] = []
    patients: list[OwnedPatientRecord] = []
    evidence: list[OwnedEvidenceRecord] = []
    components: list[AnswerComponent] = []
    decision = event_time(persona_seed, 20)
    base = event_time(persona_seed, 2)
    key = key_token(scenario_seed, "primary")
    current_values: list[str] = []
    answer_type = StructuredAnswerType.EXACT_TOKEN
    safe_abstention = False
    query_tool_ids: tuple[str, ...] = ()
    cue_seed = (scenario_seed + surface_variant_seed) % 4

    def add_component(fact: LatentFact, order: int | None = None) -> None:
        components.append(AnswerComponent(
            component_id=f"AC-{fact.fact_id}", required_fact_ids=(fact.fact_id,),
            order=len(components) if order is None else order,
        ))
        current_values.append(fact.value)

    def add_patient(namespace: str, *, when=base, valid_until=None,
                    revision_of=None, record_key: str | None = None) -> LatentFact:
        record_id = f"PR-{stable_code(scenario_seed, namespace, 12)}"
        record_key = record_key or key
        fact = _new_fact(
            facts, seed=scenario_seed, namespace=namespace,
            location=FactLocation.PATIENT_STATE, artifact=record_id,
            when=when, valid_until=valid_until, revision_of=revision_of,
        )
        _append_patient(patients, fact=fact, subject_id=subject_id,
                        record_id=record_id, key=record_key, timestamp=when)
        return fact

    def add_external(namespace: str, family: str = "PUBLIC_HEALTH", *,
                     when=base, record_key: str | None = None) -> LatentFact:
        source_id = f"ER-{stable_code(scenario_seed, namespace, 12)}"
        record_key = record_key or key
        fact = _new_fact(
            facts, seed=scenario_seed, namespace=namespace,
            location=FactLocation.EXTERNAL_EVIDENCE, artifact=source_id, when=when,
        )
        _append_evidence(evidence, fact=fact, source_id=source_id,
                         family=family, key=record_key, publication_time=when)
        return fact

    # Small recurring distractors make capability availability independent of
    # whether the answer itself needs Memory or external retrieval.
    def add_distractor(namespace: str, location: FactLocation, ordinal: int) -> None:
        distractor_key = key_token(scenario_seed, f"distractor-{namespace}-{ordinal}")
        if location == FactLocation.PATIENT_STATE:
            add_patient(namespace, record_key=distractor_key,
                        when=event_time(persona_seed, 1 + ordinal))
        elif location == FactLocation.EXTERNAL_EVIDENCE:
            add_external(namespace, family=("GUIDELINE", "LITERATURE", "PUBLIC_HEALTH")[ordinal % 3],
                         record_key=distractor_key,
                         when=event_time(persona_seed, 1 + ordinal))

    current_fact: LatentFact | None = None
    if scenario_family == "CURRENT_ONLY":
        current_fact = _new_fact(
            facts, seed=scenario_seed, namespace="current", location=FactLocation.CURRENT_CONTEXT,
            artifact=f"QUERY-{episode_id}", when=base,
        )
        add_component(current_fact)
    elif scenario_family in {"MEMORY_LOOKUP", "MEMORY_MULTI_RECORD"}:
        count = 2 if scenario_family == "MEMORY_MULTI_RECORD" else 1
        for ordinal in range(count):
            fact_key = key if ordinal == 0 else key_token(scenario_seed, f"secondary-{ordinal}")
            fact = add_patient(f"memory-{ordinal}",
                               when=event_time(persona_seed, 4 + ordinal), record_key=fact_key)
            add_component(fact)
        if count > 1:
            answer_type = StructuredAnswerType.EXACT_SET
    elif scenario_family == "MEMORY_REVISION":
        revised_at = event_time(persona_seed, 12)
        old_key = key_token(scenario_seed, "revision-v1")
        new_key = key_token(scenario_seed, "revision-v2")
        old_record = f"PR-{stable_code(scenario_seed, 'revision-old', 12)}"
        old = _new_fact(facts, seed=scenario_seed, namespace="revision-v1",
                        location=FactLocation.PATIENT_STATE, artifact=old_record,
                        when=event_time(persona_seed, 5), valid_until=revised_at)
        _append_patient(patients, fact=old, subject_id=subject_id, record_id=old_record,
                        key=old_key, timestamp=event_time(persona_seed, 5))
        current_key = old_key if diagnostic_mode == "revision-before" else new_key
        if diagnostic_mode == "revision-before":
            decision = revised_at - timedelta(seconds=1)
            add_component(old)
        else:
            new_record = f"PR-{stable_code(scenario_seed, 'revision-new', 12)}"
            new = _new_fact(facts, seed=scenario_seed, namespace="revision-v2",
                            location=FactLocation.PATIENT_STATE, artifact=new_record,
                            when=revised_at, revision_of=old.fact_id)
            _append_patient(patients, fact=new, subject_id=subject_id, record_id=new_record,
                            key=new_key, timestamp=revised_at)
            add_component(new)
        key = current_key
    elif scenario_family in {"MEMORY_TEMPORAL_COMPARE", "TEMPORAL_BOUNDARY"}:
        if scenario_family == "TEMPORAL_BOUNDARY":
            boundary = event_time(persona_seed, 10)
            target_key = key_token(scenario_seed, "boundary")
            # A visible unrelated record keeps the Memory action available on T-1.
            add_distractor("boundary-background", FactLocation.PATIENT_STATE, 0)
            fact = add_patient("boundary-target", when=boundary, record_key=target_key)
            add_component(fact)
            key = target_key
            if diagnostic_mode == "boundary" and diagnostic_slot in (0, 1, 2, 3):
                boundary_offsets = (-1, 0, 1, 0)
                decision = boundary + timedelta(seconds=boundary_offsets[diagnostic_slot])
        else:
            for ordinal, offset in enumerate((6, 13)):
                record_key = key if ordinal == 0 else key_token(scenario_seed, "temporal-second")
                fact = add_patient(f"temporal-{ordinal}",
                                   when=event_time(persona_seed, offset), record_key=record_key)
                add_component(fact)
            answer_type = StructuredAnswerType.ORDERED_SEQUENCE
    elif scenario_family == "EXTERNAL_LOOKUP":
        add_component(add_external("external-lookup", "PUBLIC_HEALTH"))
    elif scenario_family == "EXTERNAL_VERSIONED":
        published_at = event_time(persona_seed, 12)
        old_key = key_token(scenario_seed, "evidence-v1")
        new_key = key_token(scenario_seed, "evidence-v2")
        old = add_external("external-v1", "GUIDELINE",
                           when=event_time(persona_seed, 5), record_key=old_key)
        if diagnostic_mode == "evidence-before":
            decision = published_at - timedelta(seconds=1)
            key = old_key
            add_component(old)
        else:
            new = add_external("external-v2", "GUIDELINE",
                               when=published_at, record_key=new_key)
            key = new_key
            add_component(new)
    elif scenario_family == "EXTERNAL_MULTI_SOURCE":
        for ordinal, family in enumerate(("PUBLIC_HEALTH", "LITERATURE")):
            record_key = key if ordinal == 0 else key_token(scenario_seed, "source-two")
            fact = add_external(f"external-source-{ordinal}", family, record_key=record_key)
            add_component(fact)
        answer_type = StructuredAnswerType.EXACT_SET
    elif scenario_family in {"MEMORY_EXTERNAL_JOIN", "MEMORY_EXTERNAL_CONFLICT"}:
        personal = add_patient("cross-plane-personal", record_key=key)
        evidence_key = key_token(scenario_seed, "cross-plane-evidence")
        external = add_external("cross-plane-external", "PUBLIC_HEALTH", record_key=evidence_key)
        add_component(personal)
        add_component(external)
        answer_type = StructuredAnswerType.EXACT_SET
    elif scenario_family == "INSUFFICIENT_EVIDENCE":
        missing = _new_fact(
            facts, seed=scenario_seed, namespace="unavailable",
            location=FactLocation.UNAVAILABLE, artifact=f"MISSING-{episode_id}", when=base,
        )
        add_component(missing)
        safe_abstention = True
        answer_type = StructuredAnswerType.ABSTAIN
    elif scenario_family == "DISTRACTOR_HEAVY":
        # No requested answer lives in the distractor stores; it is in the
        # current note. Five+ irrelevant rows make length a poor proxy.
        current_fact = _new_fact(
            facts, seed=scenario_seed, namespace="distractor-heavy-current",
            location=FactLocation.CURRENT_CONTEXT, artifact=f"QUERY-{episode_id}", when=base,
        )
        add_component(current_fact)
        for ordinal in range(3):
            add_distractor(f"heavy-memory-{ordinal}", FactLocation.PATIENT_STATE, ordinal)
            add_distractor(f"heavy-external-{ordinal}", FactLocation.EXTERNAL_EVIDENCE, ordinal)
    elif scenario_family == "COMPOSITIONAL_MULTI_FACT":
        personal = add_patient("composition-personal", record_key=key)
        evidence_key = key_token(scenario_seed, "composition-evidence")
        external = add_external("composition-external", "LITERATURE", record_key=evidence_key)
        tool_value = "7"
        tool_fact = make_fact(
            fact_id=f"LF-{stable_code(scenario_seed, 'composition-tool', 12)}",
            value=tool_value, location=FactLocation.TOOL_OUTPUT,
            source_artifact_id="query_part_a+query_part_b",
            valid_from=base,
        )
        facts.append(tool_fact)
        add_component(personal)
        add_component(external)
        add_component(tool_fact)
        query_tool_ids = ("query_part_a", "query_part_b")
        answer_type = StructuredAnswerType.EXACT_SET
    else:
        raise AssertionError(f"unhandled scenario family: {scenario_family}")

    # Pad ordinary cases to a minimum of two personal and two evidence rows,
    # keeping resource-availability and length cues decoupled from labels.
    heavy = scenario_family == "DISTRACTOR_HEAVY"
    desired_rows = 6 if heavy else 2
    while len(patients) < desired_rows:
        add_distractor(f"pad-patient-{len(patients)}", FactLocation.PATIENT_STATE, len(patients))
    while len(evidence) < desired_rows:
        add_distractor(f"pad-evidence-{len(evidence)}", FactLocation.EXTERNAL_EVIDENCE,
                       len(evidence))

    if not components:
        raise AssertionError("generated scenario must have at least one answer component")
    if current_fact is not None:
        # Surface cue templates deliberately carry already-present current truth;
        # MEMORY/RETRIEVAL resources, when available, remain irrelevant.
        query, _, cue_variant = render_current_cue(
            cue_seed, key, current_fact.value, template_family
        )
        surface_variant = f"{surface_variant_seed % 2 + 1:02d}-{cue_variant}"
    else:
        query, _, lexical_variant = render_query(
            template_family, surface_variant_seed, key
        )
        # A small grammar variation inserts a second required key but never an
        # architecture or capability label.
        if len(current_values) > 1:
            second_key = next((record.retrieval_terms[0] for record in (*patients, *evidence)
                               if record.retrieval_terms[0] != key
                               and set(record.latent_fact_ids) & set(
                                   {fact_id for c in components for fact_id in c.required_fact_ids})), None)
            if second_key:
                query = f"{query} Include the item indexed by {second_key}."
        surface_variant = f"{surface_variant_seed % 2 + 1:02d}-{lexical_variant}"

    if query_tool_ids:
        query += " Synthetic inputs: part_a=3; part_b=4; sum."
    if scenario_family in {"INSUFFICIENT_EVIDENCE", "TEMPORAL_BOUNDARY"} \
            or diagnostic_mode == "matched-abstain":
        query = f"Please give a supported answer for {key}."

    graph = DependencyGraph(tuple(facts), tuple(components))
    world = LatentWorld(
        world_id=f"LW-{episode_id}", subject_id=subject_id,
        scenario_family=scenario_family, template_family=template_family,
        surface_variant=surface_variant, counterfactual_family_id=counterfactual_family_id,
        persona_seed=persona_seed, scenario_seed=scenario_seed, decision_time=decision,
        query=query, graph=graph, patient_records=tuple(patients),
        evidence_records=tuple(evidence),
        distractor_count=max(0, len(patients) + len(evidence)
                             - sum(fact.location in {FactLocation.PATIENT_STATE,
                                                    FactLocation.EXTERNAL_EVIDENCE}
                                   for fact_id in graph.required_fact_ids
                                   for fact in (graph.fact_index[fact_id],))),
        split_role=split_role, budget_class=budget_class, deadline_class=deadline_class,
        answer_type=answer_type, safe_abstention=safe_abstention,
        tool_ids=query_tool_ids,
    )
    # Ensure the graph-derived policy mask is computed from the final timepoint.
    _ = derive_capability_requirement(world)
    return world
