"""U2-F scale generation over the qualified U2-E latent contract."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from itertools import pairwise
from typing import Any

from .facts import key_token, make_fact, stable_code, value_token
from .realization_text import realize_evidence_record, realize_patient_record
from .scenarios import SCENARIO_FAMILIES, build_world
from .schema import (
    AnswerComponent,
    DependencyGraph,
    FactLocation,
    FactValidityInterval,
    LatentFact,
    LatentWorld,
    OwnedEvidenceRecord,
    OwnedPatientRecord,
    StructuredAnswerType,
)
from .timeline import subject_epoch

PATIENT_RECORD_TYPES = ("PROFILE", "EVENT", "MEASUREMENT", "EXAM", "CONVERSATION")
CONVERSATION_KINDS = (
    "USER_PREFERENCE", "PREVIOUS_INSTRUCTION", "PREVIOUS_PLAN_CODE", "PREVIOUS_CORRECTION",
)
EXTERNAL_FAMILIES = ("PUBLIC_HEALTH", "GUIDELINE", "LITERATURE")
TRAIN_TEMPLATE_FAMILIES = (
    "TRAIN_LOOKUP", "TRAIN_TEMPORAL", "TRAIN_UPDATE", "TRAIN_AGGREGATE",
    "TRAIN_SOURCE", "TRAIN_BOOLEAN", "TRAIN_SEQUENCE", "TRAIN_NUMERIC",
    "TRAIN_ABSTAIN", "TRAIN_COMPOSE", "TRAIN_CONVERSATION", "TRAIN_TREND",
    "TRAIN_CURRENT",
)
STRUCTURAL_TEMPLATE_FAMILIES = (
    "DEV_STRUCTURAL_ORDER", "DEV_STRUCTURAL_COMPOSE",
    "DEV_STRUCTURAL_CONDITION", "DEV_STRUCTURAL_CONTRAST",
)
HISTORY_DAY_CHOICES = {
    "SHORT": (13,),
    "MEDIUM": (40, 60, 120),
    "LONG": (180, 240, 300, 350),
    "SATURATED": (450, 600, 730),
}

NEUTRAL_QUERY_CONTEXTS = (
    "This exercise uses an opaque synthetic identifier.",
    "All values belong to a project-owned research world.",
    "The generated example has deterministic structured truth.",
    "No real patient observation is represented here.",
    "The response is checked against a frozen contract.",
    "Dates and quantities are fictional placeholders.",
    "The scenario remains separate from clinical guidance.",
    "Each generated record has traceable synthetic lineage.",
)
NEUTRAL_QUERY_SURFACES = (
    "Return the requested synthetic result.",
    "Provide the requested synthetic result.",
)


def _hash_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _weighted_choice(weights: dict[str, float], *parts: object) -> str:
    point = _hash_int(*parts) / (2**64)
    total = 0.0
    for label, weight in weights.items():
        total += float(weight)
        if point < total:
            return label
    return next(reversed(weights))


def _surface_text(world: LatentWorld, variants: tuple[str, str]) -> str:
    variant = int(world.surface_variant.split("-", 1)[0]) - 1
    return variants[variant % len(variants)]


def _balance_query_surface(world: LatentWorld) -> str:
    """Apply a requirement-independent lexical shell and reproducible context noise."""
    metadata = dict(world.structural_metadata)
    same_surface = metadata.get("same_surface_stress_group", 0) == 1
    surface_slot = 0 if same_surface else int(world.surface_variant.split("-", 1)[0]) - 1
    surface = NEUTRAL_QUERY_SURFACES[surface_slot % len(NEUTRAL_QUERY_SURFACES)]
    count = _hash_int(world.scenario_seed, "query-context-count", surface_slot) % 5
    start = _hash_int(world.scenario_seed, "query-context-start", surface_slot) % len(
        NEUTRAL_QUERY_CONTEXTS
    )
    context = [NEUTRAL_QUERY_CONTEXTS[(start + offset) % len(NEUTRAL_QUERY_CONTEXTS)]
               for offset in range(count)]
    key_instruction = "Include the item indexed by its synthetic key."
    return " ".join(part for part in (world.query, key_instruction, surface, *context) if part)


def generate_u2f_plan(plan: dict[str, Any], profile: dict[str, Any]) -> tuple[LatentWorld, ...]:
    """Create paired worlds from disjoint seed pools and frozen scenario weights."""
    role = str(plan["split_role"])
    count = int(plan["episode_count"])
    subject_count = int(plan["subject_count"])
    templates = tuple(str(item) for item in plan["template_families"])
    pair_count = count // 2
    pair_start, pair_end = (int(value) for value in plan["scenario_seed_range"])
    subject_start, subject_end = (int(value) for value in plan["persona_seed_range"])
    if count != int(profile["targets"][role]["episode_count"]) or count % 2:
        raise ValueError(f"{role} episode count must match the even frozen profile target")
    if subject_count != int(profile["targets"][role]["subject_count"]):
        raise ValueError(f"{role} subject count must match the frozen profile target")
    if pair_end - pair_start + 1 != pair_count:
        raise ValueError(f"{role} scenario seed pool must be non-overlapping and exact-sized")
    if subject_end - subject_start + 1 != subject_count:
        raise ValueError(f"{role} persona seed pool must be non-overlapping and exact-sized")
    same_surface_pairs = int(plan["same_surface_pair_count"])
    if same_surface_pairs != int(profile["same_surface_pair_count"][role]):
        raise ValueError(f"{role} same-surface count must match the frozen scale profile")
    if not 0 <= same_surface_pairs <= pair_count:
        raise ValueError("same_surface_pair_count exceeds the split pair pool")
    weights = {str(key): float(value)
               for key, value in profile["scenario_family_weights"].items()}
    if set(weights) != set(SCENARIO_FAMILIES) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("scenario family weights must cover the frozen grammar and sum to one")

    matched_indices = set(sorted(
        range(pair_count),
        key=lambda index: _hash_int(profile["global_seed"], role, index, "matched-pair"),
    )[:same_surface_pairs])
    rows: list[LatentWorld] = []
    for pair_index in range(pair_count):
        pair_seed = pair_start + pair_index
        subject_index = pair_index % subject_count
        persona_seed = subject_start + subject_index
        subject_id = f"SUBJ-{stable_code(persona_seed, 'u2f-subject', 12)}"
        counterfactual_id = f"U2F-CF-{role}-{pair_index:05d}"
        matched = pair_index in matched_indices
        if matched:
            pair_kind = _hash_int(profile["global_seed"], role, pair_index, "matched-kind") % 2
            family_pair = (
                ("MEMORY_LOOKUP", "INSUFFICIENT_EVIDENCE"),
                ("EXTERNAL_LOOKUP", "INSUFFICIENT_EVIDENCE"),
            )[pair_kind]
        else:
            family = _weighted_choice(weights, profile["global_seed"], role, pair_seed, "family")
            family_pair = (family, family)

        first_template_index = _hash_int(profile["global_seed"], role, pair_seed, "template") % len(templates)
        first_template = templates[first_template_index]
        second_template = (first_template if matched else
                           templates[(first_template_index + 1 +
                                      _hash_int(pair_seed, "template-offset") % (len(templates) - 1))
                                     % len(templates)])
        for slot, family in enumerate(family_pair):
            template = first_template if slot == 0 else second_template
            episode_id = f"U2F-{stable_code(pair_seed, f'episode-{slot}', 16)}"
            world = build_world(
                episode_id=episode_id,
                split_role=role,
                subject_id=subject_id,
                persona_seed=persona_seed,
                scenario_seed=pair_seed,
                scenario_family=family,
                template_family=template,
                surface_variant_seed=0 if matched else slot,
                counterfactual_family_id=counterfactual_id,
                budget_class=("CHEAP", "NORMAL", "RELAXED")[_hash_int(pair_seed, "budget") % 3],
                deadline_class=("TIGHT", "RELAXED")[_hash_int(pair_seed, "deadline") % 2],
                template_pool=templates,
                diagnostic_mode="u2f-matched-surface" if matched else "u2f",
            )
            if matched:
                world = replace(world, structural_metadata=tuple(sorted({
                    **dict(world.structural_metadata), "same_surface_stress_group": 1,
                }.items())))
            rows.append(world)
    return tuple(rows)


def _timeline_rows(role: str, subject_id: str, persona_seed: int) -> tuple[tuple[LatentFact, OwnedPatientRecord], ...]:
    epoch = subject_epoch(persona_seed)
    days = [1, 4, 8, 12, 15]
    days.extend(range(20, 121, 10))
    days.extend(round(125 + index * 240 / 46) for index in range(47))
    days.extend(round(390 + index * 340 / 16) for index in range(17))
    if len(days) != 80 or days != sorted(days):
        raise AssertionError("frozen longitudinal timeline must have 80 ordered record times")
    timeline_id = f"TL-{stable_code(persona_seed, f'{role}-{subject_id}', 16)}"
    records: list[tuple[LatentFact, OwnedPatientRecord]] = []
    prior_revision: str | None = None
    revision_indices = {8, 25, 45, 65}
    revision_key = key_token(persona_seed, "longitudinal-revision")
    previous_revision_index: int | None = None
    for index, day in enumerate(days):
        record_type = PATIENT_RECORD_TYPES[index % len(PATIENT_RECORD_TYPES)]
        timestamp = epoch + timedelta(days=day)
        record_id = f"{timeline_id}-R{index:03d}"
        namespace = f"longitudinal-{subject_id}-{index}"
        fact_id = f"LF-{stable_code(persona_seed, namespace, 16)}"
        if record_type == "MEASUREMENT":
            value = str(3 + _hash_int(persona_seed, index, "measurement") % 97)
        else:
            value = value_token(persona_seed, namespace)
        revision_of = prior_revision if index in revision_indices else None
        if index in revision_indices and previous_revision_index is not None:
            previous_fact, previous_record = records[previous_revision_index]
            records[previous_revision_index] = (
                replace(previous_fact, validity=FactValidityInterval(
                    previous_fact.validity.valid_from, timestamp)),
                previous_record,
            )
        fact = make_fact(
            fact_id=fact_id,
            value=value,
            location=FactLocation.PATIENT_STATE,
            source_artifact_id=record_id,
            valid_from=timestamp,
            valid_until=None,
            revision_of=revision_of,
        )
        if index in revision_indices:
            prior_revision = fact.fact_id
            previous_revision_index = len(records)
        key = revision_key if index in revision_indices else key_token(
            persona_seed, f"timeline-record-{index}"
        )
        record = realize_patient_record(
            record_id=record_id,
            subject_id=subject_id,
            timestamp=timestamp,
            fact_id=fact_id,
            value=value,
            key=key,
            record_type=record_type,
            conversation_kind=CONVERSATION_KINDS[_hash_int(
                persona_seed, index, "conversation-kind"
            ) % len(CONVERSATION_KINDS)] if record_type == "CONVERSATION"
            else "USER_CONFIRMED_STATE",
        )
        records.append((fact, record))
    return tuple(records)


def _regime_for_history(record_count: int, span_days: int) -> str:
    if 2 <= record_count <= 6 and 1 <= span_days <= 14:
        return "SHORT"
    if 8 <= record_count <= 24 and 30 <= span_days <= 120:
        return "MEDIUM"
    if 24 <= record_count <= 64 and 120 <= span_days <= 365:
        return "LONG"
    if 64 <= record_count <= 128 and 365 <= span_days <= 730:
        return "SATURATED"
    return "OUT_OF_PROFILE"


def _allocate_subject_timelines(
    pairs_by_subject: dict[str, list[str]], weights: dict[str, float],
) -> tuple[dict[str, str], dict[str, int]]:
    """Assign progressively later history regimes within every subject timeline."""
    if set(weights) != set(HISTORY_DAY_CHOICES) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("history regime weights must cover all four frozen regimes and sum to one")
    thresholds: list[tuple[float, str]] = []
    cumulative = 0.0
    for regime, weight in weights.items():
        cumulative += float(weight)
        thresholds.append((cumulative, regime))
    regimes: dict[str, str] = {}
    target_days: dict[str, int] = {}
    for pair_ids in pairs_by_subject.values():
        ordered = sorted(pair_ids, key=lambda pair_id: int(pair_id.rsplit("-", 1)[-1]))
        assigned: list[tuple[str, str]] = []
        totals: Counter[str] = Counter()
        for rank, pair_id in enumerate(ordered):
            point = (rank + 0.5) / len(ordered)
            regime = next(label for boundary, label in thresholds if point < boundary)
            assigned.append((pair_id, regime))
            totals[regime] += 1
        seen: Counter[str] = Counter()
        for pair_id, regime in assigned:
            regimes[pair_id] = regime
            options = HISTORY_DAY_CHOICES[regime]
            ordinal = seen[regime]
            option_index = min(len(options) - 1, ordinal * len(options) // totals[regime])
            target_days[pair_id] = options[option_index]
            seen[regime] += 1
    return regimes, target_days


def _revision_world(world: LatentWorld) -> LatentWorld:
    """Replace a shallow personal revision example with a validity-aware chain."""
    if world.scenario_family != "MEMORY_REVISION":
        return world
    old_required = set(world.graph.required_fact_ids)
    revision_fact_ids = set(old_required)
    changed = True
    while changed:
        expanded = revision_fact_ids | {
            fact.fact_id for fact in world.graph.facts
            if fact.revision_of in revision_fact_ids
        }
        expanded.update(
            fact.revision_of for fact in world.graph.facts
            if fact.fact_id in revision_fact_ids and fact.revision_of is not None
        )
        changed = expanded != revision_fact_ids
        revision_fact_ids = expanded
    old_patient_ids = {row.record_id for row in world.patient_records
                       if revision_fact_ids.intersection(row.latent_fact_ids)}
    kept_records = tuple(row for row in world.patient_records if row.record_id not in old_patient_ids)
    kept_facts = [fact for fact in world.graph.facts if fact.fact_id not in revision_fact_ids]
    epoch = subject_epoch(world.persona_seed)
    depth = 1 + _hash_int(world.scenario_seed, "revision-depth") % 3
    version_count = int(depth) + 1
    all_days = (5, 12, 120, 365)[:version_count]
    keys = tuple(key_token(world.scenario_seed, f"revision-version-{index}")
                 for index in range(version_count))
    versions: list[tuple[LatentFact, OwnedPatientRecord]] = []
    previous: str | None = None
    for index, day in enumerate(all_days):
        timestamp = epoch + timedelta(days=day)
        record_id = f"PR-U2F-REV-{stable_code(world.scenario_seed, f'version-{index}', 12)}"
        fact_id = f"LF-U2F-REV-{stable_code(world.scenario_seed, f'version-{index}', 16)}"
        next_day = all_days[index + 1] if index + 1 < version_count else None
        fact = make_fact(
            fact_id=fact_id,
            value=value_token(world.scenario_seed, f"revision-value-{index}"),
            location=FactLocation.PATIENT_STATE,
            source_artifact_id=record_id,
            valid_from=timestamp,
            valid_until=epoch + timedelta(days=next_day) if next_day else None,
            revision_of=previous,
        )
        record_type = PATIENT_RECORD_TYPES[_hash_int(world.scenario_seed, index, "rev-type")
                                           % len(PATIENT_RECORD_TYPES)]
        record = realize_patient_record(
            record_id=record_id, subject_id=world.subject_id, timestamp=timestamp,
            fact_id=fact_id, value=fact.value, key=keys[index], record_type=record_type,
        )
        versions.append((fact, record))
        previous = fact_id
    visible = [item for item in versions if item[0].validity.valid_from <= world.decision_time]
    if not visible:
        visible = [versions[0]]
    mode = _hash_int(world.scenario_seed, "revision-query-mode") % 4
    if mode == 1 and len(visible) > 1:
        selected = [visible[-2]]
        requested_at = selected[0][0].validity.valid_from + timedelta(days=1)
        components = (AnswerComponent(f"AC-{selected[0][0].fact_id}",
                                      (selected[0][0].fact_id,), 0,
                                      requested_as_of=requested_at),)
        query = _surface_text(world, (
            f"At the earlier valid time, report the value for {keys[versions.index(selected[0])]}.",
            f"Which value was valid at the earlier requested time for {keys[versions.index(selected[0])]}?",
        ))
    elif mode == 2 and len(visible) > 1:
        selected = visible
        components = tuple(AnswerComponent(
            f"AC-{fact.fact_id}", (fact.fact_id,), order,
            requested_as_of=fact.validity.valid_from + timedelta(seconds=1),
        ) for order, (fact, _) in enumerate(selected))
        selected_keys = [keys[versions.index(item)] for item in selected]
        selected_text = " ".join(selected_keys)
        query = _surface_text(world, (
            f"Return the change sequence in time order: {selected_text}.",
            f"List each linked value in chronological order for {selected_text}.",
        ))
    else:
        selected = [visible[-1]]
        index = versions.index(selected[0])
        components = (AnswerComponent(f"AC-{selected[0][0].fact_id}",
                                      (selected[0][0].fact_id,), 0),)
        query = _surface_text(world, (
            f"What is the latest valid value for {keys[index]}?",
            f"Report the current value valid at the query time for {keys[index]}.",
        ))
    facts = tuple(sorted((*kept_facts, *(fact for fact, _ in versions)), key=lambda item: item.fact_id))
    edges = tuple((versions[index][0].fact_id, versions[index + 1][0].fact_id)
                  for index in range(version_count - 1))
    graph = DependencyGraph(facts, components, edges)
    return replace(
        world, graph=graph, patient_records=tuple(sorted((*kept_records,
            *(record for _, record in versions)), key=lambda item: (item.timestamp, item.record_id))),
        query=query, answer_type=(StructuredAnswerType.ORDERED_SEQUENCE
                                  if len(components) > 1 else world.answer_type),
        structural_metadata=tuple(sorted({**dict(world.structural_metadata),
            "revision_depth": int(depth), "revision_chain_length": version_count}.items())),
    )


def _versioned_external_world(world: LatentWorld) -> LatentWorld:
    """Build publication/effectivity-independent synthetic evidence revisions."""
    if world.scenario_family != "EXTERNAL_VERSIONED":
        return world
    old_required = set(world.graph.required_fact_ids)
    old_ids = {row.source_id for row in world.evidence_records
               if old_required.intersection(row.latent_fact_ids)}
    kept_records = tuple(row for row in world.evidence_records if row.source_id not in old_ids)
    kept_facts = [fact for fact in world.graph.facts if fact.fact_id not in old_required]
    epoch = subject_epoch(world.persona_seed)
    # E2 remains valid through day 40; E3 becomes effective on day 45 and
    # publishes on day 55, independently exercising effectivity and publication.
    days = (2, 12, 45)
    publication_days = (1, 10, 55)
    keys = tuple(key_token(world.scenario_seed, f"external-version-{index}")
                 for index in range(len(days)))
    versions: list[tuple[LatentFact, OwnedEvidenceRecord]] = []
    for index, (effective_day, published_day) in enumerate(zip(days, publication_days, strict=True)):
        source_id = f"ER-U2F-EV-{stable_code(world.scenario_seed, f'ext-version-{index}', 12)}"
        fact_id = f"LF-U2F-EV-{stable_code(world.scenario_seed, f'ext-version-{index}', 16)}"
        end_day = days[index + 1] if index + 1 < len(days) else None
        fact = make_fact(
            fact_id=fact_id,
            value=value_token(world.scenario_seed, f"external-version-value-{index}"),
            location=FactLocation.EXTERNAL_EVIDENCE,
            source_artifact_id=source_id,
            valid_from=epoch + timedelta(days=effective_day),
            valid_until=epoch + timedelta(days=end_day) if end_day else None,
            revision_of=versions[-1][0].fact_id if versions else None,
        )
        record = realize_evidence_record(
            source_id=source_id,
            source_family="GUIDELINE",
            publication_time=epoch + timedelta(days=published_day),
            effective_time=epoch + timedelta(days=effective_day),
            effective_until=epoch + timedelta(days=end_day) if end_day else None,
            fact_id=fact_id,
            value=fact.value,
            key=keys[index],
        )
        versions.append((fact, record))
    usable = [(fact, record) for fact, record in versions
              if fact.available_at(world.decision_time)
              and record.publication_time <= world.decision_time
              and (record.effective_time is None or record.effective_time <= world.decision_time)
              and (record.effective_until is None
                   or world.decision_time < record.effective_until)]
    if not usable:
        # Every U2-F decision date is at least day 13; this guard protects edited plans.
        raise ValueError("versioned external scenario has no published effective evidence")
    # Query the current version end-to-end. Historical version intervals remain
    # represented and are exhaustively checked in the temporal audit.
    selected = [usable[-1]]
    index = versions.index(selected[0])
    components = (AnswerComponent(f"AC-{selected[0][0].fact_id}",
                                  (selected[0][0].fact_id,), 0),)
    query = _surface_text(world, (
        f"Which published synthetic source value is effective now for {keys[index]}?",
        f"At query time, return the value for {keys[index]} that is both published and effective.",
    ))
    facts = tuple(sorted((*kept_facts, *(fact for fact, _ in versions)), key=lambda item: item.fact_id))
    edges = tuple((versions[index][0].fact_id, versions[index + 1][0].fact_id)
                  for index in range(len(versions) - 1))
    graph = DependencyGraph(facts, components, edges)
    return replace(
        world,
        graph=graph,
        evidence_records=tuple(sorted((*kept_records, *(record for _, record in versions)),
                                      key=lambda item: (item.publication_time, item.source_id))),
        query=query,
        answer_type=(StructuredAnswerType.ORDERED_SEQUENCE if len(components) > 1
                     else world.answer_type),
        structural_metadata=tuple(sorted({**dict(world.structural_metadata),
            "external_revision_depth": len(versions) - 1,
            "external_versions": len(versions)}.items())),
    )


def _insufficient_world(world: LatentWorld) -> LatentWorld:
    if world.scenario_family != "INSUFFICIENT_EVIDENCE":
        return world
    facts = [fact for fact in world.graph.facts
             if fact.fact_id not in set(world.graph.required_fact_ids)]
    records = list(world.patient_records)
    evidence = list(world.evidence_records)
    components: list[AnswerComponent] = []
    epoch = subject_epoch(world.persona_seed)
    subtype = ("FUTURE_ONLY", "CONFLICT_UNRESOLVED", "MISSING_SET_MEMBER",
               "WRONG_SOURCE_FAMILY_ONLY", "STALE_PERSONAL_STATE")[_hash_int(
                   world.scenario_seed, "insufficient-subtype") % 5]
    key = key_token(world.scenario_seed, "primary")
    missing = make_fact(
        fact_id=f"LF-U2F-MISSING-{stable_code(world.scenario_seed, subtype, 14)}",
        value=value_token(world.scenario_seed, f"missing-{subtype}"),
        location=FactLocation.UNAVAILABLE,
        source_artifact_id=f"MISSING-U2F-{stable_code(world.scenario_seed, subtype, 10)}",
        valid_from=epoch + timedelta(days=1),
    )
    facts.append(missing)
    if subtype == "FUTURE_ONLY":
        missing = replace(missing, location=FactLocation.PATIENT_STATE,
                          validity=FactValidityInterval(epoch + timedelta(days=731)))
        facts[-1] = missing
        future = realize_patient_record(
            record_id=missing.source_artifact_id, subject_id=world.subject_id,
            timestamp=epoch + timedelta(days=731), fact_id=missing.fact_id,
            value=missing.value, key=key, record_type="EVENT",
        )
        records.append(future)
        selected_ids = (missing.fact_id,)
    elif subtype == "STALE_PERSONAL_STATE":
        stale = replace(missing, location=FactLocation.PATIENT_STATE,
                        fact_id=f"LF-U2F-STALE-{stable_code(world.scenario_seed, subtype, 14)}",
                        source_artifact_id=f"PR-U2F-STALE-{stable_code(world.scenario_seed, subtype, 10)}",
                        validity=FactValidityInterval(epoch + timedelta(days=1),
                                                     epoch + timedelta(days=12)))
        facts[-1] = stale
        records.append(realize_patient_record(
            record_id=stale.source_artifact_id, subject_id=world.subject_id,
            timestamp=epoch + timedelta(days=1), fact_id=stale.fact_id,
            value=stale.value, key=key_token(world.scenario_seed, "stale-obsolete"),
            record_type="EVENT",
        ))
        missing = stale
        extra = make_fact(
            fact_id=f"LF-U2F-STALE-MISSING-{stable_code(world.scenario_seed, 'stale-missing', 14)}",
            value=value_token(world.scenario_seed, "stale-missing"),
            location=FactLocation.UNAVAILABLE,
            source_artifact_id=f"MISSING-U2F-{stable_code(world.scenario_seed, 'stale-missing', 10)}",
            valid_from=epoch + timedelta(days=13),
        )
        facts.append(extra)
        selected_ids = (stale.fact_id, extra.fact_id)
    elif subtype == "MISSING_SET_MEMBER":
        partial = replace(missing, location=FactLocation.PATIENT_STATE,
                          fact_id=f"LF-U2F-PARTIAL-{stable_code(world.scenario_seed, subtype, 14)}",
                          source_artifact_id=f"PR-U2F-PARTIAL-{stable_code(world.scenario_seed, subtype, 10)}",
                          validity=FactValidityInterval(epoch + timedelta(days=1)))
        facts[-1] = partial
        records.append(realize_patient_record(
            record_id=partial.source_artifact_id, subject_id=world.subject_id,
            timestamp=epoch + timedelta(days=1), fact_id=partial.fact_id,
            value=partial.value, key=key,
            record_type="PROFILE",
        ))
        missing = make_fact(
            fact_id=f"LF-U2F-MISSING-SECOND-{stable_code(world.scenario_seed, subtype, 14)}",
            value=value_token(world.scenario_seed, "missing-second"),
            location=FactLocation.UNAVAILABLE,
            source_artifact_id=f"MISSING-U2F-{stable_code(world.scenario_seed, 'missing-second', 10)}",
            valid_from=epoch + timedelta(days=1),
        )
        facts.append(missing)
        selected_ids = (partial.fact_id, missing.fact_id)
    elif subtype == "CONFLICT_UNRESOLVED":
        conflicts = []
        for index, family in enumerate(("PUBLIC_HEALTH", "GUIDELINE")):
            source_id = f"ER-U2F-CONFLICT-{stable_code(world.scenario_seed, index, 10)}"
            conflict = make_fact(
                fact_id=f"LF-U2F-CONFLICT-{stable_code(world.scenario_seed, index, 14)}",
                value=value_token(world.scenario_seed, f"conflict-{index}"),
                location=FactLocation.EXTERNAL_EVIDENCE,
                source_artifact_id=source_id,
                valid_from=epoch + timedelta(days=1),
            )
            facts.append(conflict)
            conflicts.append(conflict)
            evidence.append(realize_evidence_record(
                source_id=source_id, source_family=family,
                publication_time=epoch + timedelta(days=2),
                effective_time=epoch + timedelta(days=2), fact_id=conflict.fact_id,
                value=conflict.value, key=key,
            ))
        # The conflicting values are evidence to inspect, not a jointly valid
        # answer. The requested adjudicated value remains explicitly unavailable.
        selected_ids = (missing.fact_id,)
    elif subtype == "WRONG_SOURCE_FAMILY_ONLY":
        wrong = make_fact(
            fact_id=f"LF-U2F-WRONG-FAMILY-{stable_code(world.scenario_seed, subtype, 14)}",
            value=value_token(world.scenario_seed, "wrong-source-family"),
            location=FactLocation.EXTERNAL_EVIDENCE,
            source_artifact_id=f"ER-U2F-WRONG-FAMILY-{stable_code(world.scenario_seed, subtype, 10)}",
            valid_from=epoch + timedelta(days=1),
        )
        facts.append(wrong)
        evidence.append(realize_evidence_record(
            source_id=wrong.source_artifact_id, source_family="PUBLIC_HEALTH",
            publication_time=epoch + timedelta(days=2),
            effective_time=epoch + timedelta(days=2), fact_id=wrong.fact_id,
            value=wrong.value, key=key,
        ))
        # The wrong-family value is a distractor; the requested value is absent
        # from the approved family and therefore remains unavailable.
        selected_ids = (missing.fact_id,)
    else:
        selected_ids = (missing.fact_id,)
    components.append(AnswerComponent(f"AC-U2F-ABSTAIN-{world.scenario_seed}", selected_ids, 0))
    graph = DependencyGraph(tuple(facts), tuple(components))
    query = f"{world.query}"
    if subtype == "FUTURE_ONLY":
        query = _surface_text(world, (
            f"Can a value valid now be established for {key}?",
            f"Is any currently valid value available for {key}?",
        ))
    elif subtype == "CONFLICT_UNRESOLVED":
        query = _surface_text(world, (
            f"The synthetic sources disagree for {key}; is a supported value available?",
            f"Given the unresolved source disagreement, can {key} be answered reliably?",
        ))
    elif subtype == "MISSING_SET_MEMBER":
        query = _surface_text(world, (
            f"Return the complete set of entries for {key}.",
            f"Are all required entries available to complete the set for {key}?",
        ))
    elif subtype == "WRONG_SOURCE_FAMILY_ONLY":
        query = _surface_text(world, (
            f"Using only an approved synthetic source family, resolve {key}.",
            f"Can an approved evidence family provide a value for {key}?",
        ))
    elif subtype == "STALE_PERSONAL_STATE":
        query = _surface_text(world, (
            f"What is the current valid personal value for {key}?",
            f"Does {key} have a non-stale personal value at the decision time?",
        ))
    query += " A supported answer must be present in the evidence."
    if subtype == "WRONG_SOURCE_FAMILY_ONLY":
        query += " Approved family:GUIDELINE."
    return replace(
        world, graph=graph, patient_records=tuple(records), evidence_records=tuple(evidence),
        query=query,
        answer_type=StructuredAnswerType.ABSTAIN, safe_abstention=True,
        structural_metadata=tuple(sorted({**dict(world.structural_metadata),
            "insufficient_subtype": subtype}.items())),
    )


def _numeric_or_boolean_world(world: LatentWorld) -> LatentWorld:
    """Author true deterministic arithmetic/control tasks over current inputs."""
    if world.scenario_family != "CURRENT_ONLY":
        return world
    selector = _hash_int(world.scenario_seed, "numeric-task") % 100
    if selector >= 42:
        return world
    inputs = (3 + _hash_int(world.scenario_seed, "number-a") % 41,
              3 + _hash_int(world.scenario_seed, "number-b") % 41)
    if selector < 7:
        operation = ("sum", "difference", "count", "minimum", "maximum", "trend")[
            _hash_int(world.scenario_seed, "numeric-operation") % 6]
        if operation == "sum":
            result = sum(inputs)
        elif operation == "difference":
            result = abs(inputs[0] - inputs[1])
        elif operation == "count":
            result = len(inputs)
        elif operation == "minimum":
            result = min(inputs)
        elif operation == "maximum":
            result = max(inputs)
        else:
            result = "UP" if inputs[1] > inputs[0] else "DOWN" if inputs[1] < inputs[0] else "STABLE"
        answer_type = (StructuredAnswerType.EXACT_TOKEN if operation == "trend"
                       else StructuredAnswerType.NUMERIC)
    elif selector < 14:
        operation = "greater_than"
        result = "TRUE" if inputs[0] > inputs[1] else "FALSE"
        answer_type = StructuredAnswerType.BOOLEAN
    else:
        return world
    artifact_id = f"TOOL-U2F-{stable_code(world.scenario_seed, operation, 12)}"
    fact = make_fact(
        fact_id=f"LF-U2F-CALC-{stable_code(world.scenario_seed, operation, 16)}",
        value=str(result), location=FactLocation.TOOL_OUTPUT,
        source_artifact_id=artifact_id, valid_from=world.decision_time,
    )
    graph = DependencyGraph(tuple(sorted((*world.graph.facts, fact), key=lambda item: item.fact_id)),
                            (AnswerComponent(f"AC-{fact.fact_id}", (fact.fact_id,), 0),))
    key = key_token(world.scenario_seed, "primary")
    query = (f"Compute the synthetic {operation} for {key}: "
             f"part_a={inputs[0]}; part_b={inputs[1]}; {operation}.")
    query = _surface_text(world, (
        query,
        (f"For {key}, calculate {operation}; part_a={inputs[0]}; "
         f"part_b={inputs[1]}; return the synthetic result."),
    ))
    if operation == "greater_than":
        query = _surface_text(world, (
            (f"Is part_a greater than part_b for {key}: part_a={inputs[0]}; "
             f"part_b={inputs[1]}; greater_than."),
            (f"For {key}, compare the inputs: part_a={inputs[0]}; "
             f"part_b={inputs[1]}; determine greater_than."),
        ))
    return replace(
        world, graph=graph, query=query, answer_type=answer_type,
        tool_ids=("query_part_a", "query_part_b"),
        structural_metadata=tuple(sorted({**dict(world.structural_metadata),
            "numeric_operation": operation,
            "numeric_inputs": inputs,
            "numeric_expected_value": str(result)}.items())),
    )


def _boolean_evidence_world(world: LatentWorld) -> LatentWorld:
    if world.scenario_family != "EXTERNAL_LOOKUP" or _hash_int(world.scenario_seed, "boolean-evidence") % 100 >= 32:
        return world
    required_ids = set(world.graph.required_fact_ids)
    selected = next((fact for fact in world.graph.facts if fact.fact_id in required_ids), None)
    if selected is None:
        return world
    value = "TRUE" if _hash_int(world.scenario_seed, "boolean-value") % 2 else "FALSE"
    new_fact = replace(selected, value=value)
    facts = tuple(new_fact if fact.fact_id == selected.fact_id else fact for fact in world.graph.facts)
    rows = []
    key = next((term for row in world.evidence_records
                if selected.fact_id in row.latent_fact_ids for term in row.retrieval_terms),
               key_token(world.scenario_seed, "primary"))
    for row in world.evidence_records:
        if selected.fact_id in row.latent_fact_ids:
            rows.append(replace(row, natural_language_content=(
                f"Synthetic {row.source_family.lower()} record verifies {key} as {value}.")))
        else:
            rows.append(row)
    return replace(
        world,
        graph=DependencyGraph(facts, world.graph.answer_components,
                              world.graph.fact_dependency_edges),
        evidence_records=tuple(rows),
        query=_surface_text(world, (
            f"Is the synthetic condition linked to {key} satisfied?",
            f"Does the synthetic condition for {key} hold?",
        )),
        answer_type=StructuredAnswerType.BOOLEAN,
        structural_metadata=tuple(sorted({**dict(world.structural_metadata),
            "boolean_target": value}.items())),
    )


def _patient_record_type_hardening(world: LatentWorld) -> LatentWorld:
    if world.scenario_family == "MEMORY_REVISION":
        return world
    required_ids = set(world.graph.required_fact_ids)
    facts = world.graph.fact_index
    selected_records = [row for row in world.patient_records
                        if required_ids.intersection(row.latent_fact_ids)
                        and any(facts[fact_id].location == FactLocation.PATIENT_STATE
                                for fact_id in row.latent_fact_ids if fact_id in facts)]
    rewritten: list[OwnedPatientRecord] = []
    type_offset = _hash_int(world.scenario_seed, "patient-record-type") % len(PATIENT_RECORD_TYPES)
    for index, row in enumerate(selected_records):
        record_type = PATIENT_RECORD_TYPES[(type_offset + index) % len(PATIENT_RECORD_TYPES)]
        fact_id = next(fact_id for fact_id in row.latent_fact_ids
                       if fact_id in facts and facts[fact_id].location == FactLocation.PATIENT_STATE)
        fact = facts[fact_id]
        key = row.retrieval_terms[0]
        conversation_kind = CONVERSATION_KINDS[_hash_int(
            world.scenario_seed, fact.fact_id, "conversation-kind"
        ) % len(CONVERSATION_KINDS)] if record_type == "CONVERSATION" else "USER_CONFIRMED_STATE"
        rewritten.append(realize_patient_record(
            record_id=row.record_id, subject_id=row.subject_id, timestamp=row.timestamp,
            fact_id=fact.fact_id, value=fact.value, key=key, record_type=record_type,
            conversation_kind=conversation_kind,
        ))
    selected_ids = {row.record_id for row in selected_records}
    remaining = tuple(row for row in world.patient_records if row.record_id not in selected_ids)
    return replace(world, patient_records=tuple(sorted((*remaining, *rewritten),
                                                        key=lambda row: (row.timestamp, row.record_id))))


def _required_key(world: LatentWorld) -> str:
    required_ids = set(world.graph.required_fact_ids)
    for record in (*world.patient_records, *world.evidence_records):
        if required_ids.intersection(record.latent_fact_ids) and record.retrieval_terms:
            return record.retrieval_terms[0]
    return key_token(world.scenario_seed, "primary")


def _expand_cross_capability_graph(
    world: LatentWorld, *, allow_extra_memory: bool = True,
) -> LatentWorld:
    if world.scenario_family in {"MEMORY_REVISION", "EXTERNAL_VERSIONED", "INSUFFICIENT_EVIDENCE"}:
        return world
    if dict(world.structural_metadata).get("numeric_operation"):
        return world
    if world.answer_type == StructuredAnswerType.BOOLEAN:
        return world
    facts = list(world.graph.facts)
    components = list(world.graph.answer_components)
    patients = list(world.patient_records)
    evidence = list(world.evidence_records)
    required_ids = set(world.graph.required_fact_ids)
    index = {fact.fact_id: fact for fact in facts}
    memory_required = any(index[fact_id].location == FactLocation.PATIENT_STATE
                          for fact_id in required_ids)
    retrieval_required = any(index[fact_id].location == FactLocation.EXTERNAL_EVIDENCE
                             for fact_id in required_ids)
    seed = world.scenario_seed
    key = _required_key(world)
    epoch = subject_epoch(world.persona_seed)
    additions = 0
    if memory_required and allow_extra_memory and _hash_int(seed, "extra-memory-branch") % 100 < 36:
        record_id = f"PR-U2F-BRANCH-{stable_code(seed, 'extra-memory', 12)}"
        fact = make_fact(
            fact_id=f"LF-U2F-BRANCH-{stable_code(seed, 'extra-memory', 16)}",
            value=value_token(seed, "extra-memory"), location=FactLocation.PATIENT_STATE,
            source_artifact_id=record_id, valid_from=epoch + timedelta(days=3),
        )
        record_type = PATIENT_RECORD_TYPES[_hash_int(seed, "extra-memory-type")
                                           % len(PATIENT_RECORD_TYPES)]
        patients.append(realize_patient_record(
            record_id=record_id, subject_id=world.subject_id,
            timestamp=fact.validity.valid_from, fact_id=fact.fact_id,
            value=fact.value, key=key, record_type=record_type,
        ))
        facts.append(fact)
        components.append(AnswerComponent(f"AC-{fact.fact_id}", (fact.fact_id,), len(components)))
        required_ids.add(fact.fact_id)
        additions += 1
    if retrieval_required and _hash_int(seed, "extra-external-branch") % 100 < 42:
        existing_families = {row.source_family for row in evidence}
        family = next((name for name in EXTERNAL_FAMILIES if name not in existing_families),
                      EXTERNAL_FAMILIES[_hash_int(seed, "extra-external-family") % 3])
        source_id = f"ER-U2F-BRANCH-{stable_code(seed, 'extra-external', 12)}"
        fact = make_fact(
            fact_id=f"LF-U2F-BRANCH-{stable_code(seed, 'extra-external', 16)}",
            value=value_token(seed, "extra-external"), location=FactLocation.EXTERNAL_EVIDENCE,
            source_artifact_id=source_id, valid_from=epoch + timedelta(days=3),
        )
        evidence.append(realize_evidence_record(
            source_id=source_id, source_family=family,
            publication_time=epoch + timedelta(days=2), effective_time=epoch + timedelta(days=3),
            fact_id=fact.fact_id, value=fact.value, key=key,
        ))
        facts.append(fact)
        components.append(AnswerComponent(f"AC-{fact.fact_id}", (fact.fact_id,), len(components)))
        required_ids.add(fact.fact_id)
        additions += 1
    answer_type = (StructuredAnswerType.EXACT_SET if additions
                   and world.answer_type != StructuredAnswerType.ABSTAIN
                   else world.answer_type)
    query = (f"{world.query} Return all supported values relevant to the request."
             if additions and not dict(world.structural_metadata).get(
                 "same_surface_stress_group"
             ) else world.query)

    # Explicit evaluator-only structure: independent, serial, or branching.
    fact_by_id = {fact.fact_id: fact for fact in facts}
    required_order = sorted(required_ids, key=lambda fact_id: (
        {FactLocation.CURRENT_CONTEXT: 0, FactLocation.PATIENT_STATE: 1,
         FactLocation.EXTERNAL_EVIDENCE: 2, FactLocation.TOOL_OUTPUT: 3,
         FactLocation.UNAVAILABLE: 4}[fact_by_id[fact_id].location], fact_id
    ))
    mode = _hash_int(seed, "dependency-shape") % 3
    edges: list[tuple[str, str]] = list(world.graph.fact_dependency_edges)
    if len(required_order) > 1 and mode == 1:
        edges.extend(pairwise(required_order))
    elif len(required_order) > 1 and mode == 2:
        edges.extend((required_order[0], fact_id) for fact_id in required_order[1:])
    graph = DependencyGraph(tuple(facts), tuple(components), tuple(edges))
    adjacency: dict[str, list[str]] = defaultdict(list)
    for parent, child in graph.fact_dependency_edges:
        adjacency[parent].append(child)

    def depth(node: str, active: frozenset[str] = frozenset()) -> int:
        if node in active:
            return 0
        return 1 + max((depth(child, active | {node}) for child in adjacency[node]), default=0)

    graph_depth = max((depth(node) for node in required_order), default=1)
    graph_width = max((len(adjacency[node]) for node in required_order), default=1)
    child_nodes = {child for _, child in graph.fact_dependency_edges}
    roots = [node for node in required_order if node not in child_nodes]
    structural = {
        **dict(world.structural_metadata),
        "dependency_depth": int(graph_depth),
        "dependency_width": int(max(1, graph_width)),
        "independent_dependency_groups": int(max(1, len(roots))),
        "serial_dependency_depth": int(graph_depth),
        "serial_dependency_count": len(graph.fact_dependency_edges),
        "evidence_branch_count": int(sum(fact_by_id[item].location == FactLocation.EXTERNAL_EVIDENCE
                                          for item in required_order)),
    }
    return replace(
        world, graph=graph,
        patient_records=tuple(sorted(patients, key=lambda row: (row.timestamp, row.record_id))),
        evidence_records=tuple(sorted(evidence, key=lambda row: (row.publication_time, row.source_id))),
        answer_type=answer_type, query=query,
        structural_metadata=tuple(sorted(structural.items())),
    )


def _add_evidence_world_records(world: LatentWorld) -> LatentWorld:
    target = (6, 16, 40)[_hash_int(world.scenario_seed, "evidence-world-size") % 3]
    rows = list(world.evidence_records)
    facts = list(world.graph.facts)
    seen_families = {row.source_family for row in rows}
    next_index = 0
    epoch = subject_epoch(world.persona_seed)
    while len(rows) < target or seen_families != set(EXTERNAL_FAMILIES):
        family = EXTERNAL_FAMILIES[next_index % len(EXTERNAL_FAMILIES)]
        namespace = f"evidence-pool-{next_index}"
        source_id = f"ER-U2F-POOL-{stable_code(world.scenario_seed, namespace, 12)}"
        fact_id = f"LF-U2F-POOL-{stable_code(world.scenario_seed, namespace, 16)}"
        publication_day = 1 + _hash_int(world.scenario_seed, namespace, "publication") % 730
        effective_day = max(1, publication_day + int(_hash_int(namespace, "effectivity") % 61) - 30)
        fact = make_fact(
            fact_id=fact_id, value=value_token(world.scenario_seed, namespace),
            location=FactLocation.EXTERNAL_EVIDENCE, source_artifact_id=source_id,
            valid_from=epoch + timedelta(days=effective_day),
        )
        record = realize_evidence_record(
            source_id=source_id, source_family=family,
            publication_time=epoch + timedelta(days=publication_day),
            effective_time=epoch + timedelta(days=effective_day),
            fact_id=fact_id, value=fact.value,
            key=key_token(world.scenario_seed, namespace),
        )
        facts.append(fact)
        rows.append(record)
        seen_families.add(family)
        next_index += 1
    if len(rows) > 64:
        raise ValueError("synthetic external evidence world exceeded the frozen 64-row limit")
    regime = "SMALL" if len(rows) <= 8 else "MEDIUM" if len(rows) <= 24 else "LARGE"
    return replace(
        world,
        graph=DependencyGraph(tuple(facts), world.graph.answer_components,
                              world.graph.fact_dependency_edges),
        evidence_records=tuple(sorted(rows, key=lambda row: (row.publication_time, row.source_id))),
        evidence_world_regime=regime,
    )


def _anchor_boundary_pair(members: tuple[LatentWorld, ...], target_day: int) -> tuple[LatentWorld, ...]:
    if len(members) != 2:
        return members
    first, second = sorted(members, key=lambda item: item.world_id)
    epoch = subject_epoch(first.persona_seed)
    target_time = epoch + timedelta(days=target_day)
    required_id = first.graph.required_fact_ids[0]
    target_fact = first.graph.fact_index[required_id]
    new_fact = replace(target_fact, validity=FactValidityInterval(target_time))
    facts = tuple(new_fact if fact.fact_id == required_id else fact for fact in first.graph.facts)
    def update(world: LatentWorld, instant) -> LatentWorld:
        patient_rows = tuple(replace(
            row, timestamp=target_time, retrieval_terms=(_required_key(world),)
        )
                             if required_id in row.latent_fact_ids else row
                             for row in world.patient_records)
        return replace(
            world,
            graph=DependencyGraph(facts, world.graph.answer_components,
                                  world.graph.fact_dependency_edges),
            patient_records=patient_rows,
            decision_time=instant,
            query=_surface_text(world, (
                f"At decision time, determine the supported value for {_required_key(world)}.",
                f"At this exact time, which value can be supported for {_required_key(world)}?",
            )),
        )
    offset_pair = _hash_int(first.scenario_seed, "boundary-pair-offset") % 2
    if offset_pair == 0:
        instants = (target_time - timedelta(seconds=1), target_time)
    else:
        instants = (target_time, target_time + timedelta(seconds=1))
    return update(first, instants[0]), update(second, instants[1])


def _graph_digest(world: LatentWorld) -> str:
    from ..contracts import stable_hash

    return "DG-" + stable_hash(world.graph.to_dict())[:24]


def build_u2f_worlds(
    *, plans: tuple[dict[str, Any], ...], profile: dict[str, Any], run_seed: int,
) -> tuple[tuple[LatentWorld, ...], dict[str, tuple[tuple[LatentFact, OwnedPatientRecord], ...]]]:
    """Generate shared timelines, longitudinal snapshots, and layered evidence worlds."""
    if run_seed != int(profile["global_seed"]):
        raise ValueError("run_seed must match the frozen global seed in the U2-F profile")
    initial = tuple(world for plan in plans
                    for world in generate_u2f_plan(plan, profile))
    pair_groups: dict[str, list[LatentWorld]] = defaultdict(list)
    for world in initial:
        pair_groups[world.counterfactual_family_id].append(world)
    plan_by_role = {str(plan["split_role"]): plan for plan in plans}
    pair_regime: dict[str, str] = {}
    pair_target_day: dict[str, int] = {}
    for role in plan_by_role:
        by_subject: dict[str, list[str]] = defaultdict(list)
        for pair_id, members in pair_groups.items():
            if members[0].split_role == role:
                by_subject[members[0].subject_id].append(pair_id)
        regimes, target_days = _allocate_subject_timelines(
            by_subject, profile["history_regime_target_weights"]
        )
        pair_regime.update(regimes)
        pair_target_day.update(target_days)

    subject_worlds: dict[str, list[LatentWorld]] = defaultdict(list)
    for world in initial:
        subject_worlds[world.subject_id].append(world)
    timelines: dict[str, tuple[tuple[LatentFact, OwnedPatientRecord], ...]] = {}
    for subject_id, members in subject_worlds.items():
        timelines[subject_id] = _timeline_rows(
            members[0].split_role, subject_id, members[0].persona_seed
        )
    episode_counts = Counter(world.subject_id for world in initial)

    result: list[LatentWorld] = []
    for pair_id, pair_members_list in sorted(pair_groups.items()):
        members = tuple(sorted(pair_members_list, key=lambda item: item.world_id))
        role = members[0].split_role
        same_surface = int(dict(members[0].structural_metadata).get(
            "same_surface_stress_group", 0
        )) == 1
        shared_query = (f"What synthetic value is supported for {_required_key(members[0])}?"
                        if same_surface else None)
        regime = pair_regime[pair_id]
        target_day = pair_target_day[pair_id]
        if members[0].scenario_family == "TEMPORAL_BOUNDARY":
            members = _anchor_boundary_pair(members, target_day)
        else:
            target_time = subject_epoch(members[0].persona_seed) + timedelta(days=target_day)
            members = tuple(replace(item, decision_time=target_time) for item in members)

        hardened: list[LatentWorld] = []
        for original in members:
            world = _revision_world(original)
            world = _versioned_external_world(world)
            world = _insufficient_world(world)
            world = _numeric_or_boolean_world(world)
            world = _boolean_evidence_world(world)
            if same_surface:
                world = replace(world, structural_metadata=tuple(sorted({
                    **dict(world.structural_metadata), "same_surface_stress_group": 1,
                }.items())))
                world = replace(world, query=shared_query)
            world = _patient_record_type_hardening(world)
            if world.scenario_family == "INSUFFICIENT_EVIDENCE" and regime == "SHORT":
                world = replace(world, patient_records=tuple(
                    row for row in world.patient_records
                    if row.record_id.startswith(("PR-U2F-", "MISSING-U2F-"))
                ))
            if world.scenario_family == "DISTRACTOR_HEAVY" and regime in {"SHORT", "MEDIUM"}:
                required_for_world = set(world.graph.required_fact_ids)
                world = replace(world, patient_records=tuple(
                    row for row in world.patient_records
                    if required_for_world.intersection(row.latent_fact_ids)
                ))
            allow_memory_branch = (
                regime != "SHORT"
                and not dict(world.structural_metadata).get("same_surface_stress_group")
            )
            world = _expand_cross_capability_graph(
                world, allow_extra_memory=allow_memory_branch
            )

            patient_rows = tuple(world.patient_records)
            visible_timeline = tuple((fact, record) for fact, record in timelines[world.subject_id]
                                     if record.timestamp <= world.decision_time)
            patient_rows = tuple(sorted((*patient_rows,
                *(record for _, record in visible_timeline)),
                key=lambda row: (row.timestamp, row.record_id)))
            facts = tuple(sorted((*world.graph.facts,
                *(fact for fact, _ in visible_timeline)), key=lambda item: item.fact_id))
            graph = DependencyGraph(facts, world.graph.answer_components,
                                    world.graph.fact_dependency_edges)
            world = replace(
                world,
                graph=graph,
                patient_records=patient_rows,
                timeline_generation_id=f"TL-{stable_code(world.persona_seed, f'{role}-{world.subject_id}', 16)}",
                timeline_record_count=len(timelines[world.subject_id]),
                timeline_span_days=(timelines[world.subject_id][-1][1].timestamp
                                    - timelines[world.subject_id][0][1].timestamp).days,
                subject_episode_count=episode_counts[world.subject_id],
            )
            world = _add_evidence_world_records(world)
            world = replace(world, query=_balance_query_surface(world))
            snapshot = tuple(row for row in world.patient_records
                             if row.timestamp <= world.decision_time)
            span_days = ((world.decision_time - snapshot[0].timestamp).days if snapshot else 0)
            history_regime = _regime_for_history(len(snapshot), span_days)
            if history_regime != regime:
                raise ValueError(
                    f"history regime drift for {world.world_id}: allocated {regime}, "
                    f"materialized {history_regime} ({len(snapshot)} records/{span_days} days)"
                )
            structural = dict(world.structural_metadata)
            structural.setdefault("dependency_depth", 1)
            structural.setdefault("dependency_width", 1)
            structural.setdefault("independent_dependency_groups", len(world.graph.required_fact_ids))
            structural.setdefault("serial_dependency_depth", 1)
            structural.setdefault("serial_dependency_count", len(world.graph.fact_dependency_edges))
            structural.setdefault("evidence_branch_count", sum(
                world.graph.fact_index[item].location == FactLocation.EXTERNAL_EVIDENCE
                for item in world.graph.required_fact_ids
            ))
            world = replace(
                world,
                history_regime=history_regime,
                dependency_graph_id=_graph_digest(world),
                structural_metadata=tuple(sorted(structural.items())),
                distractor_count=sum(
                    not set(world.graph.required_fact_ids).intersection(row.latent_fact_ids)
                    for row in snapshot
                ),
            )
            hardened.append(world)
        result.extend(hardened)
    return tuple(result), timelines
