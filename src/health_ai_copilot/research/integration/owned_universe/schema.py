"""Machine-readable contracts for the project-owned U2-E universe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from ..contracts import _aware, _nonempty

CONTENT_ORIGIN = "PROJECT_OWNED_SYNTHETIC"
WORLD_NOTICE = "SYNTHETIC_RESEARCH_WORLD; NOT_CLINICAL_GUIDANCE"


class FactLocation(StrEnum):
    CURRENT_CONTEXT = "CURRENT_CONTEXT"
    PATIENT_STATE = "PATIENT_STATE"
    EXTERNAL_EVIDENCE = "EXTERNAL_EVIDENCE"
    TOOL_OUTPUT = "TOOL_OUTPUT"
    UNAVAILABLE = "UNAVAILABLE"


class StructuredAnswerType(StrEnum):
    EXACT_TOKEN = "EXACT_TOKEN"
    EXACT_SET = "EXACT_SET"
    ORDERED_SEQUENCE = "ORDERED_SEQUENCE"
    NUMERIC = "NUMERIC"
    BOOLEAN = "BOOLEAN"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class FactValidityInterval:
    valid_from: datetime
    valid_until: datetime | None = None

    def __post_init__(self) -> None:
        _aware(self.valid_from, "valid_from")
        if self.valid_until is not None:
            _aware(self.valid_until, "valid_until")
            if self.valid_until <= self.valid_from:
                raise ValueError("valid_until must be later than valid_from")

    def contains(self, instant: datetime) -> bool:
        _aware(instant, "validity query time")
        return self.valid_from <= instant and (self.valid_until is None or instant < self.valid_until)

    def to_dict(self) -> dict[str, str | None]:
        return {"valid_from": self.valid_from.isoformat(),
                "valid_until": self.valid_until.isoformat() if self.valid_until else None}


@dataclass(frozen=True)
class LatentFact:
    fact_id: str
    value: str
    location: FactLocation | str
    source_artifact_id: str
    validity: FactValidityInterval
    revision_of: str | None = None
    content_origin: str = CONTENT_ORIGIN

    def __post_init__(self) -> None:
        for name in ("fact_id", "value", "source_artifact_id", "content_origin"):
            _nonempty(getattr(self, name), name)
        object.__setattr__(self, "location", FactLocation(self.location))
        if self.content_origin != CONTENT_ORIGIN:
            raise ValueError("U2-E facts must be project-owned synthetic")

    def available_at(self, instant: datetime) -> bool:
        return self.location != FactLocation.UNAVAILABLE and self.validity.contains(instant)

    def to_dict(self) -> dict[str, Any]:
        return {"fact_id": self.fact_id, "value": self.value,
                "location": self.location.value, "source_artifact_id": self.source_artifact_id,
                "validity": self.validity.to_dict(), "revision_of": self.revision_of,
                "content_origin": self.content_origin}


@dataclass(frozen=True)
class AnswerComponent:
    component_id: str
    required_fact_ids: tuple[str, ...]
    order: int
    requested_as_of: datetime | None = None

    def __post_init__(self) -> None:
        _nonempty(self.component_id, "component_id")
        if self.order < 0 or not self.required_fact_ids:
            raise ValueError("answer component requires dependencies and a non-negative order")
        if len(set(self.required_fact_ids)) != len(self.required_fact_ids):
            raise ValueError("answer component fact dependencies must be unique")
        if self.requested_as_of is not None:
            _aware(self.requested_as_of, "requested_as_of")

    def to_dict(self) -> dict[str, Any]:
        return {"component_id": self.component_id,
                "required_fact_ids": list(self.required_fact_ids), "order": self.order,
                "requested_as_of": (self.requested_as_of.isoformat()
                                    if self.requested_as_of else None)}


@dataclass(frozen=True)
class DependencyGraph:
    facts: tuple[LatentFact, ...]
    answer_components: tuple[AnswerComponent, ...]
    fact_dependency_edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        fact_ids = [item.fact_id for item in self.facts]
        component_ids = [item.component_id for item in self.answer_components]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("latent fact IDs must be unique")
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("answer component IDs must be unique")
        known = set(fact_ids)
        if any(not set(item.required_fact_ids).issubset(known) for item in self.answer_components):
            raise ValueError("dependency graph references an unknown latent fact")
        edges = tuple(sorted({tuple(edge) for edge in self.fact_dependency_edges}))
        if any(len(edge) != 2 or edge[0] not in known or edge[1] not in known or edge[0] == edge[1]
               for edge in edges):
            raise ValueError("fact dependency edges must connect two distinct known facts")
        adjacency: dict[str, list[str]] = {fact_id: [] for fact_id in known}
        for parent, child in edges:
            adjacency[parent].append(child)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(fact_id: str) -> None:
            if fact_id in visiting:
                raise ValueError("fact dependency graph must be acyclic")
            if fact_id in visited:
                return
            visiting.add(fact_id)
            for child in adjacency[fact_id]:
                visit(child)
            visiting.remove(fact_id)
            visited.add(fact_id)

        for fact_id in sorted(known):
            visit(fact_id)
        object.__setattr__(self, "facts", tuple(sorted(self.facts, key=lambda x: x.fact_id)))
        object.__setattr__(self, "answer_components",
                           tuple(sorted(self.answer_components, key=lambda x: (x.order, x.component_id))))
        object.__setattr__(self, "fact_dependency_edges", edges)

    @property
    def fact_index(self) -> dict[str, LatentFact]:
        return {item.fact_id: item for item in self.facts}

    @property
    def required_fact_ids(self) -> tuple[str, ...]:
        return tuple(sorted({fact_id for component in self.answer_components
                             for fact_id in component.required_fact_ids}))

    def to_dict(self) -> dict[str, Any]:
        return {"facts": [item.to_dict() for item in self.facts],
                "answer_components": [item.to_dict() for item in self.answer_components],
                "dependency_edges": [
                    {"fact_id": fact_id, "answer_component_id": component.component_id}
                    for component in self.answer_components
                    for fact_id in component.required_fact_ids
                ],
                "fact_dependency_edges": [
                    {"parent_fact_id": parent, "child_fact_id": child}
                    for parent, child in self.fact_dependency_edges
                ]}


@dataclass(frozen=True)
class OwnedPatientRecord:
    record_id: str
    subject_id: str
    timestamp: datetime
    record_type: str
    latent_fact_ids: tuple[str, ...]
    natural_language_content: str
    retrieval_terms: tuple[str, ...] = ()
    authority: str = "SYNTHETIC_RECORD"

    def __post_init__(self) -> None:
        for name in ("record_id", "subject_id", "record_type", "natural_language_content", "authority"):
            _nonempty(getattr(self, name), name)
        _aware(self.timestamp, "patient record timestamp")
        if not self.latent_fact_ids:
            raise ValueError("patient record must trace to at least one latent fact")
        if self.natural_language_content != self.natural_language_content.strip():
            raise ValueError("patient record content must be trimmed")

    def to_dict(self) -> dict[str, Any]:
        return {"record_id": self.record_id, "subject_id": self.subject_id,
                "timestamp": self.timestamp.isoformat(), "record_type": self.record_type,
                "provenance": WORLD_NOTICE, "latent_fact_ids": list(self.latent_fact_ids),
                "natural_language_content": self.natural_language_content,
                "retrieval_terms": list(self.retrieval_terms),
                "authority": self.authority,
                "content_origin": CONTENT_ORIGIN, "notice": WORLD_NOTICE}


@dataclass(frozen=True)
class OwnedEvidenceRecord:
    source_id: str
    source_family: str
    publication_time: datetime
    effective_time: datetime | None
    latent_fact_ids: tuple[str, ...]
    natural_language_content: str
    retrieval_terms: tuple[str, ...]
    effective_until: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("source_id", "source_family", "natural_language_content"):
            _nonempty(getattr(self, name), name)
        _aware(self.publication_time, "publication_time")
        if self.effective_time is not None:
            _aware(self.effective_time, "effective_time")
        if self.effective_until is not None:
            _aware(self.effective_until, "effective_until")
            if self.effective_time is None or self.effective_until <= self.effective_time:
                raise ValueError("effective_until must follow an effective_time")
        if not self.latent_fact_ids:
            raise ValueError("evidence record must trace to at least one latent fact")

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "source_family": self.source_family,
                "publication_time": self.publication_time.isoformat(),
                "effective_time": self.effective_time.isoformat() if self.effective_time else None,
                "effective_until": (self.effective_until.isoformat()
                                    if self.effective_until else None),
                "authority_metadata": {"authority": "project-owned synthetic namespace"},
                "latent_fact_ids": list(self.latent_fact_ids),
                "natural_language_content": self.natural_language_content,
                "retrieval_terms": list(self.retrieval_terms),
                "content_origin": CONTENT_ORIGIN}


@dataclass(frozen=True)
class ActionSupervisionMask:
    memory_read: bool = True
    external_retrieval: bool = True
    architecture: bool = False
    budget_class: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {"memory_read": self.memory_read,
                "external_retrieval": self.external_retrieval,
                "architecture": self.architecture,
                "budget_class": self.budget_class}


@dataclass(frozen=True)
class SupervisionStatus:
    memory_read: str = "QUALIFIED"
    external_retrieval: str = "QUALIFIED"
    architecture: str = "UNRESOLVED"
    budget_class: str = "UNRESOLVED"

    def __post_init__(self) -> None:
        if self.architecture != "UNRESOLVED" or self.budget_class != "UNRESOLVED":
            raise ValueError("U2-E does not supervise architecture or budget choice")

    def to_dict(self) -> dict[str, str]:
        return {"memory_read": self.memory_read,
                "external_retrieval": self.external_retrieval,
                "architecture": self.architecture,
                "budget_class": self.budget_class}


@dataclass(frozen=True)
class CapabilityRequirement:
    memory_required: bool
    external_retrieval_required: bool
    answerability: bool
    architecture_required: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if self.architecture_required != "UNKNOWN":
            raise ValueError("architecture requirement must remain unknown in U2-E")

    @property
    def derived_class(self) -> str:
        if not self.answerability:
            return "INSUFFICIENT"
        if self.memory_required and self.external_retrieval_required:
            return "MEMORY+RAG"
        if self.memory_required:
            return "MEMORY"
        if self.external_retrieval_required:
            return "RAG"
        return "NONE"

    def to_dict(self) -> dict[str, Any]:
        return {"memory_required": self.memory_required,
                "external_retrieval_required": self.external_retrieval_required,
                "answerability": self.answerability,
                "architecture_required": self.architecture_required,
                "derived_capability_class": self.derived_class}


@dataclass(frozen=True)
class LatentWorld:
    world_id: str
    subject_id: str
    scenario_family: str
    template_family: str
    surface_variant: str
    counterfactual_family_id: str
    persona_seed: int
    scenario_seed: int
    decision_time: datetime
    query: str
    graph: DependencyGraph
    patient_records: tuple[OwnedPatientRecord, ...]
    evidence_records: tuple[OwnedEvidenceRecord, ...]
    distractor_count: int
    split_role: str
    budget_class: str
    deadline_class: str
    answer_type: StructuredAnswerType | str
    safe_abstention: bool = False
    tool_ids: tuple[str, ...] = ()
    creation_time: str = "2026-09-30T00:00:00Z"
    timeline_generation_id: str = ""
    history_regime: str = "U2E_PILOT"
    evidence_world_regime: str = "U2E_PILOT"
    dependency_graph_id: str = ""
    subject_episode_count: int = 1
    timeline_record_count: int = 0
    timeline_span_days: int = 0
    structural_metadata: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        for name in ("world_id", "subject_id", "scenario_family", "template_family",
                     "surface_variant", "counterfactual_family_id", "split_role", "query"):
            _nonempty(getattr(self, name), name)
        _aware(self.decision_time, "decision_time")
        object.__setattr__(self, "answer_type", StructuredAnswerType(self.answer_type))
        if self.distractor_count < 0:
            raise ValueError("distractor_count must not be negative")
        if self.subject_episode_count <= 0 or self.timeline_record_count < 0 or self.timeline_span_days < 0:
            raise ValueError("longitudinal metadata counts must be non-negative and episode count positive")
        if len({key for key, _ in self.structural_metadata}) != len(self.structural_metadata):
            raise ValueError("structural metadata keys must be unique")
        fact_ids = set(self.graph.fact_index)
        if any(not set(record.latent_fact_ids).issubset(fact_ids) for record in self.patient_records):
            raise ValueError("patient record lineage references an unknown latent fact")
        if any(not set(record.latent_fact_ids).issubset(fact_ids) for record in self.evidence_records):
            raise ValueError("evidence record lineage references an unknown latent fact")

    def to_latent_dict(self) -> dict[str, Any]:
        return {"latent_world_id": self.world_id, "subject_id": self.subject_id,
                "scenario_family": self.scenario_family, "template_family": self.template_family,
                "surface_variant": self.surface_variant,
                "counterfactual_family_id": self.counterfactual_family_id,
                "persona_seed": self.persona_seed, "scenario_seed": self.scenario_seed,
                "decision_time": self.decision_time.isoformat(),
                "dependency_graph": self.graph.to_dict(),
                "distractor_count": self.distractor_count, "split_role": self.split_role,
                "budget_class": self.budget_class, "deadline_class": self.deadline_class,
                "answer_type": self.answer_type.value, "safe_abstention": self.safe_abstention,
                "tool_ids": list(self.tool_ids), "creation_time": self.creation_time,
                "timeline_generation_id": self.timeline_generation_id,
                "history_regime": self.history_regime,
                "evidence_world_regime": self.evidence_world_regime,
                "dependency_graph_id": self.dependency_graph_id,
                "subject_episode_count": self.subject_episode_count,
                "timeline_record_count": self.timeline_record_count,
                "timeline_span_days": self.timeline_span_days,
                "structural_evaluator_metadata": dict(self.structural_metadata),
                "content_origin": CONTENT_ORIGIN, "notice": WORLD_NOTICE}

    def to_realization_dict(self) -> dict[str, Any]:
        return {"latent_world_id": self.world_id, "query": self.query,
                "patient_records": [item.to_dict() for item in self.patient_records],
                "external_evidence": [item.to_dict() for item in self.evidence_records],
                "content_origin": CONTENT_ORIGIN, "notice": WORLD_NOTICE}

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_latent_dict(), **self.to_realization_dict()}


@dataclass(frozen=True)
class OwnedScenario:
    world: LatentWorld
    episode_id: str
    oracle: CapabilityRequirement
    supervision_mask: ActionSupervisionMask = ActionSupervisionMask()
    supervision_status: SupervisionStatus = SupervisionStatus()

    @property
    def split_role(self) -> str:
        return self.world.split_role

    @property
    def answer_values(self) -> tuple[str, ...]:
        index = self.world.graph.fact_index
        ordered = sorted(self.world.graph.answer_components, key=lambda item: item.order)
        return tuple(index[fact_id].value for component in ordered
                     for fact_id in component.required_fact_ids)

    def evaluator_truth_dict(self) -> dict[str, Any]:
        return {"episode_id": self.episode_id,
                "gold_answer": "INSUFFICIENT_EVIDENCE" if not self.oracle.answerability
                else "; ".join(self.answer_values),
                "answer_type": self.world.answer_type.value,
                "answer_values": list(self.answer_values),
                "answer_fact_ids": list(self.world.graph.required_fact_ids),
                "capability_requirement_oracle": self.oracle.to_dict(),
                "action_supervision_mask": self.supervision_mask.to_dict(),
                "supervision_status": self.supervision_status.to_dict(),
                "required_memory_record_ids": sorted({record.record_id
                    for record in self.world.patient_records
                    if set(record.latent_fact_ids) & set(self.world.graph.required_fact_ids)
                    and self.oracle.memory_required}),
                "required_external_evidence_ids": sorted({record.source_id
                    for record in self.world.evidence_records
                    if set(record.latent_fact_ids) & set(self.world.graph.required_fact_ids)
                    and self.oracle.external_retrieval_required}),
                "architecture_supervision": "UNRESOLVED",
                "structural_evaluator_metadata": dict(self.world.structural_metadata),
                "training_authorized": False,
                "content_origin": CONTENT_ORIGIN,
                "notice": WORLD_NOTICE}


def derive_capability_requirement(world: LatentWorld) -> CapabilityRequirement:
    """Derive requirements exclusively from answer dependencies and fact locations."""
    index = world.graph.fact_index
    needed = [index[fact_id] for fact_id in world.graph.required_fact_ids]
    requested_times: dict[str, list[datetime]] = {}
    for component in world.graph.answer_components:
        instant = component.requested_as_of or world.decision_time
        for fact_id in component.required_fact_ids:
            requested_times.setdefault(fact_id, []).append(instant)
    available = [all(fact.available_at(instant)
                     for instant in requested_times.get(fact.fact_id, [world.decision_time]))
                 for fact in needed]
    answerable = bool(needed) and all(available)
    memory = answerable and any(fact.location == FactLocation.PATIENT_STATE for fact in needed)
    retrieval = answerable and any(fact.location == FactLocation.EXTERNAL_EVIDENCE for fact in needed)
    return CapabilityRequirement(memory, retrieval, answerable)
