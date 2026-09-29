"""Immutable runtime, evaluation, and privileged-plane contracts for U1.

The three planes intentionally have separate types and serializers. An
IntegrationEpisode contains runtime inputs and references only; evaluator gold
and teacher-only information cannot be nested into its payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any


class PatientRecordType(StrEnum):
    PROFILE = "PROFILE"
    MEASUREMENT = "MEASUREMENT"
    EXAM = "EXAM"
    EVENT = "EVENT"
    CONVERSATION = "CONVERSATION"


class ExternalRetrievalLevel(StrEnum):
    OFF = "OFF"
    STANDARD = "STANDARD"


class ArchitectureMode(StrEnum):
    SINGLE = "SINGLE"
    TEAM = "TEAM"


class FailureCategory(StrEnum):
    MISSING_MEMORY_READ = "MISSING_MEMORY_READ"
    UNNECESSARY_MEMORY_READ = "UNNECESSARY_MEMORY_READ"
    MISSING_EXTERNAL_RETRIEVAL = "MISSING_EXTERNAL_RETRIEVAL"
    UNNECESSARY_EXTERNAL_RETRIEVAL = "UNNECESSARY_EXTERNAL_RETRIEVAL"
    STALE_STATE = "STALE_STATE"
    TEMPORAL_LEAKAGE = "TEMPORAL_LEAKAGE"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _unique_strings(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    normalized = tuple(sorted(values))
    if any(not isinstance(item, str) or not item.strip() for item in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True)
class ObservableState:
    """Only metadata available before an action is chosen."""

    history_exists: bool
    history_length_bucket: str
    history_time_span: str | None
    available_personal_state_types: tuple[str, ...]
    available_external_source_families: tuple[str, ...]
    available_tool_ids: tuple[str, ...]
    available_worker_capabilities: tuple[str, ...]
    budget_class: str
    deadline_class: str
    task_intent_metadata: tuple[tuple[str, str], ...] = ()

    PRE_DECISION_FIELDS = (
        "history_exists",
        "history_length_bucket",
        "history_time_span",
        "available_personal_state_types",
        "available_external_source_families",
        "available_tool_ids",
        "available_worker_capabilities",
        "budget_class",
        "deadline_class",
        "task_intent_metadata",
    )
    ALLOWED_INTENT_KEYS = frozenset({"intent", "temporal_reference", "question_mode"})

    def __post_init__(self) -> None:
        for name in ("history_length_bucket", "budget_class", "deadline_class"):
            _nonempty(getattr(self, name), name)
        for name in (
            "available_personal_state_types",
            "available_external_source_families",
            "available_tool_ids",
            "available_worker_capabilities",
        ):
            object.__setattr__(self, name, _unique_strings(tuple(getattr(self, name)), name))
        pairs = tuple(sorted(tuple(item) for item in self.task_intent_metadata))
        for key, value in pairs:
            if key not in self.ALLOWED_INTENT_KEYS or not isinstance(value, str):
                raise ValueError("task_intent_metadata contains a non-approved intent field")
        object.__setattr__(self, "task_intent_metadata", pairs)

    def observability_flags(self) -> dict[str, str]:
        """Return the required per-field pre-decision YES/NO contract."""
        return {name: "YES" for name in self.PRE_DECISION_FIELDS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_exists": self.history_exists,
            "history_length_bucket": self.history_length_bucket,
            "history_time_span": self.history_time_span,
            "available_personal_state_types": list(self.available_personal_state_types),
            "available_external_source_families": list(self.available_external_source_families),
            "available_tool_ids": list(self.available_tool_ids),
            "available_worker_capabilities": list(self.available_worker_capabilities),
            "budget_class": self.budget_class,
            "deadline_class": self.deadline_class,
            "task_intent_metadata": dict(self.task_intent_metadata),
            "pre_decision_observable": self.observability_flags(),
        }


@dataclass(frozen=True)
class PatientStateRef:
    subject_id: str
    snapshot_id: str
    record_types: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.subject_id, "subject_id")
        _nonempty(self.snapshot_id, "snapshot_id")
        object.__setattr__(self, "record_types", _unique_strings(tuple(self.record_types), "record_types"))
        unknown = set(self.record_types) - {item.value for item in PatientRecordType}
        if unknown:
            raise ValueError(f"unknown patient record types: {sorted(unknown)}")
        if not self.record_types:
            raise ValueError("patient state reference requires an explicit record-type scope")

    def to_dict(self) -> dict[str, Any]:
        return {"subject_id": self.subject_id, "snapshot_id": self.snapshot_id,
                "record_types": list(self.record_types)}


@dataclass(frozen=True)
class ExternalEvidenceWorldRef:
    world_id: str
    version: str
    source_families: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.world_id, "world_id")
        _nonempty(self.version, "version")
        object.__setattr__(self, "source_families", _unique_strings(tuple(self.source_families), "source_families"))

    def to_dict(self) -> dict[str, Any]:
        return {"world_id": self.world_id, "version": self.version,
                "source_families": list(self.source_families)}


@dataclass(frozen=True)
class WorkerManifest:
    """Fixed worker authority; workers cannot add resources to their union."""

    worker_id: str
    personal_state_scopes: tuple[str, ...] = ()
    external_source_families: tuple[str, ...] = ()
    tool_ids: tuple[str, ...] = ()
    capability_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.worker_id, "worker_id")
        for name in ("personal_state_scopes", "external_source_families", "tool_ids", "capability_domains"):
            object.__setattr__(self, name, _unique_strings(tuple(getattr(self, name)), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "personal_state_scopes": list(self.personal_state_scopes),
            "external_source_families": list(self.external_source_families),
            "tool_ids": list(self.tool_ids),
            "capability_domains": list(self.capability_domains),
        }


@dataclass(frozen=True)
class ToolSurfaceRef:
    surface_id: str
    version: str
    tool_ids: tuple[str, ...]
    workers: tuple[WorkerManifest, ...] = ()
    model_identity: str = "deterministic-scripted-v1"

    def __post_init__(self) -> None:
        _nonempty(self.surface_id, "surface_id")
        _nonempty(self.version, "version")
        _nonempty(self.model_identity, "model_identity")
        object.__setattr__(self, "tool_ids", _unique_strings(tuple(self.tool_ids), "tool_ids"))
        workers = tuple(sorted(self.workers, key=lambda item: item.worker_id))
        if len({item.worker_id for item in workers}) != len(workers):
            raise ValueError("worker IDs must be unique")
        if any(not set(item.tool_ids).issubset(self.tool_ids) for item in workers):
            raise ValueError("worker manifest grants a tool outside the shared tool surface")
        object.__setattr__(self, "workers", workers)

    def to_dict(self) -> dict[str, Any]:
        return {"surface_id": self.surface_id, "version": self.version,
                "tool_ids": list(self.tool_ids), "workers": [item.to_dict() for item in self.workers],
                "model_identity": self.model_identity}


@dataclass(frozen=True)
class EpisodeBudget:
    global_units: int
    memory_read_units: int
    external_retrieval_units: int
    team_worker_units: int
    tool_units: int
    budget_class: str
    deadline_class: str
    cost_model_version: str = "u1-abstract-cost-v1"

    def __post_init__(self) -> None:
        for name in ("global_units", "memory_read_units", "external_retrieval_units", "team_worker_units", "tool_units"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("budget_class", "deadline_class", "cost_model_version"):
            _nonempty(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {"global_units": self.global_units, "memory_read_units": self.memory_read_units,
                "external_retrieval_units": self.external_retrieval_units,
                "team_worker_units": self.team_worker_units, "tool_units": self.tool_units,
                "budget_class": self.budget_class, "deadline_class": self.deadline_class,
                "cost_model_version": self.cost_model_version}


@dataclass(frozen=True)
class EvaluatorRef:
    evaluator_id: str
    version: str

    def __post_init__(self) -> None:
        _nonempty(self.evaluator_id, "evaluator_id")
        _nonempty(self.version, "version")

    def to_dict(self) -> dict[str, str]:
        return {"evaluator_id": self.evaluator_id, "version": self.version}


@dataclass(frozen=True)
class IntegrationEpisode:
    episode_id: str
    environment_version: str
    source_provenance: str
    decision_time: datetime
    subject_id: str | None
    query: str
    observable_state: ObservableState
    patient_state_ref: PatientStateRef | None
    external_world_ref: ExternalEvidenceWorldRef | None
    tool_surface_ref: ToolSurfaceRef | None
    budget: EpisodeBudget
    evaluator_ref: EvaluatorRef

    def __post_init__(self) -> None:
        for name in ("episode_id", "environment_version", "source_provenance", "query"):
            _nonempty(getattr(self, name), name)
        _aware(self.decision_time, "decision_time")
        if self.patient_state_ref and self.subject_id != self.patient_state_ref.subject_id:
            raise ValueError("episode subject_id differs from patient state scope")
        if self.patient_state_ref and not set(self.observable_state.available_personal_state_types).issubset(
            self.patient_state_ref.record_types
        ):
            raise ValueError("observable patient record types exceed the scoped state reference")
        if self.external_world_ref and not set(self.observable_state.available_external_source_families).issubset(
            self.external_world_ref.source_families
        ):
            raise ValueError("observable source families exceed the external-world scope")
        if self.tool_surface_ref and not set(self.tool_surface_ref.tool_ids).issubset(
            self.observable_state.available_tool_ids
        ):
            raise ValueError("tool surface grants tools outside observable allowed tools")
        if self.observable_state.budget_class != self.budget.budget_class:
            raise ValueError("observable budget class differs from budget contract")
        if self.observable_state.deadline_class != self.budget.deadline_class:
            raise ValueError("observable deadline class differs from budget contract")

    def to_runtime_dict(self) -> dict[str, Any]:
        """Serialize runtime-visible fields only; no evaluator gold is reachable here."""
        return {
            "episode_id": self.episode_id,
            "environment_version": self.environment_version,
            "source_provenance": self.source_provenance,
            "decision_time": self.decision_time.isoformat(),
            "subject_id": self.subject_id,
            "query": self.query,
            "observable_state": self.observable_state.to_dict(),
            "patient_state_ref": self.patient_state_ref.to_dict() if self.patient_state_ref else None,
            "external_world_ref": self.external_world_ref.to_dict() if self.external_world_ref else None,
            "tool_surface_ref": self.tool_surface_ref.to_dict() if self.tool_surface_ref else None,
            "budget": self.budget.to_dict(),
            "evaluator_ref": self.evaluator_ref.to_dict(),
        }

    @property
    def episode_hash(self) -> str:
        return stable_hash(self.to_runtime_dict())


@dataclass(frozen=True)
class EvaluationPlane:
    """Evaluator-only labels. This object is passed only to the evaluator."""

    episode_id: str
    gold_answer: str
    required_facts: tuple[str, ...]
    required_evidence_ids: tuple[str, ...]
    task_success_predicate: str
    failure_labels: tuple[str, ...] = ()
    required_memory_facts: tuple[str, ...] = ()
    required_memory_record_ids: tuple[str, ...] = ()
    required_external_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("episode_id", "gold_answer", "task_success_predicate"):
            _nonempty(getattr(self, name), name)
        object.__setattr__(self, "required_facts", tuple(self.required_facts))
        object.__setattr__(self, "required_evidence_ids", _unique_strings(tuple(self.required_evidence_ids), "required_evidence_ids"))
        object.__setattr__(self, "failure_labels", _unique_strings(tuple(self.failure_labels), "failure_labels"))
        object.__setattr__(self, "required_memory_facts", tuple(self.required_memory_facts))
        object.__setattr__(self, "required_memory_record_ids", _unique_strings(tuple(self.required_memory_record_ids), "required_memory_record_ids"))
        object.__setattr__(self, "required_external_evidence_ids", _unique_strings(tuple(self.required_external_evidence_ids), "required_external_evidence_ids"))

    def to_evaluation_dict(self) -> dict[str, Any]:
        return {"episode_id": self.episode_id, "gold_answer": self.gold_answer,
                "required_facts": list(self.required_facts),
                "required_evidence_ids": list(self.required_evidence_ids),
                "task_success_predicate": self.task_success_predicate,
                "failure_labels": list(self.failure_labels),
                "required_memory_facts": list(self.required_memory_facts),
                "required_memory_record_ids": list(self.required_memory_record_ids),
                "required_external_evidence_ids": list(self.required_external_evidence_ids)}


@dataclass(frozen=True)
class PrivilegedTrainingPlane:
    """Future teacher/oracle information; never accepted by runtime execution."""

    episode_id: str
    counterfactual_sibling_outcomes: tuple[tuple[str, str], ...] = ()
    hidden_patient_state: tuple[str, ...] = ()
    oracle_capability_result: str | None = None
    failure_attribution: tuple[str, ...] = ()
    successful_trajectory: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.episode_id, "episode_id")
        object.__setattr__(self, "counterfactual_sibling_outcomes", tuple(sorted(self.counterfactual_sibling_outcomes)))
        object.__setattr__(self, "hidden_patient_state", tuple(self.hidden_patient_state))
        object.__setattr__(self, "failure_attribution", tuple(self.failure_attribution))
        object.__setattr__(self, "successful_trajectory", tuple(self.successful_trajectory))

    def to_privileged_dict(self) -> dict[str, Any]:
        return {"episode_id": self.episode_id,
                "counterfactual_sibling_outcomes": [list(item) for item in self.counterfactual_sibling_outcomes],
                "hidden_patient_state": list(self.hidden_patient_state),
                "oracle_capability_result": self.oracle_capability_result,
                "failure_attribution": list(self.failure_attribution),
                "successful_trajectory": list(self.successful_trajectory)}


@dataclass(frozen=True)
class ExecutionOutcome:
    task_success: bool
    safety_pass: bool
    grounding_pass: bool
    answer: str
    used_evidence_ids: tuple[str, ...] = ()
    quality_metrics: tuple[tuple[str, int | float | str | bool], ...] = ()
    provider_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    activated_capabilities: tuple[str, ...] = ()
    failure_category: FailureCategory | None = None

    def __post_init__(self) -> None:
        if self.provider_calls != 0:
            raise ValueError("U1 deterministic executor must not call a provider")
        for name in ("provider_calls", "tool_calls", "input_tokens", "output_tokens", "latency_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if len({name for name, _ in self.quality_metrics}) != len(self.quality_metrics):
            raise ValueError("quality metric names must be unique")
        object.__setattr__(self, "used_evidence_ids", tuple(self.used_evidence_ids))
        object.__setattr__(self, "activated_capabilities", tuple(sorted(set(self.activated_capabilities))))

    def to_dict(self) -> dict[str, Any]:
        return {"task_success": self.task_success, "safety_pass": self.safety_pass,
                "grounding_pass": self.grounding_pass, "answer": self.answer,
                "used_evidence_ids": list(self.used_evidence_ids),
                "quality_metrics": {name: value for name, value in self.quality_metrics},
                "provider_calls": self.provider_calls, "tool_calls": self.tool_calls,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "latency_ms": self.latency_ms,
                "activated_capabilities": list(self.activated_capabilities),
                "failure_category": self.failure_category.value if self.failure_category else None}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return sha256(stable_json(value).encode("utf-8")).hexdigest()
