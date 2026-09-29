"""Capability action schema, pre-side-effect masks, and Single/Team gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .contracts import (
    ArchitectureMode,
    EpisodeBudget,
    EvaluatorRef,
    ExternalRetrievalLevel,
    IntegrationEpisode,
    WorkerManifest,
)


class ActionKey(StrEnum):
    NONE = "NONE"
    MEMORY = "MEMORY"
    RAG = "RAG"
    TEAM = "TEAM"
    MEMORY_RAG = "MEMORY+RAG"
    MEMORY_TEAM = "MEMORY+TEAM"
    RAG_TEAM = "RAG+TEAM"
    ALL = "ALL"


@dataclass(frozen=True)
class CapabilityAction:
    memory_read: bool
    external_retrieval: ExternalRetrievalLevel | str
    architecture: ArchitectureMode | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "external_retrieval", ExternalRetrievalLevel(self.external_retrieval))
        object.__setattr__(self, "architecture", ArchitectureMode(self.architecture))

    @property
    def action_id(self) -> str:
        if (self.memory_read
                and self.external_retrieval == ExternalRetrievalLevel.STANDARD
                and self.architecture == ArchitectureMode.TEAM):
            return "ALL"
        parts: list[str] = []
        if self.memory_read:
            parts.append("MEMORY")
        if self.external_retrieval == ExternalRetrievalLevel.STANDARD:
            parts.append("RAG")
        if self.architecture == ArchitectureMode.TEAM:
            parts.append("TEAM")
        return "+".join(parts) if parts else "NONE"

    def to_dict(self) -> dict[str, str | bool]:
        return {"memory_read": self.memory_read,
                "external_retrieval": self.external_retrieval.value,
                "architecture": self.architecture.value}


ACTION_BY_KEY: dict[ActionKey, CapabilityAction] = {
    ActionKey.NONE: CapabilityAction(False, ExternalRetrievalLevel.OFF, ArchitectureMode.SINGLE),
    ActionKey.MEMORY: CapabilityAction(True, ExternalRetrievalLevel.OFF, ArchitectureMode.SINGLE),
    ActionKey.RAG: CapabilityAction(False, ExternalRetrievalLevel.STANDARD, ArchitectureMode.SINGLE),
    ActionKey.TEAM: CapabilityAction(False, ExternalRetrievalLevel.OFF, ArchitectureMode.TEAM),
    ActionKey.MEMORY_RAG: CapabilityAction(True, ExternalRetrievalLevel.STANDARD, ArchitectureMode.SINGLE),
    ActionKey.MEMORY_TEAM: CapabilityAction(True, ExternalRetrievalLevel.OFF, ArchitectureMode.TEAM),
    ActionKey.RAG_TEAM: CapabilityAction(False, ExternalRetrievalLevel.STANDARD, ArchitectureMode.TEAM),
    ActionKey.ALL: CapabilityAction(True, ExternalRetrievalLevel.STANDARD, ArchitectureMode.TEAM),
}


@dataclass(frozen=True)
class AbstractCost:
    memory_read_units: int = 0
    external_retrieval_units: int = 0
    team_worker_units: int = 0
    tool_units: int = 0
    model_version: str = "u1-abstract-cost-v1"

    @property
    def total_units(self) -> int:
        return self.memory_read_units + self.external_retrieval_units + self.team_worker_units + self.tool_units

    def to_dict(self) -> dict[str, int | str]:
        return {"memory_read_units": self.memory_read_units,
                "external_retrieval_units": self.external_retrieval_units,
                "team_worker_units": self.team_worker_units,
                "tool_units": self.tool_units,
                "total_units": self.total_units, "cost_model_version": self.model_version}


def cost_for(action: CapabilityAction, workers: tuple[WorkerManifest, ...]) -> AbstractCost:
    team_units = len(workers) if action.architecture == ArchitectureMode.TEAM else 0
    if action.architecture == ArchitectureMode.TEAM:
        tool_units = sum(sum(tool not in {"memory_read", "external_retrieval"}
                             for tool in worker.tool_ids) for worker in workers)
    else:
        tool_units = 0
    return AbstractCost(
        memory_read_units=int(action.memory_read),
        external_retrieval_units=int(action.external_retrieval == ExternalRetrievalLevel.STANDARD),
        team_worker_units=team_units,
        tool_units=tool_units,
    )


@dataclass(frozen=True)
class ActionAvailability:
    valid: tuple[ActionKey, ...]
    invalid_reasons: tuple[tuple[ActionKey, str], ...]

    @classmethod
    def for_episode(cls, episode: IntegrationEpisode) -> ActionAvailability:
        surface = episode.tool_surface_ref
        workers = surface.workers if surface else ()
        valid: list[ActionKey] = []
        invalid: list[tuple[ActionKey, str]] = []
        for key, action in ACTION_BY_KEY.items():
            reason = cls._invalid_reason(episode, action, workers)
            (valid if reason is None else invalid).append(key if reason is None else (key, reason))
        return cls(tuple(valid), tuple(invalid))  # type: ignore[arg-type]

    @staticmethod
    def _invalid_reason(
        episode: IntegrationEpisode, action: CapabilityAction, workers: tuple[WorkerManifest, ...]
    ) -> str | None:
        if action.memory_read:
            if episode.patient_state_ref is None or not episode.observable_state.history_exists:
                return "MEMORY_READ_REQUIRES_SUBJECT_HISTORY"
            if episode.tool_surface_ref is None or "memory_read" not in episode.tool_surface_ref.tool_ids:
                return "MEMORY_READ_TOOL_NOT_GRANTED"
        if action.external_retrieval == ExternalRetrievalLevel.STANDARD:
            if episode.external_world_ref is None or not episode.observable_state.available_external_source_families:
                return "EXTERNAL_RETRIEVAL_REQUIRES_EVIDENCE_WORLD"
            if episode.tool_surface_ref is None or "external_retrieval" not in episode.tool_surface_ref.tool_ids:
                return "EXTERNAL_RETRIEVAL_TOOL_NOT_GRANTED"
        if action.architecture == ArchitectureMode.TEAM and not workers:
            return "TEAM_REQUIRES_ELIGIBLE_WORKER_POOL"
        if action.architecture == ArchitectureMode.TEAM:
            report = capability_equivalence_report(episode, workers=workers)
            if not report.equivalent:
                return "SINGLE_TEAM_CAPABILITY_MISMATCH:" + ",".join(report.mismatches)
        cost = cost_for(action, workers)
        budget: EpisodeBudget = episode.budget
        if cost.memory_read_units > budget.memory_read_units:
            return "MEMORY_READ_BUDGET_UNAVAILABLE"
        if cost.external_retrieval_units > budget.external_retrieval_units:
            return "EXTERNAL_RETRIEVAL_BUDGET_UNAVAILABLE"
        if cost.team_worker_units > budget.team_worker_units:
            return "TEAM_WORKER_BUDGET_UNAVAILABLE"
        if cost.tool_units > budget.tool_units:
            return "TOOL_BUDGET_UNAVAILABLE"
        if cost.total_units > budget.global_units:
            return "GLOBAL_BUDGET_UNAVAILABLE"
        return None

    def allows(self, key: ActionKey | str) -> bool:
        return ActionKey(key) in self.valid

    def rejection_reason(self, key: ActionKey | str) -> str:
        normalized = ActionKey(key)
        if normalized in self.valid:
            return ""
        return dict(self.invalid_reasons)[normalized]

    def to_dict(self) -> dict[str, object]:
        return {"valid": [item.value for item in self.valid],
                "invalid_reasons": {key.value: reason for key, reason in self.invalid_reasons}}


@dataclass(frozen=True)
class CapabilityEnvelope:
    personal_state_scopes: tuple[str, ...]
    external_source_families: tuple[str, ...]
    tool_ids: tuple[str, ...]
    model_identity: str
    budget_regime: str
    global_budget_units: int
    evaluator_identity: str

    def to_dict(self) -> dict[str, object]:
        return {"personal_state_scopes": list(self.personal_state_scopes),
                "external_source_families": list(self.external_source_families),
                "tool_ids": list(self.tool_ids), "model_identity": self.model_identity,
                "budget_regime": self.budget_regime,
                "global_budget_units": self.global_budget_units,
                "evaluator_identity": self.evaluator_identity}


@dataclass(frozen=True)
class CapabilityEquivalenceReport:
    single: CapabilityEnvelope
    team_union: CapabilityEnvelope
    mismatches: tuple[str, ...]

    @property
    def equivalent(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, object]:
        return {"equivalent": self.equivalent, "single": self.single.to_dict(),
                "team_union": self.team_union.to_dict(), "mismatches": list(self.mismatches)}

    def require_equivalent(self) -> None:
        if not self.equivalent:
            raise ValueError("SINGLE_TEAM_CAPABILITY_MISMATCH: " + ", ".join(self.mismatches))


def capability_equivalence_report(
    episode: IntegrationEpisode,
    evaluator_ref: EvaluatorRef | None = None,
    workers: tuple[WorkerManifest, ...] | None = None,
) -> CapabilityEquivalenceReport:
    """Compare Single authority with the union of fixed Team worker manifests."""
    surface = episode.tool_surface_ref
    worker_rows = workers if workers is not None else (surface.workers if surface else ())
    single = CapabilityEnvelope(
        personal_state_scopes=episode.patient_state_ref.record_types if episode.patient_state_ref else (),
        external_source_families=episode.external_world_ref.source_families if episode.external_world_ref else (),
        tool_ids=surface.tool_ids if surface else (),
        model_identity=surface.model_identity if surface else "deterministic-scripted-v1",
        budget_regime=episode.budget.budget_class,
        global_budget_units=episode.budget.global_units,
        evaluator_identity=_evaluator_identity(evaluator_ref or episode.evaluator_ref),
    )
    team = CapabilityEnvelope(
        personal_state_scopes=tuple(sorted({scope for worker in worker_rows for scope in worker.personal_state_scopes})),
        external_source_families=tuple(sorted({family for worker in worker_rows for family in worker.external_source_families})),
        tool_ids=tuple(sorted({tool for worker in worker_rows for tool in worker.tool_ids})),
        model_identity=surface.model_identity if surface else "deterministic-scripted-v1",
        budget_regime=episode.budget.budget_class,
        global_budget_units=episode.budget.global_units,
        evaluator_identity=_evaluator_identity(evaluator_ref or episode.evaluator_ref),
    )
    mismatches = tuple(name for name in (
        "personal_state_scopes", "external_source_families", "tool_ids", "model_identity",
        "budget_regime", "global_budget_units", "evaluator_identity",
    ) if getattr(single, name) != getattr(team, name))
    return CapabilityEquivalenceReport(single, team, mismatches)


def _evaluator_identity(ref: EvaluatorRef) -> str:
    return f"{ref.evaluator_id}@{ref.version}"
