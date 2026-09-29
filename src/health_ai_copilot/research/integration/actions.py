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
from .tools import DeterministicToolRegistry, default_tool_registry


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
class ActivationCost:
    memory_activation_units: int = 0
    retrieval_activation_units: int = 0
    team_orchestration_units: int = 0

    @property
    def total_units(self) -> int:
        return self.memory_activation_units + self.retrieval_activation_units + self.team_orchestration_units

    def to_dict(self) -> dict[str, int]:
        return {"memory_activation_units": self.memory_activation_units,
                "retrieval_activation_units": self.retrieval_activation_units,
                "team_orchestration_units": self.team_orchestration_units,
                "total_units": self.total_units}


@dataclass(frozen=True)
class ObservedUsage:
    workers_started: int = 0
    tools_executed: int = 0
    memory_reads: int = 0
    retrieval_executions: int = 0
    worker_units: tuple[tuple[str, int], ...] = ()

    @property
    def total_units(self) -> int:
        return self.workers_started + self.tools_executed + self.memory_reads + self.retrieval_executions

    def to_dict(self) -> dict[str, object]:
        return {"workers_started": self.workers_started,
                "tools_executed": self.tools_executed,
                "memory_reads": self.memory_reads,
                "retrieval_executions": self.retrieval_executions,
                "worker_units": {name: units for name, units in self.worker_units},
                "total_units": self.total_units}


@dataclass(frozen=True)
class AbstractCost:
    activation: ActivationCost
    observed: ObservedUsage
    model_version: str = "u1.1-activation-observed-v1"

    @property
    def memory_read_units(self) -> int:
        return self.activation.memory_activation_units + self.observed.memory_reads

    @property
    def external_retrieval_units(self) -> int:
        return self.activation.retrieval_activation_units + self.observed.retrieval_executions

    @property
    def team_worker_units(self) -> int:
        return self.activation.team_orchestration_units + self.observed.workers_started

    @property
    def tool_units(self) -> int:
        return self.observed.tools_executed

    @property
    def total_units(self) -> int:
        return self.activation.total_units + self.observed.total_units

    def to_dict(self) -> dict[str, object]:
        return {"activation_cost": self.activation.to_dict(),
                "observed_usage": self.observed.to_dict(),
                "memory_read_units": self.memory_read_units,
                "external_retrieval_units": self.external_retrieval_units,
                "team_worker_units": self.team_worker_units,
                "tool_units": self.tool_units,
                "total_units": self.total_units,
                "cost_model_version": self.model_version}


def planned_tool_ids(episode: IntegrationEpisode, action: CapabilityAction) -> tuple[str, ...]:
    surface = episode.tool_surface_ref
    if surface is None:
        return ()
    return tuple(tool_id for tool_id in surface.tool_ids
                 if (tool_id != "memory_read" or action.memory_read)
                 and (tool_id != "external_retrieval"
                      or action.external_retrieval == ExternalRetrievalLevel.STANDARD))


def _worker_assignments(
    tool_ids: tuple[str, ...], workers: tuple[WorkerManifest, ...]
) -> dict[str, list[str]]:
    assigned = {worker.worker_id: [] for worker in workers}
    for tool_id in tool_ids:
        owners = [worker.worker_id for worker in workers if tool_id in worker.tool_ids]
        if owners:
            assigned[owners[0]].append(tool_id)
    return assigned


def cost_for(
    action: CapabilityAction,
    workers: tuple[WorkerManifest, ...],
    tool_ids: tuple[str, ...] = (),
) -> AbstractCost:
    team = action.architecture == ArchitectureMode.TEAM
    assignments = _worker_assignments(tool_ids, workers) if team else {}
    memory_reads = int(action.memory_read)
    retrievals = int(action.external_retrieval == ExternalRetrievalLevel.STANDARD)
    worker_units = tuple(
        (worker_id, 1 + len(assigned) + int("memory_read" in assigned) + int("external_retrieval" in assigned))
        for worker_id, assigned in sorted(assignments.items())
    )
    return AbstractCost(
        activation=ActivationCost(
            memory_activation_units=int(action.memory_read),
            retrieval_activation_units=retrievals,
            team_orchestration_units=int(team),
        ),
        observed=ObservedUsage(
            workers_started=len(workers) if team else 0,
            tools_executed=len(tool_ids),
            memory_reads=memory_reads,
            retrieval_executions=retrievals,
            worker_units=worker_units,
        ),
    )


@dataclass(frozen=True)
class ActionAvailability:
    valid: tuple[ActionKey, ...]
    invalid_reasons: tuple[tuple[ActionKey, str], ...]

    @classmethod
    def for_episode(
        cls,
        episode: IntegrationEpisode,
        tool_registry: DeterministicToolRegistry | None = None,
    ) -> ActionAvailability:
        surface = episode.tool_surface_ref
        workers = surface.workers if surface else ()
        registry = tool_registry or default_tool_registry()
        valid: list[ActionKey] = []
        invalid: list[tuple[ActionKey, str]] = []
        for key, action in ACTION_BY_KEY.items():
            reason = cls._invalid_reason(episode, action, workers, registry)
            (valid if reason is None else invalid).append(key if reason is None else (key, reason))
        return cls(tuple(valid), tuple(invalid))  # type: ignore[arg-type]

    @staticmethod
    def _invalid_reason(
        episode: IntegrationEpisode,
        action: CapabilityAction,
        workers: tuple[WorkerManifest, ...],
        tool_registry: DeterministicToolRegistry,
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
        surface = episode.tool_surface_ref
        declared_tool_ids = surface.tool_ids if surface else ()
        missing_tools = tool_registry.missing(declared_tool_ids)
        if missing_tools:
            return "NO_DETERMINISTIC_IMPLEMENTATION:" + ",".join(missing_tools)
        if action.architecture == ArchitectureMode.TEAM:
            executable = executable_capability_equivalence_report(episode, registry=tool_registry)
            if not executable.equivalent:
                return "SINGLE_TEAM_EXECUTABLE_CAPABILITY_MISMATCH:" + ",".join(executable.mismatches)
        cost = cost_for(action, workers, planned_tool_ids(episode, action))
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
        if action.architecture == ArchitectureMode.TEAM:
            shares = partition_global_budget(budget.global_units, workers)
            over_budget = [worker_id for worker_id, units in cost.observed.worker_units
                           if units > shares.get(worker_id, 0)]
            if over_budget:
                return "TEAM_WORKER_BUDGET_UNAVAILABLE:" + ",".join(over_budget)
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


@dataclass(frozen=True)
class ExecutableCapabilityEnvelope:
    personal_state_scopes: tuple[str, ...]
    external_source_families: tuple[str, ...]
    tool_ids: tuple[str, ...]
    tool_implementation_hashes: tuple[tuple[str, str], ...]
    unimplemented_tool_ids: tuple[str, ...]
    model_identity: str
    budget_regime: str
    global_budget_units: int
    evaluator_identity: str

    def to_dict(self) -> dict[str, object]:
        return {"personal_state_scopes": list(self.personal_state_scopes),
                "external_source_families": list(self.external_source_families),
                "tool_ids": list(self.tool_ids),
                "tool_implementation_hashes": dict(self.tool_implementation_hashes),
                "unimplemented_tool_ids": list(self.unimplemented_tool_ids),
                "model_identity": self.model_identity,
                "budget_regime": self.budget_regime,
                "global_budget_units": self.global_budget_units,
                "evaluator_identity": self.evaluator_identity}


@dataclass(frozen=True)
class ExecutableCapabilityEquivalenceReport:
    single: ExecutableCapabilityEnvelope
    team_union: ExecutableCapabilityEnvelope
    declared_report: CapabilityEquivalenceReport
    mismatches: tuple[str, ...]

    @property
    def equivalent(self) -> bool:
        return self.declared_report.equivalent and not self.mismatches

    def to_dict(self) -> dict[str, object]:
        return {"equivalent": self.equivalent,
                "single": self.single.to_dict(),
                "team_union": self.team_union.to_dict(),
                "declared_report": self.declared_report.to_dict(),
                "mismatches": list(self.mismatches)}

    def require_equivalent(self) -> None:
        if not self.equivalent:
            raise ValueError("SINGLE_TEAM_EXECUTABLE_CAPABILITY_MISMATCH: "
                             + ", ".join(self.mismatches))


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


def executable_capability_equivalence_report(
    episode: IntegrationEpisode,
    evaluator_ref: EvaluatorRef | None = None,
    *,
    registry: DeterministicToolRegistry | None = None,
    single_registry: DeterministicToolRegistry | None = None,
    team_registry: DeterministicToolRegistry | None = None,
    workers: tuple[WorkerManifest, ...] | None = None,
) -> ExecutableCapabilityEquivalenceReport:
    """Compare actual tool reachability and implementation identity by architecture."""
    surface = episode.tool_surface_ref
    single_tools = surface.tool_ids if surface else ()
    worker_rows = workers if workers is not None else (surface.workers if surface else ())
    team_tools = tuple(sorted({tool for worker in worker_rows for tool in worker.tool_ids}))
    base_registry = registry or default_tool_registry()
    single_tools_registry = single_registry or base_registry
    team_tools_registry = team_registry or base_registry

    def make_envelope(
        tool_ids: tuple[str, ...],
        tool_registry: DeterministicToolRegistry,
        personal_state_scopes: tuple[str, ...],
        external_source_families: tuple[str, ...],
    ) -> ExecutableCapabilityEnvelope:
        implemented = tuple(tool for tool in tool_ids if tool_registry.contains(tool))
        hashes = tuple((tool, tool_registry.implementation_hash(tool) or "") for tool in implemented)
        return ExecutableCapabilityEnvelope(
            personal_state_scopes=personal_state_scopes,
            external_source_families=external_source_families,
            tool_ids=implemented,
            tool_implementation_hashes=hashes,
            unimplemented_tool_ids=tool_registry.missing(tool_ids),
            model_identity=surface.model_identity if surface else "deterministic-scripted-v1",
            budget_regime=episode.budget.budget_class,
            global_budget_units=episode.budget.global_units,
            evaluator_identity=_evaluator_identity(evaluator_ref or episode.evaluator_ref),
        )

    declared = capability_equivalence_report(episode, evaluator_ref, workers=worker_rows)
    single = make_envelope(
        single_tools,
        single_tools_registry,
        episode.patient_state_ref.record_types if episode.patient_state_ref else (),
        episode.external_world_ref.source_families if episode.external_world_ref else (),
    )
    team_union = make_envelope(
        team_tools,
        team_tools_registry,
        tuple(sorted({scope for worker in worker_rows for scope in worker.personal_state_scopes})),
        tuple(sorted({family for worker in worker_rows for family in worker.external_source_families})),
    )
    mismatches = [name for name in (
        "personal_state_scopes", "external_source_families", "tool_ids",
        "tool_implementation_hashes", "model_identity", "budget_regime",
        "global_budget_units", "evaluator_identity",
    ) if getattr(single, name) != getattr(team_union, name)]
    if single.unimplemented_tool_ids:
        mismatches.append("single_unimplemented:" + ",".join(single.unimplemented_tool_ids))
    if team_union.unimplemented_tool_ids:
        mismatches.append("team_unimplemented:" + ",".join(team_union.unimplemented_tool_ids))
    mismatches.extend(f"declared:{name}" for name in declared.mismatches)
    return ExecutableCapabilityEquivalenceReport(
        single, team_union, declared, tuple(sorted(mismatches))
    )


def partition_global_budget(
    global_units: int,
    workers: tuple[WorkerManifest, ...],
) -> dict[str, int]:
    """Partition one global budget evenly; worker shares sum to that global cap."""
    if not workers:
        return {}
    quotient, remainder = divmod(global_units, len(workers))
    return {worker.worker_id: quotient + int(index < remainder)
            for index, worker in enumerate(sorted(workers, key=lambda item: item.worker_id))}


def _evaluator_identity(ref: EvaluatorRef) -> str:
    return f"{ref.evaluator_id}@{ref.version}"
