"""Offline executor using one implementation surface for Single and Team."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .actions import (
    AbstractCost,
    ActionAvailability,
    ActionKey,
    CapabilityAction,
    cost_for,
    partition_global_budget,
    planned_tool_ids,
)
from .contracts import (
    ArchitectureMode,
    ExecutionOutcome,
    ExternalRetrievalLevel,
    FailureCategory,
    IntegrationEpisode,
)
from .evidence_world import ExternalEvidenceWorld
from .state import PatientStateStore
from .tools import DeterministicToolRegistry, ToolInvocation, ToolObservation, default_tool_registry


class TraceKind(StrEnum):
    ACTION_REJECTED = "ACTION_REJECTED"
    MEMORY_READ = "MEMORY_READ"
    EXTERNAL_RETRIEVAL = "EXTERNAL_RETRIEVAL"
    TEAM_DELEGATION = "TEAM_DELEGATION"
    TOOL_CALL = "TOOL_CALL"
    TOOL_OBSERVATION = "TOOL_OBSERVATION"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class ExecutionTraceEvent:
    sequence: int
    kind: TraceKind
    namespace: str
    resource_ids: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"sequence": self.sequence, "kind": self.kind.value, "namespace": self.namespace,
                "resource_ids": list(self.resource_ids), "detail": self.detail}


@dataclass(frozen=True)
class ExecutionResources:
    patient_state_store: PatientStateStore | None = None
    external_evidence_world: ExternalEvidenceWorld | None = None


@dataclass(frozen=True)
class BudgetAudit:
    global_budget_units: int
    global_used_units: int
    worker_budgets: tuple[tuple[str, int], ...] = ()
    worker_used_units: tuple[tuple[str, int], ...] = ()

    @property
    def within_global_budget(self) -> bool:
        return self.global_used_units <= self.global_budget_units

    @property
    def worker_budgets_sum_to_global(self) -> bool:
        return not self.worker_budgets or sum(value for _, value in self.worker_budgets) == self.global_budget_units

    def to_dict(self) -> dict[str, object]:
        return {"global_budget_units": self.global_budget_units,
                "global_used_units": self.global_used_units,
                "within_global_budget": self.within_global_budget,
                "worker_budgets": dict(self.worker_budgets),
                "worker_used_units": dict(self.worker_used_units),
                "worker_budgets_sum_to_global": self.worker_budgets_sum_to_global}


@dataclass(frozen=True)
class ExecutionResult:
    outcome: ExecutionOutcome
    trace: tuple[ExecutionTraceEvent, ...]
    observed_evidence_ids: tuple[str, ...]
    tool_observations: tuple[ToolObservation, ...] = ()
    cost: AbstractCost | None = None
    budget_audit: BudgetAudit | None = None

    def to_dict(self) -> dict[str, object]:
        return {"outcome": self.outcome.to_dict(), "trace": [item.to_dict() for item in self.trace],
                "observed_evidence_ids": list(self.observed_evidence_ids),
                "tool_observations": [item.to_dict() for item in self.tool_observations],
                "cost_accounting": self.cost.to_dict() if self.cost else None,
                "budget_audit": self.budget_audit.to_dict() if self.budget_audit else None}


class DeterministicIntegrationExecutor:
    version = "u1.1-shared-tool-executor-v2"

    def __init__(self, tool_registry: DeterministicToolRegistry | None = None) -> None:
        self.tool_registry = tool_registry or default_tool_registry()

    def execute(
        self,
        episode: IntegrationEpisode,
        action: CapabilityAction,
        resources: ExecutionResources,
    ) -> ExecutionResult:
        trace: list[ExecutionTraceEvent] = []
        action_key = ActionKey(action.action_id)
        availability = ActionAvailability.for_episode(episode, self.tool_registry)
        if not availability.allows(action_key):
            return self._rejected(action_key, availability.rejection_reason(action_key))

        surface = episode.tool_surface_ref
        workers = surface.workers if surface else ()
        is_team = action.architecture == ArchitectureMode.TEAM

        # Resolve all resources and the complete call plan before the first read.
        if action.memory_read and (resources.patient_state_store is None or episode.patient_state_ref is None):
            return self._rejected(action_key, "PATIENT_STATE_RESOURCE_MISSING")
        if action.external_retrieval == ExternalRetrievalLevel.STANDARD:
            if resources.external_evidence_world is None or episode.external_world_ref is None:
                return self._rejected(action_key, "EXTERNAL_EVIDENCE_RESOURCE_MISSING")
            if (resources.external_evidence_world.world_id, resources.external_evidence_world.version) != (
                episode.external_world_ref.world_id, episode.external_world_ref.version
            ):
                return self._rejected(action_key, "EXTERNAL_WORLD_VERSION_MISMATCH")

        call_ids = planned_tool_ids(episode, action)
        missing = self.tool_registry.missing(call_ids)
        if missing:
            return self._rejected(action_key, "NO_DETERMINISTIC_IMPLEMENTATION:" + ",".join(missing))
        assignments = _assign_tools(call_ids, workers) if is_team else {}
        if is_team:
            unassigned = sorted(set(call_ids) - {tool for values in assignments.values() for tool in values})
            if unassigned:
                return self._rejected(action_key, "TEAM_TOOL_ASSIGNMENT_MISSING:" + ",".join(unassigned))

        cost = cost_for(action, workers, call_ids)
        budget = episode.budget
        if (cost.memory_read_units > budget.memory_read_units
                or cost.external_retrieval_units > budget.external_retrieval_units
                or cost.team_worker_units > budget.team_worker_units
                or cost.tool_units > budget.tool_units
                or cost.total_units > budget.global_units):
            return self._rejected(action_key, "GLOBAL_BUDGET_EXHAUSTED")
        worker_budgets = partition_global_budget(budget.global_units, workers) if is_team else {}
        worker_usage = dict(cost.observed.worker_units)
        if any(worker_usage.get(worker_id, 0) > limit for worker_id, limit in worker_budgets.items()):
            return self._rejected(action_key, "TEAM_WORKER_BUDGET_EXHAUSTED")

        if is_team:
            trace.append(ExecutionTraceEvent(
                len(trace), TraceKind.TEAM_DELEGATION, "TEAM_ORCHESTRATION",
                tuple(worker.worker_id for worker in workers),
            ))
        observations: list[ToolObservation] = []
        for tool_id in call_ids:
            worker_id = _worker_for_tool(tool_id, workers) if is_team else None
            trace.append(ExecutionTraceEvent(
                len(trace), TraceKind.TOOL_CALL, "DETERMINISTIC_TOOL_SURFACE",
                (tool_id,), worker_id or "SINGLE",
            ))
            observation = self.tool_registry.invoke(ToolInvocation(
                tool_id=tool_id,
                episode=episode,
                patient_state_store=resources.patient_state_store,
                external_evidence_world=resources.external_evidence_world,
            ))
            observations.append(observation)
            trace.append(ExecutionTraceEvent(
                len(trace), TraceKind.TOOL_OBSERVATION, "DETERMINISTIC_TOOL_SURFACE",
                observation.resource_ids, observation.output_hash,
            ))

        by_id = {item.tool_id: item for item in observations}
        memory_observation = by_id.get("memory_read")
        external_observation = by_id.get("external_retrieval")
        if memory_observation is not None:
            trace.append(ExecutionTraceEvent(len(trace), TraceKind.MEMORY_READ, "MEMORY_READ",
                                             memory_observation.resource_ids,
                                             memory_observation.output_hash))
        if external_observation is not None:
            trace.append(ExecutionTraceEvent(len(trace), TraceKind.EXTERNAL_RETRIEVAL,
                                             "EXTERNAL_RETRIEVAL",
                                             external_observation.resource_ids,
                                             external_observation.output_hash))

        query_tool_observations = [item for item in observations
                                   if item.tool_id not in {"memory_read", "external_retrieval"}]
        query_facts = [item.output for item in query_tool_observations if item.output]
        query_text = episode.query.casefold()
        numeric_operation = next((name for name in (
            "greater_than", "difference", "minimum", "maximum", "count", "trend", "sum"
        ) if name in query_text), None)
        if query_facts and numeric_operation:
            numbers = [int(value) for item in query_facts for value in _integer_strings(item)]
            if len(numbers) >= 2:
                if numeric_operation == "sum":
                    computed: str | int = sum(numbers)
                elif numeric_operation == "difference":
                    computed = abs(numbers[0] - numbers[1])
                elif numeric_operation == "count":
                    computed = len(numbers)
                elif numeric_operation == "minimum":
                    computed = min(numbers)
                elif numeric_operation == "maximum":
                    computed = max(numbers)
                elif numeric_operation == "trend":
                    computed = "UP" if numbers[1] > numbers[0] else (
                        "DOWN" if numbers[1] < numbers[0] else "STABLE"
                    )
                else:
                    computed = "TRUE" if numbers[0] > numbers[1] else "FALSE"
                query_facts = [str(computed)]

        materials = []
        if memory_observation and memory_observation.output:
            materials.append(memory_observation.output)
        if external_observation and external_observation.output:
            materials.append(external_observation.output)
        materials.extend(query_facts)
        if materials:
            answer = " ".join(materials)
        elif "insufficient" in episode.query.casefold() or "supported answer" in episode.query.casefold():
            answer = "INSUFFICIENT_EVIDENCE"
        else:
            answer = episode.query
        evidence_ids = tuple(resource_id for item in observations for resource_id in item.resource_ids)
        activated = tuple(name for enabled, name in (
            (action.memory_read, "MEMORY_READ"),
            (action.external_retrieval == ExternalRetrievalLevel.STANDARD, "EXTERNAL_RETRIEVAL"),
            (is_team, "TEAM_ORCHESTRATION"),
        ) if enabled)
        trace.append(ExecutionTraceEvent(len(trace), TraceKind.COMPLETED, "RUNTIME",
                                         evidence_ids, answer))
        final_cost = cost_for(action, workers, tuple(item.tool_id for item in observations))
        audit = BudgetAudit(
            global_budget_units=budget.global_units,
            global_used_units=final_cost.total_units,
            worker_budgets=tuple(sorted(worker_budgets.items())),
            worker_used_units=tuple(sorted(final_cost.observed.worker_units)),
        )
        outcome = ExecutionOutcome(
            task_success=False, safety_pass=True, grounding_pass=True, answer=answer,
            used_evidence_ids=evidence_ids, provider_calls=0, tool_calls=len(observations),
            input_tokens=0, output_tokens=0, latency_ms=0,
            activated_capabilities=activated,
        )
        return ExecutionResult(outcome, tuple(trace), evidence_ids, tuple(observations), final_cost, audit)

    @staticmethod
    def _rejected(action: ActionKey, reason: str) -> ExecutionResult:
        event = ExecutionTraceEvent(0, TraceKind.ACTION_REJECTED, "ACTION_MASK", (), f"{action.value}:{reason}")
        category = (FailureCategory.BUDGET_EXHAUSTED if "BUDGET" in reason
                    else FailureCategory.CONTRACT_VIOLATION)
        return ExecutionResult(
            ExecutionOutcome(False, True, True, "REJECTED", provider_calls=0,
                             failure_category=category),
            (event,), (),
        )


def _assign_tools(tool_ids: tuple[str, ...], workers) -> dict[str, list[str]]:
    assignments = {worker.worker_id: [] for worker in workers}
    for tool_id in tool_ids:
        worker_id = _worker_for_tool(tool_id, workers)
        if worker_id is not None:
            assignments[worker_id].append(tool_id)
    return assignments


def _worker_for_tool(tool_id: str, workers) -> str | None:
    owners = sorted(worker.worker_id for worker in workers if tool_id in worker.tool_ids)
    return owners[0] if owners else None


def _integer_strings(value: str) -> tuple[int, ...]:
    import re

    return tuple(int(item) for item in re.findall(r"(?<![\w.])-?\d+(?![\w.])", value))
