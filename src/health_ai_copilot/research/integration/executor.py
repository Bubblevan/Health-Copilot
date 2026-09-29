"""Offline deterministic executor. It accepts runtime inputs, never evaluator gold."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .actions import (
    ActionAvailability,
    ActionKey,
    CapabilityAction,
    capability_equivalence_report,
)
from .contracts import (
    ArchitectureMode,
    ExecutionOutcome,
    ExternalRetrievalLevel,
    FailureCategory,
    IntegrationEpisode,
    stable_hash,
)
from .evidence_world import ExternalEvidenceWorld
from .state import PatientStateRecord, PatientStateStore


class TraceKind(StrEnum):
    ACTION_REJECTED = "ACTION_REJECTED"
    MEMORY_READ = "MEMORY_READ"
    EXTERNAL_RETRIEVAL = "EXTERNAL_RETRIEVAL"
    TEAM_DELEGATION = "TEAM_DELEGATION"
    TOOL_CALL = "TOOL_CALL"
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
class ExecutionResult:
    outcome: ExecutionOutcome
    trace: tuple[ExecutionTraceEvent, ...]
    observed_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"outcome": self.outcome.to_dict(), "trace": [item.to_dict() for item in self.trace],
                "observed_evidence_ids": list(self.observed_evidence_ids)}


class DeterministicIntegrationExecutor:
    version = "u1-scripted-executor-v1"

    def execute(
        self,
        episode: IntegrationEpisode,
        action: CapabilityAction,
        resources: ExecutionResources,
    ) -> ExecutionResult:
        trace: list[ExecutionTraceEvent] = []
        action_key = ActionKey(action.action_id)
        availability = ActionAvailability.for_episode(episode)
        if not availability.allows(action_key):
            return self._rejected(action_key, availability.rejection_reason(action_key))

        surface = episode.tool_surface_ref
        workers = surface.workers if surface else ()
        if action.architecture == ArchitectureMode.TEAM:
            report = capability_equivalence_report(episode, workers=workers)
            if not report.equivalent:
                return self._rejected(action_key, "SINGLE_TEAM_CAPABILITY_MISMATCH:" + ",".join(report.mismatches))

        # Resolve every declared resource before the first read so a malformed
        # multi-capability action cannot partially execute and then fail.
        if action.memory_read and (
            resources.patient_state_store is None or episode.patient_state_ref is None
        ):
            return self._rejected(action_key, "PATIENT_STATE_RESOURCE_MISSING")
        if action.external_retrieval == ExternalRetrievalLevel.STANDARD:
            if resources.external_evidence_world is None or episode.external_world_ref is None:
                return self._rejected(action_key, "EXTERNAL_EVIDENCE_RESOURCE_MISSING")
            if (resources.external_evidence_world.world_id, resources.external_evidence_world.version) != (
                episode.external_world_ref.world_id, episode.external_world_ref.version
            ):
                return self._rejected(action_key, "EXTERNAL_WORLD_VERSION_MISMATCH")

        patient_rows: tuple[PatientStateRecord, ...] = ()
        external_rows = ()
        query_tool_outputs: list[str] = []
        tool_calls = 0
        if action.memory_read:
            ref = episode.patient_state_ref
            patient_rows = resources.patient_state_store.snapshot(ref.subject_id, episode.decision_time)
            allowed_types = set(ref.record_types)
            patient_rows = tuple(row for row in patient_rows if row.record_type.value in allowed_types)
            if action.architecture == ArchitectureMode.TEAM:
                allowed_by_workers = {
                    scope for worker in workers if "memory_read" in worker.tool_ids
                    for scope in worker.personal_state_scopes
                }
                patient_rows = tuple(row for row in patient_rows if row.record_type.value in allowed_by_workers)
            patient_rows = tuple(row for row in patient_rows if _query_matches(episode.query, row.retrieval_terms))
            trace.append(ExecutionTraceEvent(len(trace), TraceKind.MEMORY_READ, "MEMORY_READ",
                                             tuple(row.record_id for row in patient_rows)))

        if action.external_retrieval == ExternalRetrievalLevel.STANDARD:
            ref = episode.external_world_ref
            world = resources.external_evidence_world
            families = set(ref.source_families)
            if action.architecture == ArchitectureMode.TEAM:
                families &= {
                    family for worker in workers if "external_retrieval" in worker.tool_ids
                    for family in worker.external_source_families
                }
            external_rows = world.retrieve(episode.query, tuple(sorted(families)),
                                           as_of_time=episode.decision_time)
            trace.append(ExecutionTraceEvent(len(trace), TraceKind.EXTERNAL_RETRIEVAL,
                                             "EXTERNAL_RETRIEVAL",
                                             tuple(row.source_id for row in external_rows)))

        if action.architecture == ArchitectureMode.TEAM:
            trace.append(ExecutionTraceEvent(len(trace), TraceKind.TEAM_DELEGATION,
                                             "TEAM_ORCHESTRATION",
                                             tuple(worker.worker_id for worker in workers)))
            for worker in workers:
                for tool_id in worker.tool_ids:
                    if tool_id in {"memory_read", "external_retrieval"}:
                        continue
                    tool_calls += 1
                    trace.append(ExecutionTraceEvent(len(trace), TraceKind.TOOL_CALL,
                                                     "TEAM_ORCHESTRATION", (tool_id,), worker.worker_id))
                    part = _query_partition(episode.query, tool_id)
                    if part:
                        query_tool_outputs.append(part)

        query_facts: list[str] = []
        if query_tool_outputs:
            numbers = [int(match) for value in query_tool_outputs
                       for match in re.findall(r"(?<![\w.])-?\d+(?![\w.])", value)]
            if len(numbers) >= 2 and "sum" in episode.query.casefold():
                query_facts.append(str(sum(numbers)))
            else:
                query_facts.extend(query_tool_outputs)

        used_ids = tuple(row.record_id for row in patient_rows) + tuple(row.source_id for row in external_rows)
        materials = [row.content for row in patient_rows] + [row.content for row in external_rows] + query_facts
        if materials:
            answer = " ".join(materials)
        elif "insufficient" in episode.query.casefold() or "supported answer" in episode.query.casefold():
            answer = "INSUFFICIENT_EVIDENCE"
        else:
            answer = episode.query
        trace.append(ExecutionTraceEvent(len(trace), TraceKind.COMPLETED, "RUNTIME",
                                         used_ids, stable_hash({"answer": answer, "action": action.to_dict()})))
        activated = tuple(name for enabled, name in (
            (action.memory_read, "MEMORY_READ"),
            (action.external_retrieval == ExternalRetrievalLevel.STANDARD, "EXTERNAL_RETRIEVAL"),
            (action.architecture == ArchitectureMode.TEAM, "TEAM_ORCHESTRATION"),
        ) if enabled)
        outcome = ExecutionOutcome(
            task_success=False, safety_pass=True, grounding_pass=True, answer=answer,
            used_evidence_ids=used_ids, provider_calls=0, tool_calls=tool_calls,
            input_tokens=0, output_tokens=0, latency_ms=0,
            activated_capabilities=activated,
        )
        return ExecutionResult(outcome, tuple(trace), used_ids)

    @staticmethod
    def _rejected(action: ActionKey, reason: str) -> ExecutionResult:
        event = ExecutionTraceEvent(0, TraceKind.ACTION_REJECTED, "ACTION_MASK", (), f"{action.value}:{reason}")
        return ExecutionResult(
            ExecutionOutcome(False, True, True, "REJECTED", provider_calls=0,
                             failure_category=FailureCategory.CONTRACT_VIOLATION),
            (event,), (),
        )


def _query_matches(query: str, terms: tuple[str, ...]) -> bool:
    if not terms:
        return True
    text = query.casefold()
    return any(term in text for term in terms)


def _query_partition(query: str, tool_id: str) -> str:
    match = re.fullmatch(r"query_part_([a-z])", tool_id)
    if not match:
        return ""
    label = match.group(1)
    found = re.search(rf"part_{label}\s*=\s*([^;]+)", query, flags=re.IGNORECASE)
    return found.group(1).strip() if found else ""
