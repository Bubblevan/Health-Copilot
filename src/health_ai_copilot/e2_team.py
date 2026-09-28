"""Sequential heterogeneous E2 worker execution and provenance ledger.

L3 changes worker capability while remaining sequential. This module does not
implement the frozen L2 M8 orchestration or an L4 parallel scheduler.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from time import monotonic
from typing import Any

from .agent.loop import AgentLoop, AgentLoopConfig, AgentRunResult
from .agent.model import AgentModel
from .agent.session import AgentSession
from .agent.state import StopReason
from .agent.tools import ToolRegistry
from .capabilities import (
    CapabilityEligibility,
    CapabilityScopedRetriever,
    CapabilityScopedSearchTool,
    CapabilitySourceCatalog,
    E2WorkerRole,
    WorkerCapabilitySpec,
)
from .contracts import Evidence
from .runtime.budget import BudgetDenied, RunBudgetConfig
from .runtime.components import ComponentManifest
from .runtime.context import RunContext
from .runtime.trace import TraceEventType
from .verification.grounding import GroundedClaim


class E2WorkerStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class E2Task:
    """A trusted, bounded assignment; task profiles are not accepted here."""

    task_id: str
    role: E2WorkerRole | str
    objective: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", E2WorkerRole(self.role))
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError("objective must be non-empty")


@dataclass(frozen=True)
class EvidenceObservation:
    source_id: str
    origin: str
    worker_id: str | None
    worker_role: str | None
    capability_id: str | None
    task_id: str | None
    retrieval_tool_origin: str
    first_seen_order: int

    def provenance_key(self) -> tuple[str | None, str | None, str | None, str | None, str, str]:
        return (
            self.worker_id,
            self.worker_role,
            self.capability_id,
            self.task_id,
            self.origin,
            self.retrieval_tool_origin,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "origin": self.origin,
            "worker_id": self.worker_id,
            "worker_role": self.worker_role,
            "capability_id": self.capability_id,
            "task_id": self.task_id,
            "retrieval_tool_origin": self.retrieval_tool_origin,
            "first_seen_order": self.first_seen_order,
        }


@dataclass(frozen=True)
class EvidenceLedgerRecord:
    source_id: str
    evidence: Evidence
    first_observation: EvidenceObservation
    observations: tuple[EvidenceObservation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "origin": self.first_observation.origin,
            "worker_id": self.first_observation.worker_id,
            "worker_role": self.first_observation.worker_role,
            "capability_id": self.first_observation.capability_id,
            "task_id": self.first_observation.task_id,
            "retrieval_tool_origin": self.first_observation.retrieval_tool_origin,
            "first_seen_order": self.first_observation.first_seen_order,
            "observations": [item.to_dict() for item in self.observations],
            "title_sha256": _sha256(self.evidence.title),
            "excerpt_sha256": _sha256(self.evidence.excerpt),
        }


class EvidenceLedgerV2:
    """Harness-owned evidence union with ordered, worker-level provenance."""

    def __init__(self) -> None:
        self._records: dict[str, EvidenceLedgerRecord] = {}

    def add_initial(
        self,
        evidence: Sequence[Evidence],
        *,
        source_catalog: CapabilitySourceCatalog,
    ) -> None:
        for item in evidence:
            if item.source_id not in source_catalog.by_id:
                continue
            self._observe(
                item,
                EvidenceObservation(
                    item.source_id,
                    "initial",
                    None,
                    None,
                    None,
                    None,
                    "pipeline_retrieval",
                    len(self._records) + 1,
                ),
            )

    def add_worker(
        self,
        evidence: Sequence[Evidence],
        *,
        worker_id: str,
        role: E2WorkerRole,
        capability_id: str,
        task_id: str,
        allowed: Callable[[str], bool],
        retrieval_tool_origin: str,
    ) -> None:
        for item in evidence:
            if not allowed(item.source_id):
                raise ValueError("worker evidence is outside the capability source boundary")
            existing = self._records.get(item.source_id)
            first_seen_order = (
                existing.first_observation.first_seen_order
                if existing is not None
                else len(self._records) + 1
            )
            self._observe(
                item,
                EvidenceObservation(
                    item.source_id,
                    "worker",
                    worker_id,
                    role.value,
                    capability_id,
                    task_id,
                    retrieval_tool_origin,
                    first_seen_order,
                ),
            )

    def contains(self, source_id: str) -> bool:
        return source_id in self._records

    def get(self, source_id: str) -> EvidenceLedgerRecord:
        return self._records[source_id]

    @property
    def records(self) -> tuple[EvidenceLedgerRecord, ...]:
        return tuple(self._records.values())

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def final_evidence(self) -> tuple[Evidence, ...]:
        return tuple(item.evidence for item in self._records.values())

    def observations_for_worker(self, worker_id: str) -> tuple[EvidenceObservation, ...]:
        return tuple(
            observation
            for record in self._records.values()
            for observation in record.observations
            if observation.worker_id == worker_id
        )

    def _observe(self, evidence: Evidence, observation: EvidenceObservation) -> None:
        current = self._records.get(evidence.source_id)
        if current is None:
            self._records[evidence.source_id] = EvidenceLedgerRecord(
                evidence.source_id, evidence, observation, (observation,)
            )
            return
        if observation.provenance_key() in {
            item.provenance_key() for item in current.observations
        }:
            return
        self._records[evidence.source_id] = EvidenceLedgerRecord(
            current.source_id,
            current.evidence,
            current.first_observation,
            (*current.observations, observation),
        )


@dataclass(frozen=True)
class E2WorkerReport:
    worker_id: str
    role: E2WorkerRole
    capability_id: str
    capability_hash: str
    task_id: str
    status: E2WorkerStatus
    claims: tuple[GroundedClaim, ...] = ()
    citation_ids: tuple[str, ...] = ()
    observed_source_ids: tuple[str, ...] = ()
    new_unique_source_ids: tuple[str, ...] = ()
    provider_calls: int = 0
    tool_proposals: int = 0
    tool_executions: int = 0
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    elapsed_ms: float = 0.0
    stop_reason: str | None = None
    error_code: str | None = None
    verified: bool = False
    session_id_sha256: str | None = None

    @property
    def completed(self) -> bool:
        return self.status == E2WorkerStatus.COMPLETED

    @property
    def productive(self) -> bool:
        return self.completed and self.verified and bool(
            self.new_unique_source_ids or self.citation_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "role": self.role.value,
            "capability_id": self.capability_id,
            "capability_hash": self.capability_hash,
            "task_id": self.task_id,
            "status": self.status.value,
            "completed": self.completed,
            "productive": self.productive,
            "verified": self.verified,
            "claims": [
                {"text": item.text, "citation_ids": list(item.citation_ids)}
                for item in self.claims
            ],
            "citation_ids": list(self.citation_ids),
            "observed_source_ids": list(self.observed_source_ids),
            "new_unique_source_ids": list(self.new_unique_source_ids),
            "provider_calls": self.provider_calls,
            "tool_proposals": self.tool_proposals,
            "tool_executions": self.tool_executions,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "stop_reason": self.stop_reason,
            "error_code": self.error_code,
            "session_id_sha256": self.session_id_sha256,
        }


@dataclass(frozen=True)
class E2SequentialResult:
    architecture_id: str
    scheduler: str
    capability_manifest_hash: str
    component_manifest_hash: str | None
    worker_reports: tuple[E2WorkerReport, ...]
    ledger: EvidenceLedgerV2
    wall_clock_latency_ms: float

    @property
    def verified_worker_reports(self) -> tuple[E2WorkerReport, ...]:
        return tuple(item for item in self.worker_reports if item.verified)

    @property
    def lead_evidence(self) -> tuple[Evidence, ...]:
        """Ledger-backed evidence only; unverified worker output is excluded."""

        return self.ledger.final_evidence()

    def metrics(
        self,
        source_catalog: CapabilitySourceCatalog,
        *,
        required_evidence_groups: Sequence[Sequence[str]] = (),
        required_source_families: Sequence[str] = (),
    ) -> dict[str, Any]:
        return e2_metrics(
            self.worker_reports,
            self.ledger,
            source_catalog,
            wall_clock_latency_ms=self.wall_clock_latency_ms,
            required_evidence_groups=required_evidence_groups,
            required_source_families=required_source_families,
        )


class _WorkerBudgetState:
    """Per-worker counters that also consume the shared parent hard budget."""

    def __init__(self, parent: Any, spec: WorkerCapabilitySpec) -> None:
        self._parent = parent
        self._spec = spec
        self.started_at = monotonic()
        self.provider_calls_used = 0
        self.tool_executions_used = 0
        self.input_tokens_used: int | None = 0
        self.output_tokens_used: int | None = 0
        self.total_tokens_used: int | None = 0
        self._usage_records = 0
        self.parent_budget_denials = 0
        self.worker_budget_denials = 0

    @property
    def config(self) -> RunBudgetConfig:
        return self._parent.config

    @property
    def elapsed_ms(self) -> float:
        return (monotonic() - self.started_at) * 1000

    @property
    def token_counts_complete(self) -> bool:
        return self.provider_calls_used == self._usage_records

    def guard_provider(self) -> float | None:
        if self.provider_calls_used >= self._spec.max_model_turns:
            self.worker_budget_denials += 1
            raise BudgetDenied("worker_model_turn_budget_exceeded")
        try:
            remaining = self._parent.guard_provider()
        except BudgetDenied:
            self.parent_budget_denials += 1
            raise
        self.provider_calls_used += 1
        return remaining

    def guard_tool(self) -> None:
        if self.tool_executions_used >= self._spec.max_tool_calls:
            self.worker_budget_denials += 1
            raise BudgetDenied("worker_tool_call_budget_exceeded")
        try:
            self._parent.guard_tool()
        except BudgetDenied:
            self.parent_budget_denials += 1
            raise
        self.tool_executions_used += 1

    def record_usage(self, usage: Any) -> None:
        self._usage_records += 1
        self._parent.record_usage(usage)
        if usage is None:
            self.input_tokens_used = None
            self.output_tokens_used = None
            self.total_tokens_used = None
            return
        self.input_tokens_used = _add(self.input_tokens_used, usage.input_tokens)
        self.output_tokens_used = _add(self.output_tokens_used, usage.output_tokens)
        self.total_tokens_used = _add(self.total_tokens_used, usage.total_tokens)


class E2HeterogeneousSequentialRunner:
    """Fixed-pool, bounded, sequential capability-aware worker runner."""

    architecture_id = "L3_HETEROGENEOUS_SEQUENTIAL"
    scheduler = "sequential-v1"
    max_workers = 3
    _supported_tools = frozenset({"search_knowledge"})

    def __init__(
        self,
        *,
        worker_models: Mapping[E2WorkerRole | str, AgentModel],
        retriever: Any,
        retriever_profile_id: str,
        source_catalog: CapabilitySourceCatalog,
        capabilities: Sequence[WorkerCapabilitySpec],
        component_manifest: ComponentManifest | None = None,
    ) -> None:
        self.worker_models = {E2WorkerRole(key): value for key, value in worker_models.items()}
        self.retriever = retriever
        self.retriever_profile_id = retriever_profile_id
        self.source_catalog = source_catalog
        self.capabilities = tuple(sorted(capabilities, key=lambda item: item.role.value))
        if not 1 <= len(self.capabilities) <= self.max_workers:
            raise ValueError("E2 supports one to three fixed worker capabilities")
        if len({item.role for item in self.capabilities}) != len(self.capabilities):
            raise ValueError("E2 capability registry must have at most one contract per role")
        for capability in self.capabilities:
            if capability.retriever_profile_id != retriever_profile_id:
                raise ValueError("worker capability retriever profile does not match runtime")
            unsupported = set(capability.tool_ids) - self._supported_tools
            if unsupported:
                raise ValueError(f"E2 tool is not implemented: {sorted(unsupported)}")
        self.capability_by_role = {item.role: item for item in self.capabilities}
        for role, capability in self.capability_by_role.items():
            if capability.eligibility == CapabilityEligibility.ELIGIBLE and role not in self.worker_models:
                raise ValueError(f"eligible worker model is missing for role {role.value}")
        expected_refs = tuple(
            sorted(
                (item.manifest_ref() for item in self.capabilities),
                key=lambda item: item.capability_id,
            )
        )
        if component_manifest is not None and component_manifest.capability_contracts != expected_refs:
            raise ValueError("ComponentManifest is missing the frozen E2 capability hashes")
        if (
            component_manifest is not None
            and component_manifest.source_catalog_hash != source_catalog.catalog_hash
        ):
            raise ValueError("ComponentManifest is missing the resolved source-catalog hash")
        self.component_manifest = component_manifest
        self.capability_manifest_hash = _hash_json(
            {
                "source_catalog_hash": source_catalog.catalog_hash,
                "capabilities": [
                    {"capability_id": item.capability_id, "contract_hash": item.contract_hash}
                    for item in self.capabilities
                ],
            }
        )

    def run(
        self,
        question: str,
        tasks: Sequence[E2Task],
        *,
        initial_evidence: Sequence[Evidence],
        runtime: RunContext,
    ) -> E2SequentialResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty")
        if (
            self.component_manifest is not None
            and runtime.identity.component_manifest_hash != self.component_manifest.manifest_hash
        ):
            raise ValueError("RunContext identity does not match the E2 ComponentManifest")
        task_rows = tuple(tasks)
        if len(task_rows) > self.max_workers:
            raise ValueError("E2 task count exceeds the fixed three-worker limit")
        if len({item.task_id for item in task_rows}) != len(task_rows):
            raise ValueError("task IDs must be unique")
        if len({item.role for item in task_rows}) != len(task_rows):
            raise ValueError("E2 dispatch permits at most one task per fixed role")
        started = monotonic()
        ledger = EvidenceLedgerV2()
        ledger.add_initial(initial_evidence, source_catalog=self.source_catalog)
        reports: list[E2WorkerReport] = []
        if runtime.trace is not None:
            runtime.trace.emit(
                TraceEventType.E2_CAPABILITY_BOUND,
                architecture_id=self.architecture_id,
                scheduler=self.scheduler,
                capability_manifest_hash=self.capability_manifest_hash,
                source_catalog_hash=self.source_catalog.catalog_hash,
                component_manifest_hash=(
                    self.component_manifest.manifest_hash if self.component_manifest else None
                ),
                capability_count=len(self.capabilities),
            )

        for task in task_rows:
            capability = self.capability_by_role.get(task.role)
            worker_id = f"e2-worker-{task.role.value}"
            if capability is None:
                raise ValueError(f"no fixed capability exists for role {task.role.value}")
            if capability.eligibility != CapabilityEligibility.ELIGIBLE:
                reports.append(
                    self._skipped_report(task, worker_id, capability, "capability_not_yet_eligible")
                )
                continue
            report = self._run_worker(
                question,
                task,
                worker_id,
                capability,
                initial_evidence,
                ledger,
                runtime,
            )
            reports.append(report)

        latency = (monotonic() - started) * 1000
        if runtime.trace is not None:
            runtime.trace.emit(
                TraceEventType.E2_EVIDENCE_LEDGER,
                unique_source_count=len(ledger.records),
                observation_count=sum(len(item.observations) for item in ledger.records),
            )
        return E2SequentialResult(
            self.architecture_id,
            self.scheduler,
            self.capability_manifest_hash,
            self.component_manifest.manifest_hash if self.component_manifest else None,
            tuple(reports),
            ledger,
            latency,
        )

    def _run_worker(
        self,
        question: str,
        task: E2Task,
        worker_id: str,
        capability: WorkerCapabilitySpec,
        initial_evidence: Sequence[Evidence],
        ledger: EvidenceLedgerV2,
        parent_runtime: RunContext,
    ) -> E2WorkerReport:
        worker_runtime, worker_budget = _worker_runtime(parent_runtime, capability, worker_id)
        scoped_retriever = CapabilityScopedRetriever(
            self.retriever, capability, self.source_catalog
        )
        scoped_tool = CapabilityScopedSearchTool(scoped_retriever)
        registry = ToolRegistry([scoped_tool])
        visible_initial = tuple(
            item for item in initial_evidence if scoped_retriever.allows(item.source_id)
        )
        worker_session = AgentSession()
        prompt = (
            f"Original question:\n{question.strip()}\n\n"
            f"Assigned objective:\n{task.objective.strip()}\n\n"
            f"Bound capability: {capability.capability_id}.\n"
            f"Authority boundary: {capability.authority_scope}\n"
            "Return only claims supported by evidence present in this run's transcript. "
            "The runtime checks source provenance and the retrieval tool enforces this scope."
        )
        loop = AgentLoop(
            self.worker_models[capability.role],
            registry,
            config=AgentLoopConfig(
                max_model_turns=capability.max_model_turns,
                max_tool_calls=capability.max_tool_calls,
            ),
            runtime=worker_runtime,
        )
        started = monotonic()
        if parent_runtime.trace is not None:
            parent_runtime.trace.emit(
                TraceEventType.E2_WORKER_STARTED,
                worker_id=worker_id,
                role=capability.role.value,
                capability_id=capability.capability_id,
                capability_hash=capability.contract_hash,
                task_id=task.task_id,
                allowed_source_families=list(capability.allowed_source_families),
                tool_ids=list(capability.tool_ids),
            )
        try:
            result = loop.run(
                prompt,
                visible_initial,
                session=worker_session,
                runtime=worker_runtime,
            )
            report = self._verified_report(
                task,
                worker_id,
                capability,
                result,
                worker_budget,
                scoped_retriever,
                ledger,
                (monotonic() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - a worker exception is contained to its task
            report = E2WorkerReport(
                worker_id=worker_id,
                role=capability.role,
                capability_id=capability.capability_id,
                capability_hash=capability.contract_hash,
                task_id=task.task_id,
                status=E2WorkerStatus.FAILED,
                provider_calls=worker_budget.provider_calls_used,
                tool_proposals=0,
                tool_executions=worker_budget.tool_executions_used,
                input_tokens=_known_tokens(worker_budget, "input_tokens_used"),
                output_tokens=_known_tokens(worker_budget, "output_tokens_used"),
                elapsed_ms=(monotonic() - started) * 1000,
                stop_reason="worker_exception",
                error_code=_safe_error_code(exc),
            )
        if parent_runtime.trace is not None:
            parent_runtime.trace.emit(
                TraceEventType.E2_WORKER_FINISHED,
                worker_id=report.worker_id,
                task_id=report.task_id,
                status=report.status.value,
                error_code=report.error_code,
                provider_calls=report.provider_calls,
                tool_executions=report.tool_executions,
            )
            parent_runtime.trace.emit(
                TraceEventType.E2_WORKER_REPORT,
                worker_id=report.worker_id,
                task_id=report.task_id,
                status=report.status.value,
                productive=report.productive,
                observed_source_count=len(report.observed_source_ids),
                new_unique_source_count=len(report.new_unique_source_ids),
                capability_hash=report.capability_hash,
            )
        return report

    def _verified_report(
        self,
        task: E2Task,
        worker_id: str,
        capability: WorkerCapabilitySpec,
        result: AgentRunResult,
        budget: _WorkerBudgetState,
        scoped_retriever: CapabilityScopedRetriever,
        ledger: EvidenceLedgerV2,
        elapsed_ms: float,
    ) -> E2WorkerReport:
        observed = tuple(result.observed_evidence)
        observed_ids = tuple(dict.fromkeys(item.source_id for item in observed))
        observed_id_set = set(observed_ids)
        claims = tuple(result.claims)
        citation_ids = tuple(
            dict.fromkeys(source_id for claim in claims for source_id in claim.citation_ids)
        )
        invalid_claim = any(
            not claim.citation_ids
            or any(
                source_id not in observed_id_set or not scoped_retriever.allows(source_id)
                for source_id in claim.citation_ids
            )
            for claim in claims
        )
        parent_budget_denied = budget.parent_budget_denials > 0
        worker_budget_denied = budget.worker_budget_denials > 0 or result.stop_reason in {
            StopReason.MAX_MODEL_TURNS,
            StopReason.MAX_TOOL_CALLS,
        }
        normal_stop = result.stop_reason in {StopReason.FINAL, StopReason.ABSTAIN, None}
        completed = normal_stop and not invalid_claim and not parent_budget_denied and not worker_budget_denied
        error_code = None
        if invalid_claim:
            error_code = "worker_citation_provenance_violation"
        elif parent_budget_denied:
            error_code = "parent_budget_exhausted"
        elif worker_budget_denied:
            error_code = "worker_budget_exhausted"
        elif not normal_stop:
            error_code = result.stop_reason.value if result.stop_reason else "worker_failed"
        if not completed:
            claims = ()
            citation_ids = ()

        new_unique = tuple(
            source_id for source_id in observed_ids if not ledger.contains(source_id)
        ) if completed else ()
        if completed:
            tool_evidence_ids = {
                item.source_id for item in result.recovery_ranked_evidence
            }
            # The ledger is mutated only after report/citation validation. Failed
            # workers therefore cannot contribute partial evidence to a later Lead.
            for origin, rows in (
                ("search_knowledge", result.recovery_ranked_evidence),
                (
                    "initial_evidence",
                    tuple(item for item in observed if item.source_id not in tool_evidence_ids),
                ),
            ):
                ledger.add_worker(
                    rows,
                    worker_id=worker_id,
                    role=capability.role,
                    capability_id=capability.capability_id,
                    task_id=task.task_id,
                    allowed=scoped_retriever.allows,
                    retrieval_tool_origin=origin,
                )

        return E2WorkerReport(
            worker_id=worker_id,
            role=capability.role,
            capability_id=capability.capability_id,
            capability_hash=capability.contract_hash,
            task_id=task.task_id,
            status=E2WorkerStatus.COMPLETED if completed else E2WorkerStatus.FAILED,
            claims=claims,
            citation_ids=citation_ids,
            observed_source_ids=observed_ids,
            new_unique_source_ids=new_unique,
            provider_calls=budget.provider_calls_used,
            tool_proposals=result.state.tool_proposals_used,
            tool_executions=budget.tool_executions_used,
            input_tokens=_known_tokens(budget, "input_tokens_used"),
            output_tokens=_known_tokens(budget, "output_tokens_used"),
            elapsed_ms=elapsed_ms,
            stop_reason=result.stop_reason.value if result.stop_reason else None,
            error_code=error_code,
            verified=completed,
            session_id_sha256=_sha256(result.state.session.session_id),
        )

    @staticmethod
    def _skipped_report(
        task: E2Task,
        worker_id: str,
        capability: WorkerCapabilitySpec,
        reason: str,
    ) -> E2WorkerReport:
        return E2WorkerReport(
            worker_id=worker_id,
            role=capability.role,
            capability_id=capability.capability_id,
            capability_hash=capability.contract_hash,
            task_id=task.task_id,
            status=E2WorkerStatus.SKIPPED,
            stop_reason=reason,
            error_code=reason,
        )


def e2_metrics(
    reports: Sequence[E2WorkerReport],
    ledger: EvidenceLedgerV2,
    source_catalog: CapabilitySourceCatalog,
    *,
    wall_clock_latency_ms: float,
    required_evidence_groups: Sequence[Sequence[str]] = (),
    required_source_families: Sequence[str] = (),
) -> dict[str, Any]:
    """Deterministic metric schema; eval annotations stay outside the runner."""

    worker_rows = tuple(reports)
    delegated = tuple(item for item in worker_rows if item.status != E2WorkerStatus.SKIPPED)
    completed = tuple(item for item in delegated if item.completed)
    productive = tuple(item for item in delegated if item.productive)
    verified_completed = tuple(item for item in completed if item.verified)
    source_ids = set(ledger.source_ids)
    family_by_source = source_catalog.by_id

    group_rows = tuple(tuple(group) for group in required_evidence_groups)
    covered_groups = sum(bool(source_ids.intersection(group)) for group in group_rows)
    required_families = tuple(dict.fromkeys(required_source_families))
    covered_families = {
        family_by_source[source_id].source_family
        for source_id in source_ids
        if source_id in family_by_source
    }

    role_sets = [set(item.observed_source_ids) for item in verified_completed]
    overlap, unique = _worker_diversity(role_sets)
    attempts = sum(item.provider_calls for item in worker_rows)
    tool_executions = sum(item.tool_executions for item in worker_rows)
    input_tokens = _sum_optional(item.input_tokens for item in worker_rows)
    output_tokens = _sum_optional(item.output_tokens for item in worker_rows)
    sum_worker_ms = sum(item.elapsed_ms for item in worker_rows)

    def ratio(numerator: int, denominator: int) -> dict[str, Any]:
        return {
            "value": numerator / denominator if denominator else None,
            "numerator": numerator,
            "denominator": denominator,
            "version": "e2-metrics-v1",
        }

    def annotated_ratio(numerator: int, denominator: int) -> dict[str, Any]:
        return {
            "value": numerator / denominator if denominator else None,
            "numerator": numerator,
            "denominator": denominator,
            "version": "e2-metrics-v1",
            "annotation_source": "evaluation_only",
        }

    return {
        "schema_version": "e2-metrics-v1",
        "EvidenceGroupCoverage": annotated_ratio(covered_groups, len(group_rows)),
        "SourceFamilyCoverage": annotated_ratio(
            len(covered_families.intersection(required_families)),
            len(required_families),
        ),
        "WorkerCompletionRate": ratio(len(completed), len(delegated)),
        "ProductiveWorkerRate": ratio(len(productive), len(completed)),
        "WorkerEvidenceOverlap": {
            "value": overlap,
            "numerator": None,
            "denominator": None,
            "version": "e2-metrics-v1",
        },
        "WorkerUniqueEvidenceContribution": {
            "value": unique,
            "numerator": None,
            "denominator": None,
            "version": "e2-metrics-v1",
        },
        "DelegationPrecision": ratio(len(productive), len(delegated)),
        "ProviderCalls": {"value": attempts, "version": "e2-metrics-v1"},
        "ToolExecutions": {"value": tool_executions, "version": "e2-metrics-v1"},
        "InputTokens": {"value": input_tokens, "version": "e2-metrics-v1"},
        "OutputTokens": {"value": output_tokens, "version": "e2-metrics-v1"},
        "WallClockLatencyMs": {
            "value": round(wall_clock_latency_ms, 3),
            "version": "e2-metrics-v1",
        },
        "SumWorkerExecutionMs": {
            "value": round(sum_worker_ms, 3),
            "version": "e2-metrics-v1",
        },
    }


def _worker_runtime(
    parent: RunContext, spec: WorkerCapabilitySpec, worker_id: str
) -> tuple[RunContext, _WorkerBudgetState]:
    budget = _WorkerBudgetState(parent.budget, spec)
    runtime = RunContext(
        identity=parent.identity,
        budget=budget,  # type: ignore[arg-type] - the provider/tool protocols are structural
        metadata={
            **parent.metadata,
            "e2_worker_id": worker_id,
            "e2_capability_id": spec.capability_id,
            "e2_capability_hash": spec.contract_hash,
            "memory_policy": "OFF",
        },
        trace=parent.trace,
    )
    return runtime, budget


def _worker_diversity(role_sets: Sequence[set[str]]) -> tuple[float | None, float | None]:
    if len(role_sets) < 2:
        return None, None
    union = set().union(*role_sets)
    if not union:
        return 0.0, 0.0
    intersections = 0
    pair_unions = 0
    for index, left in enumerate(role_sets):
        for right in role_sets[index + 1 :]:
            intersections += len(left.intersection(right))
            pair_unions += len(left.union(right))
    occurrence_count = Counter(source_id for values in role_sets for source_id in values)
    return (
        intersections / pair_unions if pair_unions else 0.0,
        sum(value == 1 for value in occurrence_count.values()) / len(union),
    )


def _known_tokens(budget: _WorkerBudgetState, field_name: str) -> int | None:
    if not budget.token_counts_complete:
        return None
    return getattr(budget, field_name)


def _sum_optional(values: Sequence[int | None] | Any) -> int | None:
    items = tuple(values)
    if any(item is None for item in items):
        return None
    return sum(items)


def _add(current: int | None, incoming: int | None) -> int | None:
    return None if current is None or incoming is None else current + incoming


def _safe_error_code(exc: Exception) -> str:
    return getattr(exc, "code", None) or type(exc).__name__.lower()


def _sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "E2HeterogeneousSequentialRunner",
    "E2SequentialResult",
    "E2Task",
    "E2WorkerReport",
    "E2WorkerStatus",
    "EvidenceLedgerRecord",
    "EvidenceLedgerV2",
    "EvidenceObservation",
    "e2_metrics",
]
