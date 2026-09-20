"""Bounded M8 Agent Team orchestration.

This module deliberately keeps the team small and runtime-owned.  Models can
propose a decision or a worker report, but they never allocate runtime IDs,
change task state, or decide which evidence is trusted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol
from uuid import uuid4

from .agent.loop import AgentLoop, AgentLoopConfig, AgentRunResult
from .agent.messages import ToolResultMessage
from .agent.model import AgentModel
from .agent.tools import ToolRegistry
from .contracts import Evidence
from .knowledge.scope import KnowledgeScope
from .policy.evidence import EvidencePolicy
from .runtime.context import RunContext, call_with_optional_runtime
from .runtime.provider import (
    ProviderCallKind,
    ProviderExecutor,
    ProviderFailure,
    ProviderFailureKind,
    ProviderRequest,
)
from .runtime.tools import LiveToolRunner, ToolRunner
from .runtime.trace import TraceEventType
from .verification.grounding import GroundedClaim


class TeamRole(StrEnum):
    EVIDENCE = "evidence"
    GUIDELINE = "guideline"


TEAM_TOPOLOGY = "star-supervisor-v1"
TEAM_SCHEDULER = "sequential-v1"

# These contracts are intentionally role-specific even when the builder uses
# the same checkpoint, provider executor, and tool set for both workers.
TEAM_ROLE_SYSTEM_CONTRACTS: dict[TeamRole, str] = {
    TeamRole.EVIDENCE: (
        "You are the Evidence Worker. Your role contract is factual evidence acquisition, "
        "source coverage, and source-level observation. Search only for facts needed by the "
        "assigned objective, record which sources were actually observed, and return claims "
        "grounded in those sources. Avoid recommendation synthesis: do not turn facts into "
        "clinical advice, treatment instructions, or guideline recommendations."
    ),
    TeamRole.GUIDELINE: (
        "You are the Guideline Worker. Your role contract is guideline and recommendation "
        "context, including publisher and jurisdiction distinctions. Identify the issuing "
        "body, scope, population, and applicable jurisdiction when present. Avoid unsupported "
        "factual expansion: do not invent facts beyond observed sources or silently generalize "
        "a recommendation outside its stated context."
    ),
}


TEAM_ROLE_CONTRACT_IDS: dict[TeamRole, str] = {
    TeamRole.EVIDENCE: "m8-evidence-worker-v1",
    TeamRole.GUIDELINE: "m8-guideline-worker-v1",
}


class LeadDecisionKind(StrEnum):
    FINAL = "final"
    DELEGATE = "delegate"
    ABSTAIN = "abstain"


class TeamTaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TeamStopReason(StrEnum):
    FINAL = "final"
    ABSTAIN = "abstain"
    MAX_DELEGATION_ROUNDS = "max_delegation_rounds"
    MAX_TASKS = "max_tasks"
    MAX_WORKERS = "max_workers"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LEAD_ERROR = "lead_error"
    WORKER_ERROR = "worker_error"
    INVALID_DELEGATION = "invalid_delegation"
    LEAD_SECOND_DELEGATION = "lead_second_delegation"
    FINAL_VERIFICATION_FAILED = "final_verification_failed"


class TeamMessageKind(StrEnum):
    TASK_ASSIGNMENT = "task_assignment"
    WORKER_REPORT = "worker_report"
    TASK_FAILURE = "task_failure"
    RUNTIME_NOTICE = "runtime_notice"


class EvidenceOrigin(StrEnum):
    INITIAL = "initial"
    WORKER = "worker"


@dataclass(frozen=True)
class LeadTaskProposal:
    """The only task shape a lead model may propose."""

    role: TeamRole | str
    objective: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", TeamRole(self.role))
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError("task objective must be non-empty")

    @classmethod
    def from_value(cls, value: object) -> LeadTaskProposal:
        if not isinstance(value, Mapping):
            raise TypeError("task proposal must be an object")
        if set(value) - {"role", "objective"}:
            raise ValueError("task proposal contains runtime-controlled fields")
        return cls(value.get("role", ""), value.get("objective", ""))


@dataclass(frozen=True)
class LeadDecision:
    action: LeadDecisionKind | str
    claims: tuple[GroundedClaim, ...] = ()
    tasks: tuple[LeadTaskProposal, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", LeadDecisionKind(self.action))
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "tasks", tuple(self.tasks))
        if self.action == LeadDecisionKind.FINAL and not self.claims:
            raise ValueError("final lead decision needs claims")
        if self.action == LeadDecisionKind.DELEGATE and not self.tasks:
            raise ValueError("delegate lead decision needs tasks")
        if self.action != LeadDecisionKind.FINAL and self.claims:
            raise ValueError("only a final decision may contain claims")
        if self.action != LeadDecisionKind.DELEGATE and self.tasks:
            raise ValueError("only a delegate decision may contain tasks")

    @classmethod
    def from_value(cls, value: object) -> LeadDecision:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("lead decision must be an object")
        if set(value) - {"action", "claims", "tasks"}:
            raise ValueError("lead decision contains runtime-controlled fields")
        action = value.get("action")
        if action == LeadDecisionKind.FINAL.value:
            raw_claims = value.get("claims", [])
            if not isinstance(raw_claims, list):
                raise ValueError("final claims must be a list")
            claims = tuple(_claim_from_value(item) for item in raw_claims)
            return cls(action, claims=claims)
        if action == LeadDecisionKind.DELEGATE.value:
            raw_tasks = value.get("tasks", [])
            if not isinstance(raw_tasks, list):
                raise ValueError("delegate tasks must be a list")
            return cls(action, tasks=tuple(LeadTaskProposal.from_value(item) for item in raw_tasks))
        if action == LeadDecisionKind.ABSTAIN.value:
            return cls(action)
        raise ValueError("unknown lead action")


@dataclass
class TeamTask:
    task_id: str
    role: TeamRole
    objective: str
    status: TeamTaskStatus = TeamTaskStatus.PENDING
    worker_id: str | None = None
    error_code: str | None = None

    @property
    def objective_hash(self) -> str:
        return sha256(self.objective.encode("utf-8")).hexdigest()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "role": self.role.value,
            "objective_sha256": self.objective_hash,
            "status": self.status.value,
            "worker_id": self.worker_id,
            "error_code": self.error_code,
        }


class TaskStore:
    """Per-team in-memory task lifecycle owned by the orchestrator."""

    def __init__(self) -> None:
        self._tasks: dict[str, TeamTask] = {}

    def create_task(self, role: TeamRole | str, objective: str) -> TeamTask:
        task = TeamTask(f"task-{uuid4().hex}", TeamRole(role), objective)
        if task.task_id in self._tasks:
            raise RuntimeError("task ID collision")
        self._tasks[task.task_id] = task
        return task

    def assign_task(self, task_id: str, worker_id: str) -> TeamTask:
        task = self._get(task_id)
        self._require(task, TeamTaskStatus.PENDING)
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker ID must be non-empty")
        task.worker_id = worker_id
        return task

    def start_task(self, task_id: str) -> TeamTask:
        task = self._get(task_id)
        self._require(task, TeamTaskStatus.PENDING)
        task.status = TeamTaskStatus.RUNNING
        return task

    def complete_task(self, task_id: str) -> TeamTask:
        task = self._get(task_id)
        self._require(task, TeamTaskStatus.RUNNING)
        task.status = TeamTaskStatus.SUCCEEDED
        return task

    def fail_task(self, task_id: str, error_code: str = "worker_failed") -> TeamTask:
        task = self._get(task_id)
        self._require(task, TeamTaskStatus.RUNNING)
        task.status = TeamTaskStatus.FAILED
        task.error_code = error_code
        return task

    def cancel_task(self, task_id: str, error_code: str | None = None) -> TeamTask:
        task = self._get(task_id)
        self._require(task, TeamTaskStatus.RUNNING)
        task.status = TeamTaskStatus.CANCELLED
        task.error_code = error_code
        return task

    def get_task(self, task_id: str) -> TeamTask:
        return self._get(task_id)

    def list_tasks(self) -> tuple[TeamTask, ...]:
        return tuple(self._tasks.values())

    def _get(self, task_id: str) -> TeamTask:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task: {task_id}") from exc

    @staticmethod
    def _require(task: TeamTask, expected: TeamTaskStatus) -> None:
        if task.status != expected:
            raise RuntimeError(
                f"invalid task transition from {task.status.value}; expected {expected.value}"
            )


@dataclass(frozen=True)
class TeamMessage:
    message_id: str
    sender: str
    recipient: str
    kind: TeamMessageKind
    task_id: str | None
    payload: Mapping[str, Any]


class Mailbox:
    """Typed point-to-point mailbox; worker-to-worker and broadcast are forbidden."""

    def __init__(self) -> None:
        self._messages: list[TeamMessage] = []

    def send(
        self,
        sender: str,
        recipient: str,
        kind: TeamMessageKind | str,
        task_id: str | None,
        payload: Mapping[str, Any],
    ) -> TeamMessage:
        if recipient in {"", "*", "broadcast"}:
            raise ValueError("mailbox broadcast is forbidden")
        if sender.startswith("worker-") and recipient.startswith("worker-"):
            raise ValueError("worker-to-worker messages are forbidden")
        if sender.startswith("worker-") and recipient != "lead":
            raise ValueError("workers may only message the lead")
        if sender == "lead" and not recipient.startswith("worker-"):
            raise ValueError("lead may only assign workers")
        message = TeamMessage(
            f"message-{uuid4().hex}", sender, recipient, TeamMessageKind(kind), task_id, dict(payload)
        )
        self._messages.append(message)
        return message

    def list_messages(self) -> tuple[TeamMessage, ...]:
        return tuple(self._messages)

    def for_recipient(self, recipient: str) -> tuple[TeamMessage, ...]:
        return tuple(message for message in self._messages if message.recipient == recipient)


@dataclass(frozen=True)
class EvidenceProvenance:
    source_id: str
    evidence: Evidence
    origin: EvidenceOrigin
    worker_id: str | None = None
    task_id: str | None = None


class TeamEvidenceLedger:
    """Harness-owned union of evidence and its observation provenance."""

    def __init__(self) -> None:
        self._items: dict[str, EvidenceProvenance] = {}

    def add_initial(self, evidence: Sequence[Evidence]) -> None:
        for item in evidence:
            self._add(item, EvidenceOrigin.INITIAL)

    def add_worker(
        self,
        evidence: Sequence[Evidence],
        *,
        worker_id: str,
        task_id: str,
    ) -> None:
        for item in evidence:
            if item.source_id not in self._items:
                self._add(item, EvidenceOrigin.WORKER, worker_id, task_id)

    def contains(self, source_id: str) -> bool:
        return source_id in self._items

    def get(self, source_id: str) -> EvidenceProvenance:
        return self._items[source_id]

    def list_items(self) -> tuple[EvidenceProvenance, ...]:
        return tuple(self._items.values())

    def final_evidence(self) -> list[Evidence]:
        return [item.evidence for item in self._items.values()]

    def _add(
        self,
        evidence: Evidence,
        origin: EvidenceOrigin,
        worker_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        self._items.setdefault(
            evidence.source_id,
            EvidenceProvenance(evidence.source_id, evidence, origin, worker_id, task_id),
        )


@dataclass(frozen=True)
class WorkerReport:
    task_id: str
    worker_id: str
    role: TeamRole
    status: TeamTaskStatus
    claims: tuple[GroundedClaim, ...] = ()
    citation_ids: tuple[str, ...] = ()
    observed_source_ids: tuple[str, ...] = ()
    recovery_source_ids: tuple[str, ...] = ()
    provider_calls: int = 0
    tool_proposals: int = 0
    tool_executions: int = 0
    stop_reason: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", TeamRole(self.role))
        object.__setattr__(self, "status", TeamTaskStatus(self.status))
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "citation_ids", tuple(self.citation_ids))
        object.__setattr__(self, "observed_source_ids", tuple(self.observed_source_ids))
        object.__setattr__(self, "recovery_source_ids", tuple(self.recovery_source_ids))
        if self.provider_calls < 0 or self.tool_proposals < 0 or self.tool_executions < 0:
            raise ValueError("worker side-effect counts must not be negative")

    @property
    def claim_text_hashes(self) -> tuple[str, ...]:
        return tuple(sha256(claim.text.encode("utf-8")).hexdigest() for claim in self.claims)

    @property
    def final_cited_source_ids(self) -> tuple[str, ...]:
        """Sources explicitly contributed to the worker's final claims."""

        return tuple(dict.fromkeys(self.citation_ids))

    @property
    def completed(self) -> bool:
        """Whether the worker execution reached the normal completion state."""

        return self.status == TeamTaskStatus.SUCCEEDED

    @property
    def productive(self) -> bool:
        """Whether a completed worker returned claims or recovery evidence."""

        return self.completed and bool(self.claims or self.recovery_source_ids)

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "role": self.role.value,
            "status": self.status.value,
            "completed": self.completed,
            "productive": self.productive,
            "claims": [
                {"text": claim.text, "citation_ids": list(claim.citation_ids)}
                for claim in self.claims
            ],
            "citation_ids": list(self.citation_ids),
            "observed_source_ids": list(self.observed_source_ids),
            "recovery_source_ids": list(self.recovery_source_ids),
            "final_cited_source_ids": list(self.final_cited_source_ids),
            "provider_calls": self.provider_calls,
            "tool_proposals": self.tool_proposals,
            "tool_executions": self.tool_executions,
            "stop_reason": self.stop_reason,
            "error_code": self.error_code,
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "role": self.role.value,
            "status": self.status.value,
            "completed": self.completed,
            "productive": self.productive,
            "claim_count": len(self.claims),
            "claim_text_sha256": list(self.claim_text_hashes),
            "citation_count": len(self.citation_ids),
            "observed_source_count": len(self.observed_source_ids),
            "recovery_source_ids": list(self.recovery_source_ids),
            "recovery_source_count": len(self.recovery_source_ids),
            "final_cited_source_ids": list(self.final_cited_source_ids),
            "provider_calls": self.provider_calls,
            "tool_proposals": self.tool_proposals,
            "tool_executions": self.tool_executions,
            "stop_reason": self.stop_reason,
            "error_code": self.error_code,
        }


def _worker_evidence_diversity(
    reports: Sequence[WorkerReport],
    field_name: str,
) -> tuple[float | None, float | None]:
    """Return deterministic overlap and unique-contribution ratios for one field.

    The ratios are computed over role-level observed source IDs.  With fewer
    than two worker roles there is no cross-role comparison, so both values
    are ``None`` and remain outside aggregate denominators.
    """

    by_role: dict[TeamRole, set[str]] = {}
    for report in reports:
        by_role.setdefault(report.role, set()).update(getattr(report, field_name))
    role_sets = tuple(by_role.values())
    if len(role_sets) < 2:
        return None, None

    union = set().union(*role_sets)
    if not union:
        return 0.0, 0.0
    pair_intersection = 0
    pair_union = 0
    for index, left in enumerate(role_sets):
        for right in role_sets[index + 1 :]:
            pair_intersection += len(left & right)
            pair_union += len(left | right)
    overlap = pair_intersection / pair_union if pair_union else 0.0
    occurrence_count: dict[str, int] = {}
    for source_ids in role_sets:
        for source_id in source_ids:
            occurrence_count[source_id] = occurrence_count.get(source_id, 0) + 1
    unique = sum(count == 1 for count in occurrence_count.values()) / len(union)
    return overlap, unique


def worker_evidence_diversity(
    reports: Sequence[WorkerReport],
) -> tuple[float | None, float | None]:
    """Return full worker-context overlap and unique-contribution ratios."""

    return _worker_evidence_diversity(reports, "observed_source_ids")


def worker_recovery_evidence_diversity(
    reports: Sequence[WorkerReport],
) -> tuple[float | None, float | None]:
    """Return recovery-only worker overlap and unique-contribution ratios."""

    return _worker_evidence_diversity(reports, "recovery_source_ids")


@dataclass(frozen=True)
class TeamBudgetConfig:
    max_lead_calls: int = 2
    max_workers_started: int = 2
    max_tasks_created: int = 2
    max_delegation_rounds: int = 1
    max_worker_model_turns: int = 2
    max_worker_tool_calls: int = 1

    def __post_init__(self) -> None:
        for name in (
            "max_lead_calls",
            "max_workers_started",
            "max_tasks_created",
            "max_delegation_rounds",
            "max_worker_model_turns",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_worker_tool_calls < 0:
            raise ValueError("max_worker_tool_calls must not be negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_lead_calls": self.max_lead_calls,
            "max_workers_started": self.max_workers_started,
            "max_tasks_created": self.max_tasks_created,
            "max_delegation_rounds": self.max_delegation_rounds,
            "max_worker_model_turns": self.max_worker_model_turns,
            "max_worker_tool_calls": self.max_worker_tool_calls,
        }


@dataclass
class TeamRunState:
    team_id: str = field(default_factory=lambda: f"team-{uuid4().hex}")
    topology: str = TEAM_TOPOLOGY
    scheduler: str = TEAM_SCHEDULER
    allowed_roles: tuple[str, ...] = tuple(role.value for role in TeamRole)
    lead_calls_used: int = 0
    delegation_rounds_used: int = 0
    workers_started: int = 0
    tasks_created: int = 0
    task_store: TaskStore = field(default_factory=TaskStore)
    mailbox: Mailbox = field(default_factory=Mailbox)
    evidence_ledger: TeamEvidenceLedger = field(default_factory=TeamEvidenceLedger)
    worker_runs: dict[str, AgentRunResult] = field(default_factory=dict)
    worker_reports: list[WorkerReport] = field(default_factory=list)
    stop_reason: TeamStopReason | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "topology": self.topology,
            "scheduler": self.scheduler,
            "allowed_roles": list(self.allowed_roles),
            "worker_roles": [report.to_metadata() for report in self.worker_reports],
        }


@dataclass(frozen=True)
class TeamRunResult:
    state: TeamRunState
    decision: LeadDecision | None
    claims: tuple[GroundedClaim, ...]
    observed_evidence: tuple[Evidence, ...]

    @property
    def stop_reason(self) -> TeamStopReason | None:
        return self.state.stop_reason

    @property
    def reports(self) -> tuple[WorkerReport, ...]:
        return tuple(self.state.worker_reports)


class TeamLeadModel(Protocol):
    def decide(
        self,
        question: str,
        initial_evidence: Sequence[Evidence],
        worker_reports: Sequence[WorkerReport],
        team_state_summary: Mapping[str, Any],
        *,
        runtime: RunContext | None = None,
    ) -> LeadDecision:
        ...


class AgentTeamOrchestrator:
    """One lead, at most one delegation wave, sequential workers."""

    def __init__(
        self,
        lead_model: TeamLeadModel,
        worker_model: AgentModel | Mapping[TeamRole | str, AgentModel],
        tool_registry: ToolRegistry,
        *,
        config: TeamBudgetConfig | None = None,
        evidence_policy: EvidencePolicy | None = None,
        knowledge_scope: KnowledgeScope | None = None,
        tool_runner: ToolRunner | None = None,
        allowed_roles: Sequence[TeamRole | str] | None = None,
        topology: str = TEAM_TOPOLOGY,
        scheduler: str = TEAM_SCHEDULER,
    ) -> None:
        self.lead_model = lead_model
        self.worker_model = worker_model
        self.tool_registry = tool_registry
        self.config = config or TeamBudgetConfig()
        self.evidence_policy = evidence_policy
        self.knowledge_scope = knowledge_scope
        self.tool_runner = tool_runner or LiveToolRunner(tool_registry)
        raw_allowed_roles = tuple(TeamRole) if allowed_roles is None else tuple(allowed_roles)
        self.allowed_roles = tuple(TeamRole(role) for role in raw_allowed_roles)
        if not self.allowed_roles:
            raise ValueError("allowed_roles must not be empty")
        if len(set(self.allowed_roles)) != len(self.allowed_roles):
            raise ValueError("allowed_roles must not contain duplicates")
        if not isinstance(topology, str) or not topology.strip():
            raise ValueError("topology must be a non-empty string")
        if not isinstance(scheduler, str) or not scheduler.strip():
            raise ValueError("scheduler must be a non-empty string")
        self.topology = topology
        self.scheduler = scheduler

    def run(
        self,
        question: str,
        initial_evidence: Sequence[Evidence],
        *,
        runtime: RunContext,
    ) -> TeamRunResult:
        state = TeamRunState(
            topology=self.topology,
            scheduler=self.scheduler,
            allowed_roles=tuple(role.value for role in self.allowed_roles),
        )
        state.evidence_ledger.add_initial(initial_evidence)
        self._trace(runtime, TraceEventType.TEAM_STARTED, state, initial_source_count=len(initial_evidence))
        if initial_evidence:
            self._trace(
                runtime,
                TraceEventType.EVIDENCE_ADDED,
                state,
                origin=EvidenceOrigin.INITIAL.value,
                source_count=len(initial_evidence),
            )

        first = self._lead_call(state, question, initial_evidence, (), runtime)
        if first is None:
            return self._result(state, None)
        if first.action == LeadDecisionKind.FINAL:
            state.stop_reason = TeamStopReason.FINAL
            self._trace(runtime, TraceEventType.TEAM_FINAL_PROPOSED, state, claim_count=len(first.claims))
            return self._result(state, first)
        if first.action == LeadDecisionKind.ABSTAIN:
            state.stop_reason = TeamStopReason.ABSTAIN
            return self._result(state, first)

        if self.config.max_delegation_rounds < 1:
            return self._stop(state, TeamStopReason.MAX_DELEGATION_ROUNDS, runtime)
        state.delegation_rounds_used = 1
        try:
            tasks = self._validate_delegation(first.tasks)
        except ValueError:
            return self._stop(state, TeamStopReason.INVALID_DELEGATION, runtime)
        if len(tasks) > self.config.max_tasks_created:
            return self._stop(state, TeamStopReason.MAX_TASKS, runtime)

        for proposal in tasks:
            if state.tasks_created >= self.config.max_tasks_created:
                return self._stop(state, TeamStopReason.MAX_TASKS, runtime)
            task = state.task_store.create_task(proposal.role, proposal.objective)
            state.tasks_created += 1
            worker_id = f"worker-{proposal.role.value}"
            state.task_store.assign_task(task.task_id, worker_id)
            self._trace(
                runtime,
                TraceEventType.TASK_CREATED,
                state,
                task_id=task.task_id,
                worker_id=worker_id,
                role=task.role.value,
                objective_sha256=task.objective_hash,
            )
            self._trace(
                runtime,
                TraceEventType.TASK_ASSIGNED,
                state,
                task_id=task.task_id,
                worker_id=worker_id,
                role=task.role.value,
            )
            state.mailbox.send(
                "lead",
                worker_id,
                TeamMessageKind.TASK_ASSIGNMENT,
                task.task_id,
                {"role": task.role.value, "objective": task.objective},
            )
            self._trace(
                runtime,
                TraceEventType.MESSAGE_DELIVERED,
                state,
                task_id=task.task_id,
                message_kind=TeamMessageKind.TASK_ASSIGNMENT.value,
                message_count=1,
            )
            if state.workers_started >= self.config.max_workers_started:
                return self._stop(state, TeamStopReason.MAX_WORKERS, runtime)
            self._run_worker(state, task, question, initial_evidence, runtime)
            if state.worker_reports and state.worker_reports[-1].error_code == "team_budget_exhausted":
                return self._stop(state, TeamStopReason.BUDGET_EXHAUSTED, runtime)

        second = self._lead_call(
            state,
            question,
            initial_evidence,
            tuple(state.worker_reports),
            runtime,
        )
        if second is None:
            return self._result(state, None)
        if second.action == LeadDecisionKind.DELEGATE:
            return self._stop(state, TeamStopReason.LEAD_SECOND_DELEGATION, runtime)
        state.stop_reason = (
            TeamStopReason.FINAL if second.action == LeadDecisionKind.FINAL else TeamStopReason.ABSTAIN
        )
        if second.action == LeadDecisionKind.FINAL:
            self._trace(runtime, TraceEventType.TEAM_FINAL_PROPOSED, state, claim_count=len(second.claims))
        return self._result(state, second)

    def _lead_call(
        self,
        state: TeamRunState,
        question: str,
        initial_evidence: Sequence[Evidence],
        reports: Sequence[WorkerReport],
        runtime: RunContext,
    ) -> LeadDecision | None:
        if state.lead_calls_used >= self.config.max_lead_calls:
            return self._stop(state, TeamStopReason.MAX_DELEGATION_ROUNDS, runtime).decision
        state.lead_calls_used += 1
        self._trace(
            runtime,
            TraceEventType.LEAD_DECISION,
            state,
            lead_call=state.lead_calls_used,
            worker_report_count=len(reports),
        )
        try:
            decision = call_with_optional_runtime(
                self.lead_model.decide,
                question,
                tuple(initial_evidence),
                tuple(reports),
                self._summary(state),
                runtime=runtime,
            )
            return LeadDecision.from_value(decision)
        except Exception as exc:  # noqa: BLE001 - lead boundary fails closed
            if _is_budget_failure(exc, runtime):
                self._stop(state, TeamStopReason.BUDGET_EXHAUSTED, runtime)
            else:
                self._stop(state, TeamStopReason.LEAD_ERROR, runtime)
            return None

    def _run_worker(
        self,
        state: TeamRunState,
        task: TeamTask,
        question: str,
        initial_evidence: Sequence[Evidence],
        runtime: RunContext,
    ) -> None:
        worker_id = task.worker_id or f"worker-{task.role.value}"
        state.workers_started += 1
        state.task_store.start_task(task.task_id)
        self._trace(
            runtime,
            TraceEventType.WORKER_STARTED,
            state,
            task_id=task.task_id,
            worker_id=worker_id,
            role=task.role.value,
        )
        self._trace(
            runtime,
            TraceEventType.TASK_STARTED,
            state,
            task_id=task.task_id,
            worker_id=worker_id,
        )
        worker_prompt = (
            f"Original user question:\n{question}\n\n"
            f"Assigned role: {task.role.value}\n"
            f"Task objective:\n{task.objective}"
        )
        model = self._model_for(task.role)
        loop = AgentLoop(
            model,
            self.tool_registry,
            config=AgentLoopConfig(
                max_model_turns=self.config.max_worker_model_turns,
                max_tool_calls=self.config.max_worker_tool_calls,
            ),
            evidence_policy=self.evidence_policy,
            knowledge_scope=self.knowledge_scope,
            tool_runner=self.tool_runner,
        )
        provider_calls_before = runtime.budget.provider_calls_used
        try:
            result = loop.run(worker_prompt, initial_evidence, runtime=runtime)
            state.worker_runs[worker_id] = result
            report = self._report_from_run(
                task,
                worker_id,
                result,
                provider_calls=runtime.budget.provider_calls_used - provider_calls_before,
            )
            if (
                result.stop_reason is not None
                and result.stop_reason.value == "model_error"
                and self._global_budget_exhausted(runtime)
            ) or any(
                isinstance(message, ToolResultMessage)
                and message.result.error is not None
                and message.result.error.code == "runtime_budget_denied"
                for message in result.state.session.messages
            ):
                report = WorkerReport(
                    report.task_id,
                    report.worker_id,
                    report.role,
                    TeamTaskStatus.FAILED,
                    claims=report.claims,
                    citation_ids=report.citation_ids,
                    observed_source_ids=report.observed_source_ids,
                    recovery_source_ids=report.recovery_source_ids,
                    provider_calls=report.provider_calls,
                    tool_proposals=report.tool_proposals,
                    tool_executions=report.tool_executions,
                    stop_reason=report.stop_reason,
                    error_code="team_budget_exhausted",
                )
            observed_recovery = result.recovery_ranked_evidence
            if report.error_code == "worker_citation_provenance_violation":
                state.task_store.fail_task(task.task_id, report.error_code)
                self._deliver_report(state, report, runtime, failure=True)
                return
            if report.status == TeamTaskStatus.SUCCEEDED:
                state.evidence_ledger.add_worker(
                    observed_recovery, worker_id=worker_id, task_id=task.task_id
                )
                if observed_recovery:
                    self._trace(
                        runtime,
                        TraceEventType.EVIDENCE_ADDED,
                        state,
                        origin=EvidenceOrigin.WORKER.value,
                        task_id=task.task_id,
                        worker_id=worker_id,
                        source_count=len(observed_recovery),
                    )
                state.task_store.complete_task(task.task_id)
                self._deliver_report(state, report, runtime, failure=False)
            else:
                state.task_store.fail_task(task.task_id, report.error_code or "worker_failed")
                self._deliver_report(state, report, runtime, failure=True)
        except Exception:  # noqa: BLE001 - worker failure is a task result
            state.task_store.fail_task(task.task_id, "worker_failed")
            report = WorkerReport(
                task.task_id,
                worker_id,
                task.role,
                TeamTaskStatus.FAILED,
                provider_calls=runtime.budget.provider_calls_used - provider_calls_before,
                error_code="worker_failed",
            )
            self._deliver_report(state, report, runtime, failure=True)
        finally:
            self._trace(
                runtime,
                TraceEventType.WORKER_FINISHED,
                state,
                task_id=task.task_id,
                worker_id=worker_id,
                status=state.task_store.get_task(task.task_id).status.value,
            )

    def _report_from_run(
        self,
        task: TeamTask,
        worker_id: str,
        result: AgentRunResult,
        *,
        provider_calls: int = 0,
    ) -> WorkerReport:
        observed_ids = tuple(item.source_id for item in result.observed_evidence)
        recovery_ids = tuple(item.source_id for item in result.recovery_ranked_evidence)
        citation_ids = tuple(source_id for claim in result.claims for source_id in claim.citation_ids)
        error_code = None
        status = TeamTaskStatus.SUCCEEDED
        if result.stop_reason not in {None, result.stop_reason.FINAL, result.stop_reason.ABSTAIN}:
            status = TeamTaskStatus.FAILED
            error_code = result.stop_reason.value if result.stop_reason else "worker_failed"
        if any(source_id not in set(observed_ids) for source_id in citation_ids):
            status = TeamTaskStatus.FAILED
            error_code = "worker_citation_provenance_violation"
        return WorkerReport(
            task.task_id,
            worker_id,
            task.role,
            status,
            claims=tuple(result.claims),
            citation_ids=citation_ids,
            observed_source_ids=observed_ids,
            recovery_source_ids=recovery_ids,
            provider_calls=provider_calls,
            tool_proposals=result.state.tool_proposals_used,
            tool_executions=result.state.tool_calls_used,
            stop_reason=result.stop_reason.value if result.stop_reason else None,
            error_code=error_code,
        )

    def _deliver_report(
        self,
        state: TeamRunState,
        report: WorkerReport,
        runtime: RunContext,
        *,
        failure: bool,
    ) -> None:
        state.worker_reports.append(report)
        kind = TeamMessageKind.TASK_FAILURE if failure else TeamMessageKind.WORKER_REPORT
        message = state.mailbox.send(
            report.worker_id,
            "lead",
            kind,
            report.task_id,
            report.to_model_dict(),
        )
        self._trace(
            runtime,
            TraceEventType.WORKER_REPORT,
            state,
            task_id=report.task_id,
            worker_id=report.worker_id,
            status=report.status.value,
            error_code=report.error_code,
        )
        self._trace(
            runtime,
            TraceEventType.MESSAGE_DELIVERED,
            state,
            task_id=message.task_id,
            message_kind=message.kind.value,
            message_count=1,
        )

    def _model_for(self, role: TeamRole) -> AgentModel:
        if isinstance(self.worker_model, Mapping):
            try:
                return self.worker_model[role]
            except KeyError:
                return self.worker_model[role.value]
        return self.worker_model

    def _validate_delegation(self, tasks: Sequence[LeadTaskProposal]) -> tuple[LeadTaskProposal, ...]:
        proposals = tuple(
            item if isinstance(item, LeadTaskProposal) else LeadTaskProposal.from_value(item)
            for item in tasks
        )
        seen: set[TeamRole] = set()
        for proposal in proposals:
            if proposal.role not in self.allowed_roles:
                raise ValueError(f"worker role is not allowed: {proposal.role.value}")
            if proposal.role in seen:
                raise ValueError("duplicate worker role")
            seen.add(proposal.role)
        return proposals

    @staticmethod
    def _summary(state: TeamRunState) -> dict[str, Any]:
        return {
            "team_id": state.team_id,
            "topology": state.topology,
            "scheduler": state.scheduler,
            "allowed_roles": list(state.allowed_roles),
            "lead_calls_used": state.lead_calls_used,
            "delegation_rounds_used": state.delegation_rounds_used,
            "tasks_created": state.tasks_created,
            "workers_started": state.workers_started,
            "task_statuses": [task.status.value for task in state.task_store.list_tasks()],
            "worker_report_count": len(state.worker_reports),
            "evidence_count": len(state.evidence_ledger.list_items()),
        }

    def _result(self, state: TeamRunState, decision: LeadDecision | None) -> TeamRunResult:
        return TeamRunResult(
            state,
            decision,
            tuple(decision.claims) if decision and decision.action == LeadDecisionKind.FINAL else (),
            tuple(state.evidence_ledger.final_evidence()),
        )

    def _stop(
        self, state: TeamRunState, reason: TeamStopReason, runtime: RunContext
    ) -> TeamRunResult:
        state.stop_reason = reason
        self._trace(runtime, TraceEventType.TEAM_STOPPED, state, reason=reason.value)
        return self._result(state, None)

    @staticmethod
    def _global_budget_exhausted(runtime: RunContext) -> bool:
        cfg = runtime.budget.config
        return bool(
            (cfg.max_provider_calls is not None and runtime.budget.provider_calls_used >= cfg.max_provider_calls)
            or (cfg.max_tool_executions is not None and runtime.budget.tool_executions_used >= cfg.max_tool_executions)
            or (cfg.max_total_tokens is not None and runtime.budget.total_tokens_used is not None and runtime.budget.total_tokens_used >= cfg.max_total_tokens)
        )

    @staticmethod
    def _trace(runtime: RunContext, event_type: TraceEventType, state: TeamRunState, **fields: Any) -> None:
        if runtime.trace is None:
            return
        runtime.trace.emit(event_type, team_id=state.team_id, **fields)


class OpenAICompatibleTeamLeadModel:
    """Provider adapter for the lead's narrow JSON decision contract."""

    def __init__(self, provider_executor: ProviderExecutor, model: str) -> None:
        self.provider_executor = provider_executor
        self.model = model
        self.contract_version = "team-lead-v1"

    def decide(
        self,
        question: str,
        initial_evidence: Sequence[Evidence],
        worker_reports: Sequence[WorkerReport],
        team_state_summary: Mapping[str, Any],
        *,
        runtime: RunContext | None = None,
    ) -> LeadDecision:
        payload = {
            "question": question,
            "initial_evidence": [_evidence_dict(item) for item in initial_evidence],
            "worker_reports": [report.to_model_dict() for report in worker_reports],
            "team_state": dict(team_state_summary),
        }
        request = ProviderRequest.create(
            kind=ProviderCallKind.TEAM_LEAD,
            model=self.model,
            messages=(
                {
                    "role": "system",
                    "content": (
                        "You are the bounded Health-Copilot Team Lead. Return JSON only. "
                        "Choose exactly one action: final, delegate, or abstain. "
                        "A delegate task contains only role evidence|guideline and objective. "
                        "Do not create IDs. A final contains claim-first factual claims and "
                        "citation_ids. The runtime validates all citations and support."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ),
            response_format={"type": "json_object"},
            temperature=0,
            timeout_seconds=30.0,
        )
        try:
            response = self.provider_executor.execute(request, runtime or RunContext.create("team_lead"))
            return LeadDecision.from_value(json.loads(response.content or "{}"))
        except Exception as exc:
            raise RuntimeError("team lead model request failed") from exc


def _claim_from_value(value: object) -> GroundedClaim:
    if not isinstance(value, Mapping):
        raise TypeError("claim must be an object")
    text = value.get("text")
    citation_ids = value.get("citation_ids")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("claim text must be non-empty")
    if not isinstance(citation_ids, list) or not all(isinstance(item, str) for item in citation_ids):
        raise ValueError("claim citation_ids must be a string list")
    return GroundedClaim(text, tuple(citation_ids))


def _evidence_dict(item: Evidence) -> dict[str, Any]:
    return {
        "source_id": item.source_id,
        "title": item.title,
        "excerpt": item.excerpt,
        "score": item.score,
    }


def _is_budget_failure(exc: Exception, runtime: RunContext) -> bool:
    if isinstance(exc, ProviderFailure) and exc.kind == ProviderFailureKind.BUDGET_DENIED:
        return True
    return AgentTeamOrchestrator._global_budget_exhausted(runtime)


__all__ = [
    "TEAM_ROLE_CONTRACT_IDS",
    "TEAM_ROLE_SYSTEM_CONTRACTS",
    "TEAM_SCHEDULER",
    "TEAM_TOPOLOGY",
    "AgentTeamOrchestrator",
    "EvidenceOrigin",
    "EvidenceProvenance",
    "LeadDecision",
    "LeadDecisionKind",
    "LeadTaskProposal",
    "Mailbox",
    "OpenAICompatibleTeamLeadModel",
    "TaskStore",
    "TeamBudgetConfig",
    "TeamEvidenceLedger",
    "TeamLeadModel",
    "TeamMessage",
    "TeamMessageKind",
    "TeamRole",
    "TeamRunResult",
    "TeamRunState",
    "TeamStopReason",
    "TeamTask",
    "TeamTaskStatus",
    "WorkerReport",
    "worker_evidence_diversity",
    "worker_recovery_evidence_diversity",
]
