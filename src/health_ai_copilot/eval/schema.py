"""Versioned, JSON-compatible contracts for the M7 evaluation runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..runtime.components import RuntimeComponents


class EvalError(RuntimeError):
    """Base error for evaluation configuration, execution, or artifacts."""


class EvalConfigurationError(EvalError):
    """The requested evaluation cannot start with the supplied configuration."""


class EvalTargetKind(StrEnum):
    PIPELINE = "pipeline"
    RETRIEVER = "retriever"
    POLICY = "policy"
    VERIFIER = "verifier"
    REPLAY = "replay"
    DETERMINISTIC_GATE = "deterministic_gate"


class EvalExecutionMode(StrEnum):
    OFFLINE = "offline"
    LIVE = "live"
    REPLAY = "replay"


class CaseRunStatus(StrEnum):
    COMPLETE = "complete"
    ERROR = "error"
    SKIPPED = "skipped"


class GraderStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNGRADED = "ungraded"
    ERROR = "error"


class FailureStage(StrEnum):
    INPUT = "input"
    SAFETY = "safety"
    RETRIEVAL = "retrieval"
    ACTION_SELECTION = "action_selection"
    POLICY = "policy"
    TOOL_EXECUTION = "tool_execution"
    GROUNDING = "grounding"
    TERMINATION = "termination"
    BUDGET = "budget"
    PROVIDER = "provider"
    REPLAY = "replay"
    EVAL_INFRA = "eval_infra"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_hash(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class EvalCase:
    """Narrow wrapper; suite-specific payload fields remain untouched."""

    case_id: str
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("eval case_id must be non-empty")
        if not isinstance(self.payload, Mapping):
            raise TypeError("eval case payload must be a mapping")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "metadata", _mapping(self.metadata))

    @property
    def question(self) -> str | None:
        value = self.payload.get("question")
        return value if isinstance(value, str) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvalSuite:
    suite_id: str
    version: str
    target_kind: EvalTargetKind
    dataset_path: str
    supported_execution_modes: tuple[EvalExecutionMode, ...]
    default_profile_id: str | None
    grader_ids: tuple[str, ...]
    metric_definition_version: str
    content_policy: str = "metadata_only"
    public_content_allowed: bool = False
    expected_dataset_sha256: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.suite_id.strip() or not self.version.strip():
            raise ValueError("suite_id and version must be non-empty")
        object.__setattr__(self, "target_kind", EvalTargetKind(self.target_kind))
        modes = tuple(EvalExecutionMode(item) for item in self.supported_execution_modes)
        if not modes:
            raise ValueError("suite must support at least one execution mode")
        object.__setattr__(self, "supported_execution_modes", modes)
        object.__setattr__(self, "grader_ids", tuple(self.grader_ids))
        if self.expected_dataset_sha256 is not None and len(self.expected_dataset_sha256) != 64:
            raise ValueError("expected_dataset_sha256 must be a SHA-256 digest or null")
        if not isinstance(self.public_content_allowed, bool):
            raise TypeError("public_content_allowed must be boolean")
        object.__setattr__(self, "provenance", _mapping(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "version": self.version,
            "target_kind": self.target_kind.value,
            "dataset_path": self.dataset_path,
            "supported_execution_modes": [item.value for item in self.supported_execution_modes],
            "default_profile_id": self.default_profile_id,
            "grader_ids": list(self.grader_ids),
            "metric_definition_version": self.metric_definition_version,
            "content_policy": self.content_policy,
            "public_content_allowed": self.public_content_allowed,
            "expected_dataset_sha256": self.expected_dataset_sha256,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class EvalRunSpec:
    """Canonical identity of one unified M7 run."""

    suite_id: str
    execution_mode: EvalExecutionMode
    profile_id: str | None
    dataset_path: str
    dataset_sha256: str
    trials: int = 1
    trace_content_policy: str = "metadata_only"
    budget_overrides: Mapping[str, Any] = field(default_factory=dict)
    output_root: str = "runs/m7"
    replay_source: str | None = None
    replay_run_dir: str | None = None
    code_commit: str | None = None
    component_manifest_hash: str | None = None
    profile_config_hash: str | None = None
    schema_version: str = "m7-run-spec-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_mode", EvalExecutionMode(self.execution_mode))
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if len(self.dataset_sha256) != 64:
            raise ValueError("dataset_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "budget_overrides", _mapping(self.budget_overrides))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "execution_mode": self.execution_mode.value,
            "profile_id": self.profile_id,
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "trials": self.trials,
            "trace_content_policy": self.trace_content_policy,
            "budget_overrides": dict(self.budget_overrides),
            "output_root": self.output_root,
            "replay_source": self.replay_source,
            "replay_run_dir": self.replay_run_dir,
            "code_commit": self.code_commit,
            "component_manifest_hash": self.component_manifest_hash,
            "profile_config_hash": self.profile_config_hash,
        }

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def spec_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def bind_runtime(self, components: RuntimeComponents) -> EvalRunSpec:
        return replace(
            self,
            profile_id=components.profile.profile_id,
            code_commit=components.component_manifest.code_commit,
            component_manifest_hash=components.manifest_hash,
            profile_config_hash=components.profile.config_hash,
        )


@dataclass(frozen=True)
class CaseRunRecord:
    suite_id: str
    case_id: str
    trial: int
    run_id: str | None
    execution_mode: EvalExecutionMode
    profile_id: str | None
    component_manifest_hash: str | None
    code_commit: str | None
    status: CaseRunStatus
    route: str | None = None
    agent_stop_reason: str | None = None
    harness_disposition: str | None = None
    provider_calls_used: int | None = None
    tool_executions_used: int | None = None
    input_tokens_used: int | None = None
    output_tokens_used: int | None = None
    total_tokens_used: int | None = None
    elapsed_ms: float | None = None
    trace_reference: str | None = None
    observed: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    # Evaluation identity covers the whole eval specification.  ``run_id`` is
    # deliberately reserved for the execution-local RunContext identity.
    eval_run_id: str | None = None
    eval_spec_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_mode", EvalExecutionMode(self.execution_mode))
        object.__setattr__(self, "status", CaseRunStatus(self.status))
        object.__setattr__(self, "observed", _mapping(self.observed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "case_id": self.case_id,
            "trial": self.trial,
            "run_id": self.run_id,
            "execution_run_id": self.run_id,
            "eval_run_id": self.eval_run_id,
            "eval_spec_hash": self.eval_spec_hash,
            "execution_mode": self.execution_mode.value,
            "profile_id": self.profile_id,
            "component_manifest_hash": self.component_manifest_hash,
            "code_commit": self.code_commit,
            "status": self.status.value,
            "route": self.route,
            "agent_stop_reason": self.agent_stop_reason,
            "harness_disposition": self.harness_disposition,
            "provider_calls_used": self.provider_calls_used,
            "tool_executions_used": self.tool_executions_used,
            "input_tokens_used": self.input_tokens_used,
            "output_tokens_used": self.output_tokens_used,
            "total_tokens_used": self.total_tokens_used,
            "elapsed_ms": self.elapsed_ms,
            "trace_reference": self.trace_reference,
            "observed": dict(self.observed),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class TrajectoryRecord:
    suite_id: str
    case_id: str
    trial: int
    schema_version: str = "trajectory_v1"
    content_policy: str = "metadata_only"
    events: tuple[Mapping[str, Any], ...] = ()
    eval_run_id: str | None = None
    execution_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "case_id": self.case_id,
            "trial": self.trial,
            "content_policy": self.content_policy,
            "eval_run_id": self.eval_run_id,
            "execution_run_id": self.execution_run_id,
            "events": [dict(event) for event in self.events],
        }


@dataclass(frozen=True)
class GraderResult:
    grader_id: str
    grader_version: str
    case_id: str
    trial: int
    status: GraderStatus
    score: float | None = None
    expected_summary: Any = None
    observed_summary: Any = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", GraderStatus(self.status))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "grader_id": self.grader_id,
            "grader_version": self.grader_version,
            "case_id": self.case_id,
            "trial": self.trial,
            "status": self.status.value,
            "score": self.score,
            "expected_summary": self.expected_summary,
            "observed_summary": self.observed_summary,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    definition_version: str
    value: float | int | None
    numerator: float | int | None
    denominator: int
    unit: str
    scope: str
    aggregation: str

    def __post_init__(self) -> None:
        if self.denominator < 0:
            raise ValueError("metric denominator must not be negative")
        if self.denominator == 0 and self.value is not None:
            raise ValueError("zero-denominator metric value must be null")

    @classmethod
    def ratio(
        cls,
        metric_id: str,
        definition_version: str,
        numerator: int,
        denominator: int,
        *,
        unit: str = "rate",
        scope: str = "all",
    ) -> MetricResult:
        return cls(
            metric_id,
            definition_version,
            numerator / denominator if denominator else None,
            numerator,
            denominator,
            unit,
            scope,
            "numerator/denominator",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "definition_version": self.definition_version,
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "unit": self.unit,
            "scope": self.scope,
            "aggregation": self.aggregation,
        }


@dataclass(frozen=True)
class FailureRecord:
    suite_id: str
    case_id: str
    trial: int
    stage: FailureStage
    failure_code: str
    source: str
    expected: Any = None
    observed: Any = None
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", FailureStage(self.stage))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "case_id": self.case_id,
            "trial": self.trial,
            "stage": self.stage.value,
            "failure_code": self.failure_code,
            "source": self.source,
            "expected": self.expected,
            "observed": self.observed,
            "reason_codes": list(self.reason_codes),
        }
