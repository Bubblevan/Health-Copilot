"""E0 benchmark foundation contracts.

These contracts intentionally stop at dataset identity, normalization and
review metadata.  They do not execute providers, retrievers or judges.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FLOATING_REVISIONS = {"main", "master", "latest", "head", "HEAD"}


class BenchmarkContractError(ValueError):
    """Raised when a source-controlled E0 contract is invalid."""


class DatasetAdmissibility(StrEnum):
    APPROVED = "APPROVED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    RESTRICTED = "RESTRICTED"
    UNAVAILABLE = "UNAVAILABLE"


class PrivacyStatus(StrEnum):
    PUBLIC_SYNTHETIC = "PUBLIC_SYNTHETIC"
    PUBLIC_DEIDENTIFIED = "PUBLIC_DEIDENTIFIED"
    PUBLIC_REALISTIC = "PUBLIC_REALISTIC"
    SENSITIVE_OR_RESTRICTED = "SENSITIVE_OR_RESTRICTED"
    UNKNOWN = "UNKNOWN"


class MedicalContentClass(StrEnum):
    RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
    MEDICAL_QA = "MEDICAL_QA"
    PATIENT_FACING_DIALOGUE = "PATIENT_FACING_DIALOGUE"
    CLINICAL_EXAM_STYLE = "CLINICAL_EXAM_STYLE"
    BIOMEDICAL_LITERATURE = "BIOMEDICAL_LITERATURE"


class ReviewStatus(StrEnum):
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    FROZEN = "frozen"
    REVIEW_PENDING = "review_pending"


def canonical_json(value: object) -> str:
    """Serialize JSON-compatible values with one stable representation."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_hash(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkContractError(f"{field_name} must be a non-empty string")
    return value


def _validate_sha(value: str | None, field_name: str) -> None:
    if value is not None and not SHA256_RE.fullmatch(value):
        raise BenchmarkContractError(f"{field_name} must be a lowercase SHA-256 digest or null")


@dataclass(frozen=True)
class RawArtifactIdentity:
    name: str
    url: str
    sha256: str | None
    size_bytes: int | None = None
    downloaded_at: str | None = None
    content_type: str | None = None
    upstream_etag: str | None = None
    upstream_last_modified: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "raw artifact name")
        _require_text(self.url, "raw artifact URL")
        _validate_sha(self.sha256, "raw artifact sha256")
        if self.size_bytes is not None and (
            not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise BenchmarkContractError("raw artifact size_bytes must be a non-negative integer or null")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RawArtifactIdentity:
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "downloaded_at": self.downloaded_at,
            "content_type": self.content_type,
            "upstream_etag": self.upstream_etag,
            "upstream_last_modified": self.upstream_last_modified,
        }


@dataclass(frozen=True)
class LicenseReview:
    license_name: str | None
    license_url: str | None
    license_source: str | None
    license_review_status: DatasetAdmissibility
    redistribution_allowed: bool | None
    derived_artifact_commit_allowed: bool | None
    notes: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "license_review_status", DatasetAdmissibility(self.license_review_status))
        if not isinstance(self.notes, str):
            raise BenchmarkContractError("license notes must be text")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LicenseReview:
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "license_name": self.license_name,
            "license_url": self.license_url,
            "license_source": self.license_source,
            "license_review_status": self.license_review_status.value,
            "redistribution_allowed": self.redistribution_allowed,
            "derived_artifact_commit_allowed": self.derived_artifact_commit_allowed,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class BenchmarkManifest:
    benchmark_id: str
    benchmark_family: str
    version: str
    source_name: str
    canonical_url: str
    paper_url: str | None
    upstream_revision: str
    adapter_version: str
    raw_artifacts: tuple[RawArtifactIdentity, ...]
    license: LicenseReview
    privacy: PrivacyStatus
    medical_content_policy: tuple[MedicalContentClass, ...]
    normalization_schema_version: str
    split_manifest: Mapping[str, Any]
    gold_contract: Mapping[str, Any]
    metric_protocol: Mapping[str, Any]
    judge_protocol: Mapping[str, Any] | None
    created_with_commit: str
    status: DatasetAdmissibility
    evidence_urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "benchmark_id",
            "benchmark_family",
            "version",
            "source_name",
            "canonical_url",
            "upstream_revision",
            "adapter_version",
            "normalization_schema_version",
            "created_with_commit",
        ):
            _require_text(getattr(self, name), name)
        if self.upstream_revision in FLOATING_REVISIONS:
            raise BenchmarkContractError("upstream_revision must not be floating")
        object.__setattr__(self, "privacy", PrivacyStatus(self.privacy))
        object.__setattr__(
            self,
            "medical_content_policy",
            tuple(MedicalContentClass(item) for item in self.medical_content_policy),
        )
        if not self.medical_content_policy:
            raise BenchmarkContractError("medical_content_policy must not be empty")
        object.__setattr__(self, "status", DatasetAdmissibility(self.status))
        object.__setattr__(self, "raw_artifacts", tuple(self.raw_artifacts))
        object.__setattr__(self, "evidence_urls", tuple(self.evidence_urls))
        if not self.raw_artifacts:
            raise BenchmarkContractError("a benchmark must declare at least one raw artifact")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BenchmarkManifest:
        data = dict(value)
        data["raw_artifacts"] = tuple(
            RawArtifactIdentity.from_dict(item) for item in data.get("raw_artifacts", [])
        )
        data["license"] = LicenseReview.from_dict(data["license"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_family": self.benchmark_family,
            "version": self.version,
            "source_name": self.source_name,
            "canonical_url": self.canonical_url,
            "paper_url": self.paper_url,
            "upstream_revision": self.upstream_revision,
            "adapter_version": self.adapter_version,
            "raw_artifacts": [item.to_dict() for item in self.raw_artifacts],
            "license": self.license.to_dict(),
            "privacy": self.privacy.value,
            "medical_content_policy": [item.value for item in self.medical_content_policy],
            "normalization_schema_version": self.normalization_schema_version,
            "split_manifest": dict(self.split_manifest),
            "gold_contract": dict(self.gold_contract),
            "metric_protocol": dict(self.metric_protocol),
            "judge_protocol": dict(self.judge_protocol) if self.judge_protocol else None,
            "created_with_commit": self.created_with_commit,
            "status": self.status.value,
            "evidence_urls": list(self.evidence_urls),
        }

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def validate_freeze_requirements(self) -> list[str]:
        errors: list[str] = []
        if not self.canonical_url or not self.upstream_revision:
            errors.append("missing canonical source identity")
        if any(item.sha256 is None for item in self.raw_artifacts):
            errors.append("unknown raw hash")
        if self.status == DatasetAdmissibility.APPROVED and (
            self.license.license_review_status != DatasetAdmissibility.APPROVED
        ):
            errors.append("approved benchmark has unresolved license review")
        if self.judge_protocol and self.judge_protocol.get("reference_implementation_revision") in {
            "main",
            "latest",
            "HEAD",
        }:
            errors.append("floating evaluator revision")
        return errors


@dataclass(frozen=True)
class NormalizedDatasetIdentity:
    benchmark_id: str
    raw_artifact_sha256: tuple[str, ...]
    adapter_version: str
    schema_version: str
    case_count: int
    normalized_sha256: str | None
    split_manifest_sha256: str | None
    extraction_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.case_count < 0:
            raise BenchmarkContractError("case_count must not be negative")
        for digest in self.raw_artifact_sha256:
            _validate_sha(digest, "raw_artifact_sha256")
        _validate_sha(self.normalized_sha256, "normalized_sha256")
        _validate_sha(self.split_manifest_sha256, "split_manifest_sha256")
        if self.extraction_provenance is not None:
            object.__setattr__(self, "extraction_provenance", dict(self.extraction_provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "raw_artifact_sha256": list(self.raw_artifact_sha256),
            "adapter_version": self.adapter_version,
            "schema_version": self.schema_version,
            "case_count": self.case_count,
            "normalized_sha256": self.normalized_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "extraction_provenance": dict(self.extraction_provenance)
            if self.extraction_provenance
            else None,
        }


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    benchmark_id: str
    payload: Mapping[str, Any]
    gold: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_profile: Mapping[str, Any] | None = None
    source_provenance: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.case_id, "case_id")
        _require_text(self.benchmark_id, "benchmark_id")
        if not isinstance(self.payload, Mapping) or not isinstance(self.gold, Mapping):
            raise BenchmarkContractError("case payload and gold must be mappings")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "gold", dict(self.gold))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "source_provenance", tuple(dict(item) for item in self.source_provenance))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BenchmarkCase:
        return cls(
            case_id=value["case_id"],
            benchmark_id=value["benchmark_id"],
            payload=value["payload"],
            gold=value["gold"],
            metadata=value.get("metadata", {}),
            task_profile=value.get("task_profile"),
            source_provenance=tuple(value.get("source_provenance", [])),
        )

    def to_dict(self, *, include_task_profile: bool = True) -> dict[str, Any]:
        value = {
            "case_id": self.case_id,
            "benchmark_id": self.benchmark_id,
            "payload": dict(self.payload),
            "gold": dict(self.gold),
            "metadata": dict(self.metadata),
            "source_provenance": [dict(item) for item in self.source_provenance],
        }
        if include_task_profile and self.task_profile is not None:
            value["task_profile"] = dict(self.task_profile)
        return value


@dataclass(frozen=True)
class TaskProfile:
    task_profile_version: str
    case_id: str
    task_family: str
    answerability: str
    required_evidence_groups: tuple[tuple[str, ...], ...]
    source_families: tuple[str, ...]
    independent_subtasks: tuple[Mapping[str, Any], ...]
    dependency_edges: tuple[tuple[str, str], ...]
    required_capability_domains: tuple[str, ...]
    conflict_structure: Mapping[str, Any]
    temporal_structure: Mapping[str, Any]
    expected_min_sources: int
    ood_reason: str | None
    annotation_status: ReviewStatus
    reviewer: str | None
    split: str
    source_group_id: str
    question_family_id: str
    text_hash: str

    def __post_init__(self) -> None:
        for field_name in ("task_profile_version", "case_id", "task_family", "answerability", "split"):
            _require_text(getattr(self, field_name), field_name)
        if self.split not in {"DEV", "TEST", "CALIBRATION"}:
            raise BenchmarkContractError("split must be DEV, TEST or CALIBRATION")
        if self.expected_min_sources < 0:
            raise BenchmarkContractError("expected_min_sources must not be negative")
        object.__setattr__(self, "annotation_status", ReviewStatus(self.annotation_status))
        object.__setattr__(self, "required_evidence_groups", tuple(tuple(group) for group in self.required_evidence_groups))
        object.__setattr__(self, "source_families", tuple(self.source_families))
        object.__setattr__(self, "independent_subtasks", tuple(dict(item) for item in self.independent_subtasks))
        object.__setattr__(self, "dependency_edges", tuple(tuple(edge) for edge in self.dependency_edges))
        node_ids = {item.get("id") for item in self.independent_subtasks}
        if None in node_ids or any(not isinstance(item, str) or not item for item in node_ids):
            raise BenchmarkContractError("independent_subtasks need non-empty string ids")
        for source, target in self.dependency_edges:
            if source not in node_ids or target not in node_ids:
                raise BenchmarkContractError("dependency edge references an unknown subtask")
        if self.has_cycle():
            raise BenchmarkContractError("dependency graph contains a cycle")
        _validate_sha(self.text_hash, "task profile text_hash")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskProfile:
        return cls(
            task_profile_version=value["task_profile_version"],
            case_id=value["case_id"],
            task_family=value["task_family"],
            answerability=value["answerability"],
            required_evidence_groups=tuple(tuple(group) for group in value.get("required_evidence_groups", [])),
            source_families=tuple(value.get("source_families", [])),
            independent_subtasks=tuple(value.get("independent_subtasks", [])),
            dependency_edges=tuple(tuple(edge) for edge in value.get("dependency_edges", [])),
            required_capability_domains=tuple(value.get("required_capability_domains", [])),
            conflict_structure=value.get("conflict_structure", {}),
            temporal_structure=value.get("temporal_structure", {}),
            expected_min_sources=value.get("expected_min_sources", 0),
            ood_reason=value.get("ood_reason"),
            annotation_status=value.get("annotation_status", ReviewStatus.CANDIDATE),
            reviewer=value.get("reviewer"),
            split=value["split"],
            source_group_id=value["source_group_id"],
            question_family_id=value["question_family_id"],
            text_hash=value["text_hash"],
        )

    def has_cycle(self) -> bool:
        adjacency: dict[str, list[str]] = {item["id"]: [] for item in self.independent_subtasks}
        for source, target in self.dependency_edges:
            adjacency[source].append(target)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(visit(child) for child in adjacency[node]):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in adjacency)

    @property
    def derived_features(self) -> dict[str, int]:
        adjacency: dict[str, list[str]] = {item["id"]: [] for item in self.independent_subtasks}
        incoming = {item["id"]: 0 for item in self.independent_subtasks}
        for source, target in self.dependency_edges:
            adjacency[source].append(target)
            incoming[target] += 1
        depth: dict[str, int] = {node: 1 for node, count in incoming.items() if count == 0}
        queue = list(depth)
        while queue:
            node = queue.pop(0)
            for child in adjacency[node]:
                depth[child] = max(depth.get(child, 1), depth[node] + 1)
                incoming[child] -= 1
                if incoming[child] == 0:
                    queue.append(child)
        return {
            "subtask_count": len(self.independent_subtasks),
            "independent_width": sum(1 for item in self.independent_subtasks if not any(item["id"] == edge[1] for edge in self.dependency_edges)),
            "serial_depth": max(depth.values(), default=0),
            "source_family_count": len(set(self.source_families)),
            "evidence_group_count": len(self.required_evidence_groups),
            "conflict_count": int(bool(self.conflict_structure.get("present"))),
            "temporal_dependency_count": int(self.temporal_structure.get("dependency_count", 0)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_profile_version": self.task_profile_version,
            "case_id": self.case_id,
            "task_family": self.task_family,
            "answerability": self.answerability,
            "required_evidence_groups": [list(group) for group in self.required_evidence_groups],
            "source_families": list(self.source_families),
            "independent_subtasks": [dict(item) for item in self.independent_subtasks],
            "dependency_edges": [list(edge) for edge in self.dependency_edges],
            "required_capability_domains": list(self.required_capability_domains),
            "conflict_structure": dict(self.conflict_structure),
            "temporal_structure": dict(self.temporal_structure),
            "expected_min_sources": self.expected_min_sources,
            "ood_reason": self.ood_reason,
            "annotation_status": self.annotation_status.value,
            "reviewer": self.reviewer,
            "split": self.split,
            "source_group_id": self.source_group_id,
            "question_family_id": self.question_family_id,
            "text_hash": self.text_hash,
            "derived_features": self.derived_features,
        }


@dataclass(frozen=True)
class ExperimentBudgetContract:
    regime: str
    max_provider_calls: int | None
    max_tool_executions: int | None
    max_observed_tokens: int | None
    deadline_ms: int | None
    concurrency_limit: int | None
    cost_model_version: str

    def __post_init__(self) -> None:
        if self.regime not in {"NATIVE_BUDGET", "COST_MATCHED", "DEADLINE_MATCHED"}:
            raise BenchmarkContractError("unknown fairness regime")
        for name in (
            "max_provider_calls",
            "max_tool_executions",
            "max_observed_tokens",
            "deadline_ms",
            "concurrency_limit",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise BenchmarkContractError(f"{name} must be a non-negative integer or null")
        _require_text(self.cost_model_version, "cost_model_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "max_provider_calls": self.max_provider_calls,
            "max_tool_executions": self.max_tool_executions,
            "max_observed_tokens": self.max_observed_tokens,
            "deadline_ms": self.deadline_ms,
            "concurrency_limit": self.concurrency_limit,
            "cost_model_version": self.cost_model_version,
        }


@dataclass(frozen=True)
class CostModelManifest:
    cost_model_version: str
    token_accounting: str
    wall_clock_accounting: str
    provider_call_accounting: str
    tool_call_accounting: str
    price_table: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_model_version": self.cost_model_version,
            "token_accounting": self.token_accounting,
            "wall_clock_accounting": self.wall_clock_accounting,
            "provider_call_accounting": self.provider_call_accounting,
            "tool_call_accounting": self.tool_call_accounting,
            "price_table": self.price_table,
        }


@dataclass(frozen=True)
class JudgeProtocolManifest:
    judge_protocol_id: str
    benchmark_id: str
    reference_implementation_revision: str
    grader_prompt_hash: str
    expected_output_schema: Mapping[str, Any]
    aggregation_version: str
    reference_grader_model: str
    reference_grader_revision: str
    temperature: float
    sampling_parameters: Mapping[str, Any]
    retry_policy: Mapping[str, Any]
    parser_behavior: Mapping[str, Any]
    cost_accounting_policy: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_sha(self.grader_prompt_hash, "grader_prompt_hash")
        if self.reference_implementation_revision in FLOATING_REVISIONS:
            raise BenchmarkContractError("judge reference revision must not float")

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_protocol_id": self.judge_protocol_id,
            "benchmark_id": self.benchmark_id,
            "reference_implementation_revision": self.reference_implementation_revision,
            "grader_prompt_hash": self.grader_prompt_hash,
            "expected_output_schema": dict(self.expected_output_schema),
            "aggregation_version": self.aggregation_version,
            "reference_grader_model": self.reference_grader_model,
            "reference_grader_revision": self.reference_grader_revision,
            "temperature": self.temperature,
            "sampling_parameters": dict(self.sampling_parameters),
            "retry_policy": dict(self.retry_policy),
            "parser_behavior": dict(self.parser_behavior),
            "cost_accounting_policy": dict(self.cost_accounting_policy),
        }

    @property
    def protocol_hash(self) -> str:
        return canonical_hash(self.to_dict())
