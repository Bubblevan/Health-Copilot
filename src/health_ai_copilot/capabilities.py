"""Typed, harness-enforced capability contracts for the E2 research track.

These contracts are separate from the frozen M8 role prompts and orchestration.
They describe evidence/tool authority and execution budgets; prompt text is only
an explanatory view of the same immutable contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol

from .agent.tools import ToolResult, ToolSpec
from .contracts import Evidence
from .knowledge.scope import KnowledgeScope
from .runtime.components import CapabilityContractRef, ComponentManifest
from .tools.search_knowledge import SearchKnowledgeTool


class E2WorkerRole(StrEnum):
    PUBLIC_HEALTH = "public_health"
    GUIDELINE = "guideline"
    LITERATURE = "literature"


class CapabilityEligibility(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    NOT_YET_ELIGIBLE = "NOT_YET_ELIGIBLE"


@dataclass(frozen=True)
class WorkerCapabilitySpec:
    """Canonical immutable boundary for one fixed E2 worker type."""

    capability_id: str
    role: E2WorkerRole | str
    allowed_source_families: tuple[str, ...]
    allowed_capability_domains: tuple[str, ...]
    tool_ids: tuple[str, ...]
    retriever_profile_id: str
    authority_scope: str
    objective_contract_id: str
    max_model_turns: int
    max_tool_calls: int
    eligibility: CapabilityEligibility | str = CapabilityEligibility.ELIGIBLE
    eligibility_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", E2WorkerRole(self.role))
        object.__setattr__(self, "eligibility", CapabilityEligibility(self.eligibility))
        for name in (
            "capability_id",
            "retriever_profile_id",
            "authority_scope",
            "objective_contract_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "allowed_source_families",
            "allowed_capability_domains",
            "tool_ids",
        ):
            values = tuple(getattr(self, name))
            if not values or any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"{name} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, tuple(sorted(values)))
        if self.max_model_turns < 1 or self.max_model_turns > 2:
            raise ValueError("E2 worker max_model_turns must be between one and two")
        if self.max_tool_calls < 0 or self.max_tool_calls > 1:
            raise ValueError("E2 worker max_tool_calls must be zero or one")
        if self.eligibility == CapabilityEligibility.NOT_YET_ELIGIBLE and not (
            isinstance(self.eligibility_reason, str) and self.eligibility_reason.strip()
        ):
            raise ValueError("ineligible capability requires an eligibility_reason")
        if self.eligibility == CapabilityEligibility.ELIGIBLE and self.eligibility_reason:
            raise ValueError("eligible capability must not have an eligibility_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "role": self.role.value,
            "allowed_source_families": list(self.allowed_source_families),
            "allowed_capability_domains": list(self.allowed_capability_domains),
            "tool_ids": list(self.tool_ids),
            "retriever_profile_id": self.retriever_profile_id,
            "authority_scope": self.authority_scope,
            "objective_contract_id": self.objective_contract_id,
            "max_model_turns": self.max_model_turns,
            "max_tool_calls": self.max_tool_calls,
            "eligibility": self.eligibility.value,
            "eligibility_reason": self.eligibility_reason,
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @property
    def contract_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def manifest_ref(self) -> CapabilityContractRef:
        return CapabilityContractRef(self.capability_id, self.contract_hash)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerCapabilitySpec:
        return cls(
            capability_id=value["capability_id"],
            role=value["role"],
            allowed_source_families=tuple(value["allowed_source_families"]),
            allowed_capability_domains=tuple(value["allowed_capability_domains"]),
            tool_ids=tuple(value["tool_ids"]),
            retriever_profile_id=value["retriever_profile_id"],
            authority_scope=value["authority_scope"],
            objective_contract_id=value["objective_contract_id"],
            max_model_turns=value["max_model_turns"],
            max_tool_calls=value["max_tool_calls"],
            eligibility=value.get("eligibility", CapabilityEligibility.ELIGIBLE),
            eligibility_reason=value.get("eligibility_reason"),
        )


@dataclass(frozen=True)
class SourceMetadata:
    """Trusted source classification supplied by the harness, never the model."""

    source_id: str
    source_family: str
    capability_domains: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.source_family.strip():
            raise ValueError("source_id and source_family must be non-empty")
        domains = tuple(sorted(set(self.capability_domains)))
        if not domains or any(not item.strip() for item in domains):
            raise ValueError("source metadata must contain non-empty capability domains")
        object.__setattr__(self, "capability_domains", domains)


@dataclass(frozen=True)
class CapabilitySourceCatalog:
    """Closed source-ID index used by retrieval, provenance, and metrics."""

    sources: tuple[SourceMetadata, ...]

    def __post_init__(self) -> None:
        sources = tuple(sorted(self.sources, key=lambda item: item.source_id))
        if not all(isinstance(item, SourceMetadata) for item in sources):
            raise TypeError("sources must contain SourceMetadata values")
        if len({item.source_id for item in sources}) != len(sources):
            raise ValueError("source catalog contains duplicate source IDs")
        object.__setattr__(self, "sources", sources)

    @property
    def by_id(self) -> dict[str, SourceMetadata]:
        return {item.source_id: item for item in self.sources}

    def to_dict(self) -> dict[str, Any]:
        return {"sources": [
            {
                "source_id": item.source_id,
                "source_family": item.source_family,
                "capability_domains": list(item.capability_domains),
            }
            for item in self.sources
        ]}

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @property
    def catalog_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def allows(self, source_id: str, spec: WorkerCapabilitySpec) -> bool:
        source = self.by_id.get(source_id)
        return bool(
            source
            and source.source_family in spec.allowed_source_families
            and set(source.capability_domains).intersection(spec.allowed_capability_domains)
        )

    @classmethod
    def from_reviewed_knowledge(
        cls,
        cards: Sequence[Any],
        knowledge_scope: KnowledgeScope,
        *,
        publisher_families: Mapping[str, str],
    ) -> CapabilitySourceCatalog:
        """Build an explicit source index from reviewed cards and reviewed topic IDs.

        Unknown publishers remain ``unclassified`` and therefore match no fixed
        E2 capability. No URL or source-ID substring guessing is performed.
        """

        domains_by_source: dict[str, set[str]] = {}
        for topic in knowledge_scope.topics:
            for source_id in topic.source_ids:
                domains_by_source.setdefault(source_id, set()).add(topic.id)
        records = []
        for card in cards:
            source_id = getattr(card, "id", None)
            publisher = getattr(card, "publisher", None)
            domains = domains_by_source.get(source_id, set())
            if not isinstance(source_id, str) or not domains:
                continue
            records.append(
                SourceMetadata(
                    source_id=source_id,
                    source_family=publisher_families.get(publisher, "unclassified"),
                    capability_domains=tuple(domains),
                )
            )
        return cls(tuple(records))


def production_worker_capabilities() -> tuple[WorkerCapabilitySpec, ...]:
    """Frozen first-pass contracts, reflecting the currently reviewed corpus."""

    public_domains = (
        "definition_thresholds",
        "diagnostic_confirmation",
        "symptoms",
        "measurement",
        "home_monitoring",
        "risk_factors",
        "lifestyle_prevention",
        "medication_safety",
        "complications",
        "care_management",
    )
    common = {
        "tool_ids": ("search_knowledge",),
        "retriever_profile_id": "bm25-v1",
        "max_model_turns": 2,
        "max_tool_calls": 1,
    }
    return (
        WorkerCapabilitySpec(
            capability_id="e2-public-health-v1",
            role=E2WorkerRole.PUBLIC_HEALTH,
            allowed_source_families=("public_health",),
            allowed_capability_domains=public_domains,
            authority_scope=(
                "Reviewed WHO, CDC, and NHC patient-education sources only. Report public "
                "information and its jurisdiction; do not synthesize clinical guidelines, "
                "prescribe, diagnose, or extend beyond observed sources."
            ),
            objective_contract_id="e2-public-health-objective-v1",
            eligibility=CapabilityEligibility.ELIGIBLE,
            **common,
        ),
        WorkerCapabilitySpec(
            capability_id="e2-guideline-v1",
            role=E2WorkerRole.GUIDELINE,
            allowed_source_families=("reviewed_guideline",),
            allowed_capability_domains=("recommendation_scope", "population_jurisdiction"),
            authority_scope=(
                "Only reviewed guideline and recommendation records may support claims. "
                "Preserve publisher, jurisdiction, population, date, and recommendation scope; "
                "do not generalize outside those boundaries."
            ),
            objective_contract_id="e2-guideline-objective-v1",
            eligibility=CapabilityEligibility.NOT_YET_ELIGIBLE,
            eligibility_reason="No reviewed guideline/recommendation corpus is present in the active retriever.",
            **common,
        ),
        WorkerCapabilitySpec(
            capability_id="e2-literature-v1",
            role=E2WorkerRole.LITERATURE,
            allowed_source_families=("scholarly_literature",),
            allowed_capability_domains=("study_population", "study_design", "literature_findings"),
            authority_scope=(
                "Use only an approved closed scholarly-literature corpus. Identify study type, "
                "population, and limitations; do not turn observational results into guidance."
            ),
            objective_contract_id="e2-literature-objective-v1",
            eligibility=CapabilityEligibility.NOT_YET_ELIGIBLE,
            eligibility_reason=(
                "No approved scholarly-paper corpus is connected to the active retriever; "
                "raw textbook files do not satisfy this boundary."
            ),
            **common,
        ),
    )


def e2_component_manifest(
    base: ComponentManifest,
    capabilities: Sequence[WorkerCapabilitySpec],
    *,
    source_catalog: CapabilitySourceCatalog | None = None,
) -> ComponentManifest:
    """Bind capability and resolved source-catalog hashes into a runtime manifest."""

    refs = tuple(item.manifest_ref() for item in capabilities)
    return replace(
        base,
        capability_contracts=refs,
        source_catalog_hash=source_catalog.catalog_hash if source_catalog else None,
    )


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        ...


class CapabilityScopedRetriever:
    """Filtered retriever surface passed to exactly one worker.

    The existing Retriever protocol does not expose corpus filters. This
    wrapper obtains candidates locally, then returns only source IDs that the
    immutable contract and trusted source catalog both allow. Unclassified or
    mismatched results fail closed before entering a ToolResult or transcript.
    """

    def __init__(
        self,
        retriever: Retriever,
        spec: WorkerCapabilitySpec,
        source_catalog: CapabilitySourceCatalog,
    ) -> None:
        if spec.eligibility != CapabilityEligibility.ELIGIBLE:
            raise ValueError("cannot construct retrieval for an ineligible capability")
        self.__retriever = retriever
        self.spec = spec
        self.source_catalog = source_catalog

    def allows(self, source_id: str) -> bool:
        return self.source_catalog.allows(source_id, self.spec)

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be non-empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        # Ask for the closed catalog size so an out-of-scope top result cannot
        # hide a lower-ranked in-scope result before the wrapper filters it.
        candidate_count = max(top_k, len(self.source_catalog.sources))
        candidates = self.__retriever.search(query.strip(), top_k=candidate_count)
        allowed: list[Evidence] = []
        seen: set[str] = set()
        for evidence in candidates:
            if evidence.source_id in seen or not self.allows(evidence.source_id):
                continue
            allowed.append(evidence)
            seen.add(evidence.source_id)
            if len(allowed) >= top_k:
                break
        return allowed


class CapabilityScopedSearchTool:
    """The existing search tool with a private, capability-scoped retriever."""

    name = "search_knowledge"

    def __init__(self, retriever: CapabilityScopedRetriever, *, top_k: int = 3) -> None:
        if self.name not in retriever.spec.tool_ids:
            raise ValueError("capability does not grant search_knowledge")
        self.__delegate = SearchKnowledgeTool(retriever, top_k=top_k)
        self.__retriever = retriever

    @property
    def spec(self) -> ToolSpec:
        capability = self.__retriever.spec
        return ToolSpec(
            name=self.name,
            description=(
                "Read-only search over the assigned source family and domains. "
                f"Authority: {capability.authority_scope}"
            ),
            input_schema=self.__delegate.spec.input_schema,
        )

    def validate_arguments(self, arguments: object) -> dict[str, str]:
        return self.__delegate.validate_arguments(arguments)

    def execute(self, arguments: object) -> ToolResult:
        result = self.__delegate.execute(arguments)
        if not result.ok:
            return result
        if any(not self.__retriever.allows(item.source_id) for item in result.observed_evidence):
            return ToolResult.failure(
                "capability_boundary_violation",
                "retrieval returned evidence outside the worker capability",
            )
        return result


__all__ = [
    "CapabilityEligibility",
    "CapabilityScopedRetriever",
    "CapabilityScopedSearchTool",
    "CapabilitySourceCatalog",
    "E2WorkerRole",
    "SourceMetadata",
    "WorkerCapabilitySpec",
    "e2_component_manifest",
    "production_worker_capabilities",
]
