"""Specialized skill adapters over the existing owned-universe tool surface."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
from typing import Protocol

from ..research.integration.contracts import IntegrationEpisode
from ..research.integration.executor import ExecutionResources
from ..research.integration.tools import (
    DeterministicToolRegistry,
    ToolInvocation,
    ToolObservation,
    default_tool_registry,
)
from .contracts import WorkerRole


@dataclass(frozen=True)
class HospitalEvidence:
    source_id: str
    title: str
    content: str


class HospitalKnowledgeProvider(Protocol):
    async def search(self, query: str, top_k: int = 5) -> tuple[HospitalEvidence, ...]:
        """Search an approved hospital knowledge adapter."""


class DisabledHospitalKnowledgeProvider:
    """Interface-only adapter. It intentionally has no hospital corpus."""

    async def search(self, query: str, top_k: int = 5) -> tuple[HospitalEvidence, ...]:
        del query, top_k
        return ()


@dataclass(frozen=True)
class SourceMaterial:
    source_id: str
    excerpt: str
    tool_id: str


@dataclass(frozen=True)
class SkillResult:
    skill_name: str
    output: str
    sources: tuple[SourceMaterial, ...] = ()
    observation: ToolObservation | None = None
    error: str | None = None
    tool_id: str | None = None
    input_hash: str = ""
    output_hash: str = ""
    resource_versions: tuple[tuple[str, str], ...] = ()

    @property
    def tool_calls(self) -> int:
        return int(self.tool_id is not None)


@dataclass(frozen=True)
class SkillContext:
    query: str
    role: WorkerRole
    episode: IntegrationEpisode | None
    resources: ExecutionResources | None
    observed_context: tuple[str, ...] = ()
    patient_id: str | None = None
    as_of_time: datetime | None = None


class MemoryProvider(Protocol):
    async def lookup(self, context: SkillContext) -> SkillResult:
        """Return patient-specific longitudinal observations for this request."""


class ExternalEvidenceProvider(Protocol):
    async def search(self, context: SkillContext) -> SkillResult:
        """Return source-backed medical evidence for this request."""


class MedicalSkill(Protocol):
    name: str

    async def execute(self, context: SkillContext) -> SkillResult:
        ...


class _ToolSkill:
    tool_id: str

    def __init__(self, registry: DeterministicToolRegistry | None = None) -> None:
        self._registry = registry or default_tool_registry()

    @property
    def name(self) -> str:
        return (
            "PatientStateLookupSkill" if self.tool_id == "memory_read"
            else "ExternalEvidenceSearchSkill"
        )

    async def _invoke(self, context: SkillContext) -> SkillResult:
        if context.episode is None or context.resources is None:
            return SkillResult(self.name, "", error="RUNTIME_DATA_UNAVAILABLE")
        episode = replace(context.episode, query=context.query)
        invocation = ToolInvocation(
            tool_id=self.tool_id,
            episode=episode,
            patient_state_store=context.resources.patient_state_store,
            external_evidence_world=context.resources.external_evidence_world,
        )
        try:
            observation = await asyncio.to_thread(self._registry.invoke, invocation)
        except Exception as exc:  # noqa: BLE001 - tool implementations are vendor-specific
            return SkillResult(self.name, "", error=f"TOOL_ERROR:{type(exc).__name__}")
        records: dict[str, str] = {}
        if self.tool_id == "memory_read" and context.resources.patient_state_store is not None:
            records = {
                row.record_id: row.content
                for row in context.resources.patient_state_store.records
            }
        elif (self.tool_id == "external_retrieval"
              and context.resources.external_evidence_world is not None):
            records = {
                row.source_id: row.content
                for row in context.resources.external_evidence_world.records
            }
        sources = tuple(
            SourceMaterial(source_id, records.get(source_id, ""), self.tool_id)
            for source_id in observation.resource_ids
        )
        return SkillResult(
            self.name, observation.output, sources, observation,
            tool_id=self.tool_id,
            input_hash=observation.input_hash,
            output_hash=observation.output_hash,
            resource_versions=observation.resource_versions,
        )


class OwnedLongitudinalMemoryProvider(_ToolSkill):
    """Current deterministic adapter; replace this provider for a future Memory backend."""

    tool_id = "memory_read"

    async def lookup(self, context: SkillContext) -> SkillResult:
        return await self._invoke(context)


class OwnedMedicalEvidenceProvider(_ToolSkill):
    """Current owned-world adapter; product deployments can inject the frozen RAG backend."""

    tool_id = "external_retrieval"

    async def search(self, context: SkillContext) -> SkillResult:
        return await self._invoke(context)


class PatientStateLookupSkill(_ToolSkill):
    name = "PatientStateLookupSkill"
    tool_id = "memory_read"

    def __init__(self, provider: MemoryProvider | None = None) -> None:
        self._provider = provider or OwnedLongitudinalMemoryProvider()

    async def execute(self, context: SkillContext) -> SkillResult:
        return await self._provider.lookup(context)


class TimelineCompareSkill(PatientStateLookupSkill):
    name = "TimelineCompareSkill"


class MedicalKnowledgeSearchSkill(_ToolSkill):
    name = "MedicalKnowledgeSearchSkill"
    tool_id = "external_retrieval"

    def __init__(self, provider: ExternalEvidenceProvider | None = None) -> None:
        self._provider = provider or OwnedMedicalEvidenceProvider()

    async def execute(self, context: SkillContext) -> SkillResult:
        return await self._provider.search(context)


class ExternalEvidenceSearchSkill(MedicalKnowledgeSearchSkill):
    name = "ExternalEvidenceSearchSkill"


class HospitalKnowledgeSearchSkill:
    name = "HospitalKnowledgeSearchSkill"

    def __init__(self, provider: HospitalKnowledgeProvider | None = None) -> None:
        self._provider = provider or DisabledHospitalKnowledgeProvider()

    async def execute(self, context: SkillContext) -> SkillResult:
        try:
            rows = await self._provider.search(context.query, top_k=5)
        except Exception as exc:  # noqa: BLE001 - provider adapters are vendor-specific
            return SkillResult(self.name, "", error=f"HOSPITAL_PROVIDER_ERROR:{type(exc).__name__}")
        input_hash = sha256(context.query.encode()).hexdigest()
        output = " ".join(row.content for row in rows)
        output_hash = sha256(output.encode()).hexdigest()
        return SkillResult(
            self.name,
            output,
            tuple(SourceMaterial(row.source_id, row.content, self.name) for row in rows),
            tool_id="hospital_knowledge_search",
            input_hash=input_hash,
            output_hash=output_hash,
        )


class RiskAssessmentSkill:
    name = "RiskAssessmentSkill"

    async def execute(self, context: SkillContext) -> SkillResult:
        text = f"{context.query} {' '.join(context.observed_context)}".casefold()
        urgent_markers = (
            "chest pain", "shortness of breath", "stroke", "unconscious",
            "呼吸困难", "胸痛", "意识丧失", "大出血",
        )
        flags = ["urgent_symptom_requires_human_care"] if any(
            marker in text for marker in urgent_markers
        ) else []
        output = "Potential urgent symptoms were found; seek local emergency care." if flags else (
            "No deterministic urgent-symptom marker was detected; this is not a clinical ruling."
        )
        return SkillResult(self.name, output + (" [" + ",".join(flags) + "]" if flags else ""))


class AnswerabilitySkill:
    name = "AnswerabilitySkill"

    async def execute(self, context: SkillContext) -> SkillResult:
        has_observations = any(item.strip() for item in context.observed_context)
        return SkillResult(
            self.name,
            "Harness has observed relevant context." if has_observations
            else "No relevant observation has been supplied to this skill.",
        )


WORKER_SKILLS: dict[WorkerRole, frozenset[str]] = {
    WorkerRole.PATIENT_CONTEXT: frozenset({
        "PatientStateLookupSkill", "TimelineCompareSkill",
    }),
    WorkerRole.EVIDENCE: frozenset({
        "MedicalKnowledgeSearchSkill", "ExternalEvidenceSearchSkill",
        "HospitalKnowledgeSearchSkill",
    }),
    WorkerRole.CARE: frozenset({"RiskAssessmentSkill", "AnswerabilitySkill"}),
}


class SkillRegistry:
    """Harness capability registry that denies skills outside each agent scope."""

    def __init__(
        self,
        skills: tuple[MedicalSkill, ...] | None = None,
        *,
        memory_provider: MemoryProvider | None = None,
        external_evidence_provider: ExternalEvidenceProvider | None = None,
        hospital_provider: HospitalKnowledgeProvider | None = None,
    ) -> None:
        rows = skills or (
            PatientStateLookupSkill(memory_provider),
            TimelineCompareSkill(memory_provider),
            MedicalKnowledgeSearchSkill(external_evidence_provider),
            ExternalEvidenceSearchSkill(external_evidence_provider),
            HospitalKnowledgeSearchSkill(hospital_provider),
            RiskAssessmentSkill(),
            AnswerabilitySkill(),
        )
        self._skills = {skill.name: skill for skill in rows}
        self.memory_provider_configured = memory_provider is not None
        self.external_evidence_provider_configured = external_evidence_provider is not None
        self.hospital_provider_configured = hospital_provider is not None
        if len(self._skills) != len(rows):
            raise ValueError("skill names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._skills))

    def allowed_for(self, role: WorkerRole) -> tuple[str, ...]:
        return tuple(sorted(WORKER_SKILLS[role]))

    async def execute(self, role: WorkerRole, name: str, context: SkillContext) -> SkillResult:
        if name not in WORKER_SKILLS[role]:
            raise PermissionError(f"SKILL_SCOPE_DENIED:{role.value}:{name}")
        try:
            skill = self._skills[name]
        except KeyError as exc:
            raise ValueError(f"SKILL_NOT_REGISTERED:{name}") from exc
        return await skill.execute(context)
