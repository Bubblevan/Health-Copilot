import asyncio

import pytest

from health_ai_copilot.multi_agent.contracts import WorkerRole
from health_ai_copilot.multi_agent.skills import SkillContext, SkillRegistry


def test_patient_worker_cannot_call_evidence_skill() -> None:
    async def run():
        await SkillRegistry().execute(
            WorkerRole.PATIENT_CONTEXT,
            "ExternalEvidenceSearchSkill",
            SkillContext("question", WorkerRole.PATIENT_CONTEXT, None, None),
        )

    with pytest.raises(PermissionError, match="SKILL_SCOPE_DENIED"):
        asyncio.run(run())


def test_worker_skill_registry_exposes_only_specialized_surface() -> None:
    registry = SkillRegistry()
    assert set(registry.allowed_for(WorkerRole.PATIENT_CONTEXT)) == {
        "PatientStateLookupSkill", "TimelineCompareSkill",
    }
    assert "RiskAssessmentSkill" not in registry.allowed_for(WorkerRole.EVIDENCE)


def test_default_owned_adapters_report_unavailable_runtime_data() -> None:
    async def run():
        registry = SkillRegistry()
        patient = await registry.execute(
            WorkerRole.PATIENT_CONTEXT,
            "PatientStateLookupSkill",
            SkillContext("question", WorkerRole.PATIENT_CONTEXT, None, None),
        )
        evidence = await registry.execute(
            WorkerRole.EVIDENCE,
            "ExternalEvidenceSearchSkill",
            SkillContext("question", WorkerRole.EVIDENCE, None, None),
        )
        return patient, evidence

    patient, evidence = asyncio.run(run())
    assert (patient.skill_name, patient.error) == (
        "PatientStateLookupSkill", "RUNTIME_DATA_UNAVAILABLE",
    )
    assert (evidence.skill_name, evidence.error) == (
        "ExternalEvidenceSearchSkill", "RUNTIME_DATA_UNAVAILABLE",
    )
