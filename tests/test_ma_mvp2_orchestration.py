from __future__ import annotations

import asyncio
import json

import pytest

from health_ai_copilot.multi_agent.contracts import (
    CoverageStatus,
    MedicalAgentRequest,
    PlannedAspect,
    TriageDecision,
    WorkerRole,
    WorkerStatus,
)
from health_ai_copilot.multi_agent.orchestration import (
    artifact_from_worker_output,
    harness_task_ledger,
    make_coverage_ledger,
    parse_aspect_plan,
    parse_coverage_judgment,
)
from health_ai_copilot.multi_agent.providers import ModelReply
from health_ai_copilot.multi_agent.routing import (
    JEV_TRIAGE_QUESTIONS,
    JevTriageProvider,
    LocalTriageProvider,
    route_from_triage,
)
from health_ai_copilot.multi_agent.runtime import (
    MedicalAgentRuntime,
    RuntimeConfig,
    _mvp2_answer_fragment,
)
from health_ai_copilot.multi_agent.shared_context import EvidenceLedgerEntry
from health_ai_copilot.multi_agent.skills import SkillResult, SourceMaterial
from health_ai_copilot.routing.jev import JevResult


def run(awaitable):
    return asyncio.run(awaitable)


class StubJevClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    async def evaluate(self, *, state, questions):
        self.calls.append((state, questions))
        if self.fail:
            raise RuntimeError("offline")
        return JevResult(
            model="jev-test-v1",
            answers={
                "need_patient_context": {"type": "noul", "noul": 0.9},
                "need_external_evidence": {"type": "noul", "noul": 0.1},
                "need_care_analysis": {"type": "noul", "noul": 0.2},
                "complexity": {"type": "score", "score": 1.4},
            },
            input_tokens=10,
            output_tokens=4,
            latency_ms=2,
        )


def test_lead_aspect_plan_has_no_model_owned_ids_and_harness_assigns_ids() -> None:
    planned = parse_aspect_plan(
        '{"aspects":[{"aspect":"查患者趋势","worker":"patient_context",'
        '"expected_output":"列出趋势"}]}',
        allowed_workers=(WorkerRole.PATIENT_CONTEXT,),
    )
    ledger, by_worker = harness_task_ledger(
        "request-1", planned, {WorkerRole.PATIENT_CONTEXT: "worker-harness-1"},
    )

    assert ledger.aspects[0].aspect_id.startswith("aspect-")
    assert ledger.assignments[0].assignment_id.startswith("assignment-")
    assert ledger.assignments[0].worker_id == "worker-harness-1"
    assert by_worker == {WorkerRole.PATIENT_CONTEXT: (ledger.aspects[0].aspect_id,)}
    with pytest.raises(ValueError, match="INVALID_ASPECT"):
        parse_aspect_plan(
            '{"aspects":[{"aspect":"a","worker":"evidence",'
            '"expected_output":"b","aspect_id":"forged"}]}',
            allowed_workers=(WorkerRole.EVIDENCE,),
        )


def test_worker_artifact_facts_and_refs_are_harness_derived() -> None:
    excerpt = "patient record: SYNKEY-ABC SYNVAL-1234567890"
    skill = SkillResult(
        "PatientStateLookupSkill",
        excerpt,
        sources=(SourceMaterial("record-real", excerpt, "memory_read"),),
        tool_id="memory_read",
    )
    entry = EvidenceLedgerEntry(
        evidence_id="evidence-real", source_id="record-real", worker_id="worker-real",
        role="patient_context", tool_call_id="tool-real", tool_id="memory_read",
        excerpt=excerpt, input_hash="input", output_hash="output", resource_versions=(),
    )
    artifact = artifact_from_worker_output(
        worker_id="worker-real",
        role=WorkerRole.PATIENT_CONTEXT,
        status=WorkerStatus.COMPLETE,
        raw_output=(
            '{"findings":["verified"],"facts":["SYNVAL-1234567890",'
            '"fabricated SYNVAL-AAAAAAAAAA"],"unresolved":[],'
            '"answer_fragment":"SYNVAL-1234567890","source_id":"fake"}'
        ).replace(',"source_id":"fake"', ""),
        skill_results=[skill],
        evidence_entries=(entry,),
        aspect_ids=("aspect-harness",),
    )

    assert {fact.text for fact in artifact.facts} == {
        "SYNVAL-1234567890", "patient record: SYNKEY-ABC SYNVAL-1234567890",
    }
    assert artifact.patient_record_refs == ("record-real",)
    assert artifact.evidence_refs == ()
    assert artifact.aspect_ids == ("aspect-harness",)
    assert all("AAAAAAAAAA" not in fact.text for fact in artifact.facts)


def test_invalid_structured_worker_payload_is_not_echoed_as_answer() -> None:
    skill = SkillResult("MedicalKnowledgeSearchSkill", "")
    artifact = artifact_from_worker_output(
        worker_id="worker-1",
        role=WorkerRole.EVIDENCE,
        status=WorkerStatus.COMPLETE,
        raw_output='{"answer_fragment":"synthetic unsupported answer","fake_id":"forged"}',
        skill_results=[skill],
        evidence_entries=(),
        aspect_ids=(),
    )

    assert artifact.answer_fragment == ""
    assert "worker_output_invalid_structured_json" in artifact.unresolved
    assert _mvp2_answer_fragment('{"answer_fragment":"bad","fake_id":"x"}') == ""


def test_coverage_judgment_is_bounded_and_falls_back_to_partial() -> None:
    planned = (PlannedAspect("ask this", WorkerRole.EVIDENCE, "answer this"),)
    ledger, _ = harness_task_ledger("request-1", planned, {WorkerRole.EVIDENCE: "worker-1"})
    report = type("Report", (), {
        "worker_id": "worker-1",
        "status": WorkerStatus.PARTIAL,
        "answer_text": "some information",
        "artifact": None,
    })()

    assert parse_coverage_judgment('{"statuses":["COVERED"]}', 1) == (
        CoverageStatus.COVERED,
    )
    with pytest.raises(ValueError, match="COUNT_MISMATCH"):
        parse_coverage_judgment('{"statuses":[]}', 1)
    coverage = make_coverage_ledger(ledger, [report])
    assert coverage.items[0].status == CoverageStatus.PARTIAL


def test_jev_triage_is_one_four_question_call_and_local_fallback_is_automatic() -> None:
    client = StubJevClient()
    provider = JevTriageProvider(client=client)  # type: ignore[arg-type]
    decision = run(provider.triage("compare my history", "observable"))

    assert len(client.calls) == 1
    assert len(client.calls[0][1]) == 4
    assert client.calls[0][1] == JEV_TRIAGE_QUESTIONS
    assert decision.need_patient_context == 0.9
    assert decision.complexity == 0.7
    assert decision.provider == "jev"
    failed = JevTriageProvider(client=StubJevClient(fail=True))  # type: ignore[arg-type]
    fallback = run(failed.triage("compare my history", "observable"))
    assert fallback.provider == "local_fallback"
    assert fallback.fallback_reason == "RuntimeError"


def test_single_fast_path_only_when_one_specialist_and_low_complexity() -> None:
    single = route_from_triage(TriageDecision(0.8, 0.1, 0.1, 0.1))
    team = route_from_triage(TriageDecision(0.8, 0.1, 0.1, 0.8))

    assert single.mode.value == "SINGLE"
    assert team.mode.value == "TEAM"
    assert single.predicted_capabilities == (WorkerRole.PATIENT_CONTEXT,)


def test_serialized_local_triage_model_scores_stably_and_keeps_scores_bounded() -> None:
    model = {
        "version": "test-logreg-v1",
        "vocabulary": {"patient": 0, "patient history": 1},
        "idf": [1.0, 1.0],
        "classifiers": {
            "need_patient_context": {"intercept": 0.0, "coefficients": [2.0, 1.0]},
            "need_external_evidence": {"intercept": -2.0, "coefficients": [0.0, 0.0]},
            "need_care_analysis": {"intercept": -2.0, "coefficients": [0.0, 0.0]},
            "complexity": {"intercept": -2.0, "coefficients": [0.0, 0.0]},
        },
    }
    provider = LocalTriageProvider(trained_model=model)
    decision = run(provider.triage("patient history", "ignored context"))
    repeated = run(provider.triage("patient history", "different context"))

    assert decision.provider_version == "test-logreg-v1"
    assert decision.need_patient_context > 0.8
    assert decision.need_external_evidence < 0.2
    assert decision.to_dict() == repeated.to_dict()


class ScriptedMVP2Provider:
    def __init__(self) -> None:
        self.calls = []

    async def complete(
        self, *, system_prompt, user_prompt, max_output_tokens, timeout_seconds, json_mode=False,
    ):
        self.calls.append((system_prompt, user_prompt))
        del user_prompt, max_output_tokens, timeout_seconds
        if "把用户请求拆成所有需要独立回答的 aspect" in system_prompt:
            content = (
                '{"aspects":[{"aspect":"取得来源支持的结果","worker":"evidence",'
                '"expected_output":"给出支持的结果"}]}'
            )
        elif "只根据当前外部医学证据回答" in system_prompt:
            content = '{"findings":["已检查可用证据"],"facts":[],"unresolved":[],"answer_fragment":"观察到的结果"}'
        elif "受限的任务覆盖检查器" in system_prompt:
            content = '{"statuses":["COVERED"]}'
        else:
            content = "按任务清单回答：观察到的结果。"
        return ModelReply(content=content, input_tokens=20, output_tokens=10, model="scripted")


class TeamTriage:
    async def triage(self, query, observable_context):
        del query, observable_context
        return TriageDecision(0.1, 0.9, 0.1, 0.9)


def test_mvp2_runtime_and_fastapi_keep_the_stable_response_contract() -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from health_ai_copilot.multi_agent.api import create_fastapi_app

    runtime = MedicalAgentRuntime(
        ScriptedMVP2Provider(),  # type: ignore[arg-type]
        triage_provider=TeamTriage(),  # type: ignore[arg-type]
        config=RuntimeConfig(
            max_worker_output_tokens=192,
            max_lead_output_tokens=160,
            max_single_output_tokens=96,
        ),
    )
    app = create_fastapi_app(runtime)
    assert isinstance(app, fastapi.FastAPI)
    with TestClient(app) as client:
        response = client.post("/medical/answer", json={"query": "synthetic evidence request"})

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "answer", "route_mode", "workers_used", "citations", "safety_flags", "trace_id",
        "latency_ms",
    }
    assert payload["route_mode"] == "TEAM"
    execution = run(runtime.execute(MedicalAgentRequest(query="synthetic evidence request")))
    assert execution.task_ledger is not None
    assert execution.coverage_ledger is not None
    assert execution.worker_reports[0].artifact is not None
    expected_aspect_ids = tuple(item.aspect_id for item in execution.task_ledger.aspects)
    assert execution.worker_reports[0].artifact.aspect_ids == expected_aspect_ids
    worker_input = execution.trajectory["worker_inputs"][0]
    assert json.loads(worker_input["user_prompt"])["assigned_aspects"] == [
        {"aspect": execution.task_ledger.aspects[0].aspect,
         "expected_output": execution.task_ledger.aspects[0].expected_output}
    ]
    assert execution.trace["task_ledger"] is not None
    assert any(item["step"] == "triage_decision" for item in execution.trace["timeline"])


def test_mvp2_finalizer_receives_harness_safety_flags() -> None:
    provider = ScriptedMVP2Provider()
    runtime = MedicalAgentRuntime(
        provider,  # type: ignore[arg-type]
        triage_provider=TeamTriage(),  # type: ignore[arg-type]
    )
    run(runtime.execute(MedicalAgentRequest(query="I have chest pain")))

    final_prompt = next(
        user_prompt for system_prompt, user_prompt in provider.calls
        if system_prompt.startswith("你是 Medical Lead finalizer")
    )
    assert "urgent_symptom_requires_human_care" in json.loads(final_prompt)["safety_flags"]
