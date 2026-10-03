import asyncio

from health_ai_copilot.multi_agent.contracts import MedicalAgentRequest, WorkerStatus
from health_ai_copilot.multi_agent.providers import ModelReply
from health_ai_copilot.multi_agent.runtime import MedicalAgentRuntime, RuntimeConfig


class OneWorkerFailsProvider:
    async def complete(self, *, system_prompt, user_prompt, max_output_tokens,
                       timeout_seconds, json_mode=False):
        del max_output_tokens, timeout_seconds, json_mode
        if "只做任务拆解" in system_prompt:
            return ModelReply(
                '{"mode":"team","tasks":['
                '{"worker":"patient_context","objective":"查纵向记录"},'
                '{"worker":"evidence","objective":"查外部证据"}]}',
                20, 10, 1.0, "fake",
            )
        if "EvidenceAgent" in system_prompt:
            raise ConnectionError("injected provider failure")
        if "PatientContextAgent" in system_prompt:
            return ModelReply("患者记录支持 SYNVAL-ABCDEF1234。", 20, 12, 2.0, "fake")
        if "综合已完成 WorkerReport" in system_prompt:
            assert '"status":"failed"' in user_prompt
            return ModelReply("保留成功 worker 的患者记录，并说明外部检索失败。", 40, 12, 1.0, "fake")
        return ModelReply("单 Agent 降级答案。", 10, 5, 1.0, "fake")


def test_worker_provider_failure_is_reported_and_other_result_survives() -> None:
    async def run():
        runtime = MedicalAgentRuntime(OneWorkerFailsProvider(), config=RuntimeConfig(
            worker_timeout_seconds=2, planning_timeout_seconds=2,
            synthesis_timeout_seconds=2,
        ))
        return await runtime.execute(MedicalAgentRequest(
            "How has my previous record changed, and what do current guidelines recommend?"
        ))

    result = asyncio.run(run())
    statuses = {report.role.value: report.status for report in result.worker_reports}
    assert statuses["patient_context"] in {WorkerStatus.COMPLETE, WorkerStatus.PARTIAL}
    assert statuses["evidence"] == WorkerStatus.FAILED
    assert '"status":"failed"' in result.trajectory["lead_synthesis_input"]["user_prompt"]
    assert "外部检索失败" in result.response.answer
