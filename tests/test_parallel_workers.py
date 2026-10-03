import asyncio

from health_ai_copilot.multi_agent.contracts import MedicalAgentRequest, RouteMode
from health_ai_copilot.multi_agent.providers import ModelReply
from health_ai_copilot.multi_agent.runtime import MedicalAgentRuntime, RuntimeConfig


class ConcurrentFakeProvider:
    def __init__(self) -> None:
        self.active_workers = 0
        self.max_active_workers = 0
        self.lock = asyncio.Lock()

    async def complete(self, *, system_prompt, user_prompt, max_output_tokens,
                       timeout_seconds, json_mode=False):
        del user_prompt, max_output_tokens, timeout_seconds, json_mode
        if "只做任务拆解" in system_prompt:
            return ModelReply(
                '{"mode":"team","tasks":['
                '{"worker":"patient_context","objective":"查纵向记录"},'
                '{"worker":"evidence","objective":"查外部证据"}]}',
                30, 20, 1.0, "fake",
            )
        if "PatientContextAgent" in system_prompt or "EvidenceAgent" in system_prompt:
            async with self.lock:
                self.active_workers += 1
                self.max_active_workers = max(self.max_active_workers, self.active_workers)
            await asyncio.sleep(0.04)
            async with self.lock:
                self.active_workers -= 1
            label = "患者记录" if "PatientContextAgent" in system_prompt else "外部证据"
            return ModelReply(label, 20, 8, 40.0, "fake")
        return ModelReply("综合已观察到的两项信息。", 60, 12, 1.0, "fake")


def test_independent_workers_execute_concurrently() -> None:
    async def run():
        provider = ConcurrentFakeProvider()
        runtime = MedicalAgentRuntime(provider, config=RuntimeConfig(
            worker_timeout_seconds=2, planning_timeout_seconds=2,
            synthesis_timeout_seconds=2,
        ))
        result = await runtime.execute(MedicalAgentRequest(
            "How has my previous record changed, and what do current guidelines recommend?"
        ))
        assert result.response.route_mode == RouteMode.TEAM
        assert provider.max_active_workers == 2
        assert result.worker_wave_wall_ms < result.sequential_worker_latency_ms
        assert len(result.worker_reports) == 2
        return result

    result = asyncio.run(run())
    assert len(result.response.workers_used) == 2
