import asyncio

from health_ai_copilot.multi_agent.contracts import MedicalAgentRequest
from health_ai_copilot.multi_agent.providers import ModelReply
from health_ai_copilot.multi_agent.runtime import MedicalAgentRuntime


class HallucinatedCitationProvider:
    async def complete(self, *, system_prompt, user_prompt, max_output_tokens,
                       timeout_seconds, json_mode=False):
        del system_prompt, user_prompt, max_output_tokens, timeout_seconds, json_mode
        return ModelReply("依据虚构来源 [evidence-id:fake]，我无法确认。", 10, 12, 1.0, "fake")


def test_model_cannot_create_response_citations_or_ledger_entries() -> None:
    async def run():
        return await MedicalAgentRuntime(HallucinatedCitationProvider()).execute(
            MedicalAgentRequest("What is a simple direct education question?")
        )

    result = asyncio.run(run())
    assert result.response.citations == ()
    assert result.trace["evidence_ledger"] == []
    assert "fake" not in result.response.to_dict()["citations"]
