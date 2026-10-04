import asyncio

from health_ai_copilot.harness.contracts import AnswerSchema
from health_ai_copilot.providers.model import ModelReply
from health_ai_copilot.reasoning.adaptive_mdt import AdaptiveMDTReasoner
from health_ai_copilot.reasoning.base import ReasoningContext


class BasicMDTProvider:
    def __init__(self):
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return ModelReply("1)" if self.calls < 3 else "A. answer", "fake", 1, 1)


def test_adaptive_adapter_runs_over_the_injected_model_provider() -> None:
    provider = BasicMDTProvider()
    result = asyncio.run(AdaptiveMDTReasoner(provider).reason(
        ReasoningContext("case", (), (), (), AnswerSchema.SINGLE_CHOICE)
    ))
    assert provider.calls >= 3
    assert result.answer_text.startswith("A")
    assert result.reasoning_events[0]["event"] == "reasoning_adaptive_mdt"
    event_names = {item["event"] for item in result.reasoning_events}
    assert "adaptive_worker_assignment" in event_names
    assert "adaptive_worker_status" in event_names
    assert "adaptive_complexity_decided" in event_names
