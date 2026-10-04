import asyncio

from health_ai_copilot.harness.contracts import AnswerSchema
from health_ai_copilot.providers.model import ModelReply
from health_ai_copilot.reasoning.base import ReasoningContext
from health_ai_copilot.reasoning.single import _SINGLE_PROMPT, SingleReasoner


class PromptCapture:
    request = None

    async def complete(self, request):
        self.request = request
        return ModelReply("A", "fake", 1, 1)


def test_harness_single_keeps_existing_strong_single_system_prompt() -> None:
    provider = PromptCapture()
    asyncio.run(SingleReasoner(provider).reason(
        ReasoningContext("synthetic", (), (), (), AnswerSchema.SINGLE_CHOICE)
    ))
    assert provider.request.messages[0]["content"] == _SINGLE_PROMPT
