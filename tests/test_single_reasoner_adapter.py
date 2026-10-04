import asyncio

from health_ai_copilot.harness.contracts import AnswerSchema
from health_ai_copilot.providers.model import ModelReply
from health_ai_copilot.reasoning.base import ReasoningContext
from health_ai_copilot.reasoning.single import _SINGLE_PROMPT, SingleReasoner


class CaptureProvider:
    def __init__(self):
        self.request = None

    async def complete(self, request):
        self.request = request
        return ModelReply("A. answer", "fake", 4, 2)


def test_single_adapter_uses_frozen_strong_single_prompt_and_supplied_context() -> None:
    provider = CaptureProvider()
    context = ReasoningContext("question", ("history",), (), (), AnswerSchema.SINGLE_CHOICE)
    result = asyncio.run(SingleReasoner(provider).reason(context))
    assert provider.request.messages[0]["content"] == _SINGLE_PROMPT
    assert '"answer_schema":"single_choice"' in provider.request.messages[1]["content"]
    assert result.answer_text == "A. answer"
