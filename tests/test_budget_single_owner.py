import asyncio

from health_ai_copilot.harness.contracts import AnswerSchema, HarnessRequest, RuntimeResources
from health_ai_copilot.harness.profiles import (
    MemoryMode,
    ModelVariant,
    ReasoningMode,
    RetrievalMode,
    SystemProfile,
)
from health_ai_copilot.harness.runtime import HealthCopilotHarness
from health_ai_copilot.providers.model import ModelReply


class OneCallProvider:
    def __init__(self):
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return ModelReply("answer", "fake", 5, 2)


def test_harness_is_the_only_provider_budget_accountant() -> None:
    provider = OneCallProvider()
    harness = HealthCopilotHarness(model_providers={ModelVariant.QWEN3_8B_BASE: provider})
    profile = SystemProfile("smoke", ModelVariant.QWEN3_8B_BASE, RetrievalMode.OFF,
                            MemoryMode.OFF, ReasoningMode.SINGLE)
    response = asyncio.run(harness.execute(
        profile, HarnessRequest("r1", "safe question", AnswerSchema.FREE_TEXT),
    ))
    assert provider.calls == response.provider_calls == 1
    assert response.input_tokens == 5 and response.output_tokens == 2


def test_zero_output_budget_denies_before_model_side_effect() -> None:
    provider = OneCallProvider()
    harness = HealthCopilotHarness(model_providers={ModelVariant.QWEN3_8B_BASE: provider})
    profile = SystemProfile("smoke", ModelVariant.QWEN3_8B_BASE, RetrievalMode.OFF,
                            MemoryMode.OFF, ReasoningMode.SINGLE)
    response = asyncio.run(harness.execute(
        profile,
        HarnessRequest(
            "r1", "safe question", AnswerSchema.FREE_TEXT,
            runtime_resources=RuntimeResources(max_output_tokens=0),
        ),
    ))
    assert provider.calls == 0
    assert "output_token_budget_exceeded" in response.safety_flags
