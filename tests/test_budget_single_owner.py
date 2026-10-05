import asyncio

from health_ai_copilot.harness.budget import BudgetLedger, BudgetLimits
from health_ai_copilot.harness.contracts import AnswerSchema, HarnessRequest, RuntimeResources
from health_ai_copilot.harness.profiles import (
    MemoryMode,
    ModelVariant,
    ReasoningMode,
    RetrievalMode,
    SystemProfile,
)
from health_ai_copilot.harness.runtime import HealthCopilotHarness, _BudgetedModelProvider
from health_ai_copilot.harness.trace import ExecutionTrace
from health_ai_copilot.providers.model import ModelReply, ModelRequest


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


def test_deadline_recap_preserves_structured_output_schema() -> None:
    class CaptureProvider:
        def __init__(self):
            self.request = None

        async def complete(self, request):
            self.request = request
            return ModelReply('{"teams":[]}', "fake", 1, 1)

    provider = CaptureProvider()
    schema = {
        "name": "team_recruitment",
        "strict": True,
        "schema": {"type": "object", "required": ["teams"]},
    }
    request = ModelRequest(
        messages=({"role": "user", "content": "recruit teams"},),
        timeout_seconds=120,
        json_schema=schema,
    )
    metered = _BudgetedModelProvider(
        provider,
        BudgetLedger(BudgetLimits(deadline_ms=1000)),
        ExecutionTrace(query_sha256="0" * 64, profile_id="test"),
    )

    asyncio.run(metered.complete(request))

    assert provider.request is not request
    assert provider.request.timeout_seconds < request.timeout_seconds
    assert provider.request.json_schema == schema
