import pytest

from health_ai_copilot.harness.profiles import (
    DEFAULT_PRODUCT_PROFILE,
    MemoryMode,
    ModelVariant,
    ReasoningMode,
    RetrievalMode,
    SystemProfile,
)
from health_ai_copilot.harness.runtime import HealthCopilotHarness
from health_ai_copilot.multi_agent.api import create_fastapi_app
from health_ai_copilot.providers.model import ModelReply


class ApiProvider:
    calls = 0

    async def complete(self, request):
        self.calls += 1
        prompt = "\n".join(item["content"] for item in request.messages)
        answer = "basic" if "Please indicate the difficulty/complexity" in prompt else "Safe answer"
        return ModelReply(answer, "api-smoke", 2, 2)


def test_product_entry_rejects_static_single_reasoner() -> None:
    provider = ApiProvider()
    harness = HealthCopilotHarness(model_providers={ModelVariant.QWEN3_8B_BASE: provider})
    single_profile = SystemProfile(
        profile_id="evaluation-control",
        model_variant=ModelVariant.QWEN3_8B_BASE,
        retrieval_mode=RetrievalMode.OFF,
        memory_mode=MemoryMode.OFF,
        reasoning_mode=ReasoningMode.SINGLE,
    )
    with pytest.raises(ValueError, match="Single is evaluation-only"):
        create_fastapi_app(harness, profile=single_profile)

    assert DEFAULT_PRODUCT_PROFILE.reasoning_mode is ReasoningMode.ADAPTIVE_MDT


def test_fastapi_product_entry_uses_harness_and_rejects_gold() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    provider = ApiProvider()
    harness = HealthCopilotHarness(model_providers={ModelVariant.QWEN3_8B_BASE: provider})
    with TestClient(create_fastapi_app(harness)) as client:
        rejected = client.post("/medical/answer", json={"query": "question", "gold": "A"})
        answer = client.post("/medical/answer", json={"query": "safe question"})
    assert rejected.status_code == 422
    assert answer.status_code == 200
    assert answer.json()["answer"] == "Safe answer"
    assert set(answer.json()) == {
        "answer", "route_mode", "workers_used", "citations", "safety_flags", "trace_id", "latency_ms",
    }
    assert provider.calls == 3
    assert harness.metrics_snapshot()["reasoning_modes"] == {"adaptive_mdt": 1}
