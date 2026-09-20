from health_ai_copilot.config import OpenAIConfig
from health_ai_copilot.generation.openai_compatible import OpenAICompatibleGenerator
from health_ai_copilot.runtime import (
    FakeProviderExecutor,
    OpenAICompatibleProviderExecutor,
    ProviderCallKind,
    ProviderResponse,
    RunContext,
)


def test_openai_client_disables_sdk_hidden_retries(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    OpenAICompatibleProviderExecutor(OpenAIConfig(api_key="fixture", model="fixture"))

    assert captured["max_retries"] == 0


def test_m0_generator_uses_the_bounded_provider_timeout() -> None:
    executor = FakeProviderExecutor(
        [
            ProviderResponse(
                "",
                ProviderCallKind.GENERATOR,
                "fixture",
                '{"answer":"fixture","citation_ids":["source"],"abstain":false}',
            )
        ]
    )
    generator = OpenAICompatibleGenerator(
        provider_executor=executor,
        model="fixture",
        runtime=RunContext.create("test"),
    )

    generator.generate("question", [])

    assert executor.requests[0].timeout_seconds == 30.0
