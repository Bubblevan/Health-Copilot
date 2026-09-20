import pytest

from health_ai_copilot.contracts import GenerationDraft
from health_ai_copilot.generation.base import GenerationError
from health_ai_copilot.generation.openai_compatible import OpenAICompatibleGenerator
from health_ai_copilot.runtime import (
    FakeProviderExecutor,
    ProviderCallKind,
    ProviderFailure,
    ProviderFailureKind,
    ProviderResponse,
)


def test_generator_parser_accepts_only_the_expected_shape() -> None:
    draft = OpenAICompatibleGenerator._parse(
        '{"answer":"grounded","citation_ids":["source-a"],"abstain":false}'
    )

    assert draft == GenerationDraft("grounded", ["source-a"])


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        '{"answer":"answer","citation_ids":["source-a"]}',
        '{"answer":"","citation_ids":[],"abstain":false}',
    ],
)
def test_generator_parser_rejects_malformed_or_incomplete_output(content: str) -> None:
    with pytest.raises(GenerationError):
        OpenAICompatibleGenerator._parse(content)


def test_generator_uses_injected_provider_executor_with_typed_request() -> None:
    executor = FakeProviderExecutor(
        [
            ProviderResponse(
                call_id="recorded",
                kind=ProviderCallKind.GENERATOR,
                model="fixture-model",
                content='{"answer":"grounded","citation_ids":["source-a"],"abstain":false}',
            )
        ]
    )
    generator = OpenAICompatibleGenerator(
        provider_executor=executor,
        model="fixture-model",
        temperature=0,
    )

    draft = generator.generate("question sentinel", [])

    assert draft == GenerationDraft("grounded", ["source-a"])
    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.kind == ProviderCallKind.GENERATOR
    assert request.model == "fixture-model"
    assert request.response_format == {"type": "json_object"}


def test_generator_maps_normalized_provider_failure_to_generation_error() -> None:
    generator = OpenAICompatibleGenerator(
        provider_executor=FakeProviderExecutor(
            failure=ProviderFailure(ProviderFailureKind.TIMEOUT)
        )
    )

    with pytest.raises(GenerationError, match="generation request failed"):
        generator.generate("question", [])
