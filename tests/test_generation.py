import pytest

from health_ai_copilot.contracts import GenerationDraft
from health_ai_copilot.generation.base import GenerationError
from health_ai_copilot.generation.openai_compatible import OpenAICompatibleGenerator


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
