import asyncio

from health_ai_copilot.harness.contracts import AnswerSchema
from health_ai_copilot.multi_agent.mdagents_style import (
    Complexity,
    MDAgentsStyleConfig,
    _parse_complexity,
)
from health_ai_copilot.providers.model import ModelReply
from health_ai_copilot.reasoning.adaptive_mdt import AdaptiveMDTReasoner
from health_ai_copilot.reasoning.base import ReasoningContext


class BasicMDTProvider:
    def __init__(self):
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        return ModelReply("1)" if self.calls < 3 else "A. answer", "fake", 1, 1)


class InvalidRecruitmentProvider:
    def __init__(self):
        self.outputs = iter((
            "I can assess the question.",
            "3) advanced",
            '{"teams":"not-a-list"}',
        ))

    async def complete(self, request):
        return ModelReply(next(self.outputs), "fake", 1, 1)


class StructuredRecruitmentProvider:
    def __init__(self):
        self.requests = []
        self.calls = 0

    async def complete(self, request):
        self.requests.append(request)
        self.calls += 1
        if self.calls == 1:
            content = "classification"
        elif self.calls == 2:
            content = "3) advanced"
        elif self.calls == 3:
            content = (
                '{"teams":[{"name":"acute care","specialists":['
                '{"name":"cardiologist","focus":"rhythm diagnosis"},'
                '{"name":"electrophysiologist","focus":"conduction mechanisms"}]},'
                '{"name":"internal medicine","specialists":['
                '{"name":"internist","focus":"differential diagnosis"},'
                '{"name":"pharmacologist","focus":"medication effects"}]}]}'
            )
        else:
            content = "A. answer"
        return ModelReply(content, "fake", 1, 1)


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


def test_adaptive_diagnostics_capture_raw_recruitment_and_validation_failure() -> None:
    result = asyncio.run(AdaptiveMDTReasoner(
        InvalidRecruitmentProvider(),
        config=MDAgentsStyleConfig(capture_validation_outputs=True),
    ).reason(ReasoningContext("case", (), (), (), AnswerSchema.SINGLE_CHOICE)))

    audit = next(
        event for event in result.reasoning_events
        if event["event"] == "adaptive_validation_audit"
    )
    assert audit["raw_complexity_output"] == "3) advanced"
    assert audit["raw_recruitment_output"] == '{"teams":"not-a-list"}'
    assert audit["validation_error"] == "TypeError:invalid_team_recruitment"
    assert audit["fallback_reason"] == "TypeError:invalid_team_recruitment"


def test_complexity_parser_uses_final_output_after_qwen_thinking_block() -> None:
    assert _parse_complexity(
        "<think>Options include basic and intermediate.</think>3) advanced"
    ) is Complexity.ADVANCED


def test_advanced_recruitment_uses_constrained_team_schema() -> None:
    provider = StructuredRecruitmentProvider()
    result = asyncio.run(AdaptiveMDTReasoner(provider).reason(
        ReasoningContext("case", (), (), (), AnswerSchema.SINGLE_CHOICE)
    ))

    request = provider.requests[2]
    assert request.json_schema["name"] == "mdagents_team_recruitment"
    assert request.json_schema["strict"] is True
    assert request.json_schema["schema"]["properties"]["teams"]["minItems"] == 2
    assert result.failure_reason is None
    assert result.answer_text.startswith("A")
