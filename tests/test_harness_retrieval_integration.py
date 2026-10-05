import asyncio
from datetime import UTC, datetime

from health_ai_copilot.harness.contracts import AnswerSchema, HarnessRequest
from health_ai_copilot.harness.profiles import (
    MemoryMode,
    ModelVariant,
    ReasoningMode,
    RetrievalMode,
    SystemProfile,
)
from health_ai_copilot.harness.runtime import HealthCopilotHarness
from health_ai_copilot.providers.model import ModelReply
from health_ai_copilot.providers.retrieval import RetrievalResult, RetrievedEvidence


class CaptureModelProvider:
    def __init__(self) -> None:
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelReply(content="A", model="qwen3-8b-test", input_tokens=10, output_tokens=1)


class CaptureRetrievalProvider:
    def __init__(self) -> None:
        self.calls = []

    async def retrieve(self, *, query, context):
        self.calls.append((query, context))
        return RetrievalResult(
            evidence=(RetrievedEvidence("e1", "source-1", "Guidance", "Observed evidence."),),
            corpus_id="common-kb-test",
            index_hash="a" * 64,
        )


def test_harness_passes_request_identity_and_retrieved_evidence() -> None:
    model = CaptureModelProvider()
    retrieval = CaptureRetrievalProvider()
    harness = HealthCopilotHarness(
        model_providers={ModelVariant.QWEN3_8B_BASE: model},
        retrieval_provider=retrieval,
    )
    profile = SystemProfile(
        "B1-test",
        ModelVariant.QWEN3_8B_BASE,
        RetrievalMode.STANDARD,
        MemoryMode.OFF,
        ReasoningMode.SINGLE,
    )
    request = HarnessRequest(
        request_id="eval-request-1",
        query="Choose the best answer.",
        answer_schema=AnswerSchema.SINGLE_CHOICE,
        subject_id="subject-1",
        as_of_time=datetime(2026, 10, 5, tzinfo=UTC),
    )

    response = asyncio.run(harness.execute(profile, request))

    assert response.parsed_answer == "A"
    assert response.safety_flags == ()
    assert response.tool_calls == 1
    query, context = retrieval.calls[0]
    assert query == request.query
    assert context.request_id == request.request_id
    assert context.subject_id == request.subject_id
    assert context.as_of_time == request.as_of_time
    assert "Observed evidence." in model.requests[0].messages[1]["content"]
