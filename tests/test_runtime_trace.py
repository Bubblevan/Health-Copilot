import json

import pytest

from health_ai_copilot.runtime import (
    FakeProviderExecutor,
    ProviderCallKind,
    ProviderRequest,
    ProviderResponse,
    RunContext,
)
from health_ai_copilot.runtime.trace import RunTrace, TraceContentPolicy, TraceEventType


def test_metadata_trace_records_control_plane_without_provider_content(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    trace = RunTrace(path)
    runtime = RunContext.create("m4", trace=trace)
    executor = FakeProviderExecutor(
        [ProviderResponse("ignored-secret-content", ProviderCallKind.GENERATOR, "fixture", "{}")]
    )
    request = ProviderRequest.create(
        kind=ProviderCallKind.GENERATOR,
        model="fixture",
        messages=({"role": "user", "content": "private question"},),
    )

    executor.execute(request, runtime)
    trace.close(status="complete")

    text = path.read_text(encoding="utf-8")
    assert "private question" not in text
    assert "ignored-secret-content" not in text
    events = [json.loads(line) for line in text.splitlines()]
    assert [event["event_type"] for event in events] == [
        TraceEventType.RUN_START.value,
        TraceEventType.PROVIDER_START.value,
        TraceEventType.PROVIDER_END.value,
        TraceEventType.RUN_END.value,
    ]
    assert events[1]["fields"]["request_fingerprint"]


def test_metadata_trace_rejects_content_bearing_fields() -> None:
    trace = RunTrace()

    with pytest.raises(ValueError, match="metadata-only"):
        trace.emit(TraceEventType.HARNESS_DISPOSITION, answer="not allowed")


def test_public_eval_trace_can_explicitly_store_reviewed_content() -> None:
    trace = RunTrace(content_policy=TraceContentPolicy.PUBLIC_EVAL_CONTENT)
    trace.emit(TraceEventType.HARNESS_DISPOSITION, answer="reviewed fixture")

    assert trace.events[0].fields["answer"] == "reviewed fixture"
