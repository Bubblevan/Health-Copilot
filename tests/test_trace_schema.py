from health_ai_copilot.harness.trace import ExecutionTrace


def test_trace_uses_shared_event_schema_and_hash_only_query_identity() -> None:
    trace = ExecutionTrace(query_sha256="a" * 64, profile_id="B0")
    trace.emit("request_started", request_id="r1")
    payload = trace.to_dict()
    assert payload["request"] == {"query_sha256": "a" * 64}
    assert payload["events"][0]["sequence"] == 1
