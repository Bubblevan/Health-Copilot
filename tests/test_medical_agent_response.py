from health_ai_copilot.multi_agent.contracts import (
    Citation,
    MedicalAgentResponse,
    RouteMode,
)


def test_response_serializes_fastapi_friendly_contract() -> None:
    response = MedicalAgentResponse(
        answer="有来源支持。",
        route_mode=RouteMode.TEAM,
        workers_used=("evidence",),
        citations=(Citation(
            "evidence-1", "source-1", "external_retrieval", "excerpt", "worker-1",
            "external_retrieval",
        ),),
        safety_flags=(),
        trace_id="trace-1",
        latency_ms=12,
    )
    payload = response.to_dict()
    assert set(payload) == {
        "answer", "route_mode", "workers_used", "citations", "safety_flags",
        "trace_id", "latency_ms",
    }
    assert payload["route_mode"] == "TEAM"
    assert payload["citations"][0]["source_id"] == "source-1"
