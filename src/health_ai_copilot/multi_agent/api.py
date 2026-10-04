"""Optional FastAPI adapter for the stable Health-Copilot answer contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from ..harness.contracts import AnswerSchema, HarnessRequest, RuntimeResources
from ..harness.profiles import (
    DEFAULT_PRODUCT_PROFILE,
    ReasoningMode,
    SystemProfile,
)
from ..harness.runtime import HealthCopilotHarness


def create_fastapi_app(
    harness: HealthCopilotHarness,
    *,
    profile: SystemProfile | None = None,
):
    """Build the product API on Harness V1 and adapt to the stable response shape."""
    if not isinstance(harness, HealthCopilotHarness):
        raise TypeError("create_fastapi_app requires HealthCopilotHarness")
    selected_profile = profile or DEFAULT_PRODUCT_PROFILE
    if selected_profile.reasoning_mode is not ReasoningMode.ADAPTIVE_MDT:
        raise ValueError("the product API requires adaptive reasoning; Single is evaluation-only")

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError("install health-ai-copilot[api] to enable FastAPI") from exc

    app = FastAPI(title="Health-Copilot Medical Agent API", version="1.0")

    @app.post("/medical/answer")
    async def answer(payload: dict[str, Any]) -> dict[str, Any]:
        forbidden = {
            "gold", "gold_answer", "evaluator_truth", "required_capabilities",
            "expected_worker_set", "benchmark_label", "oracle_action",
        }
        allowed_fields = {
            "query", "answer_schema", "request_id", "conversation_context", "subject_id",
            "as_of_time", "benchmark_case_id", "runtime_resources",
        }
        if forbidden.intersection(payload) or set(payload) - allowed_fields:
            raise HTTPException(status_code=422, detail="evaluator-only fields are forbidden")
        try:
            resource_payload = payload.get("runtime_resources")
            if resource_payload is not None and not isinstance(resource_payload, dict):
                raise TypeError("runtime_resources must be an object")
            allowed_resources = {
                "max_provider_calls", "max_tool_calls", "max_input_tokens",
                "max_output_tokens", "deadline_ms",
            }
            if resource_payload is not None and set(resource_payload) - allowed_resources:
                raise ValueError("unknown runtime resource fields")
            if not isinstance(payload.get("query"), str):
                raise TypeError("query must be a string")
            raw_context = payload.get("conversation_context", ())
            if not isinstance(raw_context, (list, tuple)) or any(
                not isinstance(item, str) for item in raw_context
            ):
                raise TypeError("conversation_context must be a list of strings")
            request = HarnessRequest(
                request_id=str(payload.get("request_id") or f"product-{uuid4().hex}"),
                query=payload["query"],
                answer_schema=AnswerSchema(payload.get("answer_schema", "free_text")),
                conversation_context=tuple(raw_context),
                subject_id=str(payload["subject_id"]) if payload.get("subject_id") else None,
                as_of_time=(datetime.fromisoformat(str(payload["as_of_time"]))
                            if payload.get("as_of_time") else None),
                benchmark_case_id=(str(payload["benchmark_case_id"])
                                   if payload.get("benchmark_case_id") else None),
                runtime_resources=(RuntimeResources(**resource_payload)
                                   if resource_payload is not None else None),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="invalid Harness request") from exc
        response = await harness.execute(selected_profile, request)
        trace = harness.trace_for(response.trace_id) or {}
        mdt_detail = next((
            item.get("fields", {}) for item in trace.get("events", [])
            if item.get("kind") == "reasoning_detail"
            and item.get("fields", {}).get("event") == "reasoning_adaptive_mdt"
        ), {})
        actual_route = mdt_detail.get("route_mode")
        if actual_route not in {"SINGLE", "TEAM"}:
            actual_route = (
                "TEAM" if selected_profile.reasoning_mode is ReasoningMode.ADAPTIVE_MDT else "SINGLE"
            )
        workers = [item for item in str(mdt_detail.get("workers", "")).split(",") if item]
        return {
            "answer": response.answer_text,
            "route_mode": actual_route,
            "workers_used": workers,
            "citations": [
                {
                    "source_id": item.source_id,
                    "title": item.source,
                    "excerpt": item.excerpt,
                    "source_url": item.source,
                }
                for item in response.citations
            ],
            "safety_flags": list(response.safety_flags),
            "trace_id": response.trace_id,
            "latency_ms": max(0, int(response.latency_ms)),
        }

    @app.get("/medical/metrics")
    async def metrics() -> dict[str, Any]:
        return harness.metrics_snapshot()

    return app
