"""Optional FastAPI adapter for the stable MedicalAgentRequest/Response contract."""

from __future__ import annotations

from typing import Any

from .contracts import MedicalAgentRequest
from .runtime import MedicalAgentRuntime


def create_fastapi_app(runtime: MedicalAgentRuntime):
    """Expose `/medical/answer` and `/medical/metrics` when the API extra is installed."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError("install health-ai-copilot[api] to enable FastAPI") from exc

    app = FastAPI(title="Health-Copilot Medical Agent API", version="1.0")

    @app.post("/medical/answer")
    async def answer(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = MedicalAgentRequest.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="invalid medical-agent request") from exc
        response = await runtime.respond(request)
        return response.to_dict()

    @app.get("/medical/metrics")
    async def metrics() -> dict[str, Any]:
        return runtime.observability.snapshot()

    return app
