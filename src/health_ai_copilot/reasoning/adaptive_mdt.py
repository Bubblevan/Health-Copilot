"""Adapter for the Health-Copilot MDAgents-style clinical reasoning runtime."""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Any

from ..multi_agent.mdagents_style import MDAgentsStyleConfig, MDAgentsStyleOrchestrator
from ..providers.model import ModelProvider, ModelReply, ModelRequest
from .base import ReasoningContext, ReasoningResult


class _HarnessProviderBridge:
    """Expose Harness-owned model calls to the adaptive MDAgents strategy."""

    def __init__(self, provider: ModelProvider, model: str | None = None) -> None:
        self.provider = provider
        self.model = model

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        timeout_seconds: float,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> Any:
        reply = await self.provider.complete(ModelRequest(
            model=self.model,
            messages=(
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ),
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            json_mode=json_mode,
            json_schema=json_schema,
        ))
        return ModelReply(
            content=reply.content,
            input_tokens=reply.input_tokens or 0,
            output_tokens=reply.output_tokens or 0,
            latency_ms=reply.latency_ms,
            model=reply.model,
            finish_reason=reply.finish_reason,
        )

    async def complete_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_output_tokens: int,
        timeout_seconds: float,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> Any:
        reply = await self.provider.complete(ModelRequest(
            model=self.model,
            messages=tuple(messages),
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            json_mode=json_mode,
            json_schema=json_schema,
        ))
        return ModelReply(
            content=reply.content,
            input_tokens=reply.input_tokens or 0,
            output_tokens=reply.output_tokens or 0,
            latency_ms=reply.latency_ms,
            model=reply.model,
            finish_reason=reply.finish_reason,
        )


class AdaptiveMDTReasoner:
    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str | None = None,
        config: MDAgentsStyleConfig | None = None,
        orchestrator: MDAgentsStyleOrchestrator | None = None,
    ) -> None:
        actual_provider = getattr(provider, "provider", provider)
        model_name = model or getattr(actual_provider, "model", "harness_configured_model")
        self.orchestrator = orchestrator or MDAgentsStyleOrchestrator(
            _HarnessProviderBridge(provider, model),
            config=config or MDAgentsStyleConfig(model_name=model_name),
        )

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        execution = await self.orchestrator.run_context(context)
        response = execution.response
        citations = tuple(dict.fromkeys(
            match.group(1) for match in re.finditer(
                r"\[([A-Za-z0-9_.:-]+)\]", response.answer
            )
        ))
        trace_events = tuple(
            {
                "event": f"adaptive_{event.event_type.value}",
                "sequence": event.sequence,
                **dict(event.fields),
            }
            for event in execution.trace_events
        )
        assignments = tuple(
            {
                "event": "adaptive_worker_assignment",
                "task_id": str(task["task_id"]),
                "worker_id": str(task["worker_id"]),
                "role": str(task["role"]),
                "objective_sha256": sha256(
                    str(task["objective"]).encode("utf-8")
                ).hexdigest(),
            }
            for task in execution.shared_context.get("tasks", ())
        )
        worker_statuses = tuple(
            {
                "event": "adaptive_worker_status",
                "worker_id": str(worker_id),
                "status": str(status),
            }
            for worker_id, status in sorted(
                execution.shared_context.get("worker_status", {}).items()
            )
        )
        timeline = tuple(
            {"event": f"adaptive_{row['step']}", **dict(row)}
            for row in execution.shared_context.get("timeline", ())
        )
        validation_audit = ()
        if execution.diagnostic_outputs:
            complexity_outputs = [
                item for item in execution.diagnostic_outputs
                if item["stage"].startswith("complexity_classifier")
            ]
            recruitment_outputs = [
                item for item in execution.diagnostic_outputs
                if item["stage"] in {"dynamic_recruitment", "multi_team_recruitment"}
            ]
            validation_audit = ({
                "event": "adaptive_validation_audit",
                "raw_complexity_output": next((
                    item["raw_output"] for item in reversed(complexity_outputs)
                    if item["stage"] == "complexity_classifier"
                ), None),
                "raw_complexity_outputs": complexity_outputs,
                "raw_recruitment_output": (
                    recruitment_outputs[-1]["raw_output"] if recruitment_outputs else None
                ),
                "raw_recruitment_outputs": recruitment_outputs,
                "validation_error": next((
                    item["validation_error"] for item in reversed(execution.diagnostic_outputs)
                    if item.get("validation_error")
                ), None),
                "validation_errors": [
                    {"stage": item["stage"], "error": item["validation_error"]}
                    for item in execution.diagnostic_outputs if item.get("validation_error")
                ],
                "fallback_reason": execution.failure_reason,
            },)
        return ReasoningResult(
            answer_text=response.answer,
            citation_ids=citations,
            safety_flags=response.safety_flags,
            provider_calls=execution.provider_calls,
            tool_calls=0,
            input_tokens=execution.input_tokens,
            output_tokens=execution.output_tokens,
            reasoning_events=(
                {
                    "event": "reasoning_adaptive_mdt",
                    "complexity": execution.complexity.value,
                    "route_mode": response.route_mode.value,
                    "workers": ",".join(response.workers_used),
                },
                *trace_events,
                *assignments,
                *worker_statuses,
                *timeline,
                *validation_audit,
            ),
            failure_reason=(execution.failure_reason.split(":", 1)[0]
                            if execution.failure_reason else None),
        )
