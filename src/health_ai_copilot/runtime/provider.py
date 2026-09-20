"""Provider-neutral model execution boundary used by domain adapters."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

from ..config import OpenAIConfig
from .budget import BudgetDenied
from .context import RunContext
from .trace import TraceEventType, canonical_json_sha256


class ProviderCallKind(StrEnum):
    GENERATOR = "generator"
    AGENT = "agent"
    POLICY = "policy"
    GROUNDING_VERIFIER = "grounding_verifier"
    CLAIM_SUPPORT_VERIFIER = "claim_support_verifier"


class ProviderFailureKind(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    CONNECTION = "connection"
    MALFORMED_RESPONSE = "malformed_response"
    PROVIDER_ERROR = "provider_error"
    BUDGET_DENIED = "budget_denied"
    REPLAY_MISMATCH = "replay_mismatch"


class ProviderFailure(RuntimeError):
    """Normalized non-content provider failure safe for domain adapters to map."""

    def __init__(self, kind: ProviderFailureKind) -> None:
        self.kind = kind
        super().__init__(kind.value)


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ProviderRequest:
    call_id: str
    kind: ProviderCallKind
    model: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...] = ()
    response_format: Mapping[str, Any] | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    timeout_seconds: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: ProviderCallKind,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> "ProviderRequest":
        return cls(
            call_id=f"provider-{uuid4().hex}",
            kind=kind,
            model=model,
            messages=tuple(messages),
            **kwargs,
        )


@dataclass(frozen=True)
class ProviderResponse:
    call_id: str
    kind: ProviderCallKind
    model: str
    content: str | None
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    finish_reason: str | None = None
    usage: ProviderUsage | None = None
    latency_ms: float | None = None
    provider_request_id: str | None = None


class ProviderExecutor(Protocol):
    """Owns provider side effects; adapters retain domain parsing."""

    def execute(self, request: ProviderRequest, runtime: RunContext) -> ProviderResponse:
        ...


def provider_request_fingerprint(request: ProviderRequest) -> str:
    """Hash all behaviorally relevant request fields for deterministic replay matching."""

    return canonical_json_sha256(
        {
            "kind": request.kind.value,
            "model": request.model,
            "messages": request.messages,
            "tools": request.tools,
            "response_format": request.response_format,
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
        }
    )


class OpenAICompatibleProviderExecutor:
    """The only M4.1 location that creates an OpenAI-compatible live client."""

    def __init__(self, config: OpenAIConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderFailure(ProviderFailureKind.PROVIDER_ERROR) from exc
        # SDK retries must remain disabled. Any future retry policy belongs in
        # this harness, where each attempt can consume budget and be traced.
        kwargs: dict[str, Any] = {"api_key": config.api_key, "max_retries": 0}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = OpenAI(**kwargs)

    def execute(self, request: ProviderRequest, runtime: RunContext) -> ProviderResponse:
        fingerprint = provider_request_fingerprint(request)
        try:
            remaining_seconds = runtime.budget.guard_provider()
        except BudgetDenied as exc:
            _trace_budget_denied(runtime, "provider", str(exc))
            raise ProviderFailure(ProviderFailureKind.BUDGET_DENIED) from exc
        _trace_provider_start(runtime, request, fingerprint)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": list(request.messages),
        }
        if request.tools:
            kwargs["tools"] = list(request.tools)
            kwargs["tool_choice"] = "auto"
        if request.response_format is not None:
            kwargs["response_format"] = dict(request.response_format)
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        timeout = request.timeout_seconds
        if remaining_seconds is not None:
            timeout = min(timeout, remaining_seconds) if timeout is not None else remaining_seconds
        if timeout is not None:
            kwargs["timeout"] = timeout
        started = monotonic()
        try:
            raw = self._client.chat.completions.create(**kwargs)
            message = raw.choices[0].message
        except Exception as exc:  # SDK exception classes remain vendor-specific here.
            failure = ProviderFailure(_failure_kind(exc))
            _trace_provider_error(runtime, request, fingerprint, failure.kind)
            raise failure from exc
        response = ProviderResponse(
            call_id=request.call_id,
            kind=request.kind,
            model=request.model,
            content=getattr(message, "content", None),
            tool_calls=tuple(_normalize_tool_call(item) for item in (getattr(message, "tool_calls", None) or ())),
            finish_reason=getattr(raw.choices[0], "finish_reason", None),
            usage=_usage(getattr(raw, "usage", None)),
            latency_ms=(monotonic() - started) * 1000,
            provider_request_id=getattr(raw, "_request_id", None),
        )
        runtime.budget.record_usage(response.usage)
        _trace_provider_end(runtime, response, fingerprint)
        return response


class FakeProviderExecutor:
    """Deterministic executor for offline adapter tests."""

    def __init__(self, responses: Sequence[ProviderResponse] = (), failure: ProviderFailure | None = None) -> None:
        self._responses = list(responses)
        self._failure = failure
        self.requests: list[ProviderRequest] = []

    def execute(self, request: ProviderRequest, runtime: RunContext) -> ProviderResponse:
        fingerprint = provider_request_fingerprint(request)
        try:
            runtime.budget.guard_provider()
        except BudgetDenied as exc:
            _trace_budget_denied(runtime, "provider", str(exc))
            raise ProviderFailure(ProviderFailureKind.BUDGET_DENIED) from exc
        self.requests.append(request)
        _trace_provider_start(runtime, request, fingerprint)
        if self._failure is not None:
            _trace_provider_error(runtime, request, fingerprint, self._failure.kind)
            raise self._failure
        if not self._responses:
            failure = ProviderFailure(ProviderFailureKind.MALFORMED_RESPONSE)
            _trace_provider_error(runtime, request, fingerprint, failure.kind)
            raise failure
        response = self._responses.pop(0)
        normalized = ProviderResponse(
            call_id=request.call_id,
            kind=request.kind,
            model=request.model,
            content=response.content,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason,
            usage=response.usage,
            latency_ms=response.latency_ms,
            provider_request_id=response.provider_request_id,
        )
        runtime.budget.record_usage(normalized.usage)
        _trace_provider_end(runtime, normalized, fingerprint)
        return normalized


def _usage(raw: object) -> ProviderUsage | None:
    if raw is None:
        return None
    return ProviderUsage(
        input_tokens=getattr(raw, "prompt_tokens", None),
        output_tokens=getattr(raw, "completion_tokens", None),
        total_tokens=getattr(raw, "total_tokens", None),
    )


def _normalize_tool_call(raw: object) -> Mapping[str, Any]:
    function = getattr(raw, "function", None)
    return {
        "id": getattr(raw, "id", ""),
        "type": "function",
        "function": {
            "name": getattr(function, "name", ""),
            "arguments": getattr(function, "arguments", ""),
        },
    }


def _failure_kind(exc: Exception) -> ProviderFailureKind:
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return ProviderFailureKind.TIMEOUT
    if "rate" in name and "limit" in name:
        return ProviderFailureKind.RATE_LIMIT
    if "auth" in name or "permission" in name:
        return ProviderFailureKind.AUTH
    if "connection" in name or "connect" in name:
        return ProviderFailureKind.CONNECTION
    return ProviderFailureKind.PROVIDER_ERROR


def _trace_provider_start(
    runtime: RunContext, request: ProviderRequest, fingerprint: str
) -> None:
    if runtime.trace is None:
        return
    runtime.trace.emit(
        TraceEventType.PROVIDER_START,
        call_id=request.call_id,
        kind=request.kind.value,
        model=request.model,
        request_fingerprint=fingerprint,
        tool_count=len(request.tools),
        response_format=bool(request.response_format),
        temperature=request.temperature,
    )


def _trace_provider_end(
    runtime: RunContext, response: ProviderResponse, fingerprint: str
) -> None:
    if runtime.trace is None:
        return
    runtime.trace.emit(
        TraceEventType.PROVIDER_END,
        call_id=response.call_id,
        kind=response.kind.value,
        model=response.model,
        request_fingerprint=fingerprint,
        finish_reason=response.finish_reason,
        tool_call_count=len(response.tool_calls),
        usage_total=response.usage.total_tokens if response.usage else None,
        latency_ms=response.latency_ms,
    )


def _trace_provider_error(
    runtime: RunContext,
    request: ProviderRequest,
    fingerprint: str,
    failure_kind: ProviderFailureKind,
) -> None:
    if runtime.trace is None:
        return
    runtime.trace.emit(
        TraceEventType.PROVIDER_ERROR,
        call_id=request.call_id,
        kind=request.kind.value,
        model=request.model,
        request_fingerprint=fingerprint,
        failure_kind=failure_kind.value,
    )


def _trace_budget_denied(runtime: RunContext, side_effect: str, reason: str) -> None:
    if runtime.trace is not None:
        runtime.trace.emit(
            TraceEventType.BUDGET_DENIED,
            side_effect=side_effect,
            reason=reason,
        )
