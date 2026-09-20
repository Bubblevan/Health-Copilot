"""Provider-neutral model execution boundary used by domain adapters."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

from ..config import OpenAIConfig
from .context import RunContext


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


class OpenAICompatibleProviderExecutor:
    """The only M4.1 location that creates an OpenAI-compatible live client."""

    def __init__(self, config: OpenAIConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderFailure(ProviderFailureKind.PROVIDER_ERROR) from exc
        kwargs: dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = OpenAI(**kwargs)

    def execute(self, request: ProviderRequest, runtime: RunContext) -> ProviderResponse:
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
        if request.timeout_seconds is not None:
            kwargs["timeout"] = request.timeout_seconds
        started = monotonic()
        try:
            raw = self._client.chat.completions.create(**kwargs)
            message = raw.choices[0].message
        except Exception as exc:  # SDK exception classes remain vendor-specific here.
            raise ProviderFailure(_failure_kind(exc)) from exc
        return ProviderResponse(
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


class FakeProviderExecutor:
    """Deterministic executor for offline adapter tests."""

    def __init__(self, responses: Sequence[ProviderResponse] = (), failure: ProviderFailure | None = None) -> None:
        self._responses = list(responses)
        self._failure = failure
        self.requests: list[ProviderRequest] = []

    def execute(self, request: ProviderRequest, runtime: RunContext) -> ProviderResponse:
        self.requests.append(request)
        if self._failure is not None:
            raise self._failure
        if not self._responses:
            raise ProviderFailure(ProviderFailureKind.MALFORMED_RESPONSE)
        response = self._responses.pop(0)
        return ProviderResponse(
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
