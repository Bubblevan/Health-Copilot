"""One async model boundary shared by every reasoning strategy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Protocol


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[Mapping[str, str], ...]
    model: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 512
    timeout_seconds: float = 120.0
    json_mode: bool = False
    json_schema: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        messages = tuple(dict(item) for item in self.messages)
        if not messages:
            raise ValueError("model request needs at least one message")
        if any(item.get("role") not in {"system", "user", "assistant", "tool"}
               or not isinstance(item.get("content"), str) for item in messages):
            raise ValueError("model messages require a supported role and string content")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
            or not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or not isfinite(self.temperature)
        ):
            raise ValueError("model output limit and timeout must be positive")
        if self.model is not None and not self.model.strip():
            raise ValueError("model must be nonempty or None")
        object.__setattr__(self, "messages", messages)
        if self.json_schema is not None:
            schema = dict(self.json_schema)
            if not schema:
                raise ValueError("json_schema must not be empty")
            object.__setattr__(self, "json_schema", schema)


@dataclass(frozen=True)
class ModelReply:
    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    provider_request_id: str | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("model reply must contain nonempty text")
        if not self.model.strip():
            raise ValueError("model reply must identify its model")
        if not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("model latency must be finite and nonnegative")
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None")


class ModelProvider(Protocol):
    async def complete(self, request: ModelRequest) -> ModelReply:
        ...


class VllmModelProvider:
    """OpenAI-compatible vLLM provider. Client retries are disabled by design."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "local-vllm",
        default_temperature: float | None = None,
        default_top_p: float | None = None,
        max_output_tokens: int | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) OpenAI-compatible endpoint")
        if not model.strip():
            raise ValueError("model must not be empty")
        if default_temperature is not None and not isfinite(default_temperature):
            raise ValueError("default_temperature must be finite")
        if default_top_p is not None and (
            not isfinite(default_top_p) or not 0 < default_top_p <= 1
        ):
            raise ValueError("default_top_p must be in (0, 1] or None")
        if max_output_tokens is not None and (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer or None")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - configured runtime dependency
            raise RuntimeError("install health-ai-copilot dependencies for vLLM access") from exc
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.max_output_tokens = max_output_tokens
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=0,
        )

    async def complete(self, request: ModelRequest) -> ModelReply:
        started = monotonic()
        kwargs = {
            "model": request.model or self.model,
            "messages": [dict(item) for item in request.messages],
            "temperature": (
                request.temperature if self.default_temperature is None else self.default_temperature
            ),
            "max_tokens": min(request.max_output_tokens, self.max_output_tokens)
            if self.max_output_tokens is not None else request.max_output_tokens,
            "timeout": request.timeout_seconds,
        }
        if request.json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": dict(request.json_schema),
            }
        elif request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.default_top_p is not None:
            kwargs["top_p"] = self.default_top_p
        if self.chat_template_kwargs:
            kwargs["extra_body"] = {"chat_template_kwargs": self.chat_template_kwargs}
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        if not isinstance(content, str):
            raise TypeError("model response has no text content")
        usage = getattr(response, "usage", None)
        return ModelReply(
            content=content,
            model=getattr(response, "model", None) or request.model or self.model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=(monotonic() - started) * 1000,
            provider_request_id=getattr(response, "_request_id", None),
            finish_reason=getattr(choice, "finish_reason", None),
        )


class LegacyLlamaCppModelProvider(VllmModelProvider):
    """OpenAI-compatible llama.cpp adapter retained for local/legacy development only."""

    adapter_status = "LEGACY_LOCAL_DEVELOPMENT"
