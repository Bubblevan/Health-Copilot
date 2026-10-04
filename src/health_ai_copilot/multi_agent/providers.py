"""Swappable chat-model boundary, including a local llama.cpp adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from .contracts import TriageDecision


@dataclass(frozen=True)
class ModelReply:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""


class ModelProvider(Protocol):
    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        timeout_seconds: float,
        json_mode: bool = False,
    ) -> ModelReply:
        """Return one bounded completion; retries are owned by the harness."""


class TriageProvider(Protocol):
    """Classify routing needs from user text and explicitly observable context only."""

    async def triage(self, query: str, observable_context: str) -> TriageDecision:
        ...


class LocalVllmProvider:
    """OpenAI-compatible loopback vLLM provider with Qwen thinking disabled."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "qwen3-8b-local",
        api_key: str = "EMPTY",
    ) -> None:
        from urllib.parse import urlsplit

        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("LocalVllmProvider only permits an HTTP loopback endpoint")
        try:
            import httpx
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - exercised in configured installs
            raise RuntimeError("openai and httpx are required for the local vLLM provider") from exc
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._http_client = httpx.AsyncClient(trust_env=False)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=120.0,
            http_client=self._http_client,
        )

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        timeout_seconds: float,
        json_mode: bool = False,
    ) -> ModelReply:
        return await self.complete_messages(
            messages=(
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ),
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            json_mode=json_mode,
        )

    async def complete_messages(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        max_output_tokens: int,
        timeout_seconds: float,
        json_mode: bool = False,
    ) -> ModelReply:
        """Complete a provider-visible conversation without flattening its role history."""
        kwargs = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": 0,
            "max_tokens": max_output_tokens,
            "timeout": timeout_seconds,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        started = monotonic()
        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = response.usage
        return ModelReply(
            content=content,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=(monotonic() - started) * 1000,
            model=str(getattr(response, "model", self.model)),
        )

    async def healthcheck(self) -> tuple[str, ...]:
        response = await self._client.models.list()
        return tuple(str(item.id) for item in response.data)

    async def close(self) -> None:
        await self._client.close()


class LocalLlamaCppProvider:
    """OpenAI-compatible local llama.cpp provider; it never falls back to cloud."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8082/v1",
        model: str = "Qwen3-8B-Q4_K_M.gguf",
        api_key: str = "local-llama-cpp",
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - exercised in configured installs
            raise RuntimeError("openai is required for the local llama.cpp provider") from exc
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=120.0,
        )

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        timeout_seconds: float,
        json_mode: bool = False,
    ) -> ModelReply:
        kwargs = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt.rstrip()}\n/no_think"
                        if "qwen3" in self.model.casefold() else system_prompt
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max_output_tokens,
            "timeout": timeout_seconds,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        started = monotonic()
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage
        return ModelReply(
            content=content,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=(monotonic() - started) * 1000,
            model=str(getattr(response, "model", self.model)),
        )

    async def healthcheck(self) -> tuple[str, ...]:
        """Return served model IDs without sending an evaluation prompt."""
        response = await self._client.models.list()
        return tuple(str(item.id) for item in response.data)
