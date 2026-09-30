"""Plain-text guidance generation against the shared local llama.cpp server."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

SYSTEM_PROMPT = "You write concise, cautious public-health guidance for a health information system."
PROMPT_TEMPLATE = """Answer only the general guidance portion of the user's question.
Answer concisely; usually 1–3 sentences are enough. Do not diagnose, prescribe,
or personalize recommendations. Longitudinal state is materialized separately by
the runtime: do not restate or infer personal state facts.

Frozen external evidence (when present):
{EVIDENCE}

Use the supplied evidence when it supports the requested guidance. If evidence is
absent, answer the general guidance portion from your existing knowledge if possible.
Do not fabricate citations. When evidence is supplied, cite supporting statements
inline using only the bracketed aliases shown above, for example [E1].

Question:
{QUESTION}
"""
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
MAX_OUTPUT_TOKENS = 8192
CONTEXT_SIZE = 32768
CHAT_TEMPLATE_OVERHEAD_RESERVE = 256


class LlamaServerError(RuntimeError):
    """The pinned loopback inference path is unavailable or violates its contract."""


@dataclass(frozen=True, slots=True)
class GuidanceCompletion:
    text: str
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    request_payload: Mapping[str, Any]


def render_guidance_prompt(
    question: str, passages: Sequence[Mapping[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("guidance question must be non-empty text")
    aliases: list[dict[str, Any]] = []
    blocks: list[str] = []
    if len(passages) > 5:
        raise ValueError("B4 guidance may receive only the frozen B2 top-five passages")
    for index, passage in enumerate(passages, start=1):
        chunk_id, text = passage.get("chunk_id"), passage.get("text")
        if not isinstance(chunk_id, str) or not isinstance(text, str) or not text.strip():
            raise ValueError("each frozen passage must have a chunk ID and non-empty text")
        alias = f"E{index}"
        aliases.append(
            {
                "alias": alias,
                "chunk_id": chunk_id,
                "source_id": passage.get("source_id"),
                "recommendation_id": passage.get("recommendation_id"),
                "section_path": passage.get("section_path", []),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
        blocks.append(f"[{alias}]\n{text}")
    evidence = "\n\n".join(blocks) if blocks else "No external evidence is available."
    return PROMPT_TEMPLATE.format(EVIDENCE=evidence, QUESTION=question), aliases


def chat_payload(prompt: str) -> dict[str, Any]:
    """Frozen same-model decoding request; no JSON schema or retry controls."""
    return {
        "model": "local-qwen3-8b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "cache_prompt": False,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


class LlamaServerClient:
    """Minimal client that never starts, stops, or reconfigures the shared server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8081", timeout_seconds: int = 1800):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("B4 model endpoint must remain bound to 127.0.0.1")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request_json(
        self, path: str, *, method: str = "GET", payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                value = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LlamaServerError(f"loopback request failed at {path}: {type(exc).__name__}") from exc
        if not isinstance(value, (dict, list)):
            raise LlamaServerError(f"loopback endpoint returned an invalid payload at {path}")
        return value

    def health(self) -> Mapping[str, Any]:
        value = self._request_json("/health")
        if not isinstance(value, dict) or value.get("status") != "ok":
            raise LlamaServerError("shared llama.cpp server is not healthy")
        return value

    def props(self) -> Mapping[str, Any]:
        value = self._request_json("/props")
        if not isinstance(value, dict):
            raise LlamaServerError("shared llama.cpp /props response is invalid")
        return value

    def slots(self) -> Sequence[Mapping[str, Any]]:
        value = self._request_json("/slots")
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise LlamaServerError("shared llama.cpp /slots response is invalid")
        return value

    def count_prompt_tokens(self, prompt: str) -> int:
        token_input = f"{SYSTEM_PROMPT}\n{prompt}"
        value = self._request_json(
            "/tokenize",
            method="POST",
            payload={"content": token_input, "add_special": True, "parse_special": True},
        )
        tokens = value.get("tokens") if isinstance(value, dict) else None
        if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
            raise LlamaServerError("llama.cpp tokenizer returned an invalid token sequence")
        return len(tokens)

    def complete(self, prompt: str) -> GuidanceCompletion:
        payload = chat_payload(prompt)
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LlamaServerError(
                f"shared llama.cpp generation failed: {type(exc).__name__}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        choices = result.get("choices") if isinstance(result, dict) else None
        if not isinstance(choices, list) or len(choices) != 1:
            raise LlamaServerError("llama.cpp response must contain exactly one choice")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LlamaServerError("llama.cpp response has no plain-text message content")
        reasoning = message.get("reasoning_content")
        text = message["content"]
        if (isinstance(reasoning, str) and reasoning.strip()) or "<think>" in text.casefold():
            raise LlamaServerError("server returned reasoning despite the frozen disabled setting")
        usage = result.get("usage") or {}
        input_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        output_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        return GuidanceCompletion(
            text=text,
            finish_reason=choice.get("finish_reason"),
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            latency_ms=latency_ms,
            request_payload=payload,
        )


__all__ = [
    "CHAT_TEMPLATE_OVERHEAD_RESERVE",
    "CONTEXT_SIZE",
    "LlamaServerClient",
    "LlamaServerError",
    "MAX_OUTPUT_TOKENS",
    "PROMPT_TEMPLATE_SHA256",
    "SYSTEM_PROMPT_SHA256",
    "chat_payload",
    "render_guidance_prompt",
]
