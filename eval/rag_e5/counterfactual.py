"""Teacher-blind runtime helpers for the frozen E5-B2 counterfactual study."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from eval.rag_e5.overlay import ACTION_ORDER, READER_PROMPT, READER_SCHEMA


class CompletionTransportError(RuntimeError):
    """The local model service failed; stop rather than retry an experimental call."""


class CompletionClient(Protocol):
    def complete(
        self, prompt: str, *, json_schema: Mapping[str, Any] | None = None
    ) -> Completion: ...


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None


class LoopbackCompletionClient:
    """One stateless, no-prompt-cache llama.cpp request per call."""

    def __init__(self, base_url: str, *, timeout_seconds: int = 600) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("model endpoint must be plain HTTP on 127.0.0.1")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout_seconds = timeout_seconds

    def complete(
        self, prompt: str, *, json_schema: Mapping[str, Any] | None = None
    ) -> Completion:
        body: dict[str, Any] = {
            "model": "local-qwen3-8b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 256,
            "cache_prompt": False,
            "stream": False,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "e5_reader_output",
                    "strict": True,
                    "schema": dict(json_schema),
                },
            }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise CompletionTransportError(
                f"loopback llama.cpp request failed: {type(exc).__name__}"
            ) from exc
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise CompletionTransportError("llama.cpp response has invalid choices")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content")
        if not isinstance(text, str):
            raise CompletionTransportError("llama.cpp response has no text content")
        usage = payload.get("usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        return Completion(
            text=text,
            finish_reason=choice.get("finish_reason"),
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        )


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_id_for(case_id: str, action: str, lock_sha256: str) -> str:
    if action not in ACTION_ORDER:
        raise ValueError(f"unsupported retrieval action: {action}")
    return canonical_sha256([case_id, action, lock_sha256])


def expected_arm_specs(case_ids: Sequence[str], lock_sha256: str) -> list[dict[str, str]]:
    return [
        {
            "case_id": case_id,
            "action": action,
            "run_id": run_id_for(case_id, action, lock_sha256),
        }
        for case_id in sorted(case_ids)
        for action in ACTION_ORDER
    ]


def render_reader_prompt(
    *, question: str, state_packet: Mapping[str, Any] | None, passages: Sequence[Mapping[str, Any]]
) -> str:
    state_text = (
        json.dumps(state_packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if state_packet is not None
        else "No longitudinal state packet is supplied."
    )
    passage_blocks = [
        f"[{item['chunk_id']}]\n{item['text']}"
        for item in passages
        if isinstance(item.get("chunk_id"), str) and isinstance(item.get("text"), str)
    ]
    return READER_PROMPT.format(
        QUESTION=question,
        STATE_PACKET=state_text,
        PASSAGES="\n\n".join(passage_blocks) if passage_blocks else "None.",
    )


def parse_reader_output(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse once; schema-invalid or malformed outputs are never repaired/retried."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None, False
    if not isinstance(value, dict) or set(value) != set(READER_SCHEMA["required"]):
        return None, False
    state_facts = value.get("state_facts")
    guidance_facts = value.get("guidance_facts")
    citations = value.get("citations")
    if not isinstance(state_facts, list) or not isinstance(guidance_facts, list):
        return None, False
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        return None, False
    if not isinstance(value.get("answer"), str):
        return None, False
    for item in state_facts:
        if not isinstance(item, dict) or set(item) != {"field", "value"}:
            return None, False
        if any(not isinstance(item.get(key), str) for key in ("field", "value")):
            return None, False
    for item in guidance_facts:
        if not isinstance(item, dict) or set(item) != {"statement"}:
            return None, False
        if not isinstance(item.get("statement"), str):
            return None, False
    return value, True


def format_feedback_block(passages: Sequence[str]) -> str:
    return "\n".join(
        f"[{index}]. {passage.replace(chr(10), ' ').strip()}"
        for index, passage in enumerate(passages[:10], start=1)
    )


def render_lamer_prompt(template: str, query: str, passages: Sequence[str]) -> str:
    return template.format(TEXT=query, PASSAGE=format_feedback_block(passages))


def write_complete_arm(path: Any, payload: Mapping[str, Any]) -> str:
    """Write an immutable arm atomically; an existing completed arm is never replaced."""
    import os
    from pathlib import Path

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"completed counterfactual arm is immutable: {target.name}")
    content = dict(payload)
    digest = canonical_sha256(content)
    content["completion_sha256"] = digest
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(content, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return digest


def verify_complete_arm(path: Any, *, expected_run_id: str, expected_lock_sha256: str) -> dict[str, Any]:
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = payload.pop("completion_sha256", None)
    if payload.get("run_id") != expected_run_id or payload.get("lock_sha256") != expected_lock_sha256:
        raise ValueError("completed arm belongs to another run or protocol lock")
    if digest != canonical_sha256(payload):
        raise ValueError("completed arm payload hash mismatch")
    payload["completion_sha256"] = digest
    return payload
