"""Same-generator R2MED GAR prompts and single-call local generation."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from eval.r2med_crb_data import PINNED_UPSTREAM_COMMIT, UPSTREAM_PROMPT_FAMILY, sha256_file

EXPECTED_UPSTREAM_FILES = {
    "src/eval_BM25.py": "01b6f4d68baea7f3c8d4c0f177ca1c6a17512c694cf9ba0f64ea7078b9fe882d",
    "src/eval_retrieval.py": "dce42ec72f8a2d5e9b71e25bbf773cf27268b2b7e0940fe69494840b38ba7229",
    "src/generate_hypothetical_doc.py": "cc207039a7b5a6937740eac055a48445c1a7d520daed6efe0ba8be63d5687f1a",
    "src/instrcution.py": "8e767d77f22330beb4417b6baebc29be5e7ca59f386e5106162bea56882902a7",
    "src/example.py": "a1636976765b6ce4ce365349e04183bf1d1f40337d2f5f4342f00de03f2363af",
}
GENERATION_CONFIG = {
    "temperature": 0.0,
    "reasoning": "disabled",
    "max_output_tokens": 256,
    "calls_per_query_per_method": 1,
    "retry_on_error": False,
}
METHODS = ("hyde", "query2doc", "lamer", "crb_q", "crb_prf")
CRB_PROMPT = """You are generating a retrieval bridge for a biomedical search system.

Your task is NOT to answer the medical question and NOT to select an answer option.

Transform the query into terminology and a short hypothetical evidence passage that could help a retriever find relevant biomedical documents.

Use only information that can reasonably be inferred from the query. Do not claim certainty about a diagnosis.

Return JSON with exactly:
- canonical_query
- key_concepts
- disambiguating_terms
- pseudo_evidence

Query:
{QUERY}"""
CRB_FEEDBACK_SUFFIX = """\n\nThe following passages are noisy retrieval candidates.
Most may be irrelevant.

Use them only as retrieval hints.
Do not assume any passage is correct.

{PASSAGES}"""
CRB_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["canonical_query", "key_concepts", "disambiguating_terms", "pseudo_evidence"],
    "properties": {
        "canonical_query": {"type": "string"},
        "key_concepts": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string"},
        },
        "disambiguating_terms": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string"},
        },
        "pseudo_evidence": {"type": "string"},
    },
}


class CompletionClient(Protocol):
    def complete(self, prompt: str, *, json_schema: dict[str, Any] | None = None) -> dict[str, Any]: ...


@dataclass(frozen=True)
class GeneratedView:
    query_id: str
    method: str
    generated_text: str
    valid: bool
    completed: bool
    output_tokens: int
    finish_reason: str | None
    truncated: bool
    fallback_original: bool
    structured: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalLlamaCppClient:
    """One-attempt OpenAI-compatible client restricted to loopback llama.cpp."""

    def __init__(self, base_url: str = "http://127.0.0.1:8089/v1", timeout_seconds: int = 600):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("generator endpoint must be plain HTTP on 127.0.0.1 only")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str, *, json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": "local-qwen3-8b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": GENERATION_CONFIG["max_output_tokens"],
            "stream": False,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "clinical_reasoning_bridge", "strict": True, "schema": json_schema},
            }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"local llama.cpp completion failed: {type(exc).__name__}") from exc
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("local llama.cpp returned an invalid choices payload")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("local llama.cpp returned no assistant text")
        usage = payload.get("usage") or {}
        return {
            "text": content,
            "finish_reason": choice.get("finish_reason"),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        }


def load_upstream_prompt_catalog(upstream_root: Path) -> dict[str, dict[str, str]]:
    """Load the pinned upstream templates after verifying commit and source hashes."""
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_UPSTREAM_COMMIT:
        raise ValueError(f"R2MED upstream commit mismatch: expected {PINNED_UPSTREAM_COMMIT}, found {commit}")
    for relative, expected_hash in EXPECTED_UPSTREAM_FILES.items():
        path = upstream_root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"pinned R2MED upstream source hash mismatch: {relative}")

    source_dir = upstream_root / "src"
    old_example = sys.modules.pop("example", None)
    sys.path.insert(0, str(source_dir))
    try:
        namespace = runpy.run_path(str(source_dir / "instrcution.py"))
    finally:
        sys.path.remove(str(source_dir))
        sys.modules.pop("example", None)
        if old_example is not None:
            sys.modules["example"] = old_example

    families = set(UPSTREAM_PROMPT_FAMILY.values())
    prompt_sets: dict[str, dict[str, str]] = {}
    for method in ("hyde", "query2doc", "lamer"):
        raw = namespace.get(method)
        if not isinstance(raw, dict) or not families.issubset(raw):
            raise ValueError(f"pinned upstream prompt family {method!r} is incomplete")
        prompt_sets[method] = {family: raw[family] for family in families}
    return prompt_sets


def _feedback_block(passages: Sequence[str] | None) -> str:
    if not passages:
        return ""
    return "\n".join(
        f"[{index}]. {passage.replace(chr(10), ' ').strip()}"
        for index, passage in enumerate(passages[:10], start=1)
    )


def _valid_crb(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != set(CRB_JSON_SCHEMA["required"]):
        return False
    if not isinstance(value["canonical_query"], str) or not value["canonical_query"].strip():
        return False
    if not isinstance(value["pseudo_evidence"], str) or not value["pseudo_evidence"].strip():
        return False
    for key in ("key_concepts", "disambiguating_terms"):
        items = value[key]
        if not isinstance(items, list) or len(items) > 5 or any(not isinstance(item, str) or not item.strip() for item in items):
            return False
    return len(value["pseudo_evidence"].split()) <= 160


class R2MedGARGenerator:
    """A configured prompt family; each call makes at most one local model request."""

    def __init__(
        self,
        prompt_family: str,
        client: CompletionClient,
        upstream_prompts: dict[str, dict[str, str]],
    ) -> None:
        self.prompt_family = prompt_family
        self.client = client
        self.upstream_prompts = upstream_prompts

    def generate(
        self,
        query_id: str,
        query_text: str,
        method: str,
        optional_feedback_passages: Sequence[str] | None = None,
    ) -> GeneratedView:
        if method not in METHODS:
            raise ValueError(f"unsupported R2MED GAR method: {method}")
        if not query_id or not query_text.strip():
            raise ValueError("query_id and query_text must be non-empty")

        is_crb = method in {"crb_q", "crb_prf"}
        if is_crb:
            prompt = CRB_PROMPT.format(QUERY=query_text)
            if method == "crb_prf":
                prompt += CRB_FEEDBACK_SUFFIX.format(PASSAGES=_feedback_block(optional_feedback_passages))
            schema = CRB_JSON_SCHEMA
        else:
            if self.prompt_family not in self.upstream_prompts[method]:
                raise ValueError(f"prompt family {self.prompt_family!r} is unavailable for {method}")
            template = self.upstream_prompts[method][self.prompt_family]
            if method == "lamer":
                prompt = template.format(TEXT=query_text, PASSAGE=_feedback_block(optional_feedback_passages))
            else:
                prompt = template.format(TEXT=query_text)
            schema = None

        try:
            response = self.client.complete(prompt, json_schema=schema)
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            fallback = {
                "canonical_query": query_text,
                "key_concepts": [],
                "disambiguating_terms": [],
                "pseudo_evidence": query_text,
            } if is_crb else None
            return GeneratedView(
                query_id=query_id,
                method=method,
                generated_text=json.dumps(fallback, ensure_ascii=False) if fallback else query_text,
                valid=False,
                completed=False,
                output_tokens=0,
                finish_reason=None,
                truncated=False,
                fallback_original=True,
                structured=fallback,
                error=type(exc).__name__,
            )

        text = str(response.get("text", ""))
        reason = response.get("finish_reason")
        truncated = reason == "length"
        output_tokens = int(response.get("output_tokens", 0))
        if not is_crb:
            valid = bool(text.strip())
            return GeneratedView(
                query_id=query_id,
                method=method,
                generated_text=text.strip() if valid else query_text,
                valid=valid,
                completed=not truncated,
                output_tokens=output_tokens,
                finish_reason=reason,
                truncated=truncated,
                fallback_original=not valid,
            )

        try:
            structured = json.loads(text)
        except json.JSONDecodeError:
            structured = None
        valid = _valid_crb(structured)
        if not valid:
            structured = {
                "canonical_query": query_text,
                "key_concepts": [],
                "disambiguating_terms": [],
                "pseudo_evidence": query_text,
            }
        return GeneratedView(
            query_id=query_id,
            method=method,
            generated_text=json.dumps(structured, ensure_ascii=False, separators=(",", ":")),
            valid=valid,
            completed=not truncated,
            output_tokens=output_tokens,
            finish_reason=reason,
            truncated=truncated,
            fallback_original=not valid,
            structured=structured,
            error=None if valid else "invalid_crb_json_or_constraints",
        )


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
