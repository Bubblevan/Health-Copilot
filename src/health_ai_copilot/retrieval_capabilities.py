"""Frozen retrieval action profiles for the RAG-E5 harness study.

These are action contracts, not a claim that every listed source family is
currently eligible. Eligibility is resolved against the active source catalog.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any


class RetrievalAction(StrEnum):
    OFF = "OFF"
    STANDARD = "STANDARD"
    STRONG = "STRONG"


R2MED_FINAL_LOCK_SHA256 = "0b80fad6668c3a57833c55beb1db359c65440cf015effd10da770c630fb3e41d"
R2MED_SOURCE_MANIFEST_SHA256 = "b70c4f01b37f58c77597f1e28cc35a52585785142f3f625f928173c81be3874e"
R2MED_UPSTREAM_COMMIT = "11244a4925a39082967a6c9d38ef01f279c316a5"
QWEN3_8B_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
BGE_LARGE_SHA256 = "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7"
LAMER_MEDICAL_EXAM_PROMPT_SHA256 = "317a15290b029422df211a96550f2f689d98fff61618a1e1847c7f54f8443122"

_SOURCE_FAMILIES = ("public_health", "reviewed_guideline")


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True, slots=True)
class RetrievalCapabilitySpec:
    """Immutable binding from one policy action to a frozen retriever profile."""

    action: RetrievalAction
    retriever_profile_id: str | None
    source_families: tuple[str, ...]
    max_external_calls: int
    requires_generator: bool
    cost_class: str
    frozen_config_json: str
    provenance_lock_sha256: str | None = None

    def __post_init__(self) -> None:
        action = self.action if isinstance(self.action, RetrievalAction) else RetrievalAction(self.action)
        object.__setattr__(self, "action", action)
        raw_families = tuple(self.source_families)
        if any(not isinstance(item, str) or not item.strip() for item in raw_families):
            raise ValueError("source_families must contain non-empty strings")
        families = tuple(sorted(set(raw_families)))
        object.__setattr__(self, "source_families", families)
        if type(self.max_external_calls) is not int or self.max_external_calls < 0:
            raise ValueError("max_external_calls must be a non-negative integer")
        if type(self.requires_generator) is not bool:
            raise TypeError("requires_generator must be boolean")
        if not isinstance(self.cost_class, str) or not self.cost_class.strip():
            raise ValueError("cost_class must be non-empty")
        if self.retriever_profile_id is not None and not self.retriever_profile_id.strip():
            raise ValueError("retriever_profile_id must be non-empty or null")
        try:
            parsed = json.loads(self.frozen_config_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("frozen_config_json must encode a JSON object") from exc
        if not isinstance(parsed, dict) or _canonical_json(parsed) != self.frozen_config_json:
            raise ValueError("frozen_config_json must be canonical JSON object text")
        if self.provenance_lock_sha256 is not None and not _is_sha256(
            self.provenance_lock_sha256
        ):
            raise ValueError("provenance_lock_sha256 must be a SHA-256 digest or null")
        if action == RetrievalAction.OFF:
            if self.retriever_profile_id is not None or families or self.max_external_calls:
                raise ValueError("OFF must not expose an external retriever or source family")
            if self.requires_generator:
                raise ValueError("OFF must not require a generator")
        elif not self.retriever_profile_id or not families or self.max_external_calls < 1:
            raise ValueError("retrieval actions need a profile, source family, and external call")

    @property
    def frozen_config(self) -> dict[str, Any]:
        return json.loads(self.frozen_config_json)

    @property
    def config_sha256(self) -> str:
        return sha256(self.frozen_config_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "retriever_profile_id": self.retriever_profile_id,
            "source_families": list(self.source_families),
            "max_external_calls": self.max_external_calls,
            "requires_generator": self.requires_generator,
            "cost_class": self.cost_class,
            "frozen_config": self.frozen_config,
            "config_sha256": self.config_sha256,
            "provenance_lock_sha256": self.provenance_lock_sha256,
        }


def retrieval_action_specs() -> tuple[RetrievalCapabilitySpec, ...]:
    """Return the three frozen resource tiers; no live corpus is implied."""

    off = {
        "kind": "no_external_retrieval",
        "version": 1,
    }
    standard = {
        "kind": "two_source_rrf",
        "provenance": {
            "final_lock_sha256": R2MED_FINAL_LOCK_SHA256,
            "source_manifest_sha256": R2MED_SOURCE_MANIFEST_SHA256,
            "upstream_commit": R2MED_UPSTREAM_COMMIT,
        },
        "bm25": {"analyzer": "Lucene", "b": 0.4, "k1": 0.9, "top_k": 100},
        "dense": {
            "model": "BAAI/bge-large-en-v1.5",
            "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
            "weights_sha256": BGE_LARGE_SHA256,
            "top_k": 100,
        },
        "rrf": {"k": 60, "weights": [1, 1], "output_depth": 100},
    }
    strong = {
        "kind": "four_view_gar_rrf",
        "method": "LameR-MV",
        "provenance": {
            "final_lock_sha256": R2MED_FINAL_LOCK_SHA256,
            "source_manifest_sha256": R2MED_SOURCE_MANIFEST_SHA256,
            "upstream_commit": R2MED_UPSTREAM_COMMIT,
            "generation_code_sha256": "cc207039a7b5a6937740eac055a48445c1a7d520daed6efe0ba8be63d5687f1a",
            "prompt_code_sha256": "8e767d77f22330beb4417b6baebc29be5e7ca59f386e5106162bea56882902a7",
        },
        "feedback_depth": 10,
        "views": [
            "BM25(original_query)",
            "BM25(generated_bridge)",
            "BGE-large(original_query)",
            "BGE-large(generated_bridge)",
        ],
        "bm25": {"analyzer": "Lucene", "b": 0.4, "k1": 0.9, "top_k": 100},
        "dense": {
            "model": "BAAI/bge-large-en-v1.5",
            "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
            "weights_sha256": BGE_LARGE_SHA256,
            "top_k": 100,
        },
        "generator": {
            "model": "Qwen/Qwen3-8B-GGUF",
            "revision": "6a569868d07d3bd59e8b97fb001bf8c0b254bb20",
            "file": "Qwen3-8B-Q4_K_M.gguf",
            "sha256": QWEN3_8B_SHA256,
            "temperature": 0.0,
            "reasoning": "disabled",
            "max_output_tokens": 256,
            "calls_per_query": 1,
            "retry_on_error": False,
        },
        "prompt_binding": {
            "family": "MedQA-Diag",
            "template_sha256": LAMER_MEDICAL_EXAM_PROMPT_SHA256,
            "mapping": "E5 integration questions use the pinned generic medical-exam LameR template without prompt edits.",
        },
        "rrf": {"k": 20, "weights": [1, 2, 1, 2], "config_id": "k20-W2", "output_depth": 100},
    }
    return (
        RetrievalCapabilitySpec(
            action=RetrievalAction.OFF,
            retriever_profile_id=None,
            source_families=(),
            max_external_calls=0,
            requires_generator=False,
            cost_class="none",
            frozen_config_json=_canonical_json(off),
        ),
        RetrievalCapabilitySpec(
            action=RetrievalAction.STANDARD,
            retriever_profile_id="r2med-bm25-bge-rrf-v1",
            source_families=_SOURCE_FAMILIES,
            max_external_calls=1,
            requires_generator=False,
            cost_class="standard_hybrid",
            frozen_config_json=_canonical_json(standard),
            provenance_lock_sha256=R2MED_FINAL_LOCK_SHA256,
        ),
        RetrievalCapabilitySpec(
            action=RetrievalAction.STRONG,
            retriever_profile_id="r2med-lamer-mv-v1",
            source_families=_SOURCE_FAMILIES,
            max_external_calls=1,
            requires_generator=True,
            cost_class="generative_retrieval",
            frozen_config_json=_canonical_json(strong),
            provenance_lock_sha256=R2MED_FINAL_LOCK_SHA256,
        ),
    )


def require_action_available(
    action: RetrievalAction | str,
    available_actions: tuple[RetrievalAction | str, ...],
    available_source_families: tuple[str, ...],
) -> RetrievalCapabilitySpec:
    """Resolve only an explicitly exposed action; fail closed on unavailable tiers."""

    selected = action if isinstance(action, RetrievalAction) else RetrievalAction(action)
    exposed = {
        item if isinstance(item, RetrievalAction) else RetrievalAction(item)
        for item in available_actions
    }
    if selected not in exposed:
        raise ValueError(f"retrieval action is not available: {selected.value}")
    spec = next(item for item in retrieval_action_specs() if item.action == selected)
    if selected != RetrievalAction.OFF and not set(spec.source_families).intersection(
        available_source_families
    ):
        raise ValueError(f"no eligible source family is available for {selected.value}")
    return spec


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


__all__ = [
    "RetrievalAction",
    "RetrievalCapabilitySpec",
    "require_action_available",
    "retrieval_action_specs",
]
