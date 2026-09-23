"""Research-only Jev priority suggestions for unprotected context candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..runtime.context_manager import ContextItem, ContextItemCategory, ContextPriority
from .architecture import _require_research_safe_classification
from .jev import JevClient, JevResult


ELIGIBLE_CATEGORIES = frozenset(
    {
        ContextItemCategory.MEMORY,
        ContextItemCategory.RECENT_HISTORY,
        ContextItemCategory.SESSION_SUMMARY,
        ContextItemCategory.TOOL_EXCHANGE,
    }
)


@dataclass(frozen=True)
class ContextSelectionDecision:
    keep_probabilities: Mapping[str, float]
    priority_hints: Mapping[str, str]
    latency_ms: int
    model: str
    input_tokens: int
    output_tokens: int
    candidate_count: int

    def metadata(self) -> dict[str, Any]:
        return {
            "keep_probabilities": dict(self.keep_probabilities),
            "priority_hints": dict(self.priority_hints),
            "latency_ms": self.latency_ms,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "candidate_count": self.candidate_count,
        }


class JevContextSelector:
    """Suggest high/low priority only; the ContextManager keeps final authority."""

    version = "jev-context-priority-v1"

    def __init__(
        self,
        client: JevClient | None = None,
        *,
        keep_threshold: float = 0.5,
        max_candidates: int = 64,
    ) -> None:
        if not 0.0 <= keep_threshold <= 1.0:
            raise ValueError("keep_threshold must be between 0 and 1")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.client = client or JevClient()
        self.keep_threshold = keep_threshold
        self.max_candidates = max_candidates

    async def advise(
        self,
        *,
        current_task: str,
        candidates: Sequence[ContextItem],
        data_classification: str,
    ) -> ContextSelectionDecision:
        _require_research_safe_classification(data_classification)
        if not current_task.strip():
            raise ValueError("current_task must be non-empty")
        grouped = _candidate_groups(candidates)
        for item in (item for items in grouped.values() for item in items):
            if item.provenance.get("research_data_classification") != data_classification:
                raise ValueError(
                    "each context candidate must be explicitly marked with the matching "
                    "research_data_classification before it can be sent to Jev"
                )
        if len(grouped) > self.max_candidates:
            raise ValueError(
                f"Jev context selection is limited to {self.max_candidates} candidates per request"
            )
        if not grouped:
            return ContextSelectionDecision({}, {}, 0, "not-called", 0, 0, 0)
        questions: dict[str, dict[str, Any]] = {}
        question_to_items: dict[str, tuple[ContextItem, ...]] = {}
        state_candidates: list[dict[str, Any]] = []
        for group_key, items in grouped.items():
            question_name = "keep_" + hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:16]
            question_to_items[question_name] = items
            state_candidates.append(
                {
                    "candidate_key": group_key,
                    "items": [
                        {
                            "item_id": item.item_id,
                            "category": item.category.value,
                            "estimated_tokens": item.estimated_tokens,
                            "priority": item.priority.value,
                            "content": item.content,
                        }
                        for item in items
                    ],
                }
            )
            questions[question_name] = {
                "type": "noul",
                "instructions": (
                    f"Keep this unprotected context candidate when preparing an answer to "
                    f"the current task? Candidate: {group_key}"
                ),
                "criteria": {
                    "true": "The candidate is likely to help answer the current task.",
                    "false": "The candidate is unrelated, redundant, stale, or low value for this task.",
                },
            }
        result = await self.client.evaluate(
            state={"current_task": current_task, "candidates": state_candidates},
            questions=questions,
        )
        return self._decision(result, question_to_items)

    def _decision(
        self,
        result: JevResult,
        question_to_items: Mapping[str, tuple[ContextItem, ...]],
    ) -> ContextSelectionDecision:
        probabilities: dict[str, float] = {}
        hints: dict[str, str] = {}
        for question_name, items in question_to_items.items():
            answer = result.answers.get(question_name)
            if not isinstance(answer, Mapping) or answer.get("type") != "noul":
                raise ValueError(f"Jev returned an invalid context answer for {question_name}")
            probability = answer.get("noul")
            if not isinstance(probability, (int, float)) or not 0 <= float(probability) <= 1:
                raise ValueError("Jev context keep probability must be between 0 and 1")
            keep_probability = float(probability)
            priority = ContextPriority.HIGH if keep_probability >= self.keep_threshold else ContextPriority.LOW
            probabilities.update({item.item_id: keep_probability for item in items})
            hints.update({item.item_id: priority.value for item in items})
        return ContextSelectionDecision(
            keep_probabilities=probabilities,
            priority_hints=hints,
            latency_ms=result.latency_ms,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            candidate_count=len(probabilities),
        )


def _candidate_groups(candidates: Sequence[ContextItem]) -> dict[str, tuple[ContextItem, ...]]:
    by_group: dict[str, list[ContextItem]] = {}
    ungrouped: list[ContextItem] = []
    for item in candidates:
        if item.group_id:
            by_group.setdefault(item.group_id, []).append(item)
        else:
            ungrouped.append(item)
    grouped: dict[str, tuple[ContextItem, ...]] = {}
    for group_id, items in by_group.items():
        if any(item.protected or item.priority == ContextPriority.PROTECTED for item in items):
            continue
        if all(item.category in ELIGIBLE_CATEGORIES for item in items):
            grouped[f"group:{group_id}"] = tuple(items)
    for item in ungrouped:
        if (
            not item.protected
            and item.priority != ContextPriority.PROTECTED
            and item.category in ELIGIBLE_CATEGORIES
        ):
            grouped[f"item:{item.item_id}"] = (item,)
    result = grouped
    for key, items in result.items():
        if key.startswith("group:") and len(items) < 2:
            raise ValueError(f"atomic context group must contain at least two items: {key}")
    return result


__all__ = ["ContextSelectionDecision", "JevContextSelector"]
