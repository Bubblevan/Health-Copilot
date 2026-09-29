"""Leakage-safe runtime contract for RAG-E5 execution policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Protocol

from .retrieval_capabilities import RetrievalAction, retrieval_action_specs

_E5_SOURCE_FAMILIES = frozenset({"public_health", "reviewed_guideline"})


@dataclass(frozen=True, slots=True)
class ExecutionPolicyObservation:
    """Decision-time facts only; evaluator labels and action outcomes are excluded."""

    query_text: str
    history_turn_count: int = 0
    history_time_span_days: int = 0
    recent_event_count: int = 0
    exam_record_count: int = 0
    active_condition_count: int = 0
    memory_available: bool = False
    memory_evidence_count: int = 0
    memory_source_types: tuple[str, ...] = ()
    memory_temporal_span_days: int = 0
    memory_conflict_flag: bool = False
    available_source_families: tuple[str, ...] = ()
    available_retrieval_actions: tuple[RetrievalAction, ...] = (RetrievalAction.OFF,)
    remaining_provider_budget: int = 0
    remaining_tool_budget: int = 0
    remaining_token_budget: int = 0
    deadline_remaining_ms: int | None = None
    previous_tool_failures: int = 0
    state_summary: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query_text, str) or not self.query_text.strip():
            raise ValueError("query_text must be non-empty")
        for name in (
            "history_turn_count",
            "history_time_span_days",
            "recent_event_count",
            "exam_record_count",
            "active_condition_count",
            "memory_evidence_count",
            "memory_temporal_span_days",
            "remaining_provider_budget",
            "remaining_tool_budget",
            "remaining_token_budget",
            "previous_tool_failures",
        ):
            _require_non_negative_int(name, getattr(self, name))
        if self.deadline_remaining_ms is not None:
            _require_non_negative_int("deadline_remaining_ms", self.deadline_remaining_ms)
        for name in ("memory_available", "memory_conflict_flag"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        for name in ("memory_source_types", "available_source_families"):
            values = _string_tuple(name, getattr(self, name))
            if name == "available_source_families" and not set(values).issubset(
                _E5_SOURCE_FAMILIES
            ):
                raise ValueError("available_source_families contains a family outside E5 v1")
            object.__setattr__(self, name, tuple(sorted(set(values))))
        actions = tuple(
            item if isinstance(item, RetrievalAction) else RetrievalAction(item)
            for item in self.available_retrieval_actions
        )
        if len(actions) != len(set(actions)):
            raise ValueError("available_retrieval_actions must not contain duplicates")
        actions = tuple(sorted(actions, key=lambda item: item.value))
        if RetrievalAction.OFF not in actions:
            raise ValueError("OFF must always remain an available fallback action")
        if any(item != RetrievalAction.OFF for item in actions) and not self.available_source_families:
            raise ValueError("external retrieval actions require an available source family")
        specs = {item.action: item for item in retrieval_action_specs()}
        for action in actions:
            if action != RetrievalAction.OFF and not set(specs[action].source_families).intersection(
                self.available_source_families
            ):
                raise ValueError(f"available source families cannot serve {action.value}")
        object.__setattr__(self, "available_retrieval_actions", actions)
        if self.state_summary is not None and not isinstance(self.state_summary, str):
            raise TypeError("state_summary must be a string or null")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionPolicyObservation:
        if not isinstance(value, Mapping):
            raise TypeError("policy observation must be a mapping")
        if any(not isinstance(key, str) for key in value):
            raise TypeError("policy observation keys must be strings")
        allowed = {item.name for item in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"policy observation contains non-runtime fields: {sorted(unknown)}")
        if "query_text" not in value:
            raise ValueError("policy observation requires query_text")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_text": self.query_text,
            "history_turn_count": self.history_turn_count,
            "history_time_span_days": self.history_time_span_days,
            "recent_event_count": self.recent_event_count,
            "exam_record_count": self.exam_record_count,
            "active_condition_count": self.active_condition_count,
            "memory_available": self.memory_available,
            "memory_evidence_count": self.memory_evidence_count,
            "memory_source_types": list(self.memory_source_types),
            "memory_temporal_span_days": self.memory_temporal_span_days,
            "memory_conflict_flag": self.memory_conflict_flag,
            "available_source_families": list(self.available_source_families),
            "available_retrieval_actions": [item.value for item in self.available_retrieval_actions],
            "remaining_provider_budget": self.remaining_provider_budget,
            "remaining_tool_budget": self.remaining_tool_budget,
            "remaining_token_budget": self.remaining_token_budget,
            "deadline_remaining_ms": self.deadline_remaining_ms,
            "previous_tool_failures": self.previous_tool_failures,
            "state_summary": self.state_summary,
        }


class ExecutionPolicy(Protocol):
    """Policy interface deliberately receives no case/evaluator object."""

    def decide(self, observation: ExecutionPolicyObservation) -> RetrievalAction:
        ...


def require_policy_observation(value: object) -> ExecutionPolicyObservation:
    if type(value) is not ExecutionPolicyObservation:
        raise TypeError("execution policy accepts only ExecutionPolicyObservation")
    return value


def _require_non_negative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _string_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a sequence of strings")
    items = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{name} must contain non-empty strings")
    return items


__all__ = ["ExecutionPolicy", "ExecutionPolicyObservation", "require_policy_observation"]
