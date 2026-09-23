"""Deterministic context selection and structured compaction for M10."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol
from uuid import uuid4

from .memory import ContextIntent, MemoryRecord
from .session import SessionEvent, SessionEventType


class ContextItemCategory(StrEnum):
    CURRENT_USER = "current_user"
    CURRENT_EVIDENCE = "current_evidence"
    MEMORY = "memory"
    RECENT_HISTORY = "recent_history"
    TOOL_EXCHANGE = "tool_exchange"
    SESSION_SUMMARY = "session_summary"
    SYSTEM_PIN = "system_pin"


class ContextPriority(StrEnum):
    PROTECTED = "protected"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ContextFailure(RuntimeError):
    pass


class ContextBudgetExceeded(ContextFailure):
    code = "context_budget_exhausted"


class ContextAtomicityViolation(ContextFailure):
    code = "context_atomicity_violation"


@dataclass(frozen=True)
class ContextBudget:
    max_estimated_input_tokens: int = 4096
    reserved_system_tokens: int = 256
    reserved_current_turn_tokens: int = 512
    max_memory_tokens: int = 1024
    max_history_tokens: int = 2048
    max_tool_observation_tokens: int = 1024

    def __post_init__(self) -> None:
        for name in (
            "max_estimated_input_tokens",
            "reserved_system_tokens",
            "reserved_current_turn_tokens",
            "max_memory_tokens",
            "max_history_tokens",
            "max_tool_observation_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"context budget {name} must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_estimated_input_tokens": self.max_estimated_input_tokens,
            "reserved_system_tokens": self.reserved_system_tokens,
            "reserved_current_turn_tokens": self.reserved_current_turn_tokens,
            "max_memory_tokens": self.max_memory_tokens,
            "max_history_tokens": self.max_history_tokens,
            "max_tool_observation_tokens": self.max_tool_observation_tokens,
        }


class TokenEstimator(Protocol):
    version: str

    def estimate(self, value: Any) -> int: ...


class DeterministicTokenEstimator:
    """A stable estimate, explicitly not provider billing usage."""

    version = "chars4-cjk1-v1"

    def estimate(self, value: Any) -> int:
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ascii_count = sum(ord(char) < 128 for char in value)
        cjk_count = len(value) - ascii_count
        return max(1, (ascii_count + 3) // 4 + cjk_count)


@dataclass(frozen=True)
class ContextItem:
    item_id: str
    category: ContextItemCategory
    content: Any
    estimated_tokens: int
    priority: ContextPriority = ContextPriority.NORMAL
    protected: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)
    group_id: str | None = None

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("context item_id must be non-empty")
        if self.estimated_tokens < 0:
            raise ValueError("context item estimated_tokens must be non-negative")
        object.__setattr__(self, "category", ContextItemCategory(self.category))
        object.__setattr__(self, "priority", ContextPriority(self.priority))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if self.category == ContextItemCategory.SYSTEM_PIN:
            object.__setattr__(self, "protected", True)
            object.__setattr__(self, "priority", ContextPriority.PROTECTED)
        elif self.protected and self.priority != ContextPriority.PROTECTED:
            object.__setattr__(self, "priority", ContextPriority.PROTECTED)

    @property
    def content_sha256(self) -> str:
        encoded = json.dumps(self.content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return sha256(encoded.encode("utf-8")).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category.value,
            "estimated_tokens": self.estimated_tokens,
            "priority": self.priority.value,
            "protected": self.protected,
            "content_sha256": self.content_sha256,
            "provenance": dict(self.provenance),
            "group_id": self.group_id,
        }


@dataclass(frozen=True)
class ContextPlan:
    plan_id: str
    session_id: str
    session_revision: int
    selected_memory_ids: tuple[str, ...] = ()
    selected_event_ids: tuple[str, ...] = ()
    compacted_event_ids: tuple[str, ...] = ()
    dropped_event_ids: tuple[str, ...] = ()
    estimated_tokens: int = 0
    memory_tokens: int = 0
    history_tokens: int = 0
    plan_hash: str = ""
    retrieval_query: str | None = None
    items: tuple[ContextItem, ...] = ()
    compaction_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_memory_ids", tuple(self.selected_memory_ids))
        object.__setattr__(self, "selected_event_ids", tuple(self.selected_event_ids))
        object.__setattr__(self, "compacted_event_ids", tuple(self.compacted_event_ids))
        object.__setattr__(self, "dropped_event_ids", tuple(self.dropped_event_ids))
        object.__setattr__(self, "items", tuple(self.items))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_revision": self.session_revision,
            "selected_memory_ids": list(self.selected_memory_ids),
            "selected_event_ids": list(self.selected_event_ids),
            "compacted_event_ids": list(self.compacted_event_ids),
            "dropped_event_ids": list(self.dropped_event_ids),
            "estimated_tokens": self.estimated_tokens,
            "memory_tokens": self.memory_tokens,
            "history_tokens": self.history_tokens,
            "retrieval_query": self.retrieval_query,
            "items": [item.metadata() for item in self.items],
        }

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        payload = self.identity_payload()
        payload.update(
            {
                "plan_id": self.plan_id,
                "plan_hash": self.plan_hash,
                "compaction_count": self.compaction_count,
                "items": [
                    {**item.metadata(), **({"content": item.content} if include_content else {})}
                    for item in self.items
                ],
            }
        )
        return payload


class Compactor(Protocol):
    version: str

    def compact(self, items: Sequence[ContextItem]) -> ContextItem: ...


class StructuredCompactorV1:
    """Loss-aware metadata compactor; it never asks a model to summarize."""

    version = "structured-compactor-v1"

    def compact(self, items: Sequence[ContextItem]) -> ContextItem:
        event_ids = [str(item.provenance.get("event_id", item.item_id)) for item in items]
        source_hashes = [item.content_sha256 for item in items]
        payload = {
            "compacted_event_ids": event_ids,
            "item_count": len(items),
            "content_hashes": source_hashes,
            "categories": sorted({item.category.value for item in items}),
        }
        return ContextItem(
            item_id="compact-" + sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16],
            category=ContextItemCategory.SESSION_SUMMARY,
            content=payload,
            estimated_tokens=max(8, len(event_ids) * 2 + 6),
            priority=ContextPriority.LOW,
            protected=False,
            provenance={"compactor": self.version, "replaced_item_ids": [item.item_id for item in items]},
        )


class ContextManager:
    """Selects provider context without executing, persisting, or authorizing anything."""

    version = "m10-context-v2"

    def __init__(
        self,
        *,
        budget: ContextBudget | None = None,
        estimator: TokenEstimator | None = None,
        compactor: Compactor | None = None,
        history_window: int = 24,
    ) -> None:
        self.budget = budget or ContextBudget()
        self.estimator = estimator or DeterministicTokenEstimator()
        self.compactor = compactor or StructuredCompactorV1()
        self.history_window = history_window
        if history_window < 0:
            raise ValueError("history_window must be non-negative")

    def priority_candidates(
        self,
        *,
        memory_records: Sequence[MemoryRecord] = (),
        history: Sequence[SessionEvent | ContextItem] = (),
        research_data_classification: str | None = None,
    ) -> tuple[ContextItem, ...]:
        """Build stable candidates; explicitly label research-safe data before external use."""
        memory_items = [self._memory_item(record) for record in memory_records]
        history_items = [self._history_item(item) for item in history]
        if self.history_window:
            history_items = history_items[-self.history_window :]
        self._validate_groups(history_items)
        items = tuple((*memory_items, *history_items))
        if research_data_classification is not None:
            if research_data_classification not in {"public", "synthetic"}:
                raise ValueError("research_data_classification must be public or synthetic")
            items = tuple(
                replace(
                    item,
                    provenance={
                        **item.provenance,
                        "research_data_classification": research_data_classification,
                    },
                )
                for item in items
            )
        return items

    def build_plan(
        self,
        *,
        session_id: str,
        session_revision: int,
        current_user: str,
        current_evidence: Sequence[Any] = (),
        memory_records: Sequence[MemoryRecord] = (),
        history: Sequence[SessionEvent | ContextItem] = (),
        system_pins: Sequence[ContextItem] = (),
        retrieval_query: str | None = None,
        extra_items: Sequence[ContextItem] = (),
        priority_hints: Mapping[str, ContextPriority | str] | None = None,
    ) -> ContextPlan:
        items: list[ContextItem] = []
        items.extend(system_pins)
        items.append(
            ContextItem(
                item_id="current-user",
                category=ContextItemCategory.CURRENT_USER,
                content=current_user,
                estimated_tokens=self.estimator.estimate(current_user),
                priority=ContextPriority.PROTECTED,
                protected=True,
                provenance={"source": "current_turn"},
            )
        )
        for index, evidence in enumerate(current_evidence):
            source_id = getattr(evidence, "source_id", f"evidence-{index}")
            content = _content(evidence)
            items.append(
                ContextItem(
                    item_id=f"evidence-{source_id}",
                    category=ContextItemCategory.CURRENT_EVIDENCE,
                    content=content,
                    estimated_tokens=self.estimator.estimate(content),
                    priority=ContextPriority.PROTECTED,
                    protected=True,
                    provenance={"source_id": source_id},
                )
            )
        items.extend(self._memory_item(record) for record in memory_records)
        history_items = [self._history_item(item) for item in history]
        if self.history_window:
            history_items = history_items[-self.history_window :]
        items.extend(history_items)
        items.extend(extra_items)
        if priority_hints:
            items = self._apply_priority_hints(items, priority_hints)
        self._validate_groups(history_items)

        units: list[list[ContextItem]] = []
        grouped: dict[str, list[ContextItem]] = {}
        for item in items:
            if item.group_id:
                grouped.setdefault(item.group_id, []).append(item)
            else:
                units.append([item])
        units.extend(grouped.values())

        protected_units = [
            unit
            for unit in units
            if any(item.protected or item.priority == ContextPriority.PROTECTED for item in unit)
        ]
        protected = [item for unit in protected_units for item in unit]
        base_tokens = self.budget.reserved_system_tokens + sum(item.estimated_tokens for item in protected)
        if base_tokens > self.budget.max_estimated_input_tokens:
            raise ContextBudgetExceeded("protected context exceeds the configured context budget")

        selected: list[ContextItem] = list(protected)
        selected_ids = {item.item_id for item in selected}
        remaining_units = [unit for unit in units if not any(item.item_id in selected_ids for item in unit)]
        memory_used = 0
        history_used = 0
        total = base_tokens
        dropped: list[ContextItem] = []

        priority_order = {
            ContextPriority.HIGH: 0,
            ContextPriority.NORMAL: 1,
            ContextPriority.LOW: 2,
            ContextPriority.PROTECTED: -1,
        }
        remaining_units.sort(key=lambda unit: (priority_order[min(item.priority for item in unit)], unit[0].item_id))
        for unit in remaining_units:
            unit_tokens = sum(item.estimated_tokens for item in unit)
            if (
                any(item.category == ContextItemCategory.MEMORY for item in unit)
                and memory_used + unit_tokens > self.budget.max_memory_tokens
            ):
                dropped.extend(unit)
                continue
            history_unit = any(
                item.category in {ContextItemCategory.RECENT_HISTORY, ContextItemCategory.TOOL_EXCHANGE, ContextItemCategory.SESSION_SUMMARY}
                for item in unit
            )
            if history_unit:
                cap = self.budget.max_tool_observation_tokens if any(item.category == ContextItemCategory.TOOL_EXCHANGE for item in unit) else self.budget.max_history_tokens
                if history_used + unit_tokens > cap:
                    dropped.extend(unit)
                    continue
            if total + unit_tokens > self.budget.max_estimated_input_tokens:
                dropped.extend(unit)
                continue
            selected.extend(unit)
            selected_ids.update(item.item_id for item in unit)
            total += unit_tokens
            if any(item.category == ContextItemCategory.MEMORY for item in unit):
                memory_used += unit_tokens
            if history_unit:
                history_used += unit_tokens

        compacted_ids: list[str] = []
        compacted_count = 0
        compact_candidates = [item for item in dropped if item.category in {ContextItemCategory.RECENT_HISTORY, ContextItemCategory.TOOL_EXCHANGE} and not item.group_id]
        if compact_candidates:
            compacted = self.compactor.compact(compact_candidates)
            if total + compacted.estimated_tokens > self.budget.max_estimated_input_tokens:
                removable = [
                    item
                    for item in selected
                    if not item.protected
                    and item.category == ContextItemCategory.RECENT_HISTORY
                    and not item.group_id
                ]
                for item in removable:
                    selected.remove(item)
                    dropped.append(item)
                    compact_candidates.append(item)
                    total -= item.estimated_tokens
                    history_used = max(0, history_used - item.estimated_tokens)
                    if total + compacted.estimated_tokens <= self.budget.max_estimated_input_tokens:
                        compacted = self.compactor.compact(compact_candidates)
                        break
            if total + compacted.estimated_tokens <= self.budget.max_estimated_input_tokens:
                selected.append(compacted)
                total += compacted.estimated_tokens
                compacted_count = 1
                compacted_ids = [str(item.provenance.get("event_id", item.item_id)) for item in compact_candidates]
                dropped = [item for item in dropped if item not in compact_candidates]

        self._validate_groups([item for item in selected if item.category == ContextItemCategory.TOOL_EXCHANGE])
        selected_memory_ids = tuple(item.provenance["memory_id"] for item in selected if item.category == ContextItemCategory.MEMORY and "memory_id" in item.provenance)
        selected_event_ids = tuple(str(item.provenance["event_id"]) for item in selected if "event_id" in item.provenance)
        dropped_event_ids = tuple(str(item.provenance["event_id"]) for item in dropped if "event_id" in item.provenance)
        plan_payload = {
            "session_id": session_id,
            "session_revision": session_revision,
            "selected_memory_ids": selected_memory_ids,
            "selected_event_ids": selected_event_ids,
            "compacted_event_ids": tuple(compacted_ids),
            "dropped_event_ids": dropped_event_ids,
            "estimated_tokens": total,
            "memory_tokens": memory_used,
            "history_tokens": history_used,
            "retrieval_query": retrieval_query,
            "items": [item.metadata() for item in selected],
        }
        plan_hash = sha256(json.dumps(plan_payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
        return ContextPlan(
            plan_id=f"plan-{uuid4().hex}",
            session_id=session_id,
            session_revision=session_revision,
            selected_memory_ids=selected_memory_ids,
            selected_event_ids=selected_event_ids,
            compacted_event_ids=tuple(compacted_ids),
            dropped_event_ids=dropped_event_ids,
            estimated_tokens=total,
            memory_tokens=memory_used,
            history_tokens=history_used,
            plan_hash=plan_hash,
            retrieval_query=retrieval_query,
            items=tuple(selected),
            compaction_count=compacted_count,
        )

    @staticmethod
    def _apply_priority_hints(
        items: Sequence[ContextItem],
        priority_hints: Mapping[str, ContextPriority | str],
    ) -> list[ContextItem]:
        normalized = {
            str(item_id): ContextPriority(priority)
            for item_id, priority in priority_hints.items()
        }
        by_id = {item.item_id: item for item in items}
        unknown = set(normalized) - set(by_id)
        if unknown:
            raise ValueError(f"priority hints refer to unknown context items: {sorted(unknown)}")
        eligible = {
            ContextItemCategory.MEMORY,
            ContextItemCategory.RECENT_HISTORY,
            ContextItemCategory.SESSION_SUMMARY,
            ContextItemCategory.TOOL_EXCHANGE,
        }
        for item_id in normalized:
            item = by_id[item_id]
            if item.protected or item.priority == ContextPriority.PROTECTED or item.category not in eligible:
                raise ValueError(f"priority hints cannot change protected context: {item_id}")
        groups: dict[str, list[ContextItem]] = {}
        for item in items:
            if item.group_id:
                groups.setdefault(item.group_id, []).append(item)
        for group_id, members in groups.items():
            hinted = [item.item_id in normalized for item in members]
            if any(hinted) and (not all(hinted) or len({normalized[item.item_id] for item in members}) != 1):
                raise ValueError(f"priority hints must cover an atomic group consistently: {group_id}")
        return [
            replace(item, priority=normalized[item.item_id])
            if item.item_id in normalized
            else item
            for item in items
        ]

    def _history_item(self, value: SessionEvent | ContextItem) -> ContextItem:
        if isinstance(value, ContextItem):
            return value
        payload = value.payload
        category = ContextItemCategory.TOOL_EXCHANGE if value.event_type in {SessionEventType.TOOL_CALL, SessionEventType.TOOL_RESULT} else ContextItemCategory.RECENT_HISTORY
        group_id = None
        if category == ContextItemCategory.TOOL_EXCHANGE and isinstance(payload, Mapping):
            call_ids = payload.get("tool_calls")
            first_call_id = (
                call_ids[0].get("id")
                if isinstance(call_ids, Sequence)
                and call_ids
                and isinstance(call_ids[0], Mapping)
                else None
            )
            group_id = str(
                payload.get("tool_call_id") or payload.get("call_id") or first_call_id or ""
            ) or None
        return ContextItem(
            item_id=f"event-{value.event_id}",
            category=category,
            content=payload,
            estimated_tokens=self.estimator.estimate(payload),
            priority=ContextPriority.NORMAL,
            protected=category == ContextItemCategory.TOOL_EXCHANGE and isinstance(payload, Mapping) and bool(payload.get("unresolved")),
            provenance={"event_id": value.event_id, "sequence": value.sequence, "event_type": value.event_type.value},
            group_id=group_id,
        )

    def _memory_item(self, record: MemoryRecord) -> ContextItem:
        content = {
            "memory_id": record.memory_id,
            "kind": record.kind.value,
            "key": record.key,
            "value": record.value,
            "source_type": record.source_type.value,
            "value_sha256": record.value_sha256,
        }
        return ContextItem(
            item_id=f"memory-{record.memory_id}",
            category=ContextItemCategory.MEMORY,
            content=content,
            estimated_tokens=self.estimator.estimate(content),
            priority=ContextPriority.HIGH,
            provenance={"memory_id": record.memory_id, "status": record.status.value},
        )

    @staticmethod
    def _validate_groups(items: Sequence[ContextItem]) -> None:
        groups: dict[str, list[ContextItem]] = {}
        for item in items:
            if item.group_id:
                groups.setdefault(item.group_id, []).append(item)
        for group_id, members in groups.items():
            if not any(item.category == ContextItemCategory.TOOL_EXCHANGE for item in members):
                continue
            call_ids: set[str] = set()
            result_ids: set[str] = set()
            for item in members:
                if not isinstance(item.content, Mapping):
                    continue
                raw_calls = item.content.get("tool_calls")
                if isinstance(raw_calls, Sequence):
                    call_ids.update(
                        str(call.get("id"))
                        for call in raw_calls
                        if isinstance(call, Mapping) and call.get("id")
                    )
                if item.content.get("tool_call_id"):
                    result_ids.add(str(item.content["tool_call_id"]))
            if len(members) < 2 or (call_ids and result_ids and not call_ids.intersection(result_ids)):
                raise ContextAtomicityViolation(f"tool exchange group is incomplete: {group_id}")


def _content(value: Any) -> Any:
    if hasattr(value, "source_id"):
        return {
            "source_id": value.source_id,
            "title": getattr(value, "title", ""),
            "excerpt": getattr(value, "excerpt", ""),
            "source_url": getattr(value, "source_url", ""),
            "score": getattr(value, "score", 0.0),
        }
    return value


def memory_context_block(records: Sequence[MemoryRecord]) -> dict[str, Any]:
    """Return a data-bearing block; callers must not merge it into system text."""

    return {
        "type": "memory_context",
        "authority": "contextual_user_session_data_only",
        "not_medical_evidence": True,
        "not_runtime_authority": True,
        "items": [
            {
                "memory_id": record.memory_id,
                "kind": record.kind.value,
                "key": record.key,
                "value": record.value,
                "source_type": record.source_type.value,
            }
            for record in records
        ],
    }


__all__ = [
    "Compactor",
    "ContextAtomicityViolation",
    "ContextBudget",
    "ContextBudgetExceeded",
    "ContextFailure",
    "ContextIntent",
    "ContextItem",
    "ContextItemCategory",
    "ContextManager",
    "ContextPlan",
    "ContextPriority",
    "DeterministicTokenEstimator",
    "StructuredCompactorV1",
    "TokenEstimator",
    "memory_context_block",
]
