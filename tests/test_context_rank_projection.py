from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import pytest

from health_ai_copilot.runtime.context_manager import (
    ContextBudget,
    ContextItem,
    ContextItemCategory,
    ContextManager,
    ContextPriority,
)
from health_ai_copilot.runtime.memory import MemoryKind, MemoryRecord, MemorySourceType


def _record(memory_id: str, value: str = "fact") -> MemoryRecord:
    return MemoryRecord.create(
        memory_id=memory_id,
        scope_id="rank-test",
        key=f"k-{memory_id}",
        kind=MemoryKind.SESSION_NOTE,
        value={"content": value, "role": "user", "session_date": "2023/01/01 (Sun) 12:00"},
        source_type=MemorySourceType.SESSION_DERIVED,
        source_session_id=f"session-{memory_id}",
        source_event_ids=[f"event-{memory_id}"],
        created_at="2023-01-01T00:00:00Z",
        valid_from="2023-01-01T00:00:00Z",
    )


def test_no_rank_hints_preserve_frozen_legacy_plan_semantics() -> None:
    records = [_record("z-rank1", "alpha"), _record("a-rank2", "bravo"), _record("m-rank3", "charlie")]
    history = [
        ContextItem(
            item_id="event-call",
            category=ContextItemCategory.TOOL_EXCHANGE,
            content={"tool_calls": [{"id": "call-x"}]},
            estimated_tokens=55,
            priority=ContextPriority.NORMAL,
            provenance={"event_id": "event-call"},
            group_id="call-x",
        ),
        ContextItem(
            item_id="event-result",
            category=ContextItemCategory.TOOL_EXCHANGE,
            content={"tool_call_id": "call-x"},
            estimated_tokens=55,
            priority=ContextPriority.NORMAL,
            provenance={"event_id": "event-result"},
            group_id="call-x",
        ),
    ]
    manager = ContextManager(
        budget=ContextBudget(
            max_estimated_input_tokens=240,
            reserved_system_tokens=2,
            max_memory_tokens=1000,
        )
    )

    plan = manager.build_plan(
        session_id="legacy",
        session_revision=7,
        current_user="ask",
        memory_records=records,
        history=history,
        retrieval_query="ask",
        selection_rank_hints=None,
    )

    assert plan.selected_memory_ids == ("a-rank2", "m-rank3", "z-rank1")
    assert plan.selected_event_ids == ()
    assert plan.dropped_event_ids == ("event-call", "event-result")
    assert plan.estimated_tokens == 199
    assert [item.item_id for item in plan.items] == [
        "current-user",
        "memory-a-rank2",
        "memory-m-rank3",
        "memory-z-rank1",
    ]
    assert plan.plan_hash == "7d42676115adf1c926be33782eb30607ddbe1e9efdb74410f9d495d82bd1a655"


class _RankTokenEstimator:
    version = "rank-projection-fixture-v1"
    token_costs: ClassVar[dict[str, int]] = {
        "z-first": 400,
        "a-second": 400,
        "m-third": 400,
        "b-fourth": 100,
    }

    def estimate(self, value: object) -> int:
        if not isinstance(value, Mapping):
            return 1
        return self.token_costs[str(value["memory_id"])]


def test_rank_hints_dominate_memory_id_order_and_greedy_skip_keeps_later_small_item() -> None:
    records = [_record(memory_id) for memory_id in ("z-first", "a-second", "m-third", "b-fourth")]
    rank_hints = {
        "memory-z-first": 1,
        "memory-a-second": 2,
        "memory-m-third": 3,
        "memory-b-fourth": 4,
    }
    manager = ContextManager(
        budget=ContextBudget(max_memory_tokens=1024),
        estimator=_RankTokenEstimator(),
    )

    plan = manager.build_plan(
        session_id="rank-test",
        session_revision=4,
        current_user="question",
        memory_records=records,
        selection_rank_hints=rank_hints,
    )

    assert plan.selected_memory_ids == ("z-first", "a-second", "b-fourth")
    assert [item.item_id for item in plan.items if item.category == ContextItemCategory.MEMORY] == [
        "memory-z-first",
        "memory-a-second",
        "memory-b-fourth",
    ]
    assert plan.memory_tokens == 900


@pytest.mark.parametrize(
    ("hints", "message"),
    [
        ({"memory-a": 0}, "positive integers"),
        ({"memory-a": True}, "positive integers"),
        ({"memory-a": 1.5}, "positive integers"),
        ({"memory-a": 1, "memory-b": 1}, "unique ranks"),
        ({"memory-missing": 1}, "unknown context items"),
        ({"memory-a": 1}, "cover every memory item"),
    ],
)
def test_invalid_rank_hints_fail_closed(hints: dict[str, object], message: str) -> None:
    manager = ContextManager()
    with pytest.raises((TypeError, ValueError), match=message):
        manager.build_plan(
            session_id="rank-test",
            session_revision=1,
            current_user="question",
            memory_records=[_record("a"), _record("b")],
            selection_rank_hints=hints,  # type: ignore[arg-type]
        )


def test_rank_hints_cannot_target_current_or_protected_items() -> None:
    manager = ContextManager()
    with pytest.raises(ValueError, match="only apply to eligible memory"):
        manager.build_plan(
            session_id="rank-test",
            session_revision=1,
            current_user="question",
            selection_rank_hints={"current-user": 1},
        )


@pytest.mark.parametrize(
    ("category", "protected"),
    [
        (ContextItemCategory.SYSTEM_PIN, True),
        (ContextItemCategory.MEMORY, True),
    ],
)
def test_rank_hints_reject_protected_extra_items(
    category: ContextItemCategory,
    protected: bool,
) -> None:
    manager = ContextManager()
    item = ContextItem(
        item_id="protected-item",
        category=category,
        content={"text": "pinned"},
        estimated_tokens=1,
        protected=protected,
    )
    with pytest.raises(ValueError, match="only apply to eligible memory"):
        manager.build_plan(
            session_id="rank-test",
            session_revision=1,
            current_user="question",
            extra_items=[item],
            selection_rank_hints={"protected-item": 1},
        )


def test_rank_hints_are_not_added_to_reader_visible_memory_serialization() -> None:
    manager = ContextManager()
    record = _record("a", "untrusted raw turn")
    plan = manager.build_plan(
        session_id="rank-test",
        session_revision=1,
        current_user="question",
        memory_records=[record],
        selection_rank_hints={"memory-a": 7},
    )

    memory = next(item for item in plan.items if item.category == ContextItemCategory.MEMORY)
    assert memory.content == {
        "memory_id": record.memory_id,
        "kind": record.kind.value,
        "key": record.key,
        "value": record.value,
        "source_type": record.source_type.value,
        "value_sha256": record.value_sha256,
    }
    assert "rank" not in memory.content
    assert "selection_rank" not in memory.provenance
