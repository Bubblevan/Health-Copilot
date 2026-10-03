from __future__ import annotations

import copy
import sys
from datetime import date
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "research" / "memory"
sys.path.insert(0, str(TOOLS))

from locomo_tvr import detect_temporal_intent, project_temporal_view  # noqa: E402


def _proposition(memory_id: str, text: str, entity: str, day: str) -> dict:
    return {
        "memory_id": memory_id,
        "text": text,
        "entity": entity,
        "date": day,
        "date_ordinal": date.fromisoformat(day).toordinal(),
    }


def test_none_preserves_vanilla_order_and_records_without_mutation() -> None:
    rows = [
        _proposition("p2", "prefers tea", "Alice", "2024-03-01"),
        _proposition("p1", "prefers coffee", "Alice", "2023-01-01"),
    ]
    before = copy.deepcopy(rows)
    result = project_temporal_view("What does Alice prefer?", rows, [[1, 0], [0.99, 0.1]])
    assert result["mode"] == "NONE"
    assert result["visible_memory_ids"] == ["p2", "p1"]
    assert result["view_text"] == ""
    assert rows == before


def test_current_prefers_latest_same_topic_but_keeps_history() -> None:
    rows = [
        _proposition("old", "prefers running", "Alice", "2023-01-01"),
        _proposition("new", "prefers swimming", "Alice", "2024-03-01"),
        _proposition("unrelated", "moved to Boston", "Alice", "2025-01-01"),
    ]
    result = project_temporal_view(
        "What is Alice's current exercise preference?",
        rows,
        [[1, 0], [0.99, 0.1], [0, 1]],
    )
    assert result["mode"] == "CURRENT"
    assert result["component_memory_ids"] == [["old", "new"]]
    assert "Current candidate" in result["view_text"]
    assert "prefers swimming" in result["view_text"]
    assert "prefers running" in result["view_text"]
    assert "moved to Boston" not in result["view_text"]


def test_change_history_is_chronological() -> None:
    rows = [
        _proposition("late", "prefers swimming", "Alice", "2024-05-01"),
        _proposition("early", "prefers running", "Alice", "2023-01-01"),
        _proposition("middle", "prefers cycling", "Alice", "2023-09-01"),
    ]
    result = project_temporal_view(
        "How did Alice's exercise preference change over time?",
        rows,
        [[1, 0], [0.99, 0.1], [0.98, 0.15]],
    )
    assert result["mode"] == "CHANGE"
    assert result["component_memory_ids"] == [["early", "middle", "late"]]
    assert result["view_text"].index("prefers running") < result["view_text"].index("prefers cycling")
    assert result["view_text"].index("prefers cycling") < result["view_text"].index("prefers swimming")


def test_as_of_selects_latest_valid_fact_and_hides_future_from_normal_context() -> None:
    rows = [
        _proposition("jan", "prefers running", "Alice", "2023-01-01"),
        _proposition("mar", "prefers cycling", "Alice", "2023-03-10"),
        _proposition("jun", "prefers swimming", "Alice", "2023-06-01"),
    ]
    before = copy.deepcopy(rows)
    result = project_temporal_view(
        "What exercise did Alice prefer as of March 20, 2023?",
        rows,
        [[1, 0], [0.99, 0.1], [0.98, 0.15]],
    )
    assert result["mode"] == "AS_OF"
    assert result["target_date"] == "2023-03-20"
    assert result["visible_memory_ids"] == ["jan", "mar"]
    assert "prefers cycling" in result["view_text"]
    assert "prefers swimming" in result["view_text"]
    assert rows == before


def test_as_of_month_and_year_targets_resolve_deterministically() -> None:
    assert detect_temporal_intent("What was current as of March 2024?") == {
        "mode": "AS_OF",
        "target_date": "2024-03-31",
    }
    assert detect_temporal_intent("What was true in 2020?") == {
        "mode": "AS_OF",
        "target_date": "2020-12-31",
    }


@pytest.mark.parametrize(
    "question",
    ["What happened before the move?", "What did she do at that time?", "What was true by someday?"],
)
def test_ambiguous_or_unparseable_time_fails_closed(question: str) -> None:
    assert detect_temporal_intent(question) == {"mode": "NONE", "target_date": None}


def test_distinct_entities_and_below_threshold_similarity_never_link() -> None:
    rows = [
        _proposition("a1", "likes running", "Alice", "2023-01-01"),
        _proposition("a2", "likes swimming", "Alice", "2024-01-01"),
        _proposition("b1", "likes running", "Bob", "2023-01-01"),
    ]
    result = project_temporal_view(
        "What is the latest preference?",
        rows,
        [[1, 0], [0.8, 0.6], [1, 0]],
    )
    assert result["mode"] == "CURRENT"
    assert result["component_memory_ids"] == []
    assert "No same-topic multi-date proposition group was retrieved." in result["view_text"]
