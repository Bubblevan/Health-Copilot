from tools.research.memory.audit_mem3b1_dev_change_predecessor_anchors_v1 import _audit_target


def _record(turns: list[str]) -> dict:
    return {
        "haystack_dates": ["2023/05/04 (Thu) 11:22"],
        "haystack_session_ids": ["session-1"],
        "haystack_sessions": [[{"role": "user", "content": text} for text in turns]],
    }


def test_old_value_inside_correction_turn_is_not_a_prior_anchor() -> None:
    result = _audit_target(
        {"q1": _record(["I moved tea break to 2:30 pm instead of 3 pm."])},
        {"question_id": "q1", "source_position": 0, "old_value_text": "3 pm", "attribute_hint": "tea_break_time"},
    )

    assert result["prior_user_turn_count"] == 0
    assert result["prior_exact_literal_match_count"] == 0


def test_exact_prior_value_is_source_positioned_as_anchor_candidate() -> None:
    result = _audit_target(
        {"q1": _record(["My commute involved taking the bus.", "I now walk instead of taking the bus."])},
        {"question_id": "q1", "source_position": 1, "old_value_text": "taking the bus", "attribute_hint": "commute"},
    )

    assert result["prior_exact_literal_match_count"] == 1
    assert result["prior_exact_literal_matches"][0]["source_position"] == 0
