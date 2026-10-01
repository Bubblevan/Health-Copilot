from tools.research.memory.audit_mem3b1_dev_explicit_time_revision_cues_v1 import (
    find_explicit_time_revision_cues,
)


def test_explicit_same_turn_time_replacement_is_mined_with_exact_values() -> None:
    turns = [
        {
            "source_position": 8,
            "session_date": "2023/05/04 (Thu) 11:22",
            "text": "I've moved my tea break to 2:30 pm instead of 3 pm.",
        }
    ]

    candidates = find_explicit_time_revision_cues("dev-1", turns)

    assert len(candidates) == 1
    assert candidates[0]["new_value_text"] == "2:30 pm"
    assert candidates[0]["old_value_text"] == "3 pm"
    assert candidates[0]["source_quote"] == turns[0]["text"]
    assert candidates[0]["model_inference_used"] is False


def test_non_temporal_comparison_is_not_a_time_revision_candidate() -> None:
    turns = [
        {
            "source_position": 2,
            "session_date": "2023/05/04 (Thu) 11:22",
            "text": "I tried honey instead of sugar in the tea.",
        }
    ]

    assert find_explicit_time_revision_cues("dev-1", turns) == []
