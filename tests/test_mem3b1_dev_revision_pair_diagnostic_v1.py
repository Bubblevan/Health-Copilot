import json

from tools.research.memory.run_mem3b1_dev_revision_pair_diagnostic_v1 import (
    _messages,
    _object_without_sensitive_fields,
    _source_turns,
    _validate_response,
)


def test_sensitive_benchmark_fields_are_removed_recursively() -> None:
    result = json.loads(
        '{"question_id":"dev-1","question_type":"knowledge-update",'
        '"question":"What is current?","answer":"gold",'
        '"nested":{"answer_session_ids":["s1"],"value":3}}',
        object_pairs_hook=_object_without_sensitive_fields,
    )

    assert result == {
        "question_id": "dev-1",
        "question_type": "knowledge-update",
        "nested": {"value": 3},
    }


def test_source_turns_keep_only_user_text_and_order_by_session_date() -> None:
    turns = _source_turns(
        {
            "haystack_dates": ["2023/02/01 (Wed) 12:00", "2023/01/01 (Sun) 12:00"],
            "haystack_session_ids": ["later", "earlier"],
            "haystack_sessions": [
                [
                    {"role": "assistant", "content": "not user memory"},
                    {"role": "user", "content": "I now prefer tea."},
                ],
                [{"role": "user", "content": "I used to prefer coffee."}],
            ],
        }
    )

    assert [turn["text"] for turn in turns] == [
        "I used to prefer coffee.",
        "I now prefer tea.",
    ]
    assert all("not user memory" not in str(turn) for turn in turns)


def test_model_messages_omit_benchmark_identifiers_and_include_source_only() -> None:
    messages = _messages(
        [{"session_date": "2023/01/01 (Sun) 12:00", "turn_index": 0, "text": "I prefer tea."}]
    )
    content = "\n".join(row["content"] for row in messages)

    assert "I prefer tea." in content
    assert "question_id" not in content
    assert "knowledge-update" not in content
    assert "gold" not in content


def test_v3_messages_carry_only_candidate_cue_and_source_quote() -> None:
    quote = "I've been having my tea break at 2:30 pm instead of 3 pm."
    messages = _messages(
        [
            {
                "session_date": "2023/05/04 (Thu) 11:22",
                "turn_index": 8,
                "source_position": 0,
                "cue_span": "2:30 pm instead of 3 pm",
                "text": quote,
            }
        ],
        protocol_version="v3",
    )
    content = "\n".join(row["content"] for row in messages)

    assert "2:30 pm instead of 3 pm" in content
    assert quote in content
    assert "question_id" not in content
    assert "knowledge-update" not in content
    assert "gold" not in content


def test_response_validation_requires_exact_dated_source_quotes_and_values() -> None:
    turns = [
        {"session_date": "2023/01/01 (Sun) 12:00", "turn_index": 0, "text": "I prefer coffee.", "source_position": 0},
        {"session_date": "2023/02/01 (Wed) 12:00", "turn_index": 0, "text": "I now prefer tea instead.", "source_position": 1},
    ]
    content = (
        '{"transitions":[{"attribute":"preferred_drink",'
        '"old_value_text":"coffee","new_value_text":"tea",'
        '"old_date":"2023/01/01 (Sun) 12:00",'
        '"new_date":"2023/02/01 (Wed) 12:00",'
        '"old_quote":"I prefer coffee.","new_quote":"I now prefer tea instead.",'
        '"relation_basis":"EXPLICIT_CORRECTION"}]}'
    )

    audit = _validate_response(content, turns)

    assert audit["source_grounded_candidate_count"] == 1
    assert audit["audit_rows"][0]["source_grounded_candidate"] is True


def test_response_validation_rejects_wrong_date_and_nonliteral_value() -> None:
    turns = [
        {"session_date": "2023/01/01 (Sun) 12:00", "turn_index": 0, "text": "I prefer coffee.", "source_position": 0},
        {"session_date": "2023/02/01 (Wed) 12:00", "turn_index": 0, "text": "I now prefer tea instead.", "source_position": 1},
    ]
    content = (
        '{"transitions":[{"attribute":"preferred_drink",'
        '"old_value_text":"espresso","new_value_text":"tea",'
        '"old_date":"2023/02/01 (Wed) 12:00",'
        '"new_date":"2023/01/01 (Sun) 12:00",'
        '"old_quote":"I prefer coffee.","new_quote":"I now prefer tea instead.",'
        '"relation_basis":"EXPLICIT_CORRECTION"}]}'
    )

    audit = _validate_response(content, turns)

    assert audit["source_grounded_candidate_count"] == 0
    assert audit["audit_rows"][0]["old_quote_exact_in_unique_dated_user_turn"] is False
    assert audit["audit_rows"][0]["chronology_increasing_or_explicit_same_turn"] is False


def test_v2_accepts_same_turn_only_for_explicit_correction() -> None:
    quote = "I've moved my tea break to 2:30 pm instead of 3 pm."
    turns = [
        {
            "session_date": "2023/05/04 (Thu) 11:22",
            "turn_index": 8,
            "source_position": 0,
            "text": quote,
        }
    ]
    content = json.dumps(
        {
            "transitions": [
                {
                    "attribute": "tea_break_time",
                    "old_value_text": "3 pm",
                    "new_value_text": "2:30 pm",
                    "old_date": "2023/05/04 (Thu) 11:22",
                    "new_date": "2023/05/04 (Thu) 11:22",
                    "old_quote": quote,
                    "new_quote": quote,
                    "relation_basis": "EXPLICIT_CORRECTION",
                }
            ]
        }
    )

    rejected_by_v1 = _validate_response(content, turns)
    accepted_by_v2 = _validate_response(content, turns, allow_intra_turn_correction=True)

    assert rejected_by_v1["source_grounded_candidate_count"] == 0
    assert accepted_by_v2["source_grounded_candidate_count"] == 1
    assert accepted_by_v2["audit_rows"][0]["same_turn_explicit_change_cue"] is True


def test_same_turn_value_comparison_is_not_a_revision_without_change_cue() -> None:
    quote = "I tried stevia in my coffee, but I think I prefer honey."
    turns = [
        {
            "session_date": "2023/05/04 (Thu) 11:22",
            "turn_index": 2,
            "source_position": 0,
            "text": quote,
        }
    ]
    content = json.dumps(
        {
            "transitions": [
                {
                    "attribute": "preferred_coffee_sweetener",
                    "old_value_text": "stevia",
                    "new_value_text": "honey",
                    "old_date": "2023/05/04 (Thu) 11:22",
                    "new_date": "2023/05/04 (Thu) 11:22",
                    "old_quote": quote,
                    "new_quote": quote,
                    "relation_basis": "EXPLICIT_CORRECTION",
                }
            ]
        }
    )

    audit = _validate_response(content, turns, allow_intra_turn_correction=True)

    assert audit["source_grounded_candidate_count"] == 0
