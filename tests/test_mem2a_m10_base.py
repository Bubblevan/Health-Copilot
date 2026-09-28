from __future__ import annotations

from copy import deepcopy

import pytest

from health_ai_copilot.runtime.memory import InMemoryMemoryStore
from tools.research.memory.mem2a_m10_base import (
    build_question_state,
    iter_json_array,
    parse_longmemeval_timestamp,
    question_from_source_fields,
)

DATASET_SHA = "a" * 64
READER_TOKENIZER = "synthetic-qwen3-reader-tokenizer-v1"


def _source() -> dict:
    return {
        "question_id": "synthetic-001",
        "question": "Where did I put the blue lantern?",
        "question_date": "2024/01/01 (Mon) 10:00",
        "haystack_session_ids": ["session-old", "session-current", "session-future"],
        "haystack_dates": [
            "2023/12/31 (Sun) 09:00",
            "2024/01/01 (Mon) 09:00",
            "2024/01/02 (Tue) 09:00",
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I stored the blue lantern in the attic.  ", "has_answer": True},
                {"role": "assistant", "content": "I will remember that exactly."},
            ],
            [
                {"role": "user", "content": "The blue lantern is now in the hall closet."},
                {"role": "assistant", "content": "Got it."},
            ],
            [{"role": "user", "content": "The blue lantern moved to the garage."}],
        ],
        "question_type": "temporal-reasoning",
        "answer": "hall closet",
        "answer_session_ids": ["session-current"],
    }


def _run(source: dict):
    question = question_from_source_fields(source)
    return build_question_state(
        question,
        store=InMemoryMemoryStore(),
        dataset_sha256=DATASET_SHA,
        reader_token_counter=lambda text: len(text.encode("utf-8").split()),
        reader_tokenizer_name=READER_TOKENIZER,
    )


def test_timestamp_parser_is_numeric_and_checks_weekday() -> None:
    assert parse_longmemeval_timestamp("2024/01/01 (Mon) 10:00") == "2024-01-01T10:00:00Z"
    assert parse_longmemeval_timestamp("2024-01-01 (Mon) 10:00:05") == "2024-01-01T10:00:05Z"
    with pytest.raises(ValueError, match="weekday"):
        parse_longmemeval_timestamp("2024/01/01 (Tue) 10:00")
    with pytest.raises(ValueError):
        parse_longmemeval_timestamp("Jan 1, 2024 10:00")


def test_json_array_streams_across_small_chunks(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text(
        '[{"a":"' + ("x" * 300) + '","answer":"secret",'
        '"question_type":"secret","turn":{"has_answer":true}}, {"b":2}]',
        encoding="utf-8",
    )
    rows = list(iter_json_array(path, chunk_size=32))
    assert rows == [{"a": "x" * 300, "turn": {}}, {"b": 2}]
    assert next(iter_json_array(path, chunk_size=32, include_gold=True))["answer"] == "secret"


def test_labels_and_has_answer_never_change_memory_pipeline() -> None:
    source_a = _source()
    source_b = deepcopy(source_a)
    source_b["answer"] = "different gold"
    source_b["question_type"] = "different category"
    source_b["answer_session_ids"] = ["session-old"]
    source_b["has_answer"] = False
    source_b["haystack_sessions"][0][0]["has_answer"] = False

    first = _run(source_a)
    second = _run(source_b)
    for field in (
        "memory_operations",
        "memory_inventory",
        "inventory_sha256",
        "retrieval_results",
        "context_plan",
        "context_plan_hash",
        "context_items",
        "context_bundle",
        "context_bundle_sha256",
    ):
        assert first[field] == second[field]
    serialized_values = [operation["value"] for operation in first["memory_operations"]]
    assert all(set(value) == {"role", "session_date", "content"} for value in serialized_values)
    assert serialized_values[0]["content"] == "I stored the blue lantern in the attic.  "
    assert all("has_answer" not in str(value) for value in serialized_values)


def test_two_replays_have_stable_ids_inventory_retrieval_plan_and_bundle() -> None:
    left = _run(_source())
    right = _run(_source())
    assert [row["memory_id"] for row in left["memory_operations"]] == [
        row["memory_id"] for row in right["memory_operations"]
    ]
    assert left["memory_inventory"] == right["memory_inventory"]
    assert left["retrieval_results"] == right["retrieval_results"]
    assert left["context_plan"] == right["context_plan"]
    assert left["context_plan_hash"] == right["context_plan_hash"]
    assert left["context_bundle"] == right["context_bundle"]
    assert left["context_bundle_sha256"] == right["context_bundle_sha256"]


def test_replaying_same_question_into_existing_store_is_idempotent() -> None:
    question = question_from_source_fields(_source())
    store = InMemoryMemoryStore()
    kwargs = {
        "store": store,
        "dataset_sha256": DATASET_SHA,
        "reader_token_counter": lambda text: len(text.encode("utf-8").split()),
        "reader_tokenizer_name": READER_TOKENIZER,
    }
    first = build_question_state(question, **kwargs)
    second = build_question_state(question, **kwargs)
    assert first["memory_operations"] == second["memory_operations"]
    assert first["memory_inventory"] == second["memory_inventory"]
    assert first["context_plan_hash"] == second["context_plan_hash"]
    assert first["context_bundle_sha256"] == second["context_bundle_sha256"]
    assert len(store.history(question.scope_id)) == first["operation_count"]


def test_future_turns_are_in_inventory_but_not_current_query_results() -> None:
    result = _run(_source())
    inventory_ids = {row["source_session_id"] for row in result["memory_inventory"]}
    retrieval_ids = {row["source_session_id"] for row in result["retrieval_results"]}
    assert "session-future" in inventory_ids
    assert "session-future" not in retrieval_ids
    assert "session-current" in retrieval_ids
    assert result["native_top_k"] == 8
    assert len(result["retrieval_results"]) <= 8


def test_context_bundle_serialization_provenance_and_reader_tokens_align() -> None:
    result = _run(_source())
    bundle = result["context_bundle"]
    assert bundle["context_embedding_tokens"] is None
    assert bundle["context_embedding_tokenizer"] == "NOT_APPLICABLE_NO_EMBEDDING"
    assert bundle["context_reader_tokenizer"] == READER_TOKENIZER
    assert bundle["context_reader_tokens"] == result["context_reader_tokens"]
    assert bundle["serialized_context"].count("[Context item ") == len(bundle["items"])
    assert [item["rank"] for item in bundle["items"]] == list(range(1, len(bundle["items"]) + 1))
    for item in bundle["items"]:
        external = next(row for row in result["context_items"] if row["rank"] == item["rank"])
        assert item["text"] == external["text"]
        assert item["source_session_ids"] == external["source_session_ids"]
        assert external["text"].startswith('{"key":')
    assert result["m10_estimated_memory_tokens"] <= 1024
    assert all(row["valid_until"] is None and row["expires_at"] is None for row in result["memory_inventory"])


def test_query_is_question_only_and_question_scopes_are_isolated() -> None:
    first_source = _source()
    first = _run(first_source)
    second_source = deepcopy(first_source)
    second_source["question_id"] = "synthetic-002"
    second = _run(second_source)
    assert first["retrieval_query"] == first_source["question"]
    assert first["scope_id"] == "longmemeval:synthetic-001"
    assert second["scope_id"] == "longmemeval:synthetic-002"
    assert first["memory_inventory"][0]["scope_id"] != second["memory_inventory"][0]["scope_id"]


def test_frozen_global_prediction_cache_resumes_without_reader_call(tmp_path, monkeypatch) -> None:
    from tools.research.memory import run_mem2a_m10_base as runner

    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)
    question_id = "cached-question"
    cache_identity = {
        "identity": {"question_id": question_id, "reader_model_sha256": "b" * 64},
        "identity_sha256": "c" * 64,
    }
    call = {
        "question_id": question_id,
        "cache_identity_sha256": cache_identity["identity_sha256"],
        "provider": "local_qwen",
        "hosted_call": False,
    }
    prediction = {
        "question_id": question_id,
        "cache_identity": cache_identity,
        "call_ledger_row": call,
        "quality_status": "OK",
        "predicted": "cached answer",
    }
    predictions_path = tmp_path / "predictions.jsonl"
    calls_path = tmp_path / "call_ledger.jsonl"
    runner._write_jsonl_atomic(predictions_path, [prediction])
    runner._write_jsonl_atomic(calls_path, [call])
    runner._freeze_sidecar(predictions_path, "predictions.sha256")
    runner._freeze_sidecar(calls_path, "call_ledger.sha256")

    class NoCallClient:
        def post(self, *args, **kwargs):
            raise AssertionError("reader must not be called when a frozen global prediction exists")

    result, call_row = runner._run_reader_once(
        {"question_id": question_id, "cache_identity": cache_identity},
        client=NoCallClient(),
        contract_sha="d" * 64,
    )
    assert result == prediction
    assert call_row == call
    assert (tmp_path / "questions" / question_id / "prediction.json").is_file()
    assert (tmp_path / "calls" / f"{question_id}.json").is_file()
