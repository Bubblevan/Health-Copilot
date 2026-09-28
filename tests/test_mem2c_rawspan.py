from __future__ import annotations

from copy import deepcopy

from health_ai_copilot.runtime.memory import FakeClock, InMemoryMemoryStore
from tools.research.memory import run_mem2c_rawspan as runner
from tools.research.memory.mem2a_m10_base import question_from_source_fields
from tools.research.memory.raw_span_memory import (
    REPRESENTATION_VERSION,
    SEGMENTER_VERSION,
    operation_for_span,
    segment_text,
    segment_turn,
)

DATASET_SHA = "d" * 64


def test_final_gate_accepts_zero_call_counters_but_rejects_nonzero_or_false_requirements() -> None:
    gate = {
        "passed": False,
        "all_required_artifacts_sha_frozen": True,
        "memory_budget_tokens": 1024,
        "exactly_ten_successful_reader_calls": True,
        "memory_internal_llm_calls": 0,
        "embedding_calls": 0,
        "judge_calls": 0,
        "hosted_calls": 0,
        "test_access": False,
    }

    assert runner._gate_requirements_passed(gate)
    assert not runner._gate_requirements_passed({**gate, "hosted_calls": 1})
    assert not runner._gate_requirements_passed({**gate, "test_access": True})
    assert not runner._gate_requirements_passed({**gate, "memory_budget_tokens": 512})


def test_efficiency_replay_ignores_only_nondeterministic_native_timing() -> None:
    frozen = {
        "raw_turn_count": 5,
        "mean_ingestion_latency_ms": 1.2,
        "per_question": [
            {
                "question_id": "q1",
                "native_retrieval_latency_ms": 0.4,
                "context_projection_latency_ms": 0.2,
                "selected_span_count": 3,
            }
        ],
    }
    replayed = {
        "raw_turn_count": 5,
        "mean_ingestion_latency_ms": 9.8,
        "per_question": [
            {
                "question_id": "q1",
                "native_retrieval_latency_ms": 4.0,
                "context_projection_latency_ms": 2.0,
                "selected_span_count": 3,
            }
        ],
    }

    assert runner._efficiency_stable_fields(frozen) == runner._efficiency_stable_fields(
        replayed
    )
    replayed["per_question"][0]["selected_span_count"] = 4
    assert runner._efficiency_stable_fields(frozen) != runner._efficiency_stable_fields(
        replayed
    )


def _source() -> dict:
    return {
        "question_id": "synthetic-rawspan-001",
        "question": "What did I say?",
        "question_date": "2024/01/01 (Mon) 10:00",
        "haystack_session_ids": ["session-a"],
        "haystack_dates": ["2024/01/01 (Mon) 09:00"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "Dr. Lin measured 3.14. Then left.\n1. First item!\n- \u201cQuoted text?\u201d Next item.\n\nUnicode: caf\u00e9\u3002 Still here\u2026\n",
                },
                {
                    "role": "assistant",
                    "content": "First paragraph.\nSecond paragraph with 2.5 mg.\n\n- Bullet A\n- Bullet B? Done.",
                },
                {"role": "user", "content": ""},
            ]
        ],
        "question_type": "single-session-user",
        "answer": "secret one",
        "answer_session_ids": ["session-a"],
        "has_answer": True,
    }


def _ingestion(source: dict):
    question = question_from_source_fields(source)
    spans = [span for turn in question.turns for span in segment_turn(turn)]
    operations = [
        operation_for_span(span, question=question, dataset_sha256=DATASET_SHA) for span in spans
    ]
    store = InMemoryMemoryStore(clock=FakeClock(question.now))
    records = [store.apply(operation, now=question.now) for operation in operations]
    inventory = [record.to_dict() for record in records if record is not None]
    return spans, [operation.to_dict() for operation in operations], inventory


def test_segmenter_is_lossless_deterministic_and_preserves_contractual_boundaries() -> None:
    source = _source()
    question = question_from_source_fields(source)
    turns = question.turns
    first = [segment_turn(turn) for turn in turns]
    second = [segment_turn(turn) for turn in turns]

    assert SEGMENTER_VERSION == "raw-span-segmenter-v1"
    assert REPRESENTATION_VERSION == "raw-span-v1"
    assert [
        [(span.char_start, span.char_end, span.content) for span in spans] for spans in first
    ] == [[(span.char_start, span.char_end, span.content) for span in spans] for spans in second]
    for turn, spans in zip(turns, first, strict=True):
        assert "".join(span.content for span in spans) == turn.content
        assert all(span.content for span in spans)
        assert [span.span_index for span in spans] == list(range(len(spans)))
        assert all(turn.content[span.char_start : span.char_end] == span.content for span in spans)

    first_turn_contents = [span.content for span in first[0]]
    assert first_turn_contents[:3] == ["Dr. ", "Lin measured 3.14. ", "Then left.\n"]
    assert any("1. First item!\n" == value for value in first_turn_contents)
    assert "- \u201cQuoted text?\u201d Next item.\n\n" in first_turn_contents
    assert "Unicode: caf\u00e9\u3002 Still here\u2026\n" in first_turn_contents
    assert first[-1] == []

    assert segment_text("\n\n") == [(0, 2, "\n\n")]
    assert segment_text("") == []


def test_spans_emit_only_add_session_notes_with_stable_provenance() -> None:
    first = _ingestion(_source())
    second = _ingestion(_source())
    spans_a, operations_a, inventory_a = first
    spans_b, operations_b, inventory_b = second

    assert operations_a == operations_b
    assert inventory_a == inventory_b
    assert [span.key for span in spans_a] == [span.key for span in spans_b]
    assert len({row["memory_id"] for row in inventory_a}) == len(inventory_a)
    assert all(row["key"].startswith("raw_span:session-a:") for row in inventory_a)
    assert all(row["version"] == 1 and row["status"] == "active" for row in inventory_a)
    assert all(row["valid_until"] is None and row["expires_at"] is None for row in inventory_a)
    assert all(
        set(row["value"])
        == {"role", "session_date", "content", "source_turn_index", "source_span_index"}
        for row in inventory_a
    )
    assert all(operation["operation"] == "add" for operation in operations_a)
    assert all("answer" not in str(row) and "question_type" not in str(row) for row in inventory_a)


def test_gold_and_category_label_changes_do_not_change_ingestion() -> None:
    source_a = _source()
    source_b = deepcopy(source_a)
    source_b["answer"] = "different gold"
    source_b["answer_session_ids"] = ["other-session"]
    source_b["question_type"] = "temporal-reasoning"
    source_b["has_answer"] = False

    spans_a, operations_a, inventory_a = _ingestion(source_a)
    spans_b, operations_b, inventory_b = _ingestion(source_b)
    assert [(span.key, span.content) for span in spans_a] == [
        (span.key, span.content) for span in spans_b
    ]
    assert operations_a == operations_b
    assert inventory_a == inventory_b


def test_question_changes_retrieval_input_but_not_span_ingestion_identity() -> None:
    source_a = _source()
    source_b = deepcopy(source_a)
    source_b["question"] = "A different question?"
    question_a = question_from_source_fields(source_a)
    question_b = question_from_source_fields(source_b)
    spans_a = [span for turn in question_a.turns for span in segment_turn(turn)]
    spans_b = [span for turn in question_b.turns for span in segment_turn(turn)]
    operations_a = [
        operation_for_span(span, question=question_a, dataset_sha256=DATASET_SHA).to_dict()
        for span in spans_a
    ]
    operations_b = [
        operation_for_span(span, question=question_b, dataset_sha256=DATASET_SHA).to_dict()
        for span in spans_b
    ]
    assert question_a.question != question_b.question
    assert operations_a == operations_b


def test_sqlite_materialization_uses_frozen_lexical_top8_and_projection(
    tmp_path, monkeypatch
) -> None:
    class TokenResponse:
        def __init__(self, count: int) -> None:
            self._count = count

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"tokens": list(range(self._count))}

    class LocalTokenizerOnly:
        def post(self, url: str, *, json: dict) -> TokenResponse:
            assert url == "http://127.0.0.1:8081/tokenize"
            return TokenResponse(len(json["content"].encode("utf-8")) // 4 + 1)

    monkeypatch.setattr(runner.d3, "_render_and_tokenize", lambda *_args: (42, "e" * 64))
    question = question_from_source_fields(_source())
    db_path = tmp_path / "state" / "memory.sqlite"
    client = LocalTokenizerOnly()
    first, _ = runner._build_question_state(
        question,
        dataset_sha256=DATASET_SHA,
        client=client,
        db_path=db_path,
    )
    replay, _ = runner._build_question_state(
        question,
        dataset_sha256=DATASET_SHA,
        client=client,
        db_path=db_path,
    )

    assert first["operations"] == replay["operations"]
    assert first["inventory"] == replay["inventory"]
    assert first["retrieval_row"] == replay["retrieval_row"]
    assert first["plan_row"] == replay["plan_row"]
    assert (
        first["bundle_row"]["context_bundle_sha256"]
        == replay["bundle_row"]["context_bundle_sha256"]
    )
    assert first["structural_diagnostic"]["sqlite_replay_passed"] is True
    assert first["retrieval_row"]["native_top_k"] == 8
    assert len(first["retrieval_row"]["results"]) <= 8
    assert all(row["operation"] == "ADD" for row in first["operations"])
    assert first["plan_row"]["memory_budget_tokens"] == 1024
