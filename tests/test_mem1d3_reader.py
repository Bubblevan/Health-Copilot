import hashlib
import importlib.util
import json
from pathlib import Path

_runner_path = Path(__file__).parents[1] / "tools" / "research" / "memory" / "run_mem1d3_reader.py"
_runner_spec = importlib.util.spec_from_file_location("mem1d3_reader_test", _runner_path)
assert _runner_spec.loader is not None
mem1d3_reader = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(mem1d3_reader)


def test_bounded_dataset_loader_decodes_only_selected_records(monkeypatch, tmp_path):
    rows = [
        {"question_id": "selected", "question_date": "2026-01-02", "question": "q"},
        {"question_id": "unselected", "secret_payload": "must-not-decode"},
    ]
    raw_records = [json.dumps(row, separators=(",", ":")).encode() for row in rows]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_bytes(b"[" + b",".join(raw_records) + b"]")

    decoded = []
    original_loads = json.loads

    def recording_loads(value, *args, **kwargs):
        decoded.append(value)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(mem1d3_reader.json, "loads", recording_loads)
    selected, hashes, skipped = mem1d3_reader.load_selected_records(dataset_path, {"selected"})

    assert set(selected) == {"selected"}
    assert skipped == 1
    assert decoded == [raw_records[0]]
    assert hashes == {"selected": hashlib.sha256(raw_records[0]).hexdigest()}


def test_v2_reader_prompt_uses_official_date_without_category_hints():
    system_template, user_template, _ = mem1d3_reader._load_template()
    messages = mem1d3_reader.build_reader_messages(
        "What happened yesterday?",
        "2026-04-09",
        "Memory: the appointment was on April 8.",
        system_template,
        user_template,
    )

    assert "Question date: 2026-04-09" in messages[0]["content"]
    assert "against this question date" in messages[0]["content"]
    assert "What happened yesterday?" in messages[1]["content"]
    assert "question_type" not in "\n".join(message["content"] for message in messages)
    assert "pick the newest fact" not in messages[0]["content"]


def test_counterfactual_keeps_temporal_and_non_temporal_pairs_distinct():
    question_dates = {
        "records": [
            {"question_id": question_id, "question_date": "2026-04-09"}
            for question_id in mem1d3_reader.QUESTION_IDS
        ]
    }
    old_by_key = {}
    new_by_key = {}
    for system in mem1d3_reader.SYSTEMS:
        for question_id in mem1d3_reader.QUESTION_IDS:
            key = (system, question_id)
            old_by_key[key] = {
                "category": "temporal-reasoning" if question_id in mem1d3_reader.TEMPORAL_IDS else "other",
                "question": "Question?",
                "ground_truth": "answer",
                "predicted": "old answer",
                "token_precision": 0.5,
                "token_recall": 0.5,
                "f1": 0.5,
                "normalized_exact_match": 0.0,
                "context_bundle_sha256": "same-bundle",
                "shared_reader_prompt_sha256": "old-prompt",
            }
            new_by_key[key] = {
                "predicted": "new answer",
                "token_precision": 0.75,
                "token_recall": 0.75,
                "f1": 0.75,
                "normalized_exact_match": 0.0,
                "context_bundle_sha256": "same-bundle",
                "shared_reader_prompt_sha256": "new-prompt",
            }

    result = mem1d3_reader._build_counterfactual(old_by_key, new_by_key, question_dates)

    assert len(result["rows"]) == 50
    assert result["temporal_cases"]["paired_rows"] == 10
    assert result["non_temporal_cases"]["paired_rows"] == 40
    assert all(row["same_context_bundle_sha256"] for row in result["rows"])
    assert all(row["semantic_review_status"] == "PENDING_HUMAN_REVIEW" for row in result["rows"])
    assert all(row["human_reflection_labels"]["outcome"] is None for row in result["rows"])
    assert all(not row["automatic_semantic_or_failure_labels"] for row in result["rows"])
