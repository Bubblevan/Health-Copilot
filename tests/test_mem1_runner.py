import importlib.util
import sys
from pathlib import Path

import pytest


_tools_path = Path(__file__).parents[1] / "tools" / "research" / "memory"
sys.path.insert(0, str(_tools_path))
_runner_spec = importlib.util.spec_from_file_location("mem1_runner_test", _tools_path / "run_mem1.py")
mem1_runner = importlib.util.module_from_spec(_runner_spec)
assert _runner_spec.loader is not None
_runner_spec.loader.exec_module(mem1_runner)


def split_manifest():
    return {
        "dataset_sha256": "dataset",
        "dev": {"question_ids": ["dev-a", "dev-b"]},
    }


def test_runner_allows_frozen_dev_and_rejects_every_non_dev_id():
    assert mem1_runner.resolve_question_ids(split_manifest=split_manifest()) == ["dev-a", "dev-b"]
    assert mem1_runner.resolve_question_ids(
        split_manifest=split_manifest(), selected_ids=["dev-b"]
    ) == ["dev-b"]
    with pytest.raises(ValueError, match="outside frozen DEV"):
        mem1_runner.resolve_question_ids(
            split_manifest=split_manifest(), selected_ids=["test-only"]
        )


def test_record_selection_only_normalizes_requested_questions():
    normalized = mem1_runner.select_records(
        [
            {"question_id": "dev-a", "payload": "allowed"},
            {"question_id": "test-a", "payload": "never selected"},
        ],
        ["dev-a"],
        lambda row: {
            "qa": [{"question_id": row["question_id"]}],
            "payload": row["payload"],
        },
    )
    assert normalized == [{"qa": [{"question_id": "dev-a"}], "payload": "allowed"}]
    with pytest.raises(ValueError, match="exactly the requested"):
        mem1_runner.select_records([], ["dev-a"], lambda row: row)


def test_latency_percentiles_and_role_accounting():
    assert mem1_runner._percentile([], 0.95) is None
    assert mem1_runner._percentile([3.0, 1.0, 2.0], 0.95) == 3.0
    summary = mem1_runner._summarize_calls(
        [
            {"system": "openclaw", "role": "reader_answer", "latency_ms": 10, "prompt_tokens": 5, "completion_tokens": 1, "success": True},
            {"system": "openclaw", "role": "embedding", "latency_ms": 4, "prompt_tokens": 8, "completion_tokens": 0, "success": True},
            {"system": "openclaw", "role": "reader_answer", "latency_ms": 20, "prompt_tokens": 6, "completion_tokens": 2, "success": False, "provider": "local_qwen"},
        ],
        ["openclaw"],
    )["openclaw"]
    assert summary["by_role"]["embedding"]["prompt_tokens"] == 8
    assert summary["by_role"]["reader_answer"]["failed_calls"] == 1
    assert summary["answer_latency_p50_ms"] == 10
    assert summary["answer_latency_p95_ms"] == 20
    assert summary["local_reader_wall_ms"] == 20


def test_exception_chain_records_types_without_error_bodies():
    try:
        try:
            raise TimeoutError("private request payload must not be recorded")
        except TimeoutError as inner:
            raise RuntimeError("adapter failed") from inner
    except RuntimeError as error:
        chain = mem1_runner._exception_chain(error)

    assert chain == [
        {"type": "RuntimeError", "http_status": None},
        {"type": "TimeoutError", "http_status": None},
    ]
    assert all("payload" not in str(item) for item in chain)


def test_generation_freezes_predictions_before_writing_metrics(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    mem1_runner.append_jsonl(
        predictions,
        {
            "system": "fullcontext",
            "question_id": "dev-a",
            "quality_status": "OK",
            "predicted": "answer",
            "f1": 0.5,
        },
    )
    mem1_runner.append_jsonl(
        tmp_path / "call_ledger.jsonl",
        {
            "system": "fullcontext",
            "question_id": "dev-a",
            "role": "reader_answer",
            "provider": "local_qwen",
            "latency_ms": 25,
            "prompt_tokens": 100,
            "completion_tokens": 4,
            "success": True,
            "truncated": False,
            "max_model_length": 131072,
        },
    )
    manifest = {
        "run_id": "smoke-test",
        "track": "main",
        "split": "DEV",
        "question_ids": ["dev-a"],
        "dataset": {"sha256": "dataset"},
        "roles": {"memory_system": {"systems": ["fullcontext"]}},
    }

    metrics = mem1_runner._finalize_generation(
        tmp_path, manifest, ["fullcontext"], ["dev-a"], None
    )

    sidecar = tmp_path / "predictions.sha256"
    assert mem1_runner.verify_hash_sidecar(predictions, sidecar)
    assert metrics["systems"]["fullcontext"]["f1_mean"] == 0.5
    assert metrics["prediction_frozen"] is True
    assert (tmp_path / "token_efficiency.json").exists()
    assert (tmp_path / "report.md").exists()

    mem1_runner.append_jsonl(predictions, {"system": "fullcontext", "question_id": "dev-b"})
    with pytest.raises(RuntimeError, match="Frozen prediction hash mismatch"):
        mem1_runner._finalize_generation(tmp_path, manifest, ["fullcontext"], ["dev-a"], None)


def test_retrieval_recall_is_null_without_session_provenance():
    rows = [
        {
            "system": "mem0",
            "question_id": "q1",
            "category": "knowledge-update",
            "answer_session_ids": ["s1"],
            "ranked_session_groups": None,
            "session_provenance_available": False,
            "retrieved_context_tokens": 23,
            "retrieval_latency_ms": 4.0,
            "ingestion_latency_ms": 5.0,
        }
    ]
    summary = mem1_runner._summarize_memory_diagnostics(rows, ["mem0"], ["q1"], [])
    assert summary["mem0"]["answer_session_recall_at_5"] is None
    assert summary["mem0"]["by_category"]["knowledge-update"]["mrr"] is None


def test_retrieval_recall_uses_source_rank_groups():
    rows = [
        {
            "system": "openclaw",
            "question_id": "q1",
            "category": "temporal-reasoning",
            "answer_session_ids": ["s1", "s3"],
            "ranked_session_groups": [["s2"], ["s1", "s3"]],
            "retrieved_context_tokens": 23,
            "retrieval_latency_ms": 4.0,
            "ingestion_latency_ms": 5.0,
        }
    ]
    summary = mem1_runner._summarize_memory_diagnostics(rows, ["openclaw"], ["q1"], [])
    assert summary["openclaw"]["answer_session_recall_at_5"] == 1.0
    assert summary["openclaw"]["mrr"] == 0.5
