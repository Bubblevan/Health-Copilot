import importlib.util
import json
import sys
import types
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


def _mem1d1_args(selection_manifest, **overrides):
    values = {
        "split_manifest": mem1_runner.SPLIT_PATH,
        "dataset": Path("frozen-dataset-placeholder"),
        "answer_track": "context_controlled",
        "question_id": None,
        "selection_manifest": selection_manifest,
        "system": list(mem1_runner.SYSTEMS),
        "execution_question_id": None,
        "execution_system": None,
        "mem1d1_frozen_10": True,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_mem1d1_selection_is_exactly_the_frozen_manifest(monkeypatch):
    selection_path = mem1_runner.D1_SELECTION_PATH
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    split = json.loads(mem1_runner.SPLIT_PATH.read_text(encoding="utf-8"))
    mem1_runner.validate_mem1d1_selection(selection, split, selection_path)

    altered = dict(selection, question_count=9)
    with pytest.raises(ValueError, match="frozen 10-case contract"):
        mem1_runner.validate_mem1d1_selection(altered, split, selection_path)

    with pytest.raises(ValueError, match="canonical frozen"):
        mem1_runner.validate_mem1d1_selection(
            selection, split, selection_path.parent / "other_selection.json"
        )


def test_mem1d1_loader_requires_the_frozen_manifest_and_refuses_manual_ids(monkeypatch):
    monkeypatch.setattr(
        mem1_runner,
        "sha256_file",
        lambda path: (
            mem1_runner.D1_SELECTION_SHA256
            if Path(path).resolve() == mem1_runner.D1_SELECTION_PATH.resolve()
            else mem1_runner.D1_DATASET_SHA256
        ),
    )
    with pytest.raises(ValueError, match="requires main_smoke_10_manifest"):
        mem1_runner._load_locked_inputs(_mem1d1_args(None))

    with pytest.raises(ValueError, match="refuses manually supplied"):
        mem1_runner._load_locked_inputs(_mem1d1_args(
            mem1_runner.D1_SELECTION_PATH, question_id=["1cea1afa"]
        ))


def test_mem1d1_execution_chunks_must_be_subsets_but_lock_all_ten(monkeypatch):
    monkeypatch.setattr(
        mem1_runner,
        "sha256_file",
        lambda path: (
            mem1_runner.D1_SELECTION_SHA256
            if Path(path).resolve() == mem1_runner.D1_SELECTION_PATH.resolve()
            else mem1_runner.D1_DATASET_SHA256
        ),
    )
    args = _mem1d1_args(
        mem1_runner.D1_SELECTION_PATH,
        execution_question_id=["1cea1afa"],
        execution_system=["simplemem"],
    )
    _, _, _, locked_ids = mem1_runner._load_locked_inputs(args)
    assert locked_ids == list(mem1_runner.D1_QUESTION_IDS)
    assert args.execution_question_ids == ["1cea1afa"]
    assert args.execution_systems == ["simplemem"]

    invalid = _mem1d1_args(
        mem1_runner.D1_SELECTION_PATH,
        execution_question_id=["test-only"],
    )
    with pytest.raises(ValueError, match="unique subset of frozen IDs"):
        mem1_runner._load_locked_inputs(invalid)


def test_mem1d1_freeze_verifies_all_four_evidence_ledgers(tmp_path):
    artifacts = (
        "predictions.jsonl",
        "context_bundles.jsonl",
        "call_ledger.jsonl",
        "baseline_warnings.jsonl",
    )
    for name in artifacts:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    assert mem1_runner._verify_frozen_evidence(tmp_path, require_ledgers=True) is False
    for name in artifacts:
        mem1_runner.write_hash_sidecar(
            tmp_path / name, tmp_path / name.replace(".jsonl", ".sha256")
        )
    assert mem1_runner._verify_frozen_evidence(tmp_path, require_ledgers=True) is True

    with (tmp_path / "call_ledger.jsonl").open("a", encoding="utf-8") as output:
        output.write("tamper\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        mem1_runner._verify_frozen_evidence(tmp_path, require_ledgers=True)


def test_context_bundle_validation_is_independent_of_sidecar_freezing(monkeypatch, tmp_path):
    bundle = {"context_bundle_sha256": "bundle-sha"}
    bundle_path = tmp_path / "context_bundles.jsonl"
    bundle_path.write_text(
        json.dumps({
            "system": "openclaw",
            "question_id": "q1",
            "context_bundle": bundle,
        }) + "\n",
        encoding="utf-8",
    )
    prediction_rows = {
        ("openclaw", "q1"): {"context_bundle_sha256": "bundle-sha"}
    }
    monkeypatch.setattr(mem1_runner, "verify_context_bundle", lambda value: True)

    assert mem1_runner._context_bundle_rows_valid(
        tmp_path, ["openclaw"], ["q1"], prediction_rows, freeze=False
    ) is True
    assert not (tmp_path / "context_bundles.sha256").exists()

    assert mem1_runner._context_bundle_rows_valid(
        tmp_path, ["openclaw"], ["q1"], prediction_rows, freeze=True
    ) is True
    assert mem1_runner.verify_hash_sidecar(
        bundle_path, tmp_path / "context_bundles.sha256"
    )


def test_context_bundle_content_and_hash_status_are_independent(tmp_path):
    bundle_path = tmp_path / "context_bundles.jsonl"
    sidecar_path = tmp_path / "context_bundles.sha256"
    bundle_path.write_text("{}\n", encoding="utf-8")

    assert mem1_runner._context_bundle_hash_frozen(tmp_path) is False

    mem1_runner.write_hash_sidecar(bundle_path, sidecar_path)
    assert mem1_runner._context_bundle_hash_frozen(tmp_path) is True

    bundle_path.write_text("{\"changed\": true}\n", encoding="utf-8")
    assert mem1_runner._context_bundle_hash_frozen(tmp_path) is False


def test_context_controlled_is_the_default_answer_track(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mem1.py",
            "--stage", "generate",
            "--memeval-root", str(tmp_path / "MemEval"),
            "--run-dir", str(tmp_path / "run"),
            "--system", "fullcontext",
        ],
    )

    assert mem1_runner._parse_args().answer_track == "context_controlled"


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


def test_baseline_warning_telemetry_separates_recoveries_from_infra_failures():
    rows = mem1_runner._baseline_warning_rows(
        "mem0",
        "synthetic-q",
        "Error: '14'\nInvalid JSON response; retrying with repair",
        [
            {
                "role": "memory_ingest",
                "success": True,
                "retry_count": 1,
                "prompt_tokens": 3,
                "completion_tokens": 1,
            },
            {"role": "embedding", "success": True, "truncated": True},
            {
                "role": "memory_reasoning",
                "success": True,
                "prompt_tokens": None,
                "completion_tokens": None,
            },
            {"role": "reader_answer", "success": False, "retry_count": 0},
        ],
        adapter_completed=True,
    )

    by_type = {row["warning_type"]: row for row in rows}
    assert by_type["MEM0_ACTION_HANDLER_WARNING"]["classification"] == "BASELINE_INTERNAL_WARNING"
    assert by_type["MEM0_ACTION_HANDLER_WARNING"]["affected_operation"] == "DELETE"
    assert by_type["MEM0_INVALID_JSON_RESPONSE"]["recovered"] is True
    assert by_type["PROVIDER_RETRY"]["count"] == 1
    assert by_type["EMBEDDING_TRUNCATION"]["classification"] == "BASELINE_INTERNAL_WARNING"
    assert by_type["MISSING_USAGE_TELEMETRY"]["phase"] == "memory_reasoning"
    assert by_type["PROVIDER_FAILURE"]["classification"] == "INFRA_FAILURE"
    assert all("Invalid JSON response" not in str(row) for row in rows)

    summary = mem1_runner._summarize_baseline_warnings(rows, ["mem0"])["mem0"]
    assert summary["baseline_internal_warning_count"] == 5
    assert summary["infra_failure_count"] == 1


def test_baseline_warning_telemetry_records_simplemem_index_and_parser_recovery():
    rows = mem1_runner._baseline_warning_rows(
        "simplemem",
        "synthetic-q",
        "FTS index creation skipped: unavailable\nWarning: Failed to parse result",
        [],
        adapter_completed=True,
    )
    assert {row["warning_type"] for row in rows} == {
        "SIMPLEMEM_FTS_INDEX_WARNING",
        "SIMPLEMEM_PARSE_RECOVERY",
    }
    assert all(row["classification"] == "BASELINE_INTERNAL_WARNING" for row in rows)


def test_simplemem_official_retrieval_trace_is_retained_in_prediction_diagnostics(tmp_path):
    trace = {
        "source_tag": "v0.1.0",
        "source_commit": mem1_runner.PINNED_SIMPLEMEM_SHA,
        "native_answer_head_invoked": False,
        "calls": {"semantic": 2, "keyword": 1, "structured": 1, "merge_deduplicate": 1},
    }
    prediction = {
        "system": "simplemem",
        "question_id": "1cea1afa",
        "memory_system_diagnostics": trace,
    }
    assert prediction["memory_system_diagnostics"]["source_commit"] == mem1_runner.PINNED_SIMPLEMEM_SHA
    assert prediction["memory_system_diagnostics"]["calls"]["keyword"] == 1
    assert prediction["memory_system_diagnostics"]["native_answer_head_invoked"] is False


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
    assert mem1_runner.verify_hash_sidecar(
        tmp_path / "call_ledger.jsonl", tmp_path / "call_ledger.sha256"
    )
    assert mem1_runner.verify_hash_sidecar(
        tmp_path / "baseline_warnings.jsonl", tmp_path / "baseline_warnings.sha256"
    )
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


def test_abstention_uses_question_id_suffix_and_preserves_category():
    summary = mem1_runner._summarize_predictions(
        [
            {
                "system": "propmem",
                "question_id": "question_abs",
                "category": "knowledge-update",
                "quality_status": "OK",
                "predicted": "None",
                "f1": 1.0,
                "token_precision": 1.0,
                "token_recall": 1.0,
                "normalized_exact_match": 1.0,
            },
            {
                "system": "propmem",
                "question_id": "question_regular",
                "category": "temporal-reasoning",
                "quality_status": "OK",
                "predicted": "None",
                "f1": 0.0,
                "token_precision": 0.0,
                "token_recall": 0.0,
                "normalized_exact_match": 0.0,
            },
        ],
        ["propmem"],
        ["question_abs", "question_regular"],
    )["propmem"]

    assert summary["abstention_n"] == 1
    assert summary["abstention_accuracy"] == 1.0
    assert summary["by_category"]["knowledge-update"]["n"] == 1
    assert summary["by_category"]["temporal-reasoning"]["n"] == 1


def test_empty_gold_f1_and_abstention_accuracy_share_refusal_semantics():
    for prediction, expected_score in (
        ("None.", 1.0),
        ("There is no evidence that this was discussed, but perhaps it happened.", 0.0),
    ):
        answer = mem1_runner._answer_metrics(prediction, "")
        summary = mem1_runner._summarize_predictions(
            [{
                "system": "propmem",
                "question_id": "question_abs",
                "category": "knowledge-update",
                "quality_status": "OK",
                "predicted": prediction,
                **answer,
            }],
            ["propmem"],
            ["question_abs"],
        )["propmem"]
        assert answer["f1"] == expected_score
        assert summary["abstention_accuracy"] == expected_score


def test_reader_context_token_counter_uses_only_loopback_llama_tokenize(monkeypatch):
    requests = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"tokens": [1, 2, 3]}

    class FakeClient:
        def __init__(self, *, timeout, trust_env):
            assert timeout == 120
            assert trust_env is False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def post(self, url, *, json):
            requests.append((url, json))
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(Client=FakeClient))
    counter = mem1_runner._reader_context_token_counter(
        types.SimpleNamespace(reader_base_url="http://127.0.0.1:8081/v1")
    )
    assert counter(["shared context"]) == 3
    assert requests == [(
        "http://127.0.0.1:8081/tokenize",
        {"content": "shared context", "add_special": False},
    )]
    with pytest.raises(RuntimeError, match="loopback"):
        mem1_runner._reader_context_token_counter(
            types.SimpleNamespace(reader_base_url="https://api.example.com/v1")
        )


def test_failure_attribution_field_is_explicitly_a_heuristic_hint():
    row = {
        "system": "propmem",
        "question_id": "question-1",
        "quality_status": "OK",
        "predicted": "answer",
        "ground_truth": "answer",
        "failure_attribution_hint": ["CONTEXT_HAS_ANSWER_READER_MISSED"],
        "failure_attribution_heuristic": True,
    }
    assert row["failure_attribution_heuristic"] is True
    assert "failure_attribution" not in row


def test_missing_stream_usage_sums_to_null_not_zero():
    summary = mem1_runner._summarize_calls(
        [
            {
                "system": "simplemem",
                "role": "reader_answer",
                "provider": "local_qwen",
                "prompt_tokens": None,
                "completion_tokens": None,
                "latency_ms": 100,
                "success": True,
            }
        ],
        ["simplemem"],
    )["simplemem"]["by_role"]["reader_answer"]

    assert summary["prompt_tokens"] is None
    assert summary["completion_tokens"] is None
    assert summary["usage_capture_status"] == "NOT_CAPTURED"
