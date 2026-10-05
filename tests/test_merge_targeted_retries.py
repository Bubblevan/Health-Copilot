import json
from hashlib import sha256
from pathlib import Path

from tools.eval.extract_failed_case_ids import extract_case_ids
from tools.eval.merge_targeted_retries import merge_targeted_retries


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_merge_replaces_only_frozen_transport_failures_without_mutating_sources(tmp_path) -> None:
    source = tmp_path / "source"
    retry = tmp_path / "retry"
    source.mkdir()
    retry.mkdir()
    full_ids_hash = sha256(b"case-1\ncase-2\n").hexdigest()
    checkpoint_hash = "f" * 64
    profile = {
        "profile_id": "B2",
        "model_variant": "qwen3_8b_base",
        "retrieval_mode": "off",
        "memory_mode": "off",
        "reasoning_mode": "adaptive_mdt",
    }
    dataset = {
        "dataset_id": "fixture",
        "source_revision": "fixture-revision",
        "combined_dataset_identity_sha256": "a" * 64,
        "ids_manifest_sha256": "b" * 64,
        "ids_sequence_sha256": full_ids_hash,
        "expected_case_count": 2,
    }
    failed = {
        "case_id": "case-1",
        "dataset_id": "fixture",
        "profile_id": "B2",
        "dataset_selection_sha256": full_ids_hash,
        "response": {"safety_flags": ["reasoning_failure:APIConnectionError"], "provider_calls": 1},
        "score": {"correct": False, "parse_success": False},
    }
    retained = {
        "case_id": "case-2",
        "dataset_id": "fixture",
        "profile_id": "B2",
        "dataset_selection_sha256": full_ids_hash,
        "response": {"answer_text": "B", "safety_flags": [], "provider_calls": 2},
        "score": {"correct": True, "parse_success": True},
    }
    (source / "cases.jsonl").write_text(
        json.dumps(failed) + "\n" + json.dumps(retained) + "\n", encoding="utf-8",
    )
    _write_json(source / "invalidation.json", {
        "status": "INVALID_INFRASTRUCTURE_FAILURE",
        "dataset_id": "fixture",
        "profile_id": "B2",
        "connection_failures": {"APIConnectionError": 1},
    })
    source_manifest = {
        "status": "INVALID_INFRASTRUCTURE_FAILURE",
        "dataset": dataset,
        "system_profile": profile,
        "model": {"checkpoint_sha256": checkpoint_hash},
        "concurrency": 2,
    }
    _write_json(source / "run_manifest.json", source_manifest)
    _write_json(source / "summary.json", {"run_wall_seconds": 10})
    (source / "traces.jsonl").write_text('{"trace_id":"source-trace"}\n', encoding="utf-8")

    selection = extract_case_ids(source)
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selection)
    retry_dataset = {
        **dataset,
        "expected_case_count": 1,
        "selected_case_ids_sha256": selection["case_ids_sha256"],
        "case_ids_file_sha256": sha256(selection_path.read_bytes()).hexdigest(),
    }
    recovered = {
        **failed,
        "dataset_selection_sha256": selection["case_ids_sha256"],
        "response": {"answer_text": "A", "safety_flags": [], "provider_calls": 3},
        "score": {"correct": True, "parse_success": True},
    }
    (retry / "cases.jsonl").write_text(json.dumps(recovered) + "\n", encoding="utf-8")
    (retry / "traces.jsonl").write_text('{"trace_id":"retry-trace"}\n', encoding="utf-8")
    _write_json(retry / "summary.json", {"run_wall_seconds": 5})
    _write_json(retry / "run_manifest.json", {
        "status": "COMPLETE",
        "dataset": retry_dataset,
        "system_profile": profile,
        "model": {"checkpoint_sha256": checkpoint_hash},
        "concurrency": 4,
    })
    source_hash_before = sha256((source / "cases.jsonl").read_bytes()).hexdigest()

    output = merge_targeted_retries(
        source_dir=source,
        retry_dir=retry,
        selection_path=selection_path,
        output_dir=tmp_path / "composite",
    )

    merged = [json.loads(line) for line in (output / "cases.jsonl").read_text().splitlines()]
    assert [row["case_id"] for row in merged] == ["case-1", "case-2"]
    assert merged[0]["response"]["answer_text"] == "A"
    assert merged[0]["record_origin"] == "targeted_transport_retry"
    assert merged[1]["response"]["answer_text"] == "B"
    assert merged[1]["record_origin"] == "original_invalidated_run"
    assert all(row["dataset_selection_sha256"] == full_ids_hash for row in merged)
    assert len((output / "traces.jsonl").read_text().splitlines()) == 2
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "COMPOSITE_RECOVERY_PENDING_RESCORE"
    assert manifest["recovery_composite"]["selected_case_count"] == 1
    assert sha256((source / "cases.jsonl").read_bytes()).hexdigest() == source_hash_before
