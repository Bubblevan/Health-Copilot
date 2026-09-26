import importlib.util
from pathlib import Path

import pytest


_module_path = Path(__file__).parents[1] / "tools" / "research" / "memory" / "mem1_artifacts.py"
_spec = importlib.util.spec_from_file_location("mem1_artifacts_test", _module_path)
mem1_artifacts = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(mem1_artifacts)


def cache_identity(**changes):
    values = {
        "system": "PropMem",
        "question_id": "1cea1afa",
        "dataset_sha256": "dataset",
        "system_config_hash": "config-a",
        "prompt_hashes": {"answer": "answer-prompt-a", "ingest": "ingest-prompt-a"},
        "reader_artifact_sha256": "reader",
        "embedding_model": "text-embedding-3-small",
        "code_patch_hash": "patch-a",
    }
    values.update(changes)
    return mem1_artifacts.make_cache_identity(**values)


def test_cache_identity_changes_when_prompt_or_config_changes():
    baseline = cache_identity()
    assert cache_identity()["identity_sha256"] == baseline["identity_sha256"]
    assert cache_identity(prompt_hashes={"answer": "changed", "ingest": "ingest-prompt-a"})[
        "identity_sha256"
    ] != baseline["identity_sha256"]
    assert cache_identity(reader_artifact_sha256="different-reader")["identity_sha256"] != baseline[
        "identity_sha256"
    ]
    assert cache_identity(system_config_hash="config-b")["identity_sha256"] != baseline[
        "identity_sha256"
    ]


def test_cache_reuse_requires_exact_identity_and_valid_quality(tmp_path):
    path = tmp_path / "predictions.jsonl"
    identity = cache_identity()
    row = {
        "cache_identity": identity,
        "quality_status": "OK",
        "predicted": "Swimming",
    }
    mem1_artifacts.append_jsonl(path, row)
    assert mem1_artifacts.find_cached_prediction(path, identity) == row

    assert (
        mem1_artifacts.find_cached_prediction(
            path,
            cache_identity(prompt_hashes={"answer": "changed", "ingest": "ingest-prompt-a"}),
        )
        is None
    )
    mem1_artifacts.append_jsonl(
        path,
        {**row, "quality_status": "INFRA_FAILURE", "predicted": None},
    )
    assert mem1_artifacts.find_cached_prediction(path, identity) == row


def test_frozen_jsonl_hash_is_stable(tmp_path):
    path = tmp_path / "predictions.jsonl"
    mem1_artifacts.append_jsonl(path, {"question_id": "q1", "predicted": "yes"})
    first = mem1_artifacts.freeze_jsonl(path)
    assert first == mem1_artifacts.freeze_jsonl(path)
    mem1_artifacts.append_jsonl(path, {"question_id": "q2", "predicted": "no"})
    assert first != mem1_artifacts.freeze_jsonl(path)


def test_hash_sidecar_detects_post_freeze_mutation(tmp_path):
    path = tmp_path / "predictions.jsonl"
    sidecar = tmp_path / "predictions.sha256"
    mem1_artifacts.append_jsonl(path, {"question_id": "q1", "predicted": "yes"})

    digest = mem1_artifacts.write_hash_sidecar(path, sidecar)
    assert digest == mem1_artifacts.sha256_file(path)
    assert mem1_artifacts.verify_hash_sidecar(path, sidecar)

    mem1_artifacts.append_jsonl(path, {"question_id": "q2", "predicted": "no"})
    assert not mem1_artifacts.verify_hash_sidecar(path, sidecar)


def test_corrupt_cache_identity_fails_closed(tmp_path):
    path = tmp_path / "predictions.jsonl"
    identity = cache_identity()
    mem1_artifacts.append_jsonl(
        path,
        {"cache_identity": {**identity, "system_config_hash": "tampered"}, "quality_status": "OK", "predicted": "yes"},
    )

    try:
        mem1_artifacts.find_cached_prediction(path, identity)
    except ValueError as error:
        assert "identity hash" in str(error)
    else:
        raise AssertionError("tampered cache identity must not be reused")


def run_manifest_values():
    return {
        "run_id": "mem1a-offline-test",
        "track": "main",
        "dataset": {"id": "longmemeval_s", "sha256": "dataset", "test_access": False},
        "split": "DEV",
        "question_ids": ["1cea1afa"],
        "roles": {
            "reader_answer_model": {"model": "Qwen3-8B", "artifact_sha256": "reader"},
            "memory_system": {"name": "PropMem", "config_sha256": "config"},
            "embedding_model": {"model": "text-embedding-3-small"},
            "judge_model": {"model": "gpt-4o"},
        },
        "system_config_hash": "config",
        "code_patch_sha256": "patch",
        "runner_code_sha256": "runner",
        "created_at": "2026-09-26T00:00:00Z",
    }


def test_run_manifest_keeps_roles_separate_and_is_immutable(tmp_path):
    path = tmp_path / "run_manifest.json"
    values = run_manifest_values()
    manifest = mem1_artifacts.write_run_manifest(path, **values)
    assert set(manifest["roles"]) == {
        "reader_answer_model",
        "memory_system",
        "embedding_model",
        "judge_model",
    }
    resumed = mem1_artifacts.write_run_manifest(
        path, **{**values, "created_at": "2026-09-27T00:00:00Z"}
    )
    assert resumed == manifest

    changed = {**values, "system_config_hash": "changed"}
    try:
        mem1_artifacts.write_run_manifest(path, **changed)
    except FileExistsError:
        pass
    else:
        raise AssertionError("frozen run manifest must not be overwritten")


def test_run_manifest_refuses_test_access_and_credentials(tmp_path):
    values = run_manifest_values()
    with pytest.raises(ValueError, match="DEV only"):
        mem1_artifacts.write_run_manifest(tmp_path / "test.json", **{**values, "split": "TEST"})

    roles = {**values["roles"], "judge_model": {"model": "gpt-4o", "api_key": "secret"}}
    with pytest.raises(ValueError, match="credentials"):
        mem1_artifacts.write_run_manifest(tmp_path / "secret.json", **{**values, "roles": roles})
