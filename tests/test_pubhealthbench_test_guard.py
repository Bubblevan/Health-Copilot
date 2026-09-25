from __future__ import annotations

import json
import subprocess

import pytest

from eval.pubhealthbench_test_guard import (
    HoldoutAuthorizationError,
    authorize_partition,
)


def test_validation_is_the_default_safe_partition() -> None:
    assert authorize_partition() == "VALIDATION_ALLOWED"


@pytest.mark.parametrize("partition", ["reviewed", "test", "full"])
def test_confirmatory_or_full_partition_is_refused_without_authorization(
    partition: str, tmp_path
) -> None:
    with pytest.raises(HoldoutAuthorizationError):
        authorize_partition(partition, lock_path=tmp_path / "missing.json")


def test_reviewed_requires_a_valid_committed_lock(tmp_path) -> None:
    lock_path = tmp_path / "final_method_lock.json"
    lock_path.write_text(json.dumps({"test_opened": False}), encoding="utf-8")
    with pytest.raises(HoldoutAuthorizationError, match="schema_version"):
        authorize_partition("reviewed", lock_path=lock_path)


def test_full_is_out_of_scope_even_if_named_as_test(tmp_path) -> None:
    lock_path = tmp_path / "final_method_lock.json"
    lock_path.write_text(json.dumps({"test_opened": False}), encoding="utf-8")
    with pytest.raises(HoldoutAuthorizationError, match="Full 7,929"):
        authorize_partition("full", lock_path=lock_path)


def test_reviewed_allows_only_a_committed_lock_with_frozen_qwen(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.invalid")
    (repo_root / "README.md").write_text("fixture\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "base")
    base_commit = git("rev-parse", "HEAD")

    lock_path = repo_root / "runs/e1_4/final_method_lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": "pubhealthbench-final-method-lock-v1",
                "repo_commit": base_commit,
                "qwen_model_sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
                "prompt_sha256": "1" * 64,
                "retriever_identities": {"retriever": "frozen"},
                "router_identity": "fixed",
                "thresholds": {"route": 0.5},
                "cost_rules": {"retrieval": 1},
                "metric_protocol": {"primary": "accuracy"},
                "dev_results_sha256": "2" * 64,
                "test_opened": False,
            }
        ),
        encoding="utf-8",
    )
    git("add", "runs/e1_4/final_method_lock.json")
    git("commit", "-qm", "lock frozen method")

    assert (
        authorize_partition("reviewed", lock_path=lock_path, repo_root=repo_root)
        == "THIS WILL CONSUME THE CONFIRMATORY HOLDOUT"
    )
