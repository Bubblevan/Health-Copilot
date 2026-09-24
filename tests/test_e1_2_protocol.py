import subprocess

import pytest

from eval.e1_2_protocol import assert_committed_artifact


def test_committed_artifact_must_match_head_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    artifact = repo / "runs" / "lock.json"
    artifact.parent.mkdir()
    artifact.write_text('{"status":"LOCKED"}\n', encoding="utf-8")
    subprocess.run(["git", "add", "runs/lock.json"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=E1.2 test",
            "-c",
            "user.email=e1-2-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "test lock",
        ],
        cwd=repo,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    assert_committed_artifact(artifact, repo_root=repo)
    artifact.write_text('{"status":"CHANGED"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="committed unchanged"):
        assert_committed_artifact(artifact, repo_root=repo)
