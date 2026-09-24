"""Shared checks for artifacts that must be committed before TEST is opened."""

from __future__ import annotations

import subprocess
from pathlib import Path


def assert_committed_artifact(path: Path, *, repo_root: Path) -> None:
    root = repo_root.resolve()
    target = path.resolve()
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("frozen protocol artifact must be inside the repository") from exc
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not target.is_file() or result.stdout != target.read_bytes():
        raise ValueError(f"protocol artifact must be committed unchanged before TEST: {relative}")
