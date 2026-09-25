"""Fail-closed partition gate for the PubHealthBench confirmatory set."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "runs/e1_4/final_method_lock.json"
LOCK_SCHEMA = "pubhealthbench-final-method-lock-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_QWEN_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"


class HoldoutAuthorizationError(RuntimeError):
    """Raised when a confirmatory partition is not safely unlocked."""


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _validate_lock(lock: dict[str, Any], lock_path: Path, repo_root: Path) -> list[str]:
    errors: list[str] = []
    if lock.get("schema_version") != LOCK_SCHEMA:
        errors.append("schema_version must identify the PubHealthBench final-method lock")
    if not isinstance(lock.get("repo_commit"), str) or not COMMIT_RE.fullmatch(lock["repo_commit"]):
        errors.append("repo_commit must be a full 40-character Git commit")
    elif _git(repo_root, "cat-file", "-e", f"{lock['repo_commit']}^{{commit}}").returncode != 0:
        errors.append("repo_commit is not present in this repository")
    for key in ("qwen_model_sha256", "prompt_sha256", "dev_results_sha256"):
        if not isinstance(lock.get(key), str) or not SHA256_RE.fullmatch(lock[key]):
            errors.append(f"{key} must be a 64-character SHA-256")
    if lock.get("qwen_model_sha256") != REQUIRED_QWEN_SHA256:
        errors.append("qwen_model_sha256 must match the frozen Qwen3-8B-Q4_K_M artifact")
    for key in ("retriever_identities", "thresholds", "cost_rules", "metric_protocol"):
        if not isinstance(lock.get(key), dict) or not lock[key]:
            errors.append(f"{key} must be a non-empty object")
    if not isinstance(lock.get("router_identity"), str) or not lock["router_identity"].strip():
        errors.append("router_identity must be a non-empty string")
    if lock.get("test_opened") is not False:
        errors.append("test_opened must be false before the first holdout run")

    try:
        relative = lock_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        errors.append("final_method_lock.json must be inside the repository")
        return errors
    tracked = _git(repo_root, "ls-files", "--error-unmatch", relative)
    if tracked.returncode != 0:
        errors.append("final_method_lock.json must be Git-tracked")
    if _git(repo_root, "diff", "--quiet", "HEAD", "--", relative).returncode != 0:
        errors.append("final_method_lock.json must be committed and unchanged")
    return errors


def authorize_partition(
    partition: str = "validation",
    *,
    lock_path: Path = DEFAULT_LOCK,
    repo_root: Path = ROOT,
) -> str:
    normalized = partition.strip().lower()
    if normalized == "validation":
        return "VALIDATION_ALLOWED"
    if normalized in {"test", "full", "pubhealthbench-full"}:
        raise HoldoutAuthorizationError(
            "The Full 7,929 partition is out of scope for this sprint and is not an allowed runner partition."
        )
    if normalized != "reviewed":
        raise HoldoutAuthorizationError("partition must be 'validation' or 'reviewed'")
    if not lock_path.is_file():
        raise HoldoutAuthorizationError(
            "REFUSED: Reviewed confirmatory holdout requires a committed "
            "runs/e1_4/final_method_lock.json."
        )
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutAuthorizationError(f"REFUSED: final-method lock is unreadable: {exc}") from exc
    if not isinstance(lock, dict):
        raise HoldoutAuthorizationError("REFUSED: final-method lock must be a JSON object.")
    errors = _validate_lock(lock, lock_path, repo_root)
    if errors:
        raise HoldoutAuthorizationError("REFUSED: invalid final-method lock: " + "; ".join(errors))
    return "THIS WILL CONSUME THE CONFIRMATORY HOLDOUT"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", default="validation", choices=("validation", "reviewed", "test", "full"))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        status = authorize_partition(args.partition, lock_path=args.lock, repo_root=args.repo_root)
    except HoldoutAuthorizationError as exc:
        parser.error(str(exc))
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
