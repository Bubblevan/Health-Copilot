"""Generate and byte-verify the full U2-F universe in isolated allowlists."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPO_SRC = ROOT / "src"
REPO_SPEC = ROOT / "benchmarks" / "integration_owned_v1"
U1_FILES = (
    "actions.py", "contracts.py", "counterfactual.py", "evaluator.py", "executor.py",
    "evidence_world.py", "replay.py", "state.py", "tools.py", "__init__.py",
)
OWNED_FILES = (
    "__init__.py", "schema.py", "facts.py", "grammar.py", "timeline.py", "scenarios.py",
    "splits.py", "realization.py", "realization_text.py", "evaluator_truth.py",
    "diagnostics.py", "lineage.py", "source_independence.py", "cli.py", "scale.py",
    "u2f_diagnostics.py", "u2f_cli.py",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _hash_tree(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    }


def _copy_allowlisted_sources(temp_root: Path) -> tuple[Path, Path]:
    temp_src = temp_root / "src"
    temp_spec = temp_root / "benchmarks" / "integration_owned_v1"
    integration_src = REPO_SRC / "health_ai_copilot" / "research" / "integration"
    integration_dst = temp_src / "health_ai_copilot" / "research" / "integration"
    owned_dst = integration_dst / "owned_universe"
    owned_dst.mkdir(parents=True)
    temp_spec.mkdir(parents=True)
    shutil.copy2(REPO_SRC / "health_ai_copilot" / "__init__.py",
                 temp_src / "health_ai_copilot" / "__init__.py")
    shutil.copy2(REPO_SRC / "health_ai_copilot" / "research" / "__init__.py",
                 temp_src / "health_ai_copilot" / "research" / "__init__.py")
    for name in U1_FILES:
        shutil.copy2(integration_src / name, integration_dst / name)
    owned_src = integration_src / "owned_universe"
    for name in OWNED_FILES:
        shutil.copy2(owned_src / name, owned_dst / name)
    for source in sorted(REPO_SPEC.glob("*.json")):
        shutil.copy2(source, temp_spec / source.name)
    return temp_src, temp_spec


def _run_isolated(
    temp_root: Path, output: Path, *, branch: str, base_commit: str,
    generator_commit: str, human_review_status: str, human_review_note: str,
) -> None:
    source_root, spec_root = _copy_allowlisted_sources(temp_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    env["PYTHONUTF8"] = "1"
    command = [
        sys.executable, "-m", "health_ai_copilot.research.integration.owned_universe.u2f_cli",
        "--spec-root", str(spec_root), "--source-root", str(source_root),
        "--output", str(output), "--branch", branch, "--base-commit", base_commit,
        "--generator-commit", generator_commit, "--determinism-verified",
        "--human-review-status", human_review_status,
        "--human-review-note", human_review_note,
    ]
    completed = subprocess.run(command, cwd=temp_root, env=env, text=True,
                               capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(f"isolated U2-F generator exited {completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "integration")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--base-commit", default=None)
    parser.add_argument("--generator-commit", default=None)
    parser.add_argument("--human-review-status", choices=("PENDING", "PASS", "FAIL"),
                        default="PENDING")
    parser.add_argument("--human-review-note", default="")
    parser.add_argument("--allow-candidate", action="store_true",
                        help="Keep a two-run deterministic candidate even when a quality gate is pending.")
    args = parser.parse_args()

    branch = args.branch or _git("branch", "--show-current")
    base_commit = args.base_commit or "ce68eb7d37463d255a8abd3ad0a1b73cfbbf518e"
    generator_commit = args.generator_commit or _git("rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(prefix="health-copilot-u2f-") as temp_name:
        temp_root = Path(temp_name)
        first = temp_root / "run-a"
        second = temp_root / "run-b"
        _run_isolated(
            temp_root / "isolated-a", first, branch=branch, base_commit=base_commit,
            generator_commit=generator_commit, human_review_status=args.human_review_status,
            human_review_note=args.human_review_note,
        )
        _run_isolated(
            temp_root / "isolated-b", second, branch=branch, base_commit=base_commit,
            generator_commit=generator_commit, human_review_status=args.human_review_status,
            human_review_note=args.human_review_note,
        )
        first_hashes = _hash_tree(first)
        second_hashes = _hash_tree(second)
        if first_hashes != second_hashes:
            raise RuntimeError("isolated full U2-F runs did not produce byte-identical artifacts")
        manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "PASS" and not args.allow_candidate:
            failed = [name for name, passed in manifest["scale_gate_details"]["gates"].items()
                      if not passed]
            raise RuntimeError(f"U2-F gates failed or remain pending: {failed}")
        suffix = manifest["run_id"]
        output = args.output_root / f"u2f-owned-v1-{suffix}"
        if output.exists():
            if _hash_tree(output) != first_hashes:
                raise FileExistsError(f"existing U2-F directory differs; refusing overwrite: {output}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(first, output)
        print(json.dumps({
            "status": manifest["status"],
            "run_id": manifest["run_id"],
            "output": str(output),
            "deterministic_regeneration": True,
            "artifact_count": len(first_hashes),
            "artifact_hashes": first_hashes,
        }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
