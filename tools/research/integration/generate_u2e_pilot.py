"""Generate the U2-E pilot in a temporary allowlisted working directory."""

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
    "diagnostics.py", "lineage.py", "source_independence.py", "cli.py",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _hash_tree(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


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


def _run_isolated(temp_root: Path, output: Path, *, branch: str, base_commit: str) -> None:
    source_root, spec_root = _copy_allowlisted_sources(temp_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    env["PYTHONUTF8"] = "1"
    command = [
        sys.executable, "-m", "health_ai_copilot.research.integration.owned_universe.cli",
        "--spec-root", str(spec_root), "--source-root", str(source_root),
        "--output", str(output), "--branch", branch, "--base-commit", base_commit,
    ]
    completed = subprocess.run(command, cwd=temp_root, env=env, text=True,
                               capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(f"isolated generator exited {completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "runs" / "integration")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--base-commit", default=None)
    args = parser.parse_args()

    branch = args.branch or _git("branch", "--show-current")
    base_commit = args.base_commit or _git("rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(prefix="health-copilot-u2e-") as temp_name:
        temp_root = Path(temp_name)
        first = temp_root / "run-a"
        second = temp_root / "run-b"
        _run_isolated(temp_root / "isolated-a", first, branch=branch, base_commit=base_commit)
        _run_isolated(temp_root / "isolated-b", second, branch=branch, base_commit=base_commit)
        first_hashes = _hash_tree(first)
        second_hashes = _hash_tree(second)
        if first_hashes != second_hashes:
            raise RuntimeError("same generator_version/seed/spec did not produce byte-identical artifacts")

        manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "QUALIFIED_PILOT":
            raise RuntimeError(f"U2-E pilot gates failed: {manifest['gates']}")
        output = args.output_root / f"u2e-pilot-{manifest['run_id']}"
        if output.exists():
            if _hash_tree(output) != first_hashes:
                raise FileExistsError(f"existing pilot directory differs; refusing overwrite: {output}")
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
