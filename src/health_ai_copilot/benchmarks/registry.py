"""Explicit source-controlled E0 benchmark registry."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .contracts import BenchmarkManifest


class BenchmarkRegistryError(RuntimeError):
    """Base registry error."""


class DuplicateBenchmarkError(BenchmarkRegistryError):
    pass


class UnknownBenchmarkError(BenchmarkRegistryError):
    pass


@dataclass(frozen=True)
class BenchmarkRegistration:
    manifest: BenchmarkManifest
    manifest_path: Path


class BenchmarkRegistry:
    """Maps explicit benchmark IDs to trusted manifests without plugin scanning."""

    def __init__(self) -> None:
        self._registrations: dict[str, BenchmarkRegistration] = {}

    def register(self, manifest: BenchmarkManifest, manifest_path: Path | None = None) -> None:
        if manifest.benchmark_id in self._registrations:
            raise DuplicateBenchmarkError(f"duplicate benchmark: {manifest.benchmark_id}")
        self._registrations[manifest.benchmark_id] = BenchmarkRegistration(
            manifest=manifest,
            manifest_path=manifest_path or Path(),
        )

    def get(self, benchmark_id: str) -> BenchmarkManifest:
        try:
            return self._registrations[benchmark_id].manifest
        except KeyError as exc:
            raise UnknownBenchmarkError(f"unknown benchmark: {benchmark_id}") from exc

    def registration(self, benchmark_id: str) -> BenchmarkRegistration:
        try:
            return self._registrations[benchmark_id]
        except KeyError as exc:
            raise UnknownBenchmarkError(f"unknown benchmark: {benchmark_id}") from exc

    def list(self) -> tuple[BenchmarkManifest, ...]:
        return tuple(
            self._registrations[key].manifest for key in sorted(self._registrations)
        )

    def inspect(self, benchmark_id: str) -> dict:
        return self.get(benchmark_id).to_dict() | {
            "manifest_hash": self.get(benchmark_id).manifest_hash,
        }


def repository_root() -> Path:
    configured = os.environ.get("HEALTH_COPILOT_REPO_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[3]


def load_benchmark_registry(root: str | Path | None = None) -> BenchmarkRegistry:
    root_path = Path(root).resolve() if root else repository_root()
    registry_path = root_path / "benchmarks" / "registry.json"
    value = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("entries"), dict):
        raise BenchmarkRegistryError("benchmark registry needs an entries mapping")
    registry = BenchmarkRegistry()
    for benchmark_id, relative_path in sorted(value["entries"].items()):
        if not isinstance(benchmark_id, str) or not isinstance(relative_path, str):
            raise BenchmarkRegistryError("registry entries need string IDs and paths")
        manifest_path = root_path / relative_path
        manifest = BenchmarkManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        if manifest.benchmark_id != benchmark_id:
            raise BenchmarkRegistryError(
                f"registry ID {benchmark_id} does not match manifest {manifest.benchmark_id}"
            )
        registry.register(manifest, manifest_path)
    return registry


def default_benchmark_registry() -> BenchmarkRegistry:
    """Load the source-controlled registry; never performs network I/O."""

    return load_benchmark_registry()
