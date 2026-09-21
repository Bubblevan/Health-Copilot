"""E0 benchmark foundation contracts and offline tooling."""

from .adapters import (
    BenchmarkAdapter,
    HealthBenchAdapter,
    MedicalMirageAdapter,
    NFCorpusAdapter,
    adapter_for,
)
from .contracts import (
    BenchmarkCase,
    BenchmarkManifest,
    DatasetAdmissibility,
    ExperimentBudgetContract,
    JudgeProtocolManifest,
    NormalizedDatasetIdentity,
    RawArtifactIdentity,
    TaskProfile,
)
from .registry import BenchmarkRegistry, default_benchmark_registry

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkCase",
    "BenchmarkManifest",
    "BenchmarkRegistry",
    "DatasetAdmissibility",
    "ExperimentBudgetContract",
    "HealthBenchAdapter",
    "JudgeProtocolManifest",
    "MedicalMirageAdapter",
    "NFCorpusAdapter",
    "NormalizedDatasetIdentity",
    "RawArtifactIdentity",
    "TaskProfile",
    "adapter_for",
    "default_benchmark_registry",
]
