import json
from pathlib import Path

import pytest

from health_ai_copilot.benchmarks.adapters import (
    HealthBenchAdapter,
    MedicalMirageAdapter,
    NFCorpusAdapter,
)
from health_ai_copilot.benchmarks.audit import audit_research_pack
from health_ai_copilot.benchmarks.contracts import (
    BenchmarkContractError,
    DatasetAdmissibility,
    RawArtifactIdentity,
)
from health_ai_copilot.benchmarks.registry import default_benchmark_registry

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "benchmarks"


def test_registry_is_explicit_and_contains_only_e0_entries() -> None:
    registry = default_benchmark_registry()
    assert [item.benchmark_id for item in registry.list()] == [
        "healthbench-v1",
        "medical-mirage-v1",
        "nfcorpus-v1",
        "research-architecture-v1",
    ]
    assert registry.get("healthbench-v1").status == DatasetAdmissibility.REVIEW_REQUIRED


def test_floating_revision_is_rejected() -> None:
    with pytest.raises(BenchmarkContractError):
        RawArtifactIdentity(name="x", url="https://example.invalid/x", sha256="main")


def test_nfcorpus_adapter_preserves_graded_qrels_and_is_deterministic(tmp_path: Path) -> None:
    raw = FIXTURES / "nfcorpus"
    first = tmp_path / "first"
    second = tmp_path / "second"
    identity_a = NFCorpusAdapter().normalize(raw, first)
    identity_b = NFCorpusAdapter().normalize(raw, second)

    assert identity_a.normalized_sha256 == identity_b.normalized_sha256
    cases = [json.loads(line) for line in (first / "cases.jsonl").read_text().splitlines()]
    assert cases[0]["gold"]["qrels"] == {"doc-1": 2}
    assert NFCorpusAdapter().validate(first) == []


def test_mirage_adapter_preserves_subdataset_and_question_only_protocol(tmp_path: Path) -> None:
    output = tmp_path / "mirage"
    MedicalMirageAdapter().normalize(FIXTURES / "medical_mirage", output)
    case = json.loads((output / "cases.jsonl").read_text().splitlines()[0])
    assert case["metadata"]["subdataset"] == "synthetic_medqa"
    assert case["payload"]["retrieval_protocol"]["question_only"] is True
    assert case["payload"]["options"]["A"] == "salt"


def test_healthbench_adapter_preserves_rubrics_without_judging(tmp_path: Path) -> None:
    output = tmp_path / "healthbench"
    HealthBenchAdapter().normalize(FIXTURES / "healthbench", output)
    case = json.loads((output / "cases.jsonl").read_text().splitlines()[0])
    assert case["case_id"] == "hb-1"
    assert case["gold"]["rubrics"][0]["weight"] == 1
    assert case["metadata"]["subset"] == "main"


def test_research_pack_has_pre_registered_balance_and_is_not_frozen() -> None:
    report = audit_research_pack(ROOT / "benchmarks" / "research_architecture_v1")
    assert report.details["case_count"] == 48
    assert set(report.details["category_counts"].values()) == {6}
    assert "pending human review" in report.errors
    assert all("expected-winner" not in error for error in report.errors)
