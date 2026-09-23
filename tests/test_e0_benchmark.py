import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from health_ai_copilot.benchmarks.adapters import (
    HealthBenchAdapter,
    MedicalMirageAdapter,
    NFCorpusAdapter,
)
from health_ai_copilot.benchmarks.audit import audit_research_pack, research_pack_hashes
from health_ai_copilot.benchmarks.contracts import (
    BenchmarkContractError,
    DatasetAdmissibility,
    RawArtifactIdentity,
)
from health_ai_copilot.benchmarks.fetch import FetchError, safe_extract_zip
from health_ai_copilot.benchmarks.registry import default_benchmark_registry
from health_ai_copilot.benchmarks.review import REVIEW_CHECKS, apply_review, freeze_research_pack

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
    assert registry.get("healthbench-v1").status == DatasetAdmissibility.APPROVED


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


def test_mirage_adapter_accepts_id_to_record_dataset_mappings(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "benchmark.json").write_text(
        json.dumps(
            {
                "medqa": {
                    "0007": {
                        "question": "What is the safe next step?",
                        "options": {"A": "Observe", "B": "Escalate"},
                        "answer": "B",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "normalized"
    identity = MedicalMirageAdapter().normalize(raw, output)
    case = json.loads((output / "cases.jsonl").read_text().splitlines()[0])
    assert identity.case_count == 1
    assert case["case_id"] == "medqa:0007"
    assert case["gold"]["answer"] == "B"


def test_safe_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape.txt", "must not extract")
    with pytest.raises(FetchError, match="escapes extraction root"):
        safe_extract_zip(archive, tmp_path / "extracted")


def test_healthbench_adapter_preserves_rubrics_without_judging(tmp_path: Path) -> None:
    output = tmp_path / "healthbench"
    HealthBenchAdapter().normalize(FIXTURES / "healthbench", output)
    case = json.loads((output / "cases.jsonl").read_text().splitlines()[0])
    assert case["case_id"] == "hb-1"
    assert case["gold"]["rubrics"][0]["weight"] == 1
    assert case["metadata"]["subset"] == "main"


def test_research_pack_has_reviewed_balance_and_frozen_identity() -> None:
    report = audit_research_pack(ROOT / "benchmarks" / "research_architecture_v1")
    assert report.details["case_count"] == 48
    assert set(report.details["category_counts"].values()) == {6}
    assert report.errors == ()
    assert report.details["annotation_status"] == "frozen"
    assert all("expected-winner" not in error for error in report.errors)


def test_review_apply_and_freeze_recompute_and_persist_identity(tmp_path: Path) -> None:
    source = ROOT / "benchmarks" / "research_architecture_v1"
    staged = tmp_path / "staged"
    shutil.copytree(source, staged)
    cases = [json.loads(line) for line in (source / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    decisions = tmp_path / "decisions.jsonl"
    rows = []
    for index, case in enumerate(cases):
        rows.append(
            {
                "case_id": case["case_id"],
                "decision": "EDIT" if index == 0 else "APPROVE",
                **{check: True for check in REVIEW_CHECKS},
                "notes": "test review",
                "proposed_edits": {"question": "Edited review question."} if index == 0 else {},
                "reviewer": "test-reviewer",
                "review_date": "2026-09-23",
            }
        )
    edited_case_id = rows[0]["case_id"]
    decisions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "reviewed"
    result = apply_review(staged, decisions, output)
    assert result["case_count"] == 48
    reviewed_cases = [
        json.loads(line) for line in (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reviewed_profiles = [
        json.loads(line)
        for line in (output / "task_profiles.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reviewed_case = next(case for case in reviewed_cases if case["case_id"] == edited_case_id)
    reviewed_profile = next(profile for profile in reviewed_profiles if profile["case_id"] == edited_case_id)
    expected_hash = hashlib.sha256(b"edited review question.").hexdigest()
    assert reviewed_case["payload"]["question"] == "Edited review question."
    assert reviewed_profile["text_hash"] == expected_hash
    assert json.loads((output / "annotation_manifest.json").read_text(encoding="utf-8"))["status"] == "reviewed"

    frozen = freeze_research_pack(output)
    annotation = json.loads((output / "annotation_manifest.json").read_text(encoding="utf-8"))
    assert frozen["frozen"] is True
    assert annotation["status"] == "frozen"
    assert set(annotation["hashes"]) == {
        "cases_sha256",
        "gold_sha256",
        "task_profiles_sha256",
        "split_sha256",
        "annotation_manifest_sha256",
        "aggregate_benchmark_sha256",
    }
    assert annotation["hashes"] == research_pack_hashes(output)
