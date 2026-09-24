import json
from pathlib import Path

from eval.e1_2_runner import (
    load_partition,
    per_case_retrieval_latency,
    select_generator_pilot,
    summarize_results,
)


def _case(case_id: str, subdataset: str) -> dict:
    return {
        "case_id": case_id,
        "metadata": {"subdataset": subdataset},
        "payload": {"question": "Q?", "options": {"A": "a", "B": "b"}},
        "gold": {"answer": "A"},
    }


def test_partition_loader_matches_manifest_ids_and_excludes_exposed_history(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    rows = [_case("medqa:a", "medqa"), _case("pubmedqa:p", "pubmedqa")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    split = {
        "cases": [
            {"case_id": "medqa:a", "subdataset": "medqa", "split": "TEST"},
            {"case_id": "pubmedqa:p", "subdataset": "pubmedqa", "split": "EXPOSED_HISTORY"},
        ],
        "counts": {"medqa": {"TEST": 1}, "medmcqa": {"TEST": 0}, "mmlu": {"TEST": 0}},
    }
    assert [row["case_id"] for row in load_partition(path, split, "TEST")] == ["medqa:a"]


def test_generator_pilot_is_stable_and_balanced_without_label_stratification():
    rows = [
        _case(f"{subset}:{index:03}", subset)
        for subset in ("medqa", "medmcqa", "mmlu")
        for index in range(35)
    ]
    first = [row["case_id"] for row in select_generator_pilot(rows)]
    second = [row["case_id"] for row in select_generator_pilot(rows)]
    assert first == second
    assert len(first) == 90
    assert sum(case_id.startswith("medqa:") for case_id in first) == 30
    assert sum(case_id.startswith("medmcqa:") for case_id in first) == 30
    assert sum(case_id.startswith("mmlu:") for case_id in first) == 30


def test_fixed_denominator_accuracy_counts_provider_failures_as_incorrect():
    summary = summarize_results(
        [
            {"case_id": "a", "subdataset": "medqa", "status": "completed", "is_correct": True},
            {"case_id": "b", "subdataset": "medqa", "status": "failed", "is_correct": False},
        ]
    )
    assert summary["accuracy_fixed_denominator"] == 0.5
    assert summary["by_subdataset"]["medqa"]["answer_coverage"] == 0.5


def test_retrieval_latency_reads_recorded_component_metrics():
    assert per_case_retrieval_latency({"retrieval_metadata": {"latency_ms": 3.5}}, "bm25") == 3.5
    assert per_case_retrieval_latency(
        {"retrieved_evidence": [{"query_latency_ms": 12}]}, "medcpt"
    ) == 12.0
