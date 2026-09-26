from __future__ import annotations

import ast
from pathlib import Path

import pytest

from eval.r2med_final_test import (
    FROZEN_ARMS,
    TEST_COUNTS,
    TEST_TOTAL,
    append_jsonl_once,
    read_lock,
    resume_claim_gates,
    validate_complete_generation,
    validate_final_lock,
    validate_generation_budget,
    validate_rankings,
)


def _valid_lock() -> dict:
    from eval.r2med_final_test import BOOTSTRAP, GENERATION_CONFIG, LOCKED_CODE_PATHS, QWEN_IDENTITY

    return {
        "schema_version": "r2med-final-public-test-lock-v1",
        "purpose": "PUBLIC_TEST_BASELINE_EVALUATION_ONLY",
        "test_status": "PUBLIC_BENCHMARK_REUSED",
        "test_counts": {**TEST_COUNTS, "total": TEST_TOTAL},
        "arms": FROZEN_ARMS,
        "generator": {
            "identity": QWEN_IDENTITY,
            "config": GENERATION_CONFIG,
            "llama_cpp_version": "test-version",
            "expected_calls": {"lamer": TEST_TOTAL, "compact_crb_q": TEST_TOTAL, "total": 606},
            "paid_api_calls": 0,
        },
        "bootstrap": BOOTSTRAP,
        "evaluation": {
            "primary_metric": "equal-subset macro nDCG@10",
            "qrels_boundary": "after_all_six_rankings_are_frozen",
        },
        "code": {"files_sha256": dict.fromkeys(LOCKED_CODE_PATHS, "0" * 64)},
    }


def test_final_lock_requires_exact_frozen_test_recipes() -> None:
    lock = _valid_lock()
    validate_final_lock(lock)
    lock["arms"] = {**FROZEN_ARMS, "B5_DualSource_RRF_lam0.5": {"crb_weight": 1.0}}
    with pytest.raises(ValueError, match="configuration"):
        validate_final_lock(lock)


@pytest.mark.parametrize(
    ("arm", "field", "value"),
    [
        ("B3_LameR_MV", "weights", [1, 1, 1, 1]),
        ("B4_Compact_CRB_Q", "weights", [2, 1, 1, 1]),
        ("B5_DualSource_RRF_lam0.5", "crb_weight", 1.0),
    ],
)
def test_test_runner_rejects_changed_frozen_weights(arm: str, field: str, value) -> None:
    lock = _valid_lock()
    altered = {key: dict(config) for key, config in FROZEN_ARMS.items()}
    altered[arm][field] = value
    lock["arms"] = altered
    with pytest.raises(ValueError, match="configuration"):
        validate_final_lock(lock)


def test_generation_checkpoint_refuses_duplicate_completed_query(tmp_path: Path) -> None:
    checkpoint = tmp_path / "partial.jsonl"
    record = {"query_id": "q1", "method": "lamer", "prompt_sha256": "p"}
    append_jsonl_once(checkpoint, record, expected_method="lamer", expected_prompt_sha="p")
    with pytest.raises(ValueError, match="already completed"):
        append_jsonl_once(checkpoint, record, expected_method="lamer", expected_prompt_sha="p")


def test_final_lock_is_required(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_lock(tmp_path / "missing-final-lock.json")


def test_lamer_and_compact_generation_budget_is_exactly_one_row_per_test_query() -> None:
    assert TEST_TOTAL == 303
    lamer = [{"query_id": str(i)} for i in range(TEST_TOTAL)]
    compact = [{"query_id": str(i)} for i in range(TEST_TOTAL)]
    validate_generation_budget(lamer, compact)
    with pytest.raises(ValueError, match="exactly 303"):
        validate_generation_budget(lamer[:-1], compact)


def test_generation_artifact_requires_frozen_prompt_and_query_order() -> None:
    rows = [
        {"query_id": "q1", "method": "crb_q_compact_repair", "prompt_sha256": "p", "error": None},
        {"query_id": "q2", "method": "crb_q_compact_repair", "prompt_sha256": "p", "error": None},
    ]
    validate_complete_generation(
        rows,
        ["q1", "q2"],
        expected_method="crb_q_compact_repair",
        expected_prompt_sha="p",
    )
    with pytest.raises(ValueError, match="query order"):
        validate_complete_generation(
            rows[::-1],
            ["q1", "q2"],
            expected_method="crb_q_compact_repair",
            expected_prompt_sha="p",
        )


def test_compact_schema_failure_falls_back_without_retry() -> None:
    from tools.generate_r2med_crb_compact_repair import _generation_record

    class InvalidOnceClient:
        calls = 0

        def complete(self, prompt: str, *, json_schema=None):
            self.calls += 1
            return {"text": "not-json", "finish_reason": "stop", "output_tokens": 2}

    client = InvalidOnceClient()
    row = _generation_record("q1", "original question", client)
    assert client.calls == 1
    assert row["valid"] is False
    assert row["fallback_original"] is True
    assert row["structured"]["canonical_query"] == "original question"


def test_ranking_rejects_duplicates_and_documents_outside_corpus() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_rankings({"q": ["d1", "d1"]}, ["q"], {"d1"})
    with pytest.raises(ValueError, match="outside the corpus"):
        validate_rankings({"q": ["d2"]}, ["q"], {"d1"})


def test_evaluator_is_the_only_qrels_reader() -> None:
    evaluator_path = Path(__file__).parents[1] / "eval/r2med_crb_evaluator.py"
    tree = ast.parse(evaluator_path.read_text(encoding="utf-8"))
    refs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(child, ast.Constant) and "qrels.jsonl" in str(child.value)
            for child in ast.walk(node)
        ):
            refs.append(node.name)
    assert refs == ["_load_qrels"]


def test_generation_and_ranking_code_do_not_name_or_open_qrels() -> None:
    root = Path(__file__).parents[1]
    for relative in (
        "tools/run_r2med_final_test.py",
        "tools/generate_r2med_crb_compact_repair.py",
        "tools/generate_r2med_gar.py",
        "eval/r2med_crb.py",
        "eval/r2med_multiview.py",
        "eval/r2med_candidate_union.py",
    ):
        assert "qrels.jsonl" not in (root / relative).read_text(encoding="utf-8")


def test_test_evaluation_happens_only_after_ranking_freeze() -> None:
    runner_path = Path(__file__).parents[1] / "tools/run_r2med_final_test.py"
    source = runner_path.read_text(encoding="utf-8")
    assert source.index("ranking_freeze = {") < source.index("report = _evaluate_and_report(")


def test_candidate_union_diagnostic_separates_pool_ceiling_from_source_only_hits(monkeypatch) -> None:
    from eval import r2med_crb_evaluator

    monkeypatch.setattr(
        r2med_crb_evaluator,
        "_load_qrels",
        lambda *args, **kwargs: {"q": {"a": 1, "b": 1, "c": 1}},
    )
    result = r2med_crb_evaluator.evaluate_candidate_complementarity(
        "TEST",
        {"tiny": {"lamer": {"q": ["a"]}, "crb": {"q": ["b"]}}},
        source_root=Path("unused"),
    )
    assert result["macro_equal_subset_weight"]["lamer_recall@100"] == pytest.approx(1 / 3)
    assert result["macro_equal_subset_weight"]["crb_recall@100"] == pytest.approx(1 / 3)
    assert result["macro_equal_subset_weight"]["raw_union_candidate_recall@100"] == pytest.approx(2 / 3)
    assert result["relevant_pool_counts"] == {"lamer_only": 1, "crb_only": 1, "both": 0, "neither": 1}


def test_resume_gates_compare_dual_against_both_basic_baselines_and_lamer() -> None:
    def summary(value: float) -> dict:
        return {"macro_equal_subset_weight": {"ndcg@10": value}}

    summaries = {
        "B0_BM25": summary(0.2),
        "B2_BM25_BGE_RRF": summary(0.3),
        "B3_LameR_MV": summary(0.32),
        "B5_DualSource_RRF_lam0.5": summary(0.325),
    }
    comparisons = {
        "DualSource_vs_BM25": {"lower_95": 0.02},
        "DualSource_vs_ordinary_RRF": {"lower_95": 0.01},
        "DualSource_vs_LameR_MV": {"lower_95": 0.001},
    }
    gates = resume_claim_gates(summaries, comparisons)
    assert gates["PUBLIC_BASIC_BASELINE_IMPROVEMENT"] is True
    assert gates["STRONG_BASIC_BASELINE_IMPROVEMENT"] is True
    assert gates["POINT_IMPROVEMENT_OVER_LAMER"] is True
    assert gates["STRONG_BASELINE_IMPROVEMENT"] is True
    comparisons["DualSource_vs_ordinary_RRF"]["lower_95"] = -0.001
    assert resume_claim_gates(summaries, comparisons)["PUBLIC_BASIC_BASELINE_IMPROVEMENT"] is False
