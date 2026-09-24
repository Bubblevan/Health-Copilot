import pytest

from eval.e1_2_r2med_metrics import (
    paired_macro_bootstrap,
    query_metrics,
    select_best_fixed_baseline,
    summarize_metrics,
)


def test_graded_ndcg_and_recall_use_qrels_and_rank_positions():
    metrics = query_metrics(["d2", "d1", "d3"], {"d1": 2, "d2": 1, "d3": 0})
    ideal = 3 + 1 / 1.584962500721156
    observed = 1 + 3 / 1.584962500721156
    assert metrics["ndcg@10"] == pytest.approx(observed / ideal)
    assert metrics["mrr@10"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0


def test_no_relevant_documents_return_zero_metrics_and_duplicates_are_rejected():
    assert query_metrics(["x"], {}) == {
        "ndcg@10": 0.0,
        "mrr@10": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
    }
    with pytest.raises(ValueError, match="duplicate"):
        query_metrics(["x", "x"], {"x": 1})


def test_macro_summary_weights_subsets_equally_not_by_query_count():
    rows = [
        {"subset": "small", "ndcg@10": 0.0, "mrr@10": 0.0, "recall@5": 0.0, "recall@10": 0.0},
        {"subset": "large", "ndcg@10": 1.0, "mrr@10": 1.0, "recall@5": 1.0, "recall@10": 1.0},
        {"subset": "large", "ndcg@10": 1.0, "mrr@10": 1.0, "recall@5": 1.0, "recall@10": 1.0},
    ]
    summary = summarize_metrics(rows)
    assert summary["macro_equal_subset_weight"]["ndcg@10"] == 0.5
    assert summary["query_count"] == 3


def test_bootstrap_is_paired_stratified_and_reproducible():
    candidate = {"a": {"ndcg@10": 1.0}, "b": {"ndcg@10": 0.5}, "c": {"ndcg@10": 0.8}}
    baseline = {"a": {"ndcg@10": 0.5}, "b": {"ndcg@10": 0.5}, "c": {"ndcg@10": 0.3}}
    strata = {"a": "x", "b": "x", "c": "y"}
    first = paired_macro_bootstrap(candidate, baseline, strata, resamples=1000)
    second = paired_macro_bootstrap(candidate, baseline, strata, resamples=1000)
    assert first == second
    assert first["macro_subset_mean_difference"] == pytest.approx(0.375, abs=0.03)
    assert first["mean_difference"] == pytest.approx(0.375)
    assert first["micro_query_mean_difference"] == pytest.approx(1 / 3)
    assert first["lower_95"] == pytest.approx(0.25)


def test_fixed_baseline_selection_is_dev_only_with_stable_tie_order():
    summary = {
        arm: {"macro_equal_subset_weight": {"ndcg@10": 0.5}}
        for arm in ("bm25", "bge_dense", "medcpt_dense")
    }
    assert select_best_fixed_baseline(summary) == "bm25"
