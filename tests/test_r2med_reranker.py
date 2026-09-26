from __future__ import annotations

import builtins
import inspect
import json
from pathlib import Path

import pytest

from eval.r2med_candidate_union import Candidate
from eval.r2med_crb_evaluator import METRICS, summarize_query_metrics
from eval.r2med_reranker import (
    AAR_ALPHAS,
    RERANK_DEPTHS,
    aar_is_eligible,
    append_score_cache,
    load_score_cache,
    pair_cache_key,
    rerank_top_k,
    select_best_dev_arm,
)


@pytest.mark.parametrize("depth", RERANK_DEPTHS)
def test_rerank_depth_is_exact_and_tail_order_is_preserved(depth: int) -> None:
    candidates = [Candidate(f"doc-{i:02d}", i + 1, None, 0.0) for i in range(50)]
    scores = {candidate.doc_id: float(50 - i) for i, candidate in enumerate(candidates)}

    result = rerank_top_k(candidates, scores, depth=depth)

    assert len(result) == len(candidates)
    assert sum(item.reranker_score is not None for item in result) == depth
    assert [item.candidate.doc_id for item in result[depth:]] == [
        candidate.doc_id for candidate in candidates[depth:]
    ]
    assert [item.candidate.doc_id for item in result[:depth]] == [
        candidate.doc_id for candidate in candidates[:depth]
    ]


def test_reranker_order_uses_scores_and_agreement_prior() -> None:
    candidates = [
        Candidate("only-lamer", 1, None, 0.0),
        Candidate("both", 2, 1, 0.0),
        *[Candidate(f"tail-{i}", i + 3, None, 0.0) for i in range(48)],
    ]
    scores = {candidate.doc_id: 0.0 for candidate in candidates}
    scores["only-lamer"] = 0.05

    plain = rerank_top_k(candidates, scores, depth=20)
    aar = rerank_top_k(candidates, scores, depth=20, agreement_alpha=0.1)

    assert plain[0].candidate.doc_id == "only-lamer"
    assert aar[0].candidate.doc_id == "both"
    assert aar[0].final_score == pytest.approx(0.1)


def test_reranking_is_qrels_blind_and_accepts_only_pair_content(monkeypatch) -> None:
    assert "qrels" not in inspect.signature(rerank_top_k).parameters
    assert "gold" not in inspect.signature(pair_cache_key).parameters
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if "qrels" in str(file).lower():
            raise AssertionError("reranking must not open qrels")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    candidates = [Candidate(f"d{i}", i + 1, None, 0.0) for i in range(20)]
    result = rerank_top_k(candidates, {candidate.doc_id: 1.0 for candidate in candidates}, depth=20)
    assert len(result) == 20


def test_aar_is_disabled_without_positive_dual_source_parent_signal() -> None:
    assert not aar_is_eligible(0.31, 0.31)
    assert not aar_is_eligible(0.30, 0.31)
    assert aar_is_eligible(0.32, 0.31)


def test_dev_selection_uses_macro_ndcg_then_frozen_tie_breaks() -> None:
    summaries = {
        "lower": {"macro_equal_subset_weight": {"ndcg@10": 0.3, "mrr@10": 0.9, "recall@10": 0.9}},
        "shorter": {"macro_equal_subset_weight": {"ndcg@10": 0.31, "mrr@10": 0.4, "recall@10": 0.5}},
        "longer": {"macro_equal_subset_weight": {"ndcg@10": 0.31, "mrr@10": 0.4, "recall@10": 0.5}},
        "lambda_close": {"macro_equal_subset_weight": {"ndcg@10": 0.31, "mrr@10": 0.4, "recall@10": 0.5}},
    }
    configs = {
        "lower": {"depth": 20, "lambda": 1.0},
        "shorter": {"depth": 20, "lambda": 0.5},
        "longer": {"depth": 30, "lambda": 1.0},
        "lambda_close": {"depth": 20, "lambda": 1.0},
    }

    assert select_best_dev_arm(summaries, configs) == "lambda_close"


def test_dev_selection_treats_absent_lambda_as_neutral_for_single_source_arm() -> None:
    summary = {
        "macro_equal_subset_weight": {"ndcg@10": 0.31, "mrr@10": 0.4, "recall@10": 0.5}
    }
    summaries = {"lamer_single_source": summary, "dual_lambda_half": summary}
    configs = {
        "lamer_single_source": {"depth": 30, "lambda": None},
        "dual_lambda_half": {"depth": 30, "lambda": 0.5},
    }

    assert select_best_dev_arm(summaries, configs) == "lamer_single_source"


def test_macro_metric_weights_subsets_equally_not_queries() -> None:
    rows = []
    for query_id, subset, value in (
        ("a", "small", 1.0),
        ("b", "large", 0.0),
        ("c", "large", 0.0),
        ("d", "large", 0.0),
    ):
        rows.append({"query_id": query_id, "subset": subset, **{metric: value for metric in METRICS}})

    summary = summarize_query_metrics(rows)

    assert summary["macro_equal_subset_weight"]["ndcg@10"] == pytest.approx(0.5)


def test_dev_reranker_grid_and_model_revision_are_frozen() -> None:
    protocol_path = Path(__file__).resolve().parents[1] / "runs/rag_r2med_rerank/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    assert protocol["reranker"]["revision"] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert protocol["candidate_sources"]["fusion"]["lambda_grid"] == [0.5, 1.0, 2.0]
    assert protocol["reranker"]["rerank_depth_grid"] == [20, 30, 50]
    assert protocol["selection"]["test_allowed_only_after_gate"] is True




def test_rerank_rejects_missing_scores_or_unfrozen_depth() -> None:
    candidates = [Candidate(f"d{i}", i + 1, None, 0.0) for i in range(20)]
    with pytest.raises(KeyError):
        rerank_top_k(candidates, {}, depth=20)
    with pytest.raises(ValueError):
        rerank_top_k(candidates, {candidate.doc_id: 0.0 for candidate in candidates}, depth=40)
    with pytest.raises(ValueError):
        rerank_top_k(
            candidates,
            {candidate.doc_id: 0.0 for candidate in candidates},
            depth=20,
            agreement_alpha=0.3,
        )
    assert AAR_ALPHAS == (0.0, 0.1, 0.2)


def test_score_cache_is_append_only_and_deterministic(tmp_path) -> None:
    path = tmp_path / "scores.jsonl"
    rows = [{"cache_key": "key-1", "query_id": "q", "doc_id": "d", "reranker_score": 0.25}]

    append_score_cache(path, rows)
    append_score_cache(path, rows)

    assert load_score_cache(path) == {"key-1": 0.25}
    with pytest.raises(ValueError):
        append_score_cache(
            path,
            [{"cache_key": "key-1", "query_id": "q", "doc_id": "d", "reranker_score": 0.5}],
        )
