import numpy as np
import pytest

from eval.e1_3_learned_router import (
    LEARNED_POLICIES,
    counterfactual_arm_outcome,
    fit_stage2_selector,
    fit_tfidf_partitions,
    nested_threshold_for_outer_fold,
    outer_fold_indices,
    validate_oof_prediction_coverage,
)
from eval.e1_3_router_dataset import FIXED_ACTIONS, ArmOutcome, RouterCase


def _tiny_protocol():
    return {
        "models": {
            "direct_tfidf": {
                "features": {
                    "word": {
                        "analyzer": "word",
                        "ngram_range": [1, 2],
                        "min_df": 1,
                        "max_features": 100,
                        "lowercase": True,
                        "strip_accents": None,
                    },
                    "char_wb": {
                        "analyzer": "char_wb",
                        "ngram_range": [3, 4],
                        "min_df": 1,
                        "max_features": 100,
                        "lowercase": True,
                        "strip_accents": None,
                    },
                    "sublinear_tf": True,
                    "norm": "l2",
                },
                "classifier": {
                    "solver": "lbfgs",
                    "C": 1.0,
                    "class_weight": "balanced",
                    "max_iter": 100,
                    "tol": 0.0001,
                },
            },
            "hierarchical": {
                "logistic_regression": {
                    "solver": "lbfgs",
                    "C": 1.0,
                    "class_weight": "balanced",
                    "max_iter": 100,
                    "tol": 0.0001,
                },
                "threshold_grid": [0.1, 0.5, 0.9],
            },
        },
        "outer_cv": {"seed": 20260925, "fold_count": 5},
        "inner_cv": {"fold_count": 3},
    }


def test_tfidf_fit_train_only() -> None:
    protocol = _tiny_protocol()
    vectorizer, train_matrix, eval_matrix = fit_tfidf_partitions(
        ["trainingonlytoken alpha", "trainingonlytoken beta"],
        ["evaluationexclusivetoken gamma"],
        protocol,
    )
    word_vocabulary = vectorizer.transformer_list[0][1].vocabulary_
    assert "trainingonlytoken" in word_vocabulary
    assert "evaluationexclusivetoken" not in word_vocabulary
    assert train_matrix.shape[0] == 2
    assert eval_matrix.shape[0] == 1


def test_question_only_features_are_plain_question_strings() -> None:
    vectorizer, train_matrix, eval_matrix = fit_tfidf_partitions(
        ["question-only token alpha", "question-only token beta"],
        ["question-only token gamma"],
        _tiny_protocol(),
    )
    assert train_matrix.shape[1] == eval_matrix.shape[1]
    assert all(name in {"word", "char_wb"} for name, _ in vectorizer.transformer_list)


def test_threshold_selection_train_only() -> None:
    count = 50
    cases = [
        RouterCase(f"medqa:{index}", "medqa", f"synthetic question {index} marker{index % 4}")
        for index in range(count)
    ]
    benefits = np.zeros(count, dtype=bool)
    benefits[:8] = True
    targets = np.full(count, None, dtype=object)
    targets[:4] = "rag_bm25"
    targets[4:8] = "rag_medcpt"
    closed = np.ones(count, dtype=bool)
    bm25 = np.zeros(count, dtype=bool)
    medcpt = np.zeros(count, dtype=bool)
    closed[:8] = False
    bm25[:4] = True
    medcpt[4:8] = True
    fixed_correct = {
        "closed_book": closed,
        "rag_bm25": bm25,
        "rag_medcpt": medcpt,
    }
    embeddings = np.random.default_rng(23).normal(size=(count, 768)).astype(np.float32)
    train_indices = np.arange(40)
    selected = nested_threshold_for_outer_fold(
        representation="bge",
        outer_train_indices=train_indices,
        cases=cases,
        benefits=benefits,
        stage2_targets=targets,
        fixed_correct=fixed_correct,
        protocol=_tiny_protocol(),
        outer_fold=0,
        bge_embeddings=embeddings,
    )

    changed_heldout = {key: value.copy() for key, value in fixed_correct.items()}
    for values in changed_heldout.values():
        values[40:] = ~values[40:]
    selected_after_heldout_change = nested_threshold_for_outer_fold(
        representation="bge",
        outer_train_indices=train_indices,
        cases=cases,
        benefits=benefits,
        stage2_targets=targets,
        fixed_correct=changed_heldout,
        protocol=_tiny_protocol(),
        outer_fold=0,
        bge_embeddings=embeddings,
    )
    assert selected == selected_after_heldout_change


def test_outer_case_never_in_train() -> None:
    folds = np.asarray([index % 5 for index in range(50)])
    for fold in range(5):
        train, evaluation = outer_fold_indices(folds, fold)
        assert set(train).isdisjoint(evaluation)
        assert np.all(folds[train] != fold)
        assert np.all(folds[evaluation] == fold)


def test_counterfactual_arm_lookup() -> None:
    expected = ArmOutcome("case-1", "medqa", True, 12, 30, 44.5, 0, "completed")
    arms = {"closed_book": {"case-1": expected}}
    assert counterfactual_arm_outcome(arms, "closed_book", "case-1") is expected
    with pytest.raises(ValueError, match="No historical result"):
        counterfactual_arm_outcome(arms, "rag_bm25", "case-1")


def test_stage2_small_class_uses_majority_fallback() -> None:
    selector = fit_stage2_selector(
        np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        ["rag_bm25", "rag_medcpt", "rag_medcpt"],
        _tiny_protocol(),
        seed=7,
    )
    actions, probabilities = selector.predict(np.zeros((2, 2), dtype=np.float32))
    assert selector.fallback
    assert list(actions) == ["rag_medcpt", "rag_medcpt"]
    assert np.all(probabilities == 0.0)


def test_oof_exactly_one_prediction_per_case_and_policy() -> None:
    case_ids = ["medqa:a", "medqa:b"]
    rows = [
        {"case_id": case_id, "policy": policy}
        for case_id in case_ids
        for policy in LEARNED_POLICIES
    ]
    validate_oof_prediction_coverage(rows, case_ids)
    with pytest.raises(ValueError, match="Duplicate OOF"):
        validate_oof_prediction_coverage(rows + [rows[0]], case_ids)
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_oof_prediction_coverage(rows[:-1], case_ids)
    with pytest.raises(ValueError, match="forbidden"):
        validate_oof_prediction_coverage(rows + [{**rows[0], "question": "synthetic"}], case_ids)


def test_oof_requires_all_predeclared_policies() -> None:
    rows = [{"case_id": "medqa:a", "policy": policy} for policy in LEARNED_POLICIES]
    validate_oof_prediction_coverage(rows, ["medqa:a"])
    assert FIXED_ACTIONS == ("closed_book", "rag_bm25", "rag_medcpt")
