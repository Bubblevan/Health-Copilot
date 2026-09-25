"""Nested-CV question-only TF-IDF and frozen-BGE capability routers."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

from eval.e1_3_router_dataset import FIXED_ACTIONS, ArmOutcome, RouterCase

LEARNED_POLICIES = (
    "tfidf_direct",
    "tfidf_hierarchical",
    "tfidf_hierarchical_always_medcpt",
    "bge_hierarchical",
    "bge_hierarchical_always_medcpt",
)
FORBIDDEN_OUTPUT_KEYS = {"question", "options", "gold", "prediction", "evidence"}


class Stage2Selector:
    def __init__(self, classifier: LogisticRegression | None, fallback_action: str | None):
        self.classifier = classifier
        self.fallback_action = fallback_action
        self.fallback = classifier is None

    def predict(self, features: Any) -> tuple[np.ndarray, np.ndarray]:
        if self.classifier is None:
            actions = np.full(features.shape[0], self.fallback_action, dtype=object)
            p_bm25 = np.full(
                features.shape[0],
                1.0 if self.fallback_action == "rag_bm25" else 0.0,
                dtype=np.float64,
            )
            return actions, p_bm25
        probabilities = self.classifier.predict_proba(features)
        classes = list(self.classifier.classes_)
        bm25_index = classes.index("rag_bm25")
        actions = self.classifier.classes_[np.argmax(probabilities, axis=1)]
        return actions.astype(object), probabilities[:, bm25_index]


def make_tfidf_vectorizer(protocol: dict[str, Any]) -> FeatureUnion:
    config = protocol["models"]["direct_tfidf"]["features"]
    word = config["word"]
    char = config["char_wb"]
    shared = {
        "lowercase": word["lowercase"],
        "strip_accents": word["strip_accents"],
        "sublinear_tf": config["sublinear_tf"],
        "norm": config["norm"],
        "dtype": np.float32,
    }
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer=word["analyzer"],
                    ngram_range=tuple(word["ngram_range"]),
                    min_df=word["min_df"],
                    max_features=word["max_features"],
                    **shared,
                ),
            ),
            (
                "char_wb",
                TfidfVectorizer(
                    analyzer=char["analyzer"],
                    ngram_range=tuple(char["ngram_range"]),
                    min_df=char["min_df"],
                    max_features=char["max_features"],
                    **shared,
                ),
            ),
        ],
        n_jobs=1,
    )


def fit_tfidf_partitions(
    training_questions: list[str],
    evaluation_questions: list[str],
    protocol: dict[str, Any],
) -> tuple[FeatureUnion, Any, Any]:
    """Fit vocabulary and IDF on training questions only, then transform evaluation text."""
    vectorizer = make_tfidf_vectorizer(protocol)
    training_features = vectorizer.fit_transform(training_questions)
    evaluation_features = vectorizer.transform(evaluation_questions)
    return vectorizer, training_features, evaluation_features


def outer_fold_indices(folds: np.ndarray, heldout_fold: int) -> tuple[np.ndarray, np.ndarray]:
    evaluation = np.flatnonzero(folds == heldout_fold)
    training = np.flatnonzero(folds != heldout_fold)
    if evaluation.size == 0 or training.size == 0 or np.intersect1d(training, evaluation).size:
        raise ValueError("Invalid or overlapping outer train/evaluation fold")
    return training, evaluation


def counterfactual_arm_outcome(
    arms: dict[str, dict[str, ArmOutcome]], action: str, case_id: str
) -> ArmOutcome:
    try:
        return arms[action][case_id]
    except KeyError as exc:
        raise ValueError(f"No historical result for selected action {action!r}") from exc


def make_logistic_regression(
    protocol: dict[str, Any], *, hierarchical: bool, seed: int
) -> LogisticRegression:
    config = (
        protocol["models"]["hierarchical"]["logistic_regression"]
        if hierarchical
        else protocol["models"]["direct_tfidf"]["classifier"]
    )
    return LogisticRegression(
        solver=config["solver"],
        C=config["C"],
        class_weight=config["class_weight"],
        max_iter=config["max_iter"],
        tol=config["tol"],
        random_state=seed,
    )


def fit_stage2_selector(
    features: Any,
    targets: list[str] | np.ndarray,
    protocol: dict[str, Any],
    *,
    seed: int,
) -> Stage2Selector:
    counts = Counter(str(target) for target in targets)
    if not counts:
        raise ValueError("Stage 2 requires at least one retrieval-rescue training case")
    ordered_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    majority_action = ordered_counts[0][0]
    if any(counts.get(action, 0) < 2 for action in ("rag_bm25", "rag_medcpt")):
        return Stage2Selector(None, majority_action)
    classifier = make_logistic_regression(protocol, hierarchical=True, seed=seed)
    classifier.fit(features, np.asarray(targets, dtype=object))
    return Stage2Selector(classifier, None)


def positive_class_probability(classifier: LogisticRegression, features: Any) -> np.ndarray:
    classes = list(classifier.classes_)
    if 1 not in classes:
        return np.zeros(features.shape[0], dtype=np.float64)
    return classifier.predict_proba(features)[:, classes.index(1)]


def choose_threshold(
    benefit_probabilities: np.ndarray,
    stage2_actions: np.ndarray,
    fixed_correct: dict[str, np.ndarray],
    candidate_thresholds: list[float],
) -> tuple[float, dict[str, float]]:
    """Select a threshold from inner-training OOF predictions only."""
    baseline_accuracy = float(np.mean(fixed_correct["rag_medcpt"]))
    scored: list[dict[str, float]] = []
    for threshold in candidate_thresholds:
        retrieve = benefit_probabilities >= threshold
        actions = np.full(len(retrieve), "closed_book", dtype=object)
        actions[retrieve] = stage2_actions[retrieve]
        selected_correct = np.fromiter(
            (fixed_correct[action][i] for i, action in enumerate(actions)),
            dtype=bool,
            count=len(actions),
        )
        scored.append(
            {
                "threshold": float(threshold),
                "accuracy": float(np.mean(selected_correct)),
                "retrieval_rate": float(np.mean(retrieve)),
            }
        )

    eligible = [row for row in scored if row["accuracy"] >= baseline_accuracy - 0.0025]
    if eligible:
        selected = min(eligible, key=lambda row: (row["retrieval_rate"], -row["threshold"]))
    else:
        selected = min(
            scored,
            key=lambda row: (-row["accuracy"], row["retrieval_rate"], -row["threshold"]),
        )
    return selected["threshold"], selected


def nested_threshold_for_outer_fold(
    *,
    representation: str,
    outer_train_indices: np.ndarray,
    cases: list[RouterCase],
    benefits: np.ndarray,
    stage2_targets: np.ndarray,
    fixed_correct: dict[str, np.ndarray],
    protocol: dict[str, Any],
    outer_fold: int,
    bge_embeddings: np.ndarray | None,
) -> tuple[float, dict[str, float]]:
    inner_count = int(protocol["inner_cv"]["fold_count"])
    base_seed = int(protocol["outer_cv"]["seed"]) + outer_fold * 100
    inner = StratifiedKFold(n_splits=inner_count, shuffle=True, random_state=base_seed)
    local_targets = benefits[outer_train_indices].astype(np.int8)
    probabilities = np.full(len(outer_train_indices), np.nan, dtype=np.float64)
    selector_actions = np.full(len(outer_train_indices), "rag_medcpt", dtype=object)

    for inner_train_local, inner_valid_local in inner.split(
        np.zeros((len(outer_train_indices), 1)), local_targets
    ):
        inner_train = outer_train_indices[inner_train_local]
        inner_valid = outer_train_indices[inner_valid_local]
        if representation == "tfidf":
            train_questions = [cases[index].question for index in inner_train]
            valid_questions = [cases[index].question for index in inner_valid]
            _, train_features, valid_features = fit_tfidf_partitions(
                train_questions, valid_questions, protocol
            )
        else:
            if bge_embeddings is None:
                raise ValueError("BGE embeddings are required for the BGE router")
            train_features = bge_embeddings[inner_train]
            valid_features = bge_embeddings[inner_valid]

        benefit_model = make_logistic_regression(
            protocol, hierarchical=True, seed=base_seed
        )
        benefit_model.fit(train_features, benefits[inner_train].astype(np.int8))
        probabilities[inner_valid_local] = positive_class_probability(
            benefit_model, valid_features
        )

        rescue_train = benefits[inner_train]
        selector = fit_stage2_selector(
            train_features[rescue_train],
            stage2_targets[inner_train][rescue_train],
            protocol,
            seed=base_seed,
        )
        selector_actions[inner_valid_local], _ = selector.predict(valid_features)

    if np.isnan(probabilities).any():
        raise AssertionError("Inner CV did not produce exactly one score per outer-train case")
    inner_correct = {
        action: fixed_correct[action][outer_train_indices] for action in FIXED_ACTIONS
    }
    return choose_threshold(
        probabilities,
        selector_actions,
        inner_correct,
        protocol["models"]["hierarchical"]["threshold_grid"],
    )


def validate_oof_prediction_coverage(
    rows: list[dict[str, Any]], case_ids: list[str], policies: tuple[str, ...] = LEARNED_POLICIES
) -> None:
    expected = {(case_id, policy) for case_id in case_ids for policy in policies}
    actual: set[tuple[str, str]] = set()
    for row in rows:
        forbidden = FORBIDDEN_OUTPUT_KEYS.intersection(row)
        if forbidden:
            raise ValueError(f"Question/answer content fields forbidden in OOF artifact: {sorted(forbidden)}")
        identity = (row.get("case_id"), row.get("policy"))
        if identity in actual:
            raise ValueError(f"Duplicate OOF prediction for {identity}")
        actual.add(identity)
    if actual != expected:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise ValueError(f"OOF coverage mismatch: missing={missing}, extra={extra}")


def run_oof_predictions(
    cases: list[RouterCase],
    oracle_rows: list[dict[str, Any]],
    arms: dict[str, dict[str, ArmOutcome]],
    fold_manifest: dict[str, Any],
    protocol: dict[str, Any],
    bge_embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    case_ids = [case.case_id for case in cases]
    if list(fold_manifest["case_id_to_fold"]) != case_ids:
        raise ValueError("Fold manifest case order does not match the question feature matrix")
    oracle_by_id = {row["case_id"]: row for row in oracle_rows}
    if set(oracle_by_id) != set(case_ids):
        raise ValueError("Cost oracle case IDs do not match the question feature matrix")
    if bge_embeddings.shape != (len(cases), 768):
        raise ValueError(f"Expected a 768-dimensional BGE matrix; got {bge_embeddings.shape}")

    folds = np.asarray(
        [fold_manifest["case_id_to_fold"][case_id] for case_id in case_ids], dtype=np.int8
    )
    benefits = np.asarray(
        [oracle_by_id[case_id]["retrieval_benefit"] for case_id in case_ids], dtype=bool
    )
    stage2_targets = np.asarray(
        [oracle_by_id[case_id]["stage2_target"] for case_id in case_ids], dtype=object
    )
    target_actions = np.asarray(
        [oracle_by_id[case_id]["cost_oracle_action"] for case_id in case_ids], dtype=object
    )
    fixed_correct = {
        action: np.asarray([arms[action][case_id].is_correct for case_id in case_ids], dtype=bool)
        for action in FIXED_ACTIONS
    }
    policy_predictions: dict[str, dict[int, dict[str, Any]]] = {
        policy: {} for policy in LEARNED_POLICIES
    }

    for outer_fold in range(int(protocol["outer_cv"]["fold_count"])):
        train_indices, eval_indices = outer_fold_indices(folds, outer_fold)
        if np.any(folds[train_indices] == outer_fold):
            raise AssertionError("An outer evaluation case entered its training fold")

        train_questions = [cases[index].question for index in train_indices]
        eval_questions = [cases[index].question for index in eval_indices]
        _, tfidf_train, tfidf_eval = fit_tfidf_partitions(
            train_questions, eval_questions, protocol
        )

        direct_model = make_logistic_regression(
            protocol, hierarchical=False, seed=int(protocol["outer_cv"]["seed"]) + outer_fold
        )
        direct_model.fit(tfidf_train, target_actions[train_indices])
        direct_probabilities = direct_model.predict_proba(tfidf_eval)
        direct_actions = direct_model.classes_[np.argmax(direct_probabilities, axis=1)]
        direct_retrieval_probability = np.sum(
            direct_probabilities[:, [i for i, action in enumerate(direct_model.classes_) if action != "closed_book"]],
            axis=1,
        )

        fold_representation_results: dict[str, dict[str, Any]] = {}
        for representation, train_features, eval_features in (
            ("tfidf", tfidf_train, tfidf_eval),
            ("bge", bge_embeddings[train_indices], bge_embeddings[eval_indices]),
        ):
            model_seed = int(protocol["outer_cv"]["seed"]) + outer_fold
            benefit_model = make_logistic_regression(
                protocol, hierarchical=True, seed=model_seed
            )
            benefit_model.fit(train_features, benefits[train_indices].astype(np.int8))
            retrieval_probabilities = positive_class_probability(benefit_model, eval_features)
            rescue_train = benefits[train_indices]
            selector = fit_stage2_selector(
                train_features[rescue_train],
                stage2_targets[train_indices][rescue_train],
                protocol,
                seed=model_seed,
            )
            selector_actions, stage2_bm25_probabilities = selector.predict(eval_features)
            threshold, threshold_selection = nested_threshold_for_outer_fold(
                representation=representation,
                outer_train_indices=train_indices,
                cases=cases,
                benefits=benefits,
                stage2_targets=stage2_targets,
                fixed_correct=fixed_correct,
                protocol=protocol,
                outer_fold=outer_fold,
                bge_embeddings=bge_embeddings,
            )
            retrieve = retrieval_probabilities >= threshold
            hierarchical_actions = np.full(len(eval_indices), "closed_book", dtype=object)
            hierarchical_actions[retrieve] = selector_actions[retrieve]
            medcpt_actions = np.full(len(eval_indices), "closed_book", dtype=object)
            medcpt_actions[retrieve] = "rag_medcpt"
            fold_representation_results[representation] = {
                "retrieval_probabilities": retrieval_probabilities,
                "stage2_actions": selector_actions,
                "stage2_bm25_probabilities": stage2_bm25_probabilities,
                "stage2_fallback": selector.fallback,
                "threshold": threshold,
                "threshold_selection": threshold_selection,
                "hierarchical_actions": hierarchical_actions,
                "always_medcpt_actions": medcpt_actions,
            }

        for local_index, case_index in enumerate(eval_indices):
            policy_predictions["tfidf_direct"][int(case_index)] = {
                "action": str(direct_actions[local_index]),
                "retrieval_probability": float(direct_retrieval_probability[local_index]),
                "stage2_bm25_probability": None,
                "threshold": None,
                "stage2_fallback": None,
            }
            for representation, values in fold_representation_results.items():
                base_name = "tfidf" if representation == "tfidf" else "bge"
                policy_predictions[f"{base_name}_hierarchical"][int(case_index)] = {
                    "action": str(values["hierarchical_actions"][local_index]),
                    "retrieval_probability": float(values["retrieval_probabilities"][local_index]),
                    "stage2_bm25_probability": float(values["stage2_bm25_probabilities"][local_index]),
                    "threshold": float(values["threshold"]),
                    "stage2_fallback": bool(values["stage2_fallback"]),
                }
                policy_predictions[f"{base_name}_hierarchical_always_medcpt"][int(case_index)] = {
                    "action": str(values["always_medcpt_actions"][local_index]),
                    "retrieval_probability": float(values["retrieval_probabilities"][local_index]),
                    "stage2_bm25_probability": float(values["stage2_bm25_probabilities"][local_index]),
                    "threshold": float(values["threshold"]),
                    "stage2_fallback": bool(values["stage2_fallback"]),
                }

    rows: list[dict[str, Any]] = []
    for policy in LEARNED_POLICIES:
        predictions = policy_predictions[policy]
        if set(predictions) != set(range(len(cases))):
            raise AssertionError(f"{policy} did not predict every case exactly once")
        for index, case in enumerate(cases):
            prediction = predictions[index]
            action = prediction["action"]
            historical = counterfactual_arm_outcome(arms, action, case.case_id)
            rows.append(
                {
                    "case_id": case.case_id,
                    "fold": int(folds[index]),
                    "subdataset": case.subdataset,
                    "cost_oracle_action": str(oracle_by_id[case.case_id]["cost_oracle_action"]),
                    "retrieval_benefit": bool(benefits[index]),
                    "policy": policy,
                    "retrieval_probability": prediction["retrieval_probability"],
                    "selected_action": action,
                    "stage2_bm25_probability": prediction["stage2_bm25_probability"],
                    "threshold": prediction["threshold"],
                    "stage2_fallback": prediction["stage2_fallback"],
                    "selected_arm_correct": historical.is_correct,
                    "retrieval_calls": historical.retrieval_calls,
                    "answer_input_tokens": historical.answer_input_tokens,
                    "context_characters": historical.context_characters,
                    "component_latency_proxy_ms": historical.component_latency_proxy_ms,
                }
            )
    validate_oof_prediction_coverage(rows, case_ids)
    return rows
