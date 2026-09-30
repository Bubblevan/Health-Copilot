"""Post-freeze evaluator for U3-R's three fixed retrieval counterfactuals."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eval.u3r_rag_transfer import (
    ACTION_ORDER,
    U2F_DATASET_ROOT_SHA256,
    U2F_MANIFEST_SHA256,
    U2F_ROOT,
    U3R_RUN_ROOT,
    canonical_json_bytes,
    load_dev_runtime_episodes,
    load_runtime_corpora,
    sha256_bytes,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
FREEZE_MANIFEST = "u3r_counterfactual_manifest.json"
BRIDGE_OUTPUT = "u3r_lamer_bridges.jsonl"
RUNTIME_OUTPUT = "u3r_counterfactuals.jsonl"
REQUIRED_FAILURES = (
    "RETRIEVAL_UNNECESSARY",
    "RETRIEVAL_REQUIRED",
    "RETRIEVAL_MISS",
    "RANKING_MISS",
    "RETRIEVAL_NOT_SUFFICIENT",
    "CONTEXT_INTERFERENCE",
    "INSUFFICIENT_EVIDENCE",
)
VALUE_TOKEN = re.compile(r"SYNVAL-[0-9A-F]{10}")
NUMBER_TOKEN = re.compile(r"(?<![\w.-])-?\d+(?![\w.])")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected an object at {path.name}:{line_number}")
            rows.append(row)
    return rows


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite frozen score artifact: {path.name}")
        return
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale score temp file requires inspection: {temporary.name}")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
    temporary.replace(path)


def _verify_freeze(run_root: Path, u2f_root: Path) -> dict[str, Any]:
    manifest_path = run_root / FREEZE_MANIFEST
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != "u3r-counterfactual-freeze-v1"
        or manifest.get("dataset_root_sha256") != U2F_DATASET_ROOT_SHA256
        or manifest.get("u2f_manifest_sha256") != U2F_MANIFEST_SHA256
        or manifest.get("episode_count") != 1024
        or manifest.get("arm_count") != 3072
        or manifest.get("evaluator_truth_opened") is not False
        or manifest.get("train_outcomes_opened") is not False
        or manifest.get("reserved_test_ood_opened") is not False
        or manifest.get("reserved_test_ood_materialized") is not False
        or manifest.get("training_started") is not False
        or manifest.get("llm_cannot_select_actions") is not True
        or manifest.get("llm_cannot_write_answers") is not True
        or manifest.get("llm_cannot_access_evaluator_truth") is not True
    ):
        raise ValueError("counterfactual freeze manifest does not satisfy U3-R gold-blind gates")
    dataset_manifest = u2f_root / "manifest.json"
    if sha256_file(dataset_manifest) != U2F_MANIFEST_SHA256:
        raise ValueError("U2-F manifest SHA-256 mismatch")
    dataset = _read_json(dataset_manifest)
    if dataset.get("dataset_root_hash") != U2F_DATASET_ROOT_SHA256:
        raise ValueError("U2-F dataset root identity mismatch")
    if dataset.get("reserved_evaluation_rows_materialized") is not False:
        raise ValueError("reserved TEST/OOD rows must remain unmaterialized")
    recorded_code = manifest.get("code_sha256")
    if not isinstance(recorded_code, dict) or not recorded_code:
        raise ValueError("freeze manifest does not contain frozen implementation hashes")
    for name, expected_hash in recorded_code.items():
        if sha256_file(ROOT / name) != expected_hash:
            raise ValueError(f"frozen implementation hash mismatch: {name}")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    frozen_commit = manifest.get("code_commit")
    if not isinstance(frozen_commit, str) or not frozen_commit:
        raise ValueError("freeze manifest does not identify its execution code commit")
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen_commit, current_commit],
        cwd=ROOT,
        check=False,
    )
    if is_ancestor.returncode != 0:
        raise ValueError("frozen execution code commit is not an ancestor of current HEAD")
    for filename, manifest_key in (
        (BRIDGE_OUTPUT, "bridge_artifact_sha256"),
        (RUNTIME_OUTPUT, "counterfactual_artifact_sha256"),
    ):
        artifact = run_root / filename
        if sha256_file(artifact) != manifest.get(manifest_key):
            raise ValueError(f"frozen artifact hash mismatch: {filename}")
    return manifest


def _observed_values(answer: str, truth: dict[str, Any]) -> tuple[str, ...]:
    expected = tuple(str(item) for item in truth["answer_values"])
    answer_type = truth["answer_type"]
    if answer_type == "BOOLEAN":
        return tuple(item.upper() for item in re.findall(
            r"\b(?:TRUE|FALSE)\b", answer, flags=re.IGNORECASE
        ))
    if expected and all(item in {"UP", "DOWN", "STABLE"} for item in expected):
        return tuple(item.upper() for item in re.findall(
            r"\b(?:UP|DOWN|STABLE)\b", answer, flags=re.IGNORECASE
        ))
    values = VALUE_TOKEN.findall(answer)
    if any(item.isdigit() for item in expected):
        values.extend(NUMBER_TOKEN.findall(answer))
    return tuple(values)


def _evaluate_arm(
    *, truth: dict[str, Any], runtime_arm: dict[str, Any], visible_doc_ids: set[str]
) -> dict[str, Any]:
    answer = str(runtime_arm["answer"])
    used = set(runtime_arm["used_evidence_ids"])
    required_memory = set(truth.get("required_memory_record_ids", ()))
    required_external = set(truth.get("required_external_evidence_ids", ()))
    missing_resources = (required_memory | required_external) - used
    requirement = truth["capability_requirement_oracle"]
    answerable = bool(requirement["answerability"])
    expected = tuple(str(item) for item in truth["answer_values"])

    if not answerable:
        abstention_correct: bool | None = (
            answer == "INSUFFICIENT_EVIDENCE" and not used
        )
        semantic_success = bool(abstention_correct)
        observed: tuple[str, ...] = ()
    else:
        abstention_correct = None
        observed = _observed_values(answer, truth)
        if truth["answer_type"] == "ORDERED_SEQUENCE":
            semantic_success = len(observed) == len(expected) and observed == expected
        elif truth["answer_type"] == "EXACT_TOKEN":
            semantic_success = len(expected) == 1 and observed == expected
        else:
            semantic_success = (
                set(observed) == set(expected) and len(observed) == len(set(observed))
            )
        semantic_success = semantic_success and not missing_resources

    task_success = bool(semantic_success)
    candidate_ids = set(runtime_arm["ranked_evidence_ids"])
    provenance_grounded = used.issubset(candidate_ids) and used.issubset(visible_doc_ids)
    required_resources_observed = not missing_resources
    grounding_pass = provenance_grounded and required_resources_observed
    fact_count = len(required_external)
    fact_coverage = (
        len(required_external & used) / fact_count if fact_count else None
    )
    answer_value_coverage = (
        len(set(observed) & set(expected)) / len(set(expected)) if expected else None
    )
    return {
        "action": runtime_arm["action"],
        "task_success": task_success,
        "grounding_pass": grounding_pass,
        "provenance_grounded": provenance_grounded,
        "required_external_fact_coverage": fact_coverage,
        "answer_value_coverage": answer_value_coverage,
        "abstention_correct": abstention_correct,
        "grounded_success": task_success and grounding_pass,
        "required_external_count": fact_count,
        "required_external_observed": len(required_external & used),
        "missing_required_resource_count": len(missing_resources),
        "used_document_count": len(used),
    }


def _action_metrics(rows: list[dict[str, Any]], action: str) -> dict[str, Any]:
    selected = [row for row in rows if row["primary_u3r"]]
    arms = [row["actions"][action] for row in selected]
    if not arms:
        return {"count": 0}

    def mean_bool(key: str) -> float:
        return float(np.mean([bool(arm[key]) for arm in arms]))

    coverages = [arm["required_external_fact_coverage"] for arm in arms
                 if arm["required_external_fact_coverage"] is not None]
    abstentions = [arm["abstention_correct"] for arm in arms
                   if arm["abstention_correct"] is not None]
    return {
        "count": len(arms),
        "task_success": mean_bool("task_success"),
        "grounded_success": mean_bool("grounded_success"),
        "grounding_pass": mean_bool("grounding_pass"),
        "provenance_grounded": mean_bool("provenance_grounded"),
        "required_external_fact_coverage": float(np.mean(coverages)) if coverages else None,
        "answer_value_coverage": float(np.mean([
            arm["answer_value_coverage"] for arm in arms
            if arm["answer_value_coverage"] is not None
        ])) if any(arm["answer_value_coverage"] is not None for arm in arms) else None,
        "abstention_correctness": float(np.mean(abstentions)) if abstentions else None,
    }


def _cost_reduction(
    baseline_cost: dict[str, int], candidate_cost: dict[str, int], key: str
) -> float:
    baseline = int(baseline_cost[key])
    if baseline <= 0:
        return 0.0
    return (baseline - int(candidate_cost[key])) / baseline


def _runtime_features(episode) -> dict[str, Any]:
    state = episode.contract.observable_state
    return {
        "history_exists": state.history_exists,
        "history_length_bucket": state.history_length_bucket,
        "history_time_span": state.history_time_span or "NONE",
        "available_personal_state_types": "|".join(state.available_personal_state_types),
        "available_external_source_families": "|".join(state.available_external_source_families),
        "available_tool_ids": "|".join(state.available_tool_ids),
        "available_worker_capabilities": "|".join(state.available_worker_capabilities),
        "budget_class": state.budget_class,
        "deadline_class": state.deadline_class,
        "task_intent_metadata": "|".join(f"{key}={value}" for key, value in state.task_intent_metadata),
    }


def _cross_validation_probe(rows: list[dict[str, Any]], episodes_by_id: dict[str, Any]) -> dict[str, Any]:
    from scipy.sparse import hstack
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold

    selected = [row for row in rows if row["primary_u3r"]]
    if not selected:
        raise ValueError("cannot run the predictability audit on an empty primary slice")
    labels = [row["minimal_action_label"] for row in selected]
    groups = [episodes_by_id[row["episode_id"]].subject_id for row in selected]
    queries = [episodes_by_id[row["episode_id"]].query for row in selected]
    metadata = [_runtime_features(episodes_by_id[row["episode_id"]]) for row in selected]
    if len(set(labels)) < 2:
        return {
            "method": "single TF-IDF + LogisticRegression; optional one-hot runtime metadata",
            "grouping": "GroupKFold by subject_id",
            "target": "post-freeze minimal sufficient action; UNRESOLVED retained",
            "class_distribution": dict(sorted(Counter(labels).items())),
            "query_only": {"accuracy": None, "macro_f1": None},
            "context_metadata": {"accuracy": None, "macro_f1": None},
            "estimable": False,
            "reason": "single target class",
            "diagnostic_only": True,
            "classifier_probe_fit": False,
            "sft_opd_grpo_started": False,
        }
    unique_groups = len(set(groups))
    folds = min(5, unique_groups)
    if folds < 2:
        raise ValueError("subject-grouped predictability audit needs at least two subjects")
    splitter = GroupKFold(n_splits=folds)
    predictions = {"query_only": [None] * len(selected), "query_plus_metadata": [None] * len(selected)}
    single_class_train_folds = 0
    for train_index, test_index in splitter.split(queries, labels, groups):
        text_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True
        )
        x_train_text = text_vectorizer.fit_transform([queries[index] for index in train_index])
        x_test_text = text_vectorizer.transform([queries[index] for index in test_index])
        metadata_vectorizer = DictVectorizer(sparse=True)
        x_train_meta = metadata_vectorizer.fit_transform([metadata[index] for index in train_index])
        x_test_meta = metadata_vectorizer.transform([metadata[index] for index in test_index])
        feature_sets = {
            "query_only": (x_train_text, x_test_text),
            "query_plus_metadata": (
                hstack((x_train_text, x_train_meta), format="csr"),
                hstack((x_test_text, x_test_meta), format="csr"),
            ),
        }
        training_labels = [labels[index] for index in train_index]
        single_class_training_fold = len(set(training_labels)) < 2
        if single_class_training_fold:
            single_class_train_folds += 1
        for name, (train_matrix, test_matrix) in feature_sets.items():
            if single_class_training_fold:
                predicted = [training_labels[0]] * len(test_index)
            else:
                classifier = LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=0
                )
                classifier.fit(train_matrix, training_labels)
                predicted = classifier.predict(test_matrix)
            for index, value in zip(test_index, predicted, strict=True):
                predictions[name][int(index)] = str(value)
    result = {}
    for name, predicted in predictions.items():
        if any(item is None for item in predicted):
            raise ValueError("grouped CV did not produce an out-of-fold prediction for every row")
        result[name] = {
            "accuracy": float(accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
            "subjects": unique_groups,
            "folds": folds,
        }
    return {
        "method": "single TF-IDF + LogisticRegression; optional one-hot runtime metadata",
        "grouping": "GroupKFold by subject_id",
        "target": "post-freeze minimal sufficient action; UNRESOLVED retained",
        "class_distribution": dict(sorted(Counter(labels).items())),
        "estimable": True,
        "single_class_train_folds": single_class_train_folds,
        "query_only": result["query_only"],
        "context_metadata": result["query_plus_metadata"],
        "diagnostic_only": True,
        "classifier_probe_fit": True,
        "sft_opd_grpo_started": False,
    }


def score_u3r(*, u2f_root: Path = U2F_ROOT, run_root: Path = U3R_RUN_ROOT) -> dict[str, Any]:
    # The freeze and every runtime artifact are authenticated before any gold path is opened.
    manifest = _verify_freeze(run_root, u2f_root)
    episodes = load_dev_runtime_episodes(u2f_root)
    episodes_by_id = {item.episode_id: item for item in episodes}
    corpus_manifest = _read_json(run_root / "u3r_runtime_corpus_manifest.json")
    corpus_path = run_root / "u3r_runtime_visible_corpus.jsonl"
    if (
        corpus_manifest.get("runtime_corpus_sha256") != sha256_file(corpus_path)
        or corpus_manifest.get("evaluator_truth_opened") is not False
    ):
        raise ValueError("gold-blind corpus artifact failed pre-score validation")
    corpora = load_runtime_corpora(corpus_path)
    runtime_rows = _read_jsonl(run_root / RUNTIME_OUTPUT)
    bridge_rows = _read_jsonl(run_root / BRIDGE_OUTPUT)
    if len(runtime_rows) != 1024 or len(bridge_rows) != 1024:
        raise ValueError("frozen U3-R artifact row count must equal 1024")
    if {row["episode_id"] for row in runtime_rows} != set(episodes_by_id):
        raise ValueError("frozen runtime artifact IDs differ from DEV runtime IDs")
    if {row["episode_id"] for row in bridge_rows} != set(episodes_by_id):
        raise ValueError("bridge artifact IDs differ from DEV runtime IDs")
    runtime_by_id = {row["episode_id"]: row for row in runtime_rows}
    for episode_id, row in runtime_by_id.items():
        if set(row.get("actions", {})) != set(ACTION_ORDER):
            raise ValueError(f"counterfactual actions are incomplete for {episode_id}")
        for action in ACTION_ORDER:
            arm = row["actions"][action]
            actual = arm.get("deterministic_execution", {}).get("outcome", {})
            if (
                actual.get("answer") != arm.get("answer")
                or actual.get("provider_calls") != 0
                or arm.get("action") != action
            ):
                raise ValueError("runtime answer/action boundary is not deterministic and frozen")

    # This is the first opening of evaluator truth in the U3-R pipeline.
    truth_by_id: dict[str, dict[str, Any]] = {}
    truth_hashes: dict[str, str] = {}
    for split, folder in (("DEV_IID", "dev_iid"), ("DEV_STRUCTURAL", "dev_structural")):
        truth_path = u2f_root / folder / "evaluator_truth.jsonl"
        truth_hashes[split] = sha256_file(truth_path)
        for row in _read_jsonl(truth_path):
            if row.get("episode_id") in truth_by_id:
                raise ValueError("duplicate evaluator-truth episode ID")
            truth_by_id[row["episode_id"]] = row
    if set(truth_by_id) != set(episodes_by_id):
        raise ValueError("DEV evaluator-truth IDs do not match frozen runtime episodes")

    scored: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    for episode in episodes:
        truth = truth_by_id[episode.episode_id]
        requirement_class = truth["capability_requirement_oracle"]["derived_capability_class"]
        primary = requirement_class in {"NONE", "RAG", "INSUFFICIENT"}
        runtime = runtime_by_id[episode.episode_id]
        if not primary:
            label_rows.append({
                "episode_id": episode.episode_id,
                "split": episode.split,
                "subject_id": episode.subject_id,
                "derived_requirement_class": requirement_class,
                "primary_u3r": False,
                "minimal_action": "NOT_PRIMARY_U3R",
                "cost_oracle_action": "OFF",
            })
            continue
        visible_ids = {item.doc_id for item in corpora[episode.episode_id]}
        actions = {
            action: _evaluate_arm(
                truth=truth,
                runtime_arm=runtime["actions"][action],
                visible_doc_ids=visible_ids,
            )
            for action in ACTION_ORDER
        }
        successes = [action for action in ACTION_ORDER if actions[action]["grounded_success"]]
        minimal = successes[0] if successes else "UNRESOLVED"
        cost_action = minimal if minimal != "UNRESOLVED" else "OFF"
        scored.append({
            "episode_id": episode.episode_id,
            "split": episode.split,
            "subject_id": episode.subject_id,
            "derived_requirement_class": requirement_class,
            "primary_u3r": True,
            "minimal_action_label": minimal,
            "cost_oracle_action": cost_action,
            "actions": actions,
            "runtime_costs": runtime["actions"],
        })
        label_rows.append({
            "episode_id": episode.episode_id,
            "split": episode.split,
            "subject_id": episode.subject_id,
            "derived_requirement_class": requirement_class,
            "primary_u3r": True,
            "minimal_action": minimal,
            "cost_oracle_action": cost_action,
        })

        for action in ACTION_ORDER:
            categories: list[str] = []
            outcome = actions[action]
            off_success = actions["OFF"]["grounded_success"]
            if requirement_class == "INSUFFICIENT" and action != "OFF":
                categories.append("INSUFFICIENT_EVIDENCE")
            if action != "OFF" and off_success:
                categories.append("RETRIEVAL_UNNECESSARY")
            if requirement_class == "RAG" and not off_success:
                categories.append("RETRIEVAL_REQUIRED")
            if requirement_class == "RAG" and action != "OFF" and not outcome["grounded_success"]:
                required = set(truth.get("required_external_evidence_ids", ()))
                candidate = set(runtime["actions"][action]["candidate_union_ids"])
                final_used = set(runtime["actions"][action]["used_evidence_ids"])
                if not required.issubset(candidate):
                    categories.append("RETRIEVAL_MISS")
                elif not required.issubset(final_used):
                    categories.append("RANKING_MISS")
                else:
                    categories.append("RETRIEVAL_NOT_SUFFICIENT")
            if action != "OFF" and not outcome["grounded_success"] and off_success:
                categories.append("CONTEXT_INTERFERENCE")
            attribution_rows.append({
                "episode_id": episode.episode_id,
                "split": episode.split,
                "action": action,
                "derived_requirement_class": requirement_class,
                "grounded_success": outcome["grounded_success"],
                "categories": sorted(set(categories)),
            })

    if not scored:
        raise ValueError("U3-R primary slice is empty")
    scored_by_id = {row["episode_id"]: row for row in scored}
    all_score_rows = []
    for episode in episodes:
        all_score_rows.append(scored_by_id.get(episode.episode_id, {
            "episode_id": episode.episode_id,
            "split": episode.split,
            "subject_id": episode.subject_id,
            "derived_requirement_class": truth_by_id[episode.episode_id][
                "capability_requirement_oracle"
            ]["derived_capability_class"],
            "primary_u3r": False,
        }))

    fixed = {
        action: _action_metrics(scored, action)
        for action in ACTION_ORDER
    }
    split_metrics = {
        split: {
            action: _action_metrics([row for row in scored if row["split"] == split], action)
            for action in ACTION_ORDER
        }
        for split in ("DEV_IID", "DEV_STRUCTURAL")
    }
    class_metrics = {
        label: {
            action: _action_metrics([
                row for row in scored if row["derived_requirement_class"] == label
            ], action)
            for action in ACTION_ORDER
        }
        for label in ("NONE", "RAG", "INSUFFICIENT")
    }

    success_by_id = {
        row["episode_id"]: {
            action: bool(row["actions"][action]["grounded_success"])
            for action in ACTION_ORDER
        }
        for row in scored
    }
    selected_cost_actions = {row["episode_id"]: row["cost_oracle_action"] for row in scored}
    cost_keys = (
        "retrieval_activations", "bridge_activations", "retrieval_search_invocations",
        "retrieved_document_count", "model_calls", "input_tokens", "output_tokens",
    )

    def cost_summary(action_by_id: dict[str, str]) -> dict[str, int]:
        result = {key: 0 for key in cost_keys}
        for row in scored:
            action = action_by_id[row["episode_id"]]
            arm = row["runtime_costs"][action]
            for key in cost_keys:
                result[key] += int(arm[key])
        return result

    always_strong_cost = cost_summary({row["episode_id"]: "STRONG" for row in scored})
    oracle_cost = cost_summary(selected_cost_actions)
    action_quality = {action: fixed[action]["grounded_success"] for action in ACTION_ORDER}
    best_fixed_action = max(ACTION_ORDER, key=lambda action: action_quality[action])
    quality_oracle = float(np.mean([
        any(success_by_id[row["episode_id"]].values()) for row in scored
    ]))
    quality_headroom = quality_oracle - action_quality[best_fixed_action]
    retrieval_call_reduction = _cost_reduction(
        always_strong_cost, oracle_cost, "retrieval_search_invocations"
    )
    retrieval_activation_reduction = _cost_reduction(
        always_strong_cost, oracle_cost, "retrieval_activations"
    )
    bridge_reduction = _cost_reduction(
        always_strong_cost, oracle_cost, "bridge_activations"
    )
    cost_quality_preserved = quality_oracle >= action_quality["STRONG"]
    cost_gate = cost_quality_preserved and (
        retrieval_call_reduction >= 0.15 or bridge_reduction >= 0.20
    )

    labels_count = Counter(row["minimal_action"] for row in label_rows if row["primary_u3r"])
    subjects_by_label: dict[str, set[str]] = defaultdict(set)
    for row in label_rows:
        if row["primary_u3r"] and row["minimal_action"] in ACTION_ORDER:
            subjects_by_label[row["minimal_action"]].add(row["subject_id"])
    primary_count = len(scored)
    diversity_classes = [
        action for action in ACTION_ORDER
        if labels_count[action] / primary_count >= 0.10
        and len(subjects_by_label[action]) >= 10
    ]
    diversity_gate = len(diversity_classes) >= 2
    unique_strong_success = sum(
        success_by_id[row["episode_id"]]["STRONG"]
        and not success_by_id[row["episode_id"]]["STANDARD"]
        and not success_by_id[row["episode_id"]]["OFF"]
        for row in scored
    )

    probe = _cross_validation_probe(scored, episodes_by_id)
    failure_counts = Counter(
        category
        for row in attribution_rows
        for category in row["categories"]
    )
    failure_by_action = {
        action: dict(sorted(Counter(
            category for row in attribution_rows if row["action"] == action
            for category in row["categories"]
        ).items()))
        for action in ACTION_ORDER
    }
    failure_report = {
        "required_categories": list(REQUIRED_FAILURES),
        "category_counts": {name: failure_counts[name] for name in REQUIRED_FAILURES},
        "category_counts_by_action": {
            action: {name: failure_by_action[action].get(name, 0) for name in REQUIRED_FAILURES}
            for action in ACTION_ORDER
        },
        "episode_attributions": attribution_rows,
    }
    fixed_report = {
        "schema_version": "u3r-fixed-action-report-v1",
        "dataset_id": "health-copilot-owned-longitudinal-v1",
        "dataset_root_sha256": U2F_DATASET_ROOT_SHA256,
        "execution_freeze_sha256": sha256_file(run_root / FREEZE_MANIFEST),
        "all_dev_episodes_executed": len(runtime_rows),
        "primary_slice_count": primary_count,
        "excluded_from_primary": len(episodes) - primary_count,
        "primary_classes": ["NONE", "RAG", "INSUFFICIENT"],
        "primary_class_counts": dict(sorted(Counter(
            row["derived_requirement_class"] for row in scored
        ).items())),
        "fixed_action_metrics": fixed,
        "split_metrics": split_metrics,
        "requirement_class_metrics": class_metrics,
        "best_fixed_action": best_fixed_action,
        "quality_oracle": quality_oracle,
        "quality_headroom_vs_best_fixed": quality_headroom,
        "unique_strong_success_count": int(unique_strong_success),
        "sft_opd_grpo_started": False,
        "reserved_test_ood_opened": False,
    }
    minimal_report = {
        "schema_version": "u3r-minimal-action-labels-v1",
        "primary_count": primary_count,
        "label_counts": {name: labels_count[name] for name in (*ACTION_ORDER, "UNRESOLVED")},
        "unique_subjects_by_action": {
            name: len(subjects_by_label[name]) for name in ACTION_ORDER
        },
        "labels": label_rows,
        "unresolved_cost_action": "OFF",
        "unresolved_training_label": "UNRESOLVED",
        "training_views_materialized": False,
        "sft_opd_grpo_started": False,
    }
    frontier_report = {
        "schema_version": "u3r-cost-frontier-v1",
        "quality_first_cost_second": True,
        "cost_tier_order": list(ACTION_ORDER),
        "quality": {
            "fixed_actions_grounded_success": action_quality,
            "best_fixed_action": best_fixed_action,
            "quality_oracle": quality_oracle,
            "quality_headroom_vs_best_fixed": quality_headroom,
            "quality_preserved_vs_always_strong": cost_quality_preserved,
        },
        "always_strong_cost": always_strong_cost,
        "minimal_cost_oracle_cost": oracle_cost,
        "cost_reduction": {
            "retrieval_search_invocation_fraction": retrieval_call_reduction,
            "retrieval_activation_fraction": retrieval_activation_reduction,
            "bridge_activation_fraction": bridge_reduction,
        },
        "cost_gate_retrieval_metric": "retrieval_search_invocations",
        "cost_aware_policy_signal": cost_gate,
        "action_diversity": {
            "classes_meeting_each_10_percent_and_10_subjects": diversity_classes,
            "subjects_by_action": {
                name: len(subjects_by_label[name]) for name in ACTION_ORDER
            },
            "gate": diversity_gate,
        },
        "strong_unique_success_count": int(unique_strong_success),
        "post_training_retrieval_policy": (
            "JUSTIFIED" if cost_gate and diversity_gate else "NOT_JUSTIFIED"
        ),
    }
    scored_content = b"".join(canonical_json_bytes(row) + b"\n" for row in all_score_rows)
    outputs = {
        "u3r_scored_outcomes.jsonl": scored_content,
        "u3r_fixed_action_report.json": canonical_json_bytes(fixed_report) + b"\n",
        "u3r_minimal_action_labels.json": canonical_json_bytes(minimal_report) + b"\n",
        "u3r_cost_frontier.json": canonical_json_bytes(frontier_report) + b"\n",
        "u3r_failure_attribution.json": canonical_json_bytes(failure_report) + b"\n",
        "u3r_predictability_probe.json": canonical_json_bytes(probe) + b"\n",
    }
    for name, content in outputs.items():
        _write_immutable(run_root / name, content)
    score_manifest = {
        "schema_version": "u3r-post-freeze-scoring-manifest-v1",
        "execution_freeze_sha256": sha256_file(run_root / FREEZE_MANIFEST),
        "execution_code_commit": manifest["code_commit"],
        "evaluator_truth_opened_after_execution_freeze": True,
        "evaluator_truth_hashes": truth_hashes,
        "evaluator_truth_rows": len(truth_by_id),
        "opened_files": ["dev_iid/evaluator_truth.jsonl", "dev_structural/evaluator_truth.jsonl"],
        "train_evaluator_truth_opened": False,
        "reserved_test_ood_opened": False,
        "sft_opd_grpo_started": False,
        "classifier_probe_fit": True,
        "output_sha256": {name: sha256_bytes(content) for name, content in outputs.items()},
        "post_training_retrieval_policy": frontier_report["post_training_retrieval_policy"],
    }
    _write_immutable(
        run_root / "u3r_scoring_manifest.json",
        canonical_json_bytes(score_manifest) + b"\n",
    )
    return {
        "primary_count": primary_count,
        "primary_classes": fixed_report["primary_class_counts"],
        "fixed_action_metrics": fixed,
        "best_fixed_action": best_fixed_action,
        "quality_oracle": quality_oracle,
        "quality_headroom": quality_headroom,
        "minimal_action_counts": minimal_report["label_counts"],
        "always_strong_cost": always_strong_cost,
        "minimal_cost_oracle_cost": oracle_cost,
        "retrieval_call_reduction": retrieval_call_reduction,
        "retrieval_activation_reduction": retrieval_activation_reduction,
        "bridge_activation_reduction": bridge_reduction,
        "predictability_probe": probe,
        "action_diversity_gate": diversity_gate,
        "cost_aware_policy_signal": cost_gate,
        "post_training_retrieval_policy": frontier_report["post_training_retrieval_policy"],
        "scoring_manifest": score_manifest,
    }
