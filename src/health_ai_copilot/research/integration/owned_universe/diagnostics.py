"""Deterministic diagnostics over generated owned pilot scenarios."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from ..actions import ActionKey
from ..contracts import stable_hash
from ..counterfactual import CounterfactualRunner
from .evaluator_truth import evaluate_structured
from .realization import MaterializedCase
from .schema import LatentWorld

SHORTCUT_WORDS = ("previous", "history", "last", "guideline", "evidence", "source")


def distribution_report(worlds: tuple[LatentWorld, ...],
                        cases: tuple[MaterializedCase, ...]) -> dict[str, Any]:
    by_role: dict[str, dict[str, Any]] = {}
    for role in sorted({world.split_role for world in worlds}):
        role_worlds = tuple(world for world in worlds if world.split_role == role)
        role_cases = tuple(case for case in cases if case.scenario.split_role == role)
        family_counts = Counter(item.scenario_family for item in role_worlds)
        template_counts = Counter(item.template_family for item in role_worlds)
        requirement_counts = Counter(case.scenario.oracle.derived_class for case in role_cases)
        lengths = [len(item.query.split()) for item in role_worlds]
        history_lengths = [len(case.resources.patient_state_store.snapshot(
            case.episode.subject_id or "", case.episode.decision_time)) for case in role_cases]
        spans = [
            (case.episode.decision_time - case.resources.patient_state_store.snapshot(
                case.episode.subject_id or "", case.episode.decision_time)[0].timestamp).total_seconds() / 86400
            for case in role_cases
            if case.resources.patient_state_store.snapshot(
                case.episode.subject_id or "", case.episode.decision_time)
        ]
        by_role[role] = {
            "episode_count": len(role_worlds),
            "subject_count": len({item.subject_id for item in role_worlds}),
            "scenario_family_distribution": dict(sorted(family_counts.items())),
            "template_family_distribution": dict(sorted(template_counts.items())),
            "derived_capability_requirement_distribution": dict(sorted(requirement_counts.items())),
            "query_word_count": _numeric_summary(lengths),
            "history_length_records": _numeric_summary(history_lengths),
            "history_timespan_days": _numeric_summary(spans),
            "distractor_count": _numeric_summary([item.distractor_count for item in role_worlds]),
            "revision_count": sum(item.scenario_family == "MEMORY_REVISION" for item in role_worlds),
            "temporal_boundary_count": sum(item.scenario_family == "TEMPORAL_BOUNDARY"
                                            for item in role_worlds),
            "training_authorized": False,
        }
    return {"by_split_role": by_role,
            "overall_episode_count": len(worlds),
            "overall_subject_count": len({world.subject_id for world in worlds}),
            "scenario_family_distribution": dict(sorted(Counter(
                world.scenario_family for world in worlds).items())),
            "derived_capability_requirement_distribution": dict(sorted(Counter(
                case.scenario.oracle.derived_class for case in cases).items())),
            "architecture_supervision": "UNRESOLVED",
            "budget_class_supervision": "UNRESOLVED",
            "model_performance_metrics": "NOT_RUN"}


def _numeric_summary(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {"count": len(ordered), "min": min(ordered), "median": median, "max": max(ordered)}


def _majority_accuracy(feature_values: list[Any], labels: list[bool]) -> float:
    groups: dict[Any, list[bool]] = defaultdict(list)
    for value, label in zip(feature_values, labels, strict=True):
        groups[value].append(label)
    correct = sum(max(sum(group), len(group) - sum(group)) for group in groups.values())
    return correct / len(labels) if labels else 0.0


def _threshold_accuracy(values: list[int], labels: list[bool]) -> float:
    if not labels:
        return 0.0
    candidates = sorted(set(values))
    predictions: list[list[bool]] = [[sum(labels) >= len(labels) / 2] * len(labels)]
    for threshold in candidates:
        predictions.extend([[value >= threshold for value in values],
                            [value < threshold for value in values]])
    return max(sum(left == right for left, right in zip(row, labels, strict=True)) / len(labels)
               for row in predictions)


def shortcut_audit(worlds: tuple[LatentWorld, ...],
                   cases: tuple[MaterializedCase, ...], threshold: float) -> dict[str, Any]:
    probes: dict[str, dict[str, float]] = {}
    for capability in ("memory_read", "external_retrieval"):
        labels = [getattr(case.scenario.oracle,
                          "memory_required" if capability == "memory_read"
                          else "external_retrieval_required") for case in cases]
        accuracies: dict[str, float] = {}
        for word in SHORTCUT_WORDS:
            present = [word in world.query.casefold() for world in worlds]
            direct = sum(a == b for a, b in zip(present, labels, strict=True)) / len(labels)
            inverse = sum((not a) == b for a, b in zip(present, labels, strict=True)) / len(labels)
            accuracies[f"contains_{word}"] = max(direct, inverse)
        accuracies["query_length_threshold"] = _threshold_accuracy(
            [len(world.query.split()) for world in worlds], labels
        )
        accuracies["history_length_bucket"] = _majority_accuracy(
            [case.episode.observable_state.history_length_bucket for case in cases], labels
        )
        probes[capability] = dict(sorted(accuracies.items()))

    maximum = max((value for rows in probes.values() for value in rows.values()), default=0.0)
    return {
        "status": "PASS" if maximum < threshold else "FAIL",
        "gate_accuracy_threshold": threshold,
        "maximum_single_feature_accuracy": maximum,
        "probes": probes,
        "interpretation": "Simple feature accuracy is a leakage diagnostic, not model performance.",
        "lexical_balance": _lexical_balance(worlds, cases),
    }


def _lexical_balance(worlds: tuple[LatentWorld, ...],
                     cases: tuple[MaterializedCase, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for word in SHORTCUT_WORDS:
        rows = [(word in world.query.casefold(), case.scenario.oracle.memory_required,
                 case.scenario.oracle.external_retrieval_required)
                for world, case in zip(worlds, cases, strict=True)]
        result[word] = {
            "occurrences": sum(item[0] for item in rows),
            "memory_required_when_present": sum(item[0] and item[1] for item in rows),
            "external_retrieval_required_when_present": sum(item[0] and item[2] for item in rows),
            "memory_required_when_absent": sum(not item[0] and item[1] for item in rows),
            "external_retrieval_required_when_absent": sum(not item[0] and item[2] for item in rows),
        }
    return result


def matched_pair_diagnostics(worlds: tuple[LatentWorld, ...],
                             cases: tuple[MaterializedCase, ...]) -> dict[str, Any]:
    by_id = {case.scenario.episode_id: case for case in cases}
    groups: dict[str, list[LatentWorld]] = defaultdict(list)
    for world in worlds:
        groups[world.counterfactual_family_id].append(world)
    same_surface_different_requirement = []
    different_surface_same_graph = []
    for group_id, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        for first, second in combinations(members, 2):
            first_case, second_case = by_id[first.world_id.removeprefix("LW-")], by_id[
                second.world_id.removeprefix("LW-")]
            first_req = first_case.scenario.oracle
            second_req = second_case.scenario.oracle
            requirement_a = (first_req.memory_required, first_req.external_retrieval_required,
                             first_req.answerability)
            requirement_b = (second_req.memory_required, second_req.external_retrieval_required,
                             second_req.answerability)
            if first.query == second.query and requirement_a != requirement_b:
                same_surface_different_requirement.append({
                    "counterfactual_family_id": group_id,
                    "episode_ids": [first_case.scenario.episode_id, second_case.scenario.episode_id],
                    "requirements": [first_req.to_dict(), second_req.to_dict()],
                })
            if first.query != second.query and stable_hash(first.graph.to_dict()) == stable_hash(
                second.graph.to_dict()
            ):
                different_surface_same_graph.append({
                    "counterfactual_family_id": group_id,
                    "episode_ids": [first_case.scenario.episode_id, second_case.scenario.episode_id],
                    "template_families": [first.template_family, second.template_family],
                    "requirement": first_req.to_dict(),
                })
    return {
        "same_surface_different_requirement_pair_count": len(same_surface_different_requirement),
        "different_surface_same_latent_graph_pair_count": len(different_surface_same_graph),
        "same_surface_different_requirement_examples": same_surface_different_requirement[:12],
        "different_surface_same_graph_examples": different_surface_same_graph[:12],
        "sibling_groups": len(groups),
        "sibling_groups_with_multiple_episodes": sum(len(rows) > 1 for rows in groups.values()),
    }


def counterfactual_contract_report(cases: tuple[MaterializedCase, ...]) -> dict[str, Any]:
    runner = CounterfactualRunner(epsilon=0)
    checked = 0
    expected_failure_count = 0
    mismatches: list[dict[str, Any]] = []
    invalid_architecture_labels = 0
    for case in cases:
        bundle = runner.run(case.episode, case.evaluation, case.resources)
        oracle = case.scenario.oracle
        if oracle.architecture_required != "UNKNOWN":
            invalid_architecture_labels += 1
        for arm in bundle.arms:
            checked += 1
            action = arm.action
            memory_on = action.memory_read
            retrieval_on = action.external_retrieval.value == "STANDARD"
            structured = evaluate_structured(
                case, arm.execution.outcome.answer, arm.execution.observed_evidence_ids
            )
            expected_success = (
                structured.success if not oracle.answerability else
                (not oracle.memory_required or memory_on)
                and (not oracle.external_retrieval_required or retrieval_on)
            )
            actual_success = arm.evaluation.outcome.task_success
            if expected_success != actual_success or expected_success != structured.success:
                expected_failure_count += int(not expected_success)
                if len(mismatches) < 30:
                    mismatches.append({
                        "episode_id": case.scenario.episode_id,
                        "action_key": arm.action_key.value,
                        "expected_success_from_latent_requirements": expected_success,
                        "u1_1_counterfactual_success": actual_success,
                        "structured_evaluator_success": structured.success,
                        "failure_category": (arm.evaluation.outcome.failure_category.value
                                             if arm.evaluation.outcome.failure_category else None),
                    })
    return {
        "status": "PASS" if not mismatches and not invalid_architecture_labels else "FAIL",
        "contract_scope": "DETERMINISTIC_CONTRACT_SMOKE_ONLY; TEAM outcomes are not architecture evidence",
        "episode_count": len(cases), "counterfactual_arm_count": checked,
        "latent_to_counterfactual_mismatches": len(mismatches),
        "mismatch_examples": mismatches,
        "failed_expected_capability_arms": expected_failure_count,
        "architecture_supervision": "UNRESOLVED",
        "architecture_labels_present": invalid_architecture_labels,
        "provider_calls": 0,
    }


def temporal_revision_audit(cases: tuple[MaterializedCase, ...]) -> dict[str, Any]:
    groups: dict[str, list[MaterializedCase]] = defaultdict(list)
    for case in cases:
        groups[case.scenario.world.counterfactual_family_id].append(case)
    runner = CounterfactualRunner(epsilon=0)
    boundary_checks = []
    revision_checks = []
    version_checks = []
    for group_id, members in sorted(groups.items()):
        family = members[0].scenario.world.scenario_family
        if family == "TEMPORAL_BOUNDARY":
            for case in members:
                target_ids = set(case.scenario.world.graph.required_fact_ids)
                target_records = [row for row in case.scenario.world.patient_records
                                  if set(row.latent_fact_ids) & target_ids]
                target_id = target_records[0].record_id
                visible = target_id in {row.record_id for row in
                                        case.resources.patient_state_store.snapshot(
                                            case.episode.subject_id or "", case.episode.decision_time)}
                should_be_visible = target_records[0].timestamp <= case.episode.decision_time
                boundary_checks.append({"episode_id": case.scenario.episode_id,
                                        "delta_seconds": int((case.episode.decision_time
                                                              - target_records[0].timestamp).total_seconds()),
                                        "target_visible": visible,
                                        "expected_visible": should_be_visible,
                                        "pass": visible == should_be_visible})
        elif family == "MEMORY_REVISION":
            for case in members:
                bundle = runner.run(case.episode, case.evaluation, case.resources)
                memory_arm = next((arm for arm in bundle.arms if arm.action_key == ActionKey.MEMORY), None)
                expected = set(case.evaluation.required_memory_record_ids)
                observed = set(memory_arm.execution.observed_evidence_ids) if memory_arm else set()
                revision_checks.append({"episode_id": case.scenario.episode_id,
                                        "required_current_revision_ids": sorted(expected),
                                        "memory_read_ids": sorted(observed),
                                        "pass": expected.issubset(observed)})
        elif family == "EXTERNAL_VERSIONED":
            for case in members:
                bundle = runner.run(case.episode, case.evaluation, case.resources)
                rag_arm = next((arm for arm in bundle.arms if arm.action_key == ActionKey.RAG), None)
                expected = set(case.evaluation.required_external_evidence_ids)
                observed = set(rag_arm.execution.observed_evidence_ids) if rag_arm else set()
                version_checks.append({"episode_id": case.scenario.episode_id,
                                       "required_as_of_evidence_ids": sorted(expected),
                                       "retrieved_as_of_ids": sorted(observed),
                                       "pass": expected.issubset(observed)})
    all_checks = boundary_checks + revision_checks + version_checks
    return {"status": "PASS" if all(row["pass"] for row in all_checks) else "FAIL",
            "boundary_checks": boundary_checks,
            "revision_checks": revision_checks,
            "external_publication_checks": version_checks,
            "timezone_aware_decisions": all(
                case.episode.decision_time.tzinfo is not None for case in cases),
            "checks_passed": sum(row["pass"] for row in all_checks),
            "checks_total": len(all_checks)}
