"""Scale, leakage, temporal, duplication, and spot-audit diagnostics for U2-F."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import pairwise
from statistics import mean, median
from typing import Any

from ..contracts import stable_hash
from ..evidence_world import ExternalEvidenceWorld
from ..state import PatientStateStore
from .realization import MaterializedCase
from .schema import FactLocation, LatentWorld

KEY_TOKEN = re.compile(r"SYNKEY-[0-9A-F]{8}")
VALUE_TOKEN = re.compile(r"SYNVAL-[0-9A-F]{10}")
WORD_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")
KEYWORD_FEATURES = (
    "earlier", "latest", "current", "history", "record", "source", "family",
    "effective", "published", "sequence", "change", "complete", "supported",
    "compute", "calculate", "sum", "difference", "count", "true", "false",
    "value", "entries", "valid", "now", "time", "set", "trend",
)


def _count_distribution(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _summary(values: list[int | float]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    if not ordered:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {"min": ordered[0], "median": median(ordered),
            "mean": round(mean(ordered), 3), "max": ordered[-1]}


def _age_bucket(days: float) -> str:
    if days < 1:
        return "<1 day"
    if days <= 7:
        return "1-7 days"
    if days <= 30:
        return "8-30 days"
    if days <= 180:
        return "31-180 days"
    if days <= 365:
        return "181-365 days"
    return ">365 days"


def _graph_metrics(world: LatentWorld) -> tuple[int, int, int]:
    required = set(world.graph.required_fact_ids)
    parents: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    for parent, child in world.graph.fact_dependency_edges:
        children[parent].add(child)
        parents[child].add(parent)
    relevant = set(required)
    stack = list(required)
    while stack:
        node = stack.pop()
        for parent in parents[node]:
            if parent not in relevant:
                relevant.add(parent)
                stack.append(parent)
    memo: dict[str, int] = {}

    def depth(node: str) -> int:
        if node not in memo:
            memo[node] = 1 + max((depth(child) for child in children[node]
                                  if child in relevant), default=0)
        return memo[node]

    graph_depth = max((depth(node) for node in relevant), default=1)
    width = max((len(children[node] & relevant) for node in relevant), default=1)
    roots = sum(not (parents[node] & relevant) for node in relevant)
    return graph_depth, max(1, width), max(1, roots)


def u2f_distribution_report(
    worlds: tuple[LatentWorld, ...], cases: tuple[MaterializedCase, ...], timelines,
) -> dict[str, Any]:
    split_rows: dict[str, list[MaterializedCase]] = defaultdict(list)
    for case in cases:
        split_rows[case.scenario.world.split_role].append(case)

    available_types: Counter[str] = Counter()
    available_types_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    required_memory_types: Counter[str] = Counter()
    required_memory_types_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    available_families: Counter[str] = Counter()
    required_external_families: Counter[str] = Counter()
    temporal_distances: Counter[str] = Counter()
    temporal_distances_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    revision_depths_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    history_count_buckets_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    evidence_regimes_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    insufficient_subtypes: Counter[str] = Counter()
    timeline_observations: dict[str, list[tuple[Any, set[str]]]] = defaultdict(list)
    for case in cases:
        world = case.scenario.world
        role = world.split_role
        visible = case.resources.patient_state_store.snapshot(
            case.episode.subject_id or "", case.episode.decision_time
        )
        available_types.update(row.record_type.value for row in visible)
        available_types_by_split[role].update(row.record_type.value for row in visible)
        visible_count = len(visible)
        history_count_buckets_by_split[role][(
            "1-4" if visible_count <= 4 else "5-7" if visible_count <= 7 else
            "8-23" if visible_count <= 23 else "24-63" if visible_count <= 63 else "64+"
        )] += 1
        evidence_regimes_by_split[role][world.evidence_world_regime] += 1
        available_families.update(row.source_family for row in world.evidence_records)
        timeline_prefix = f"{world.timeline_generation_id}-R"
        timeline_observations[world.subject_id].append((
            case.episode.decision_time,
            {row.record_id for row in visible if row.record_id.startswith(timeline_prefix)},
        ))
        required_ids = set(world.graph.required_fact_ids)
        if case.scenario.oracle.memory_required:
            for row in world.patient_records:
                if required_ids.intersection(row.latent_fact_ids):
                    required_memory_types[row.record_type] += 1
                    required_memory_types_by_split[role][row.record_type] += 1
        if case.scenario.oracle.external_retrieval_required:
            for row in world.evidence_records:
                if required_ids.intersection(row.latent_fact_ids):
                    required_external_families[row.source_family] += 1
        component_times = {
            fact_id: component.requested_as_of or world.decision_time
            for component in world.graph.answer_components
            for fact_id in component.required_fact_ids
        }
        for fact_id in (required_ids if case.scenario.oracle.memory_required else ()):
            fact = world.graph.fact_index[fact_id]
            if fact.location != FactLocation.PATIENT_STATE:
                continue
            target_time = component_times.get(fact_id, world.decision_time)
            days = (target_time - fact.validity.valid_from).total_seconds() / 86400
            distance_bucket = _age_bucket(days)
            temporal_distances[distance_bucket] += 1
            temporal_distances_by_split[role][distance_bucket] += 1
        subtype = dict(world.structural_metadata).get("insufficient_subtype")
        if subtype:
            insufficient_subtypes[str(subtype)] += 1

    subject_episode_counts: dict[str, int] = Counter(
        case.episode.subject_id or "" for case in cases
    )
    timeline_sizes = [len(rows) for rows in timelines.values()]
    timeline_spans = [((rows[-1][1].timestamp - rows[0][1].timestamp).days if rows else 0)
                      for rows in timelines.values()]
    regime_counts = _count_distribution(world.history_regime for world in worlds)
    history_record_counts = [sum(
        row.timestamp <= world.decision_time for row in world.patient_records
    ) for world in worlds]
    evidence_sizes = [len(world.evidence_records) for world in worlds]
    dependency_depth, dependency_width, dependency_groups = [], [], []
    revision_depths, distractors = [], []
    distractor_regime_by_history: dict[str, Counter[str]] = defaultdict(Counter)
    for world in worlds:
        depth, width, groups = _graph_metrics(world)
        dependency_depth.append(depth)
        dependency_width.append(width)
        dependency_groups.append(groups)
        metadata = dict(world.structural_metadata)
        if "revision_depth" in metadata:
            revision_depths.append(metadata["revision_depth"])
            revision_depths_by_split[world.split_role][str(metadata["revision_depth"])] += 1
        distractors.append(world.distractor_count)
        distractor_regime_by_history[world.history_regime][(
            "LOW" if 2 <= world.distractor_count <= 5 else
            "MEDIUM" if 6 <= world.distractor_count <= 15 else
            "HIGH" if 16 <= world.distractor_count <= 40 else
            "EXTREME" if 41 <= world.distractor_count <= 100 else "OUT_OF_PROFILE"
        )] += 1

    requirement = Counter(case.scenario.oracle.derived_class for case in cases)
    answerability_by_history: dict[str, dict[str, int]] = {}
    for exists in (False, True):
        subset = [case for case in cases if case.episode.observable_state.history_exists is exists]
        answerability_by_history[str(exists).lower()] = {
            "episodes": len(subset),
            "memory_required": sum(case.scenario.oracle.memory_required for case in subset),
            "memory_required_rate": round(sum(case.scenario.oracle.memory_required for case in subset)
                                           / len(subset), 4) if subset else 0.0,
            "insufficient": sum(not case.scenario.oracle.answerability for case in subset),
        }
    answerability_by_external = {}
    for exists in (False, True):
        subset = [case for case in cases
                  if bool(case.scenario.world.evidence_records) is exists]
        answerability_by_external[str(exists).lower()] = {
            "episodes": len(subset),
            "external_retrieval_required": sum(
                case.scenario.oracle.external_retrieval_required for case in subset
            ),
            "required_rate": round(sum(case.scenario.oracle.external_retrieval_required
                                        for case in subset) / len(subset), 4) if subset else 0.0,
        }

    case_by_world = {case.scenario.world.world_id: case for case in cases}
    pairs: dict[str, list[LatentWorld]] = defaultdict(list)
    for world in worlds:
        pairs[world.counterfactual_family_id].append(world)
    same_surface_groups = []
    different_surface_graph_groups = []
    for group_id, members in sorted(pairs.items()):
        if len(members) != 2:
            continue
        a, b = members
        ca, cb = case_by_world[a.world_id], case_by_world[b.world_id]
        requirement_a = (ca.scenario.oracle.memory_required,
                         ca.scenario.oracle.external_retrieval_required,
                         ca.scenario.oracle.answerability)
        requirement_b = (cb.scenario.oracle.memory_required,
                         cb.scenario.oracle.external_retrieval_required,
                         cb.scenario.oracle.answerability)
        if a.query == b.query and requirement_a != requirement_b:
            same_surface_groups.append(group_id)
        if a.query != b.query and stable_hash(a.graph.to_dict()) == stable_hash(b.graph.to_dict()):
            different_surface_graph_groups.append(group_id)

    split_report = {}
    for role, rows in sorted(split_rows.items()):
        counts = Counter(case.episode.subject_id for case in rows)
        split_report[role] = {
            "episodes": len(rows), "subjects": len(counts),
            "episodes_per_subject": _summary(list(counts.values())),
            "scenario_families": _count_distribution(
                case.scenario.world.scenario_family for case in rows
            ),
            "template_families": _count_distribution(
                case.scenario.world.template_family for case in rows
            ),
            "answer_types": _count_distribution(
                case.scenario.world.answer_type.value for case in rows
            ),
            "derived_capability_requirements": _count_distribution(
                case.scenario.oracle.derived_class for case in rows
            ),
            "evidence_world_regimes": _count_distribution(
                case.scenario.world.evidence_world_regime for case in rows
            ),
            "dependency_depths": _count_distribution(
                _graph_metrics(case.scenario.world)[0] for case in rows
            ),
            "history_regimes": _count_distribution(
                case.scenario.world.history_regime for case in rows
            ),
        }
    by_answer = Counter(case.scenario.world.answer_type.value for case in cases)
    distractor_regimes = Counter(
        "LOW" if 2 <= value <= 5 else
        "MEDIUM" if 6 <= value <= 15 else
        "HIGH" if 16 <= value <= 40 else
        "EXTREME" if 41 <= value <= 100 else "OUT_OF_PROFILE"
        for value in distractors
    )
    timeline_subject_checks = []
    for subject_id, observations in sorted(timeline_observations.items()):
        ordered = sorted(observations, key=lambda item: item[0])
        adjacent = list(pairwise(ordered))
        timeline_subject_checks.append({
            "subject_id": subject_id,
            "episode_count": len(ordered),
            "timeline_snapshots_monotonic": all(
                earlier[1].issubset(later[1]) for earlier, later in adjacent
            ),
            "later_snapshot_adds_records": any(
                earlier[1] < later[1] for earlier, later in adjacent
            ),
            "visible_timeline_record_counts": [len(snapshot) for _, snapshot in ordered],
        })
    progression_types = ("PROFILE", "EVENT", "MEASUREMENT", "EXAM", "CONVERSATION")
    timeline_progression = {
        "shared_subject_timelines": len(timelines),
        "subjects_with_multiple_episodes": sum(row["episode_count"] > 1
                                                  for row in timeline_subject_checks),
        "subjects_with_strict_later_record_additions": sum(
            row["later_snapshot_adds_records"] for row in timeline_subject_checks
        ),
        "all_subject_snapshots_monotonic": all(
            row["timeline_snapshots_monotonic"] for row in timeline_subject_checks
        ),
        "sample_subjects": timeline_subject_checks[:12],
        "initial_state_progression_record_types": progression_types,
        "initial_state_progression_present": all(
            len(rows) >= 5 and tuple(record.record_type for _, record in rows[:5])
            == progression_types for rows in timelines.values()
        ),
    }
    return {
        "episode_counts_by_split": {role: len(rows) for role, rows in sorted(split_rows.items())},
        "subject_counts_by_split": {role: len({case.episode.subject_id for case in rows})
                                    for role, rows in sorted(split_rows.items())},
        "episodes_per_subject_overall": _summary(list(subject_episode_counts.values())),
        "episodes_per_subject_by_split": split_report,
        "timeline_records_per_subject": _summary(timeline_sizes),
        "timeline_span_days_per_subject": _summary(timeline_spans),
        "history_records_per_episode": _summary(history_record_counts),
        "history_record_count_buckets": _count_distribution(
            "1-4" if count <= 4 else "5-7" if count <= 7 else
            "8-23" if count <= 23 else "24-63" if count <= 63 else "64+"
            for count in history_record_counts
        ),
        "history_record_count_buckets_by_split": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(history_count_buckets_by_split.items())
        },
        "history_regime_distribution": regime_counts,
        "available_patient_records_by_type": dict(sorted(available_types.items())),
        "available_patient_records_by_type_and_split": {
            role: dict(sorted(counts.items())) for role, counts in sorted(available_types_by_split.items())
        },
        "required_memory_evidence_records_by_type": dict(sorted(required_memory_types.items())),
        "required_memory_evidence_records_by_type_and_split": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(required_memory_types_by_split.items())
        },
        "available_external_records_by_family": dict(sorted(available_families.items())),
        "required_external_evidence_by_family": dict(sorted(required_external_families.items())),
        "scenario_family_distribution": _count_distribution(world.scenario_family for world in worlds),
        "answer_type_distribution": dict(sorted(by_answer.items())),
        "derived_capability_distribution": dict(sorted(requirement.items())),
        "dependency_depth_distribution": _count_distribution(dependency_depth),
        "dependency_width_distribution": _count_distribution(dependency_width),
        "independent_dependency_groups_distribution": _count_distribution(dependency_groups),
        "distractor_count_summary": _summary(distractors),
        "distractor_regime_distribution": dict(sorted(distractor_regimes.items())),
        "distractor_regime_by_history_regime": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(distractor_regime_by_history.items())
        },
        "external_evidence_record_count_summary": _summary(evidence_sizes),
        "evidence_world_regime_distribution": _count_distribution(
            world.evidence_world_regime for world in worlds
        ),
        "evidence_world_regime_distribution_by_split": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(evidence_regimes_by_split.items())
        },
        "revision_depth_distribution": _count_distribution(revision_depths),
        "revision_depth_distribution_by_split": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(revision_depths_by_split.items())
        },
        "insufficient_evidence_subtype_distribution": dict(sorted(insufficient_subtypes.items())),
        "timeline_progression": timeline_progression,
        "memory_required_temporal_distance_distribution": dict(sorted(temporal_distances.items())),
        "memory_required_temporal_distance_distribution_by_split": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(temporal_distances_by_split.items())
        },
        "history_available_conditionals": answerability_by_history,
        "external_world_available_conditionals": answerability_by_external,
        "same_surface_different_requirement": {
            "pair_count": len(same_surface_groups),
            "participating_episode_count": 2 * len(same_surface_groups),
            "participating_episode_fraction": round(2 * len(same_surface_groups) / len(worlds), 4),
        },
        "different_surface_same_latent_dependency": {
            "pair_count": len(different_surface_graph_groups),
            "participating_episode_count": 2 * len(different_surface_graph_groups),
            "participating_episode_fraction": round(2 * len(different_surface_graph_groups) / len(worlds), 4),
        },
        "same_surface_group_ids": sorted(same_surface_groups),
        "different_surface_same_graph_group_ids": sorted(different_surface_graph_groups),
    }


def _runtime_features(case: MaterializedCase, *, context: bool) -> set[str]:
    query = unicodedata.normalize("NFKC", case.episode.query).casefold()
    query = KEY_TOKEN.sub(" synthetickey ", query)
    query = VALUE_TOKEN.sub(" syntheticvalue ", query)
    tokens = WORD_TOKEN.findall(query)
    features = {f"qtoken:{token}" for token in tokens if token not in {"synthetickey", "syntheticvalue"}}
    features.add(f"qwords:{min(40, len(tokens)) // 2 * 2}")
    features.add(f"qchars:{min(240, len(query)) // 20 * 20}")
    for keyword in KEYWORD_FEATURES:
        if keyword in tokens:
            features.add(f"qkeyword:{keyword}")
    if not context:
        return features
    state = case.episode.observable_state
    features.add(f"history_exists:{state.history_exists}")
    features.add(f"history_length:{state.history_length_bucket}")
    features.add(f"history_span:{state.history_time_span or 'none'}")
    for record_type in state.available_personal_state_types:
        features.add(f"patient_type:{record_type}")
    for family in state.available_external_source_families:
        features.add(f"source_family:{family}")
    for tool in state.available_tool_ids:
        features.add(f"tool:{tool}")
    features.update((f"budget:{state.budget_class}", f"deadline:{state.deadline_class}"))
    return features


def _binary_metrics(labels: list[bool], predictions: list[bool]) -> dict[str, float]:
    if not labels:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0}
    accuracy = sum(a == b for a, b in zip(labels, predictions, strict=True)) / len(labels)
    recalls, f1s = [], []
    for positive in (False, True):
        tp = sum(a is positive and b is positive for a, b in zip(labels, predictions, strict=True))
        fn = sum(a is positive and b is not positive for a, b in zip(labels, predictions, strict=True))
        fp = sum(a is not positive and b is positive for a, b in zip(labels, predictions, strict=True))
        support = tp + fn
        if support:
            recalls.append(tp / support)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"accuracy": round(accuracy, 4),
            "balanced_accuracy": round(mean(recalls), 4) if recalls else 0.0,
            "macro_f1": round(mean(f1s), 4)}


def _naive_bayes_probe(
    train_cases: list[MaterializedCase], eval_cases: list[MaterializedCase], *, context: bool,
) -> dict[str, Any]:
    target_names = ("memory_required", "external_retrieval_required")
    reports: dict[str, Any] = {}
    for target in target_names:
        train_labels = [bool(getattr(case.scenario.oracle, target)) for case in train_cases]
        eval_labels = [bool(getattr(case.scenario.oracle, target)) for case in eval_cases]
        train_features = [_runtime_features(case, context=context) for case in train_cases]
        eval_features = [_runtime_features(case, context=context) for case in eval_cases]
        universe = sorted(set().union(*train_features)) if train_features else []
        positive_total = sum(train_labels)
        negative_total = len(train_labels) - positive_total
        smoothing = 1.0
        positive_denominator = positive_total + 2 * smoothing
        negative_denominator = negative_total + 2 * smoothing
        positive_prob: dict[str, float] = {}
        negative_prob: dict[str, float] = {}
        for feature in universe:
            positive_hits = sum(feature in row for row, label in zip(train_features, train_labels, strict=True)
                                if label)
            negative_hits = sum(feature in row for row, label in zip(train_features, train_labels, strict=True)
                                if not label)
            positive_prob[feature] = (positive_hits + smoothing) / positive_denominator
            negative_prob[feature] = (negative_hits + smoothing) / negative_denominator
        predictions = []
        for row in eval_features:
            scores = {}
            for label, prior_count, probs in (
                (True, positive_total, positive_prob),
                (False, negative_total, negative_prob),
            ):
                prior = (prior_count + smoothing) / (len(train_labels) + 2 * smoothing)
                score = math.log(prior)
                for feature, probability in probs.items():
                    score += math.log(probability if feature in row else 1 - probability)
                scores[label] = score
            predictions.append(scores[True] > scores[False])
        attribution = sorted((
            (feature, round(abs(math.log(pos / neg)), 4))
            for feature, pos, neg in ((item, positive_prob[item], negative_prob[item])
                                      for item in universe)
        ), key=lambda row: (-row[1], row[0]))[:15]
        reports[target] = {
            **_binary_metrics(eval_labels, predictions),
            "train_positive_rate": round(positive_total / len(train_labels), 4),
            "eval_positive_rate": round(sum(eval_labels) / len(eval_labels), 4),
            "feature_count": len(universe),
            "top_feature_attribution_abs_log_odds": [
                {"feature": feature, "abs_log_odds": importance}
                for feature, importance in attribution
            ],
            "predicted_positive_count": sum(predictions),
        }
    maximum = max((max(report["balanced_accuracy"], report["macro_f1"])
                   for report in reports.values()), default=0.0)
    return {"status": "GENERATOR_SHORTCUT_REVIEW_REQUIRED" if maximum >= 0.90 else "PASS",
            "feature_scope": "runtime-observable cheap features only",
            "context_features_included": context,
            "excluded_features": ["scenario_family", "gold", "required_fact_ids",
                                  "latent_dependency_graph", "counterfactual_family_id"],
            "targets": reports,
            "max_balanced_accuracy_or_macro_f1": round(maximum, 4),
            "review_threshold": 0.90}


def _best_category_accuracy(values: list[Any], labels: list[bool]) -> tuple[float, float, Any]:
    groups: dict[Any, list[bool]] = defaultdict(list)
    for value, label in zip(values, labels, strict=True):
        groups[value].append(label)
    predictions = []
    for value in values:
        group = groups[value]
        majority = sum(group) >= len(group) / 2
        predictions.append(majority)
    accuracy = sum(a == b for a, b in zip(labels, predictions, strict=True)) / len(labels)
    balanced = _binary_metrics(labels, predictions)["balanced_accuracy"]
    return accuracy, balanced, "category-majority"


def _best_threshold_accuracy(values: list[int], labels: list[bool]) -> tuple[float, float, Any]:
    candidates = sorted(set(values))
    best = (0.0, 0.0, None)
    for threshold in candidates:
        for invert in (False, True):
            predictions = [(value >= threshold) ^ invert for value in values]
            accuracy = sum(a == b for a, b in zip(labels, predictions, strict=True)) / len(labels)
            balanced = _binary_metrics(labels, predictions)["balanced_accuracy"]
            if accuracy > best[0]:
                best = (accuracy, balanced, {"threshold": threshold, "inverted": invert})
    return best


def single_feature_audit(cases: tuple[MaterializedCase, ...], threshold: float = 0.80,
                         lift_threshold: float = 0.15) -> dict[str, Any]:
    targets = {}
    for target in ("memory_required", "external_retrieval_required"):
        labels = [bool(getattr(case.scenario.oracle, target)) for case in cases]
        majority = max(sum(labels), len(labels) - sum(labels)) / len(labels)
        probes: dict[str, Any] = {}

        def test(
            name: str, values: list[Any], *, numeric: bool = False,
            labels_for_target: list[bool] = labels,
            majority_baseline: float = majority,
            probes_for_target: dict[str, Any] = probes,
        ) -> None:
            if numeric:
                accuracy, balanced, rule = _best_threshold_accuracy(
                    [int(value) for value in values], labels_for_target
                )
            else:
                accuracy, balanced, rule = _best_category_accuracy(values, labels_for_target)
            probes_for_target[name] = {
                "accuracy": round(accuracy, 4),
                "balanced_accuracy": balanced,
                "lift_over_majority": round(accuracy - majority_baseline, 4),
                "rule": rule,
            }

        obs = [case.episode.observable_state for case in cases]
        test("history_exists", [item.history_exists for item in obs])
        test("history_length_bucket", [item.history_length_bucket for item in obs])
        test("history_time_span", [item.history_time_span or "none" for item in obs])
        test("budget_class", [item.budget_class for item in obs])
        test("deadline_class", [item.deadline_class for item in obs])
        test("available_record_type_set", ["|".join(item.available_personal_state_types)
                                            for item in obs])
        test("available_external_family_set", ["|".join(item.available_external_source_families)
                                                for item in obs])
        test("available_tool_id_set", ["|".join(item.available_tool_ids) for item in obs])
        test("query_word_count_threshold", [len(case.episode.query.split()) for case in cases], numeric=True)
        test("query_character_count_threshold", [len(case.episode.query) for case in cases], numeric=True)
        test("query_has_key", [bool(KEY_TOKEN.search(case.episode.query)) for case in cases])
        test("query_has_value", [bool(VALUE_TOKEN.search(case.episode.query)) for case in cases])
        for word in KEYWORD_FEATURES:
            test(f"query_contains_{word}", [word in case.episode.query.casefold().split()
                                              for case in cases])
        token_counts = Counter(token for case in cases
                               for token in WORD_TOKEN.findall(KEY_TOKEN.sub("", case.episode.query.casefold())))
        for token, _ in token_counts.most_common(40):
            test(f"query_token_{token}", [token in case.episode.query.casefold().split()
                                           for case in cases])
        targets[target] = {
            "positive_rate": round(sum(labels) / len(labels), 4),
            "majority_baseline_accuracy": round(majority, 4),
            "balanced_accuracy_of_best_single_feature": max(
                (item["balanced_accuracy"] for item in probes.values()), default=0.0
            ),
            "best_single_feature_accuracy": round(max(item["accuracy"] for item in probes.values()), 4),
            "best_single_feature_lift_over_majority": round(max(
                item["lift_over_majority"] for item in probes.values()), 4),
            "probes": probes,
        }
    maximum_accuracy = max(row["best_single_feature_accuracy"] for row in targets.values())
    maximum_lift = max(row["best_single_feature_lift_over_majority"] for row in targets.values())
    return {
        "status": "PASS" if maximum_accuracy <= threshold and maximum_lift <= lift_threshold else "FAIL",
        "target_scope": ["memory_required", "external_retrieval_required"],
        "gate_accuracy_max": threshold,
        "gate_lift_over_majority_max": lift_threshold,
        "maximum_single_feature_accuracy": maximum_accuracy,
        "maximum_single_feature_lift_over_majority": maximum_lift,
        "targets": targets,
        "features_considered": [
            "history availability/length/span", "available record/source/tool types",
            "budget/deadline", "query token/character counts", "query keywords",
            "top query token indicators",
        ],
        "interpretation": "Univariate runtime-observable leakage diagnosis; not router performance.",
    }


def cheap_classifier_audit(cases: tuple[MaterializedCase, ...]) -> dict[str, Any]:
    train = [case for case in cases if case.scenario.world.split_role == "TRAIN"]
    result: dict[str, Any] = {"status": "PASS", "train_rows": len(train),
                              "train_to_dev_iid": {}, "train_to_dev_structural": {},
                              "query_only": {"dev_iid": {}, "dev_structural": {}}}
    for split, key in (("DEV_IID", "train_to_dev_iid"),
                       ("DEV_STRUCTURAL", "train_to_dev_structural")):
        evaluation = [case for case in cases if case.scenario.world.split_role == split]
        contextual = _naive_bayes_probe(train, evaluation, context=True)
        query_only = _naive_bayes_probe(train, evaluation, context=False)
        result[key] = contextual
        result["query_only"]["dev_iid" if split == "DEV_IID" else "dev_structural"] = query_only
        if contextual["status"] != "PASS":
            result["status"] = "GENERATOR_SHORTCUT_REVIEW_REQUIRED"
    return result


def duplicate_audit(worlds: tuple[LatentWorld, ...]) -> dict[str, Any]:
    exact_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    normalized_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    exact_groups: dict[str, set[str]] = defaultdict(set)
    graph_groups: dict[str, set[str]] = defaultdict(set)
    world_by_query: dict[str, list[LatentWorld]] = defaultdict(list)
    graph_by_hash: dict[str, list[LatentWorld]] = defaultdict(list)
    for world in worlds:
        role = world.split_role
        query_hash = stable_hash(world.query)
        exact_by_split[role][query_hash] += 1
        normalized = unicodedata.normalize("NFKC", world.query).casefold()
        normalized = KEY_TOKEN.sub("<key>", normalized)
        normalized = VALUE_TOKEN.sub("<value>", normalized)
        normalized = " ".join(normalized.split())
        normalized_by_split[role][normalized] += 1
        world_by_query[query_hash].append(world)
        graph_hash = stable_hash(world.graph.to_dict())
        graph_by_hash[graph_hash].append(world)
    intentional_exact = 0
    unapproved_exact = []
    for query_hash, rows in world_by_query.items():
        if len(rows) < 2:
            continue
        groups = {row.counterfactual_family_id for row in rows}
        exact_groups[query_hash].update(groups)
        if len(groups) == 1 and all(
            dict(row.structural_metadata).get("same_surface_stress_group") == 1 for row in rows
        ):
            intentional_exact += len(rows) - 1
        else:
            unapproved_exact.append({"query_hash": query_hash, "episode_ids": [row.world_id for row in rows]})
    for digest, rows in graph_by_hash.items():
        graph_groups[digest].update(row.counterfactual_family_id for row in rows)
    cross_split_query_hits = []
    roles_by_normalized: dict[str, set[str]] = defaultdict(set)
    for role, frequencies in normalized_by_split.items():
        for query, count in frequencies.items():
            if count:
                roles_by_normalized[query].add(role)
    for normalized, roles in sorted(roles_by_normalized.items()):
        if len(roles) > 1:
            cross_split_query_hits.append({"normalized_query": normalized,
                                           "split_roles": sorted(roles)})
    runtime_hash_counts = Counter()
    for world in worlds:
        runtime_hash_counts[stable_hash({"query": world.query,
                                        "patient_records": [row.to_dict() for row in world.patient_records],
                                        "external_evidence": [row.to_dict() for row in world.evidence_records]})] += 1
    return {
        "status": "PASS" if not unapproved_exact else "FAIL",
        "exact_query_duplicate_rows": sum(max(0, count - 1)
                                           for rows in exact_by_split.values() for count in rows.values()),
        "exact_query_duplicates_allowed_inside_matched_groups": intentional_exact,
        "unapproved_exact_query_duplicate_groups": unapproved_exact,
        "normalized_query_duplicate_rows_by_split": {
            role: sum(max(0, count - 1) for count in rows.values())
            for role, rows in sorted(normalized_by_split.items())
        },
        "cross_split_normalized_query_overlap": cross_split_query_hits,
        "exact_runtime_serialization_duplicate_rows": sum(
            max(0, count - 1) for count in runtime_hash_counts.values()
        ),
        "latent_graph_duplicate_groups": sum(len(rows) > 1 for rows in graph_by_hash.values()),
        "latent_graph_duplicates_confined_to_sibling_group": all(
            len({row.counterfactual_family_id for row in rows}) == 1
            for rows in graph_by_hash.values() if len(rows) > 1
        ),
    }


def temporal_revision_audit_u2f(cases: tuple[MaterializedCase, ...]) -> dict[str, Any]:
    future_patient_checks, external_availability_checks, revision_checks = [], [], []
    external_version_checks = []
    boundary_checks = []
    for case in cases:
        world = case.scenario.world
        decision = case.episode.decision_time
        store: PatientStateStore = case.resources.patient_state_store
        visible_ids = {row.record_id for row in store.snapshot(case.episode.subject_id or "", decision)}
        future_rows = [row for row in world.patient_records if row.timestamp > decision]
        future_patient_checks.extend({
            "episode_id": case.episode.episode_id, "record_id": row.record_id,
            "timestamp": row.timestamp.isoformat(), "hidden": row.record_id not in visible_ids,
        } for row in future_rows)
        ext_world: ExternalEvidenceWorld = case.resources.external_evidence_world
        eligible = {
            row.source_id for row in ext_world.records
            if row.publication_time <= decision
            and (row.effective_time is None or row.effective_time <= decision)
            and (row.effective_until is None or decision < row.effective_until)
        }
        for row in ext_world.records:
            visible = row.source_id in eligible
            checks = []
            if row.publication_time > decision:
                checks.append(not visible)
            if row.effective_time and row.effective_time > decision:
                checks.append(not visible)
            if row.effective_until and row.effective_until <= decision:
                checks.append(not visible)
            if checks:
                external_availability_checks.append({
                    "episode_id": case.episode.episode_id,
                    "source_id": row.source_id,
                    "publication_time": row.publication_time.isoformat(),
                    "effective_time": row.effective_time.isoformat() if row.effective_time else None,
                    "effective_until": row.effective_until.isoformat() if row.effective_until else None,
                    "hidden_by_temporal_constraints": all(checks),
                })
        if world.scenario_family == "EXTERNAL_VERSIONED":
            version_rows = [row for row in world.evidence_records
                            if row.source_id.startswith("ER-U2F-EV-")]
            available_versions = [row for row in version_rows
                                  if row.publication_time <= decision
                                  and (row.effective_time is None or row.effective_time <= decision)
                                  and (row.effective_until is None or decision < row.effective_until)]
            version_fact_ids = {
                fact.fact_id for fact in world.graph.facts
                if fact.fact_id.startswith("LF-U2F-EV-")
            }
            selected_fact_ids = set(world.graph.required_fact_ids) & version_fact_ids
            latest = max(available_versions, key=lambda row: row.effective_time or row.publication_time,
                         default=None)
            expected_fact_ids = set(latest.latent_fact_ids) if latest else set()
            external_version_checks.append({
                "episode_id": case.episode.episode_id,
                "available_published_effective_versions": len(available_versions),
                "selected_fact_ids": sorted(selected_fact_ids),
                "latest_valid_published_fact_ids": sorted(expected_fact_ids),
                "latest_version_selected": bool(expected_fact_ids)
                and selected_fact_ids == expected_fact_ids,
            })
        if world.scenario_family == "MEMORY_REVISION":
            metadata = dict(world.structural_metadata)
            versions = sorted((fact for fact in world.graph.facts
                               if fact.location == FactLocation.PATIENT_STATE
                               and fact.fact_id.startswith("LF-U2F-REV-")),
                              key=lambda fact: fact.validity.valid_from)
            chain_ok = bool(versions) and all(
                first.validity.valid_until == second.validity.valid_from
                and second.revision_of == first.fact_id
                for first, second in pairwise(versions)
            )
            component_ok = all(
                all(world.graph.fact_index[fact_id].available_at(
                    component.requested_as_of or decision
                ) for fact_id in component.required_fact_ids)
                for component in world.graph.answer_components
            )
            revision_checks.append({
                "episode_id": case.episode.episode_id,
                "declared_depth": metadata.get("revision_depth", 0),
                "version_count": metadata.get("revision_chain_length", 0),
                "validity_chain_contiguous": chain_ok,
                "requested_fact_valid_at_requested_time": component_ok,
                "pass": chain_ok and component_ok and metadata.get("revision_depth", 0) >= 1,
            })
        if world.scenario_family == "TEMPORAL_BOUNDARY":
            target_ids = set(world.graph.required_fact_ids)
            target = min((row for row in world.patient_records
                          if target_ids.intersection(row.latent_fact_ids)),
                         key=lambda row: abs((decision - row.timestamp).total_seconds()),
                         default=None)
            if target is not None:
                visible = target.record_id in visible_ids
                boundary_checks.append({
                    "episode_id": case.episode.episode_id,
                    "offset_seconds": int((decision - target.timestamp).total_seconds()),
                    "visible": visible, "expected_visible": target.timestamp <= decision,
                    "pass": visible == (target.timestamp <= decision),
                })
    all_checks_pass = (
        all(row["hidden"] for row in future_patient_checks)
        and all(row["hidden_by_temporal_constraints"] for row in external_availability_checks)
        and all(row["latest_version_selected"] for row in external_version_checks)
        and all(row["pass"] for row in revision_checks + boundary_checks)
    )
    return {
        "status": "PASS" if all_checks_pass else "FAIL",
        "future_personal_records_checked": len(future_patient_checks),
        "future_personal_records_hidden": sum(row["hidden"] for row in future_patient_checks),
        "future_personal_record_failures": [row for row in future_patient_checks if not row["hidden"]],
        "publication_effectivity_checks": len(external_availability_checks),
        "publication_effectivity_failures": [row for row in external_availability_checks
                                             if not row["hidden_by_temporal_constraints"]],
        "external_version_selection_checks": external_version_checks,
        "external_version_selection_failures": [row for row in external_version_checks
                                                if not row["latest_version_selected"]],
        "revision_checks": revision_checks,
        "decision_boundary_checks": boundary_checks,
        "timezone_aware_decisions": all(case.episode.decision_time.tzinfo is not None
                                         for case in cases),
        "all_revision_depths_over_one_present": any(
            row["declared_depth"] >= 2 for row in revision_checks
        ),
        "external_temporal_ordering_examples": [row for row in external_availability_checks
                                                if row["publication_time"] != row["effective_time"]][:12],
    }


def synthetic_key_leakage_audit(worlds: tuple[LatentWorld, ...]) -> dict[str, Any]:
    patterns = Counter()
    label_hits = []
    for world in worlds:
        for query_or_term in (world.query, *(term for row in world.patient_records
                                             for term in row.retrieval_terms),
                              *(term for row in world.evidence_records
                                for term in row.retrieval_terms)):
            for key in KEY_TOKEN.findall(query_or_term):
                patterns["SYNKEY-<8 uppercase hex>"] += 1
                if any(label in key for label in (world.split_role, world.scenario_family,
                                                   world.subject_id, world.template_family)):
                    label_hits.append({"key": key, "episode_id": world.world_id})
    return {"status": "PASS" if not label_hits else "FAIL",
            "key_format_counts": dict(patterns), "key_label_hits": label_hits,
            "key_label_leakage": "NO" if not label_hits else "YES"}


def deterministic_spot_sample(cases: tuple[MaterializedCase, ...], seed: int,
                              sample_sizes: dict[str, int]) -> dict[str, Any]:
    packet: dict[str, Any] = {"audit_seed": seed, "sample_sizes": {}, "samples": []}
    case_groups: dict[str, list[MaterializedCase]] = defaultdict(list)
    for case in cases:
        case_groups[case.scenario.world.split_role].append(case)
    for role, wanted in sample_sizes.items():
        candidates = sorted(case_groups.get(role, []), key=lambda case: sha256(
            f"{seed}|{role}|{case.episode.episode_id}".encode()
        ).hexdigest())
        selected = candidates[:wanted]
        packet["sample_sizes"][role] = len(selected)
        for case in selected:
            world = case.scenario.world
            snapshot = case.resources.patient_state_store.snapshot(
                case.episode.subject_id or "", case.episode.decision_time
            )
            packet["samples"].append({
                "split_role": role,
                "episode_id": case.episode.episode_id,
                "query": case.episode.query,
                "visible_current_state_summary": {
                    "record_count": len(snapshot),
                    "record_types": dict(sorted(Counter(row.record_type.value
                                                          for row in snapshot).items())),
                    "time_span": (max(row.timestamp for row in snapshot)
                                  - min(row.timestamp for row in snapshot)).days
                    if len(snapshot) > 1 else 0,
                    "examples": [row.content for row in snapshot[:3]],
                },
                "latent_dependency_summary": {
                    "dependency_count": len(world.graph.required_fact_ids),
                    "dependency_locations": dict(sorted(Counter(
                        world.graph.fact_index[fact_id].location.value
                        for fact_id in world.graph.required_fact_ids
                    ).items())),
                    "required_fact_ids": list(world.graph.required_fact_ids),
                },
                "expected_partial_capability_requirement": {
                    "memory_read": "READ" if case.scenario.oracle.memory_required else "OFF",
                    "external_retrieval": ("STANDARD" if case.scenario.oracle.external_retrieval_required
                                           else "OFF"),
                    "answerability": "ANSWER" if case.scenario.oracle.answerability else "ABSTAIN",
                },
            })
    packet["samples"].sort(key=lambda row: (row["split_role"], row["episode_id"]))
    return packet
