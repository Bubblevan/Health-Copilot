"""U2-F scale and distribution hardening contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

from health_ai_copilot.research.integration.owned_universe.diagnostics import (
    CounterfactualRunner,
    counterfactual_contract_report,
    matched_pair_diagnostics,
)
from health_ai_copilot.research.integration.owned_universe.evaluator_truth import (
    evaluate_structured,
)
from health_ai_copilot.research.integration.owned_universe.grammar import QueryTemplate
from health_ai_copilot.research.integration.owned_universe.lineage import lineage_row
from health_ai_copilot.research.integration.owned_universe.realization import materialize
from health_ai_copilot.research.integration.owned_universe.scale import (
    _balance_query_surface,
    _numeric_or_boolean_world,
    _required_key,
    _revision_world,
    build_u2f_worlds,
)
from health_ai_copilot.research.integration.owned_universe.scenarios import build_world
from health_ai_copilot.research.integration.owned_universe.schema import (
    StructuredAnswerType,
)
from health_ai_copilot.research.integration.owned_universe.splits import audit_splits
from health_ai_copilot.research.integration.owned_universe.u2f_cli import (
    _lineage_audit,
    _partial_policy_rows,
    _reserved_plan_audit,
    _runtime_leakage_audit,
)
from health_ai_copilot.research.integration.owned_universe.u2f_diagnostics import (
    cheap_classifier_audit,
    deterministic_spot_sample,
    duplicate_audit,
    single_feature_audit,
    synthetic_key_leakage_audit,
    temporal_revision_audit_u2f,
    u2f_distribution_report,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = ROOT / "benchmarks" / "integration_owned_v1"
ROLES = ("TRAIN", "DEV_IID", "DEV_STRUCTURAL")
PLAN_FILES = (
    "u2f_train_plan.json", "u2f_dev_iid_plan.json", "u2f_dev_structural_plan.json",
)


def _small_worlds():
    plans = []
    counts = ((80, 10), (32, 4), (32, 4))
    for index, (name, (episode_count, subject_count)) in enumerate(
        zip(PLAN_FILES, counts, strict=True)
    ):
        plan = json.loads((SPEC_ROOT / name).read_text(encoding="utf-8"))
        scenario_start = 900_000 + index * 10_000
        persona_start = 700_000 + index * 1_000
        plan.update({
            "episode_count": episode_count,
            "subject_count": subject_count,
            "scenario_seed_range": [scenario_start, scenario_start + episode_count // 2 - 1],
            "persona_seed_range": [persona_start, persona_start + subject_count - 1],
            "same_surface_pair_count": episode_count // 4,
        })
        plans.append(plan)
    profile = json.loads((SPEC_ROOT / "u2f_scale_profile.json").read_text(encoding="utf-8"))
    for role, plan in zip(ROLES, plans, strict=True):
        profile["targets"][role] = {
            "episode_count": plan["episode_count"],
            "subject_count": plan["subject_count"],
        }
        profile["same_surface_pair_count"][role] = plan["same_surface_pair_count"]
    worlds, timelines = build_u2f_worlds(
        plans=tuple(plans), profile=profile, run_seed=profile["global_seed"]
    )
    return tuple(plans), profile, worlds, timelines


def _revision_fixture(seed: int):
    return build_world(
        episode_id=f"REV-{seed}", split_role="TRAIN", subject_id="SUBJ-OPAQUE",
        persona_seed=41_005, scenario_seed=seed, scenario_family="MEMORY_REVISION",
        template_family="TRAIN_TEMPORAL", surface_variant_seed=seed % 2,
        counterfactual_family_id=f"CF-{seed}", budget_class="NORMAL",
        deadline_class="RELAXED", template_pool=("TRAIN_TEMPORAL",),
        diagnostic_mode="u2f",
    )


def test_u2f_builds_shared_longitudinal_histories_and_frozen_regimes():
    plans, profile, worlds, timelines = _small_worlds()
    cases = tuple(materialize(world) for world in worlds)
    report = u2f_distribution_report(worlds, cases, timelines)

    assert {world.history_regime for world in worlds} == {
        "SHORT", "MEDIUM", "LONG", "SATURATED",
    }
    assert report["timeline_progression"]["initial_state_progression_present"]
    assert report["timeline_progression"]["all_subject_snapshots_monotonic"]
    assert report["timeline_progression"]["subjects_with_strict_later_record_additions"] == 18
    assert set(report["available_patient_records_by_type"]) == set(profile["record_types"])
    for role in ROLES:
        assert set(report["available_patient_records_by_type_and_split"][role]) == set(
            profile["record_types"]
        )
        assert set(report["evidence_world_regime_distribution_by_split"][role]) == {
            "SMALL", "MEDIUM", "LARGE",
        }
    assert report["timeline_records_per_subject"] == {
        "min": 80, "median": 80, "mean": 80, "max": 80,
    }
    assert audit_splits(worlds, plans)["status"] == "PASS"


def test_u2f_revision_chains_cover_multiple_depths_and_temporal_validity():
    depths = set()
    for seed in range(1_000, 1_120):
        world = _revision_world(_revision_fixture(seed))
        depths.add(dict(world.structural_metadata)["revision_depth"])
        versions = sorted(
            (fact for fact in world.graph.facts if fact.fact_id.startswith("LF-U2F-REV-")),
            key=lambda fact: fact.validity.valid_from,
        )
        assert all(
            earlier.validity.valid_until == later.validity.valid_from
            and later.revision_of == earlier.fact_id
            for earlier, later in pairwise(versions)
        )
    assert depths == {1, 2, 3}


def test_u2f_counterfactual_pairs_and_synthetic_keys_are_not_shortcuts():
    _, _, worlds, _ = _small_worlds()
    cases = tuple(materialize(world) for world in worlds)
    matched = matched_pair_diagnostics(worlds, cases)
    duplicate = duplicate_audit(worlds)

    assert matched["same_surface_different_requirement_pair_count"] > 0
    assert matched["different_surface_same_latent_graph_pair_count"] > 0
    assert duplicate["status"] == "PASS"
    assert duplicate["exact_runtime_serialization_duplicate_rows"] == 0
    assert duplicate["latent_graph_duplicates_confined_to_sibling_group"]
    assert synthetic_key_leakage_audit(worlds)["key_label_leakage"] == "NO"


def test_u2f_temporal_revision_counterfactual_and_runtime_truth_are_separated():
    _, profile, worlds, timelines = _small_worlds()
    cases = tuple(materialize(world) for world in worlds)
    cf = counterfactual_contract_report(cases)
    temporal = temporal_revision_audit_u2f(cases)
    runtime = _runtime_leakage_audit([case.episode.to_runtime_dict() for case in cases])
    lineages = [lineage_row(world, "f" * 64, profile["generator_version"])
                for world in worlds]

    assert cf["counterfactual_arm_count"] == len(cases) * 8
    assert cf["status"] == "PASS"
    assert temporal["status"] == "PASS"
    assert temporal["external_version_selection_checks"]
    assert runtime["status"] == "PASS"
    assert _lineage_audit(lineages, "f" * 64, profile["generator_version"])["status"] == "PASS"
    assert len(timelines) == sum(profile["targets"][role]["subject_count"] for role in ROLES)


def test_u2f_insufficient_worlds_keep_wrong_or_conflicting_evidence_unanswerable():
    _, _, worlds, _ = _small_worlds()
    insufficient = [world for world in worlds
                    if world.scenario_family == "INSUFFICIENT_EVIDENCE"]
    cases = [materialize(world) for world in insufficient]

    assert {dict(case.scenario.world.structural_metadata)["insufficient_subtype"]
            for case in cases} == {
        "FUTURE_ONLY", "CONFLICT_UNRESOLVED", "MISSING_SET_MEMBER",
        "WRONG_SOURCE_FAMILY_ONLY", "STALE_PERSONAL_STATE",
    }
    assert all(not case.scenario.oracle.answerability for case in cases)
    assert all(not case.scenario.oracle.memory_required
               and not case.scenario.oracle.external_retrieval_required for case in cases)


def test_u2f_query_shell_is_common_and_matched_surfaces_remain_equal():
    _, _, worlds, _ = _small_worlds()
    assert all("  " not in world.query for world in worlds)
    assert all("include the requested synthetic item, indexed by its key." in world.query.casefold()
               for world in worlds)
    assert all(
        "return the requested synthetic result." in world.query.casefold()
        or "provide the requested synthetic result." in world.query.casefold()
        for world in worlds
    )
    assert all("guideline-related value" not in world.query.casefold() for world in worlds)
    same_surface_groups = {}
    for world in worlds:
        if dict(world.structural_metadata).get("same_surface_stress_group"):
            same_surface_groups.setdefault(world.counterfactual_family_id, []).append(world)
    assert same_surface_groups
    assert all(len({world.query for world in group}) == 1
               for group in same_surface_groups.values())


def test_u2f_templates_reject_missing_answer_slots_and_revision_key_tracks_truth():
    template = QueryTemplate("TEST", "01", "The answer is {value} for {key}.")
    try:
        template.render("SYNKEY-00000000")
    except ValueError as exc:
        assert "requires an answer value" in str(exc)
    else:
        raise AssertionError("value-bearing query templates must reject an omitted answer")

    world = _revision_fixture(1_500_001)
    required_records = [
        record for record in world.patient_records
        if set(world.graph.required_fact_ids).intersection(record.latent_fact_ids)
    ]
    assert required_records
    assert _required_key(world) == required_records[0].retrieval_terms[0]


def test_u2f_numeric_surface_variant_preserves_tool_partition_contract():
    operation_worlds = {}
    for seed in range(1_001, 20_001, 2):
        world = build_world(
            episode_id=f"NUM-{seed}", split_role="TRAIN", subject_id="SUBJ-NUMERIC",
            persona_seed=41_005, scenario_seed=seed, scenario_family="CURRENT_ONLY",
            template_family="TRAIN_NUMERIC", surface_variant_seed=1,
            counterfactual_family_id=f"CF-NUM-{seed}", budget_class="NORMAL",
            deadline_class="RELAXED", template_pool=("TRAIN_NUMERIC",),
            diagnostic_mode="u2f",
        )
        world = _numeric_or_boolean_world(world)
        operation = dict(world.structural_metadata).get("numeric_operation")
        if operation:
            operation_worlds.setdefault(
                operation, replace(world, query=_balance_query_surface(world))
            )
        if len(operation_worlds) == 7:
            break

    assert set(operation_worlds) == {
        "sum", "difference", "count", "minimum", "maximum", "trend", "greater_than",
    }
    for world in operation_worlds.values():
        case = materialize(world)
        arms = CounterfactualRunner(epsilon=0).run(
            case.episode, case.evaluation, case.resources
        ).arms
        assert all(evaluate_structured(
            case, arm.execution.outcome.answer, arm.execution.observed_evidence_ids
        ).success for arm in arms)


def test_u2f_numeric_answers_do_not_treat_synthetic_key_digits_as_values():
    for seed in range(1_001, 20_001, 2):
        world = build_world(
            episode_id=f"NUM-KEY-{seed}", split_role="TRAIN", subject_id="SUBJ-NUMERIC",
            persona_seed=41_005, scenario_seed=seed, scenario_family="CURRENT_ONLY",
            template_family="TRAIN_NUMERIC", surface_variant_seed=1,
            counterfactual_family_id=f"CF-NUM-KEY-{seed}", budget_class="NORMAL",
            deadline_class="RELAXED", template_pool=("TRAIN_NUMERIC",),
            diagnostic_mode="u2f",
        )
        world = _numeric_or_boolean_world(world)
        if (dict(world.structural_metadata).get("numeric_operation")
                and world.answer_type == StructuredAnswerType.NUMERIC):
            world = replace(world, query=_balance_query_surface(world))
            case = materialize(world)
            answer = f"{case.evaluation.gold_answer} Source identifier SYNKEY-96771728."
            assert evaluate_structured(case, answer, ()).success
            return
    raise AssertionError("numeric fixture seed not found in frozen search interval")


def test_u2f_partial_policy_supervision_keeps_unresolved_dimensions_unlabeled():
    _, _, worlds, _ = _small_worlds()
    cases = tuple(materialize(world) for world in worlds)
    rows = _partial_policy_rows(cases, eligible=True)

    assert len(rows) == len(cases)
    for row in rows:
        assert set(row["qualified_dimensions"]) == {
            "memory_read", "external_retrieval", "answerability",
        }
        assert row["unresolved_dimensions"] == {
            "architecture": "UNRESOLVED", "budget": "UNRESOLVED",
        }
        assert row["supervision_mask"]["architecture"] is False
        assert row["supervision_mask"]["budget"] is False
        assert row["training_run_started"] is False


def test_u2f_deterministic_regeneration_and_probe_provenance():
    plans, profile, worlds, timelines = _small_worlds()
    replay, replay_timelines = build_u2f_worlds(
        plans=plans, profile=profile, run_seed=profile["global_seed"]
    )
    cases = tuple(materialize(world) for world in worlds)
    second_cases = tuple(materialize(world) for world in replay)
    distribution = u2f_distribution_report(worlds, cases, timelines)
    single = single_feature_audit(cases)
    combined = cheap_classifier_audit(cases)

    assert [world.to_dict() for world in worlds] == [world.to_dict() for world in replay]
    assert timelines == replay_timelines
    assert [case.episode.to_runtime_dict() for case in cases] == [
        case.episode.to_runtime_dict() for case in second_cases
    ]
    assert all("majority_baseline_accuracy" in value
               for value in single["targets"].values())
    assert all("balanced_accuracy_of_best_single_feature" in value
               for value in single["targets"].values())
    expected_exclusions = [
        "scenario_family", "gold", "required_fact_ids", "latent_dependency_graph",
        "counterfactual_family_id",
    ]
    assert combined["train_to_dev_iid"]["excluded_features"] == expected_exclusions
    assert combined["train_to_dev_structural"]["excluded_features"] == expected_exclusions
    assert "query_only" in combined
    assert set(distribution["answer_type_distribution"]).issubset(
        {item.value for item in StructuredAnswerType}
    )


def test_u2f_reserved_pools_and_spot_packet_are_manifest_only():
    plans, _, worlds, _ = _small_worlds()
    reserved = json.loads((SPEC_ROOT / "u2f_reserved_test_plan.json").read_text(encoding="utf-8"))
    audit = _reserved_plan_audit(reserved, plans)
    cases = tuple(materialize(world) for world in worlds)
    packet = deterministic_spot_sample(
        cases, 7, {"TRAIN": 8, "DEV_IID": 4, "DEV_STRUCTURAL": 4}
    )

    assert audit["status"] == "PASS"
    assert audit["test_rows_materialized"] is False
    assert audit["rows_generated"] == 0
    assert packet["sample_sizes"] == {
        "TRAIN": 8, "DEV_IID": 4, "DEV_STRUCTURAL": 4,
    }
