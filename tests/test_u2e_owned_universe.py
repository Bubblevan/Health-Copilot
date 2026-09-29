from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from health_ai_copilot.research.integration.actions import ActionKey
from health_ai_copilot.research.integration.counterfactual import CounterfactualRunner
from health_ai_copilot.research.integration.owned_universe.diagnostics import (
    counterfactual_contract_report,
    matched_pair_diagnostics,
    shortcut_audit,
    temporal_revision_audit,
)
from health_ai_copilot.research.integration.owned_universe.evaluator_truth import (
    evaluate_structured,
)
from health_ai_copilot.research.integration.owned_universe.realization import materialize
from health_ai_copilot.research.integration.owned_universe.schema import (
    FactLocation,
    derive_capability_requirement,
)
from health_ai_copilot.research.integration.owned_universe.source_independence import (
    source_independence_audit,
)
from health_ai_copilot.research.integration.owned_universe.splits import audit_splits, generate_plan

ROOT = Path(__file__).parents[1]
SPEC_ROOT = ROOT / "benchmarks" / "integration_owned_v1"
GLOBAL_SEED = 20260929


def _read_json(name: str):
    return json.loads((SPEC_ROOT / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _pilot():
    spec = _read_json("universe_spec.json")
    train = _read_json("train_generation_plan.json")
    dev = _read_json("dev_generation_plan.json")["views"]
    plans = (train, *dev)
    worlds = tuple(world for plan in plans for world in generate_plan(plan, run_seed=GLOBAL_SEED))
    cases = tuple(materialize(world) for world in worlds)
    return spec, plans, worlds, cases


def test_seed_and_spec_deterministically_reproduce_worlds() -> None:
    _, plans, worlds, _ = _pilot()
    first = [world.to_dict() for world in generate_plan(plans[0], run_seed=GLOBAL_SEED)]
    second = [world.to_dict() for world in generate_plan(plans[0], run_seed=GLOBAL_SEED)]
    assert first == second
    assert len(worlds) == 216


def test_subject_template_and_counterfactual_split_policies_are_enforced() -> None:
    _, plans, worlds, _ = _pilot()
    audit = audit_splits(worlds, plans)
    assert audit["status"] == "PASS"
    assert audit["subject_split_leaks"] == []
    assert audit["persona_seed_split_leaks"] == []
    assert audit["scenario_seed_split_leaks"] == []
    assert audit["counterfactual_sibling_split_leaks"] == []
    assert audit["template_family_split_leaks"] == []
    assert audit["all_subjects_total"] == 36
    assert audit["episode_counts"] == {"DEV_IID": 36, "DEV_STRUCTURAL": 36, "TRAIN": 144}


def test_all_facts_are_synthetic_machine_readable_and_runtime_separated() -> None:
    _, _, worlds, cases = _pilot()
    forbidden_runtime_fields = {
        "latent_dependency_graph", "required_capability", "required_evidence_ids",
        "scenario_family", "independent_subtask_count", "oracle_action",
        "architecture_required", "required_memory_facts",
    }
    for world, case in zip(worlds, cases, strict=True):
        runtime = case.episode.to_runtime_dict()
        assert not forbidden_runtime_fields.intersection(runtime)
        assert case.episode.source_provenance == "SYNTHETIC_RESEARCH_WORLD; NOT_CLINICAL_GUIDANCE"
        assert world.to_dict()["content_origin"] == "PROJECT_OWNED_SYNTHETIC"
        latent = world.to_latent_dict()
        surface = world.to_realization_dict()
        assert {"query", "patient_records", "external_evidence"}.isdisjoint(latent)
        assert "query" in surface and "patient_records" in surface
        assert all(fact.content_origin == "PROJECT_OWNED_SYNTHETIC" for fact in world.graph.facts)
        assert case.scenario.oracle.architecture_required == "UNKNOWN"
        truth = case.scenario.evaluator_truth_dict()
        assert truth["training_authorized"] is False
        assert truth["architecture_supervision"] == "UNRESOLVED"


def test_capability_oracle_derives_only_from_fact_locations() -> None:
    _, _, worlds, _ = _pilot()
    by_family = {family: next(world for world in worlds if world.scenario_family == family)
                 for family in {world.scenario_family for world in worlds}}
    assert derive_capability_requirement(by_family["CURRENT_ONLY"]).derived_class == "NONE"
    assert derive_capability_requirement(by_family["MEMORY_LOOKUP"]).memory_required
    assert derive_capability_requirement(by_family["EXTERNAL_LOOKUP"]).external_retrieval_required
    joined = derive_capability_requirement(by_family["MEMORY_EXTERNAL_JOIN"])
    assert joined.memory_required and joined.external_retrieval_required
    assert not derive_capability_requirement(by_family["INSUFFICIENT_EVIDENCE"]).answerability
    assert all(fact.location != FactLocation.UNAVAILABLE
               for fact in by_family["MEMORY_LOOKUP"].graph.facts
               if fact.fact_id in by_family["MEMORY_LOOKUP"].graph.required_fact_ids)


def test_partial_supervision_never_labels_architecture_or_budget() -> None:
    _, _, _, cases = _pilot()
    for case in cases:
        mask = case.scenario.supervision_mask.to_dict()
        status = case.scenario.supervision_status.to_dict()
        assert mask == {"memory_read": True, "external_retrieval": True,
                        "architecture": False, "budget_class": False}
        assert status["architecture"] == "UNRESOLVED"
        assert status["budget_class"] == "UNRESOLVED"
        assert case.scenario.oracle.architecture_required == "UNKNOWN"


def test_matched_pairs_cover_same_surface_state_changes_and_surface_invariance() -> None:
    _, _, worlds, cases = _pilot()
    report = matched_pair_diagnostics(worlds, cases)
    assert report["same_surface_different_requirement_pair_count"] >= 3
    assert report["different_surface_same_latent_graph_pair_count"] >= 3
    by_id = {case.scenario.episode_id: case for case in cases}
    for pair in report["same_surface_different_requirement_examples"]:
        left, right = [by_id[item] for item in pair["episode_ids"]]
        assert left.episode.query == right.episode.query


def test_negative_capability_activation_and_cued_current_facts() -> None:
    _, _, _, cases = _pilot()
    current = next(case for case in cases
                   if case.scenario.world.scenario_family == "CURRENT_ONLY"
                   and any(marker in case.episode.query.casefold()
                           for marker in ("last", "guideline", "evidence")))
    assert not current.scenario.oracle.memory_required
    assert not current.scenario.oracle.external_retrieval_required
    bundle = CounterfactualRunner().run(current.episode, current.evaluation, current.resources)
    for key in (ActionKey.NONE, ActionKey.MEMORY_RAG):
        arm = next(item for item in bundle.arms if item.action_key == key)
        assert arm.evaluation.outcome.task_success
        assert evaluate_structured(current, arm.execution.outcome.answer,
                                   arm.execution.observed_evidence_ids).success


def test_revision_and_publication_boundaries_are_exact() -> None:
    _, _, _, cases = _pilot()
    audit = temporal_revision_audit(cases)
    assert audit["status"] == "PASS"
    assert audit["timezone_aware_decisions"] is True
    offsets = {row["delta_seconds"] for row in audit["boundary_checks"]}
    assert {-1, 0, 1}.issubset(offsets)
    assert all(row["pass"] for group in (
        audit["boundary_checks"], audit["revision_checks"],
        audit["external_publication_checks"],
    ) for row in group)


def test_latent_requirements_match_every_valid_u1_counterfactual_arm() -> None:
    _, _, _, cases = _pilot()
    report = counterfactual_contract_report(cases)
    assert report["status"] == "PASS", report["mismatch_examples"][:5]
    assert report["architecture_supervision"] == "UNRESOLVED"
    assert report["architecture_labels_present"] == 0
    assert report["provider_calls"] == 0
    assert report["counterfactual_arm_count"] >= 216 * 4


def test_rule_based_shortcut_probes_pass_without_external_classifier() -> None:
    spec, _, worlds, cases = _pilot()
    report = shortcut_audit(worlds, cases, spec["lexical_shortcut_gate_accuracy_threshold"])
    assert report["status"] == "PASS", report["probes"]
    assert report["maximum_single_feature_accuracy"] < 0.95


def test_source_independence_allowlist_and_generated_text_scan_pass() -> None:
    spec, _, worlds, _ = _pilot()
    report = source_independence_audit(
        worlds, source_root=ROOT / "src", spec_root=SPEC_ROOT,
        forbidden_identifiers=tuple(spec["forbidden_generated_identifiers"]),
    )
    assert report["status"] == "PASS", report
    assert report["external_benchmark_data_paths_opened"] == []
    assert report["forbidden_identifier_scan"]["hits"] == []
    assert any("universe_spec.json" in path for path in report["allowlisted_input_paths"])


def test_reserved_test_ood_pools_are_manifests_only() -> None:
    reserved = _read_json("reserved_test_plan.json")
    assert reserved["materialized"] is False
    assert len(reserved["pools"]) == 6
    assert {pool["split_role"] for pool in reserved["pools"]} == {
        "IID_TEST", "OOD_PATIENT", "OOD_TASK", "OOD_TEMPORAL", "OOD_SOURCE",
        "OOD_COMPOSITION",
    }
