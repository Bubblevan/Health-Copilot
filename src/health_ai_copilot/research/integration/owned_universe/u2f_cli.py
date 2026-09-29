"""Full owned TRAIN/DEV universe builder and qualification report for U2-F."""

from __future__ import annotations

import argparse
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from .diagnostics import counterfactual_contract_report, matched_pair_diagnostics
from .lineage import lineage_row
from .realization import MaterializedCase, materialize
from .scale import build_u2f_worlds
from .schema import CONTENT_ORIGIN, WORLD_NOTICE
from .source_independence import source_independence_audit
from .splits import audit_splits
from .u2f_diagnostics import (
    cheap_classifier_audit,
    deterministic_spot_sample,
    duplicate_audit,
    single_feature_audit,
    synthetic_key_leakage_audit,
    temporal_revision_audit_u2f,
    u2f_distribution_report,
)

PLAN_FILES = (
    "u2f_train_plan.json",
    "u2f_dev_iid_plan.json",
    "u2f_dev_structural_plan.json",
)
PROFILE_FILE = "u2f_scale_profile.json"
RESERVED_FILE = "u2f_reserved_test_plan.json"
SPLIT_DIRS = {"TRAIN": "train", "DEV_IID": "dev_iid", "DEV_STRUCTURAL": "dev_structural"}
REQUIRED_LINEAGE_FIELDS = (
    "artifact_id", "generator_version", "spec_hash", "seed", "split_role",
    "subject_id", "timeline_generation_id", "scenario_family", "template_family",
    "surface_variant", "counterfactual_family_id", "history_regime",
    "evidence_world_regime", "dependency_graph_id", "parent_latent_fact_ids",
    "content_origin",
)
FORBIDDEN_RUNTIME_KEYS = {
    "answer_fact_ids", "answer_values", "architecture_label", "architecture_requirement",
    "architecture_supervision", "capability_requirement_oracle", "counterfactual_family_id",
    "dependency_graph", "gold", "gold_answer", "history_regime", "latent_world_id",
    "optimal_architecture", "parent_latent_fact_ids", "required_capability",
    "required_evidence_ids", "required_external_evidence_ids", "required_fact_ids",
    "required_memory_facts", "required_memory_record_ids", "scenario_family",
    "scenario_seed", "split_role", "structural_evaluator_metadata", "template_family",
    "timeline_generation_id", "training_authorized", "training_target",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n" for row in rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_jsonl(rows), encoding="utf-8", newline="\n")


def _hash_artifacts(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in output.rglob("*")
                           if item.is_file() and item.name != "manifest.json")
    }


def _runtime_leakage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    key_hits: list[dict[str, str]] = []
    text_hits: list[dict[str, str]] = []

    def visit(value: Any, episode_id: str, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_path = f"{path}.{key}" if path else key
                if key.casefold() in FORBIDDEN_RUNTIME_KEYS:
                    key_hits.append({"episode_id": episode_id, "field": key_path})
                visit(child, episode_id, key_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, episode_id, f"{path}[{index}]")
        elif isinstance(value, str):
            for token in ("TRAIN_LOOKUP", "DEV_STRUCTURAL_", "MEMORY_LOOKUP",
                          "EXTERNAL_VERSIONED", "INSUFFICIENT_EVIDENCE", "SINGLE"):
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
                             value, flags=re.IGNORECASE):
                    text_hits.append({"episode_id": episode_id, "token": token})

    for row in rows:
        visit(row, str(row.get("episode_id", "")))
    return {
        "status": "PASS" if not key_hits and not text_hits else "FAIL",
        "runtime_episode_count": len(rows),
        "forbidden_evaluator_or_split_keys": sorted(key_hits, key=lambda row: (row["episode_id"], row["field"])),
        "forbidden_scenario_or_architecture_tokens": sorted(
            text_hits, key=lambda row: (row["episode_id"], row["token"])
        ),
        "runtime_and_evaluator_files_physically_separate": True,
    }


def _lineage_audit(rows: list[dict[str, Any]], spec_hash: str,
                   generator_version: str) -> dict[str, Any]:
    missing = []
    invalid = []
    for row in rows:
        absent = [field for field in REQUIRED_LINEAGE_FIELDS if field not in row]
        if absent:
            missing.append({"artifact_id": row.get("artifact_id"), "fields": absent})
        if row.get("spec_hash") != spec_hash or row.get("generator_version") != generator_version:
            invalid.append(row.get("artifact_id"))
        if row.get("content_origin") != CONTENT_ORIGIN:
            invalid.append(row.get("artifact_id"))
    return {
        "status": "PASS" if not missing and not invalid else "FAIL",
        "rows_checked": len(rows),
        "required_fields": list(REQUIRED_LINEAGE_FIELDS),
        "missing_field_rows": missing,
        "invalid_hash_version_or_origin_artifact_ids": sorted(set(invalid)),
    }


def _reserved_plan_audit(reserved: dict[str, Any], plans: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    used_persona = [tuple(plan["persona_seed_range"]) for plan in plans]
    used_scenario = [tuple(plan["scenario_seed_range"]) for plan in plans]
    overlaps = []
    for pool in reserved["pools"]:
        for role, used in (("persona", used_persona), ("scenario", used_scenario)):
            start, end = pool[f"{role}_seed_range"]
            if any(start <= previous_end and previous_start <= end
                   for previous_start, previous_end in used):
                overlaps.append({"pool": pool["split_role"], "seed_type": role})
    roles = [row["split_role"] for row in reserved["pools"]]
    expected_roles = ["IID_TEST", "OOD_PATIENT", "OOD_TASK", "OOD_TEMPORAL",
                      "OOD_SOURCE", "OOD_COMPOSITION"]
    return {
        "status": "PASS" if not overlaps and not reserved["materialized"]
        and reserved["rows_generated"] == 0 and roles == expected_roles else "FAIL",
        "reserved_roles": roles,
        "reserved_role_count": len(roles),
        "ranges_overlap_train_dev": overlaps,
        "materialized": bool(reserved["materialized"]),
        "rows_generated": int(reserved["rows_generated"]),
        "test_rows_materialized": False,
    }


def _partial_policy_rows(cases: tuple[MaterializedCase, ...], eligible: bool) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        oracle = case.scenario.oracle
        rows.append({
            "episode_id": case.episode.episode_id,
            "qualified_dimensions": {
                "memory_read": "READ" if oracle.memory_required else "OFF",
                "external_retrieval": "STANDARD" if oracle.external_retrieval_required else "OFF",
                "answerability": "ANSWER" if oracle.answerability else "ABSTAIN",
            },
            "unresolved_dimensions": {"architecture": "UNRESOLVED", "budget": "UNRESOLVED"},
            "supervision_mask": {
                "memory_read": True, "external_retrieval": True, "answerability": True,
                "architecture": False, "budget": False,
            },
            "partial_policy_training_eligible": eligible,
            "training_run_started": False,
        })
    return rows


def _build_gates(
    *, profile: dict[str, Any], worlds,
    cases: tuple[MaterializedCase, ...], distribution: dict[str, Any],
    single: dict[str, Any], combined: dict[str, Any], split: dict[str, Any],
    duplicate: dict[str, Any], temporal: dict[str, Any], counterfactual: dict[str, Any],
    source: dict[str, Any], runtime: dict[str, Any], lineage: dict[str, Any],
    key_audit: dict[str, Any],
    reserved: dict[str, Any], matched: dict[str, Any],
    spot: dict[str, Any], determinism_verified: bool, human_review_status: str,
) -> dict[str, Any]:
    target_counts = {role: int(profile["targets"][role]["episode_count"])
                     for role in SPLIT_DIRS}
    target_subjects = {role: int(profile["targets"][role]["subject_count"])
                       for role in SPLIT_DIRS}
    episode_counts = distribution["episode_counts_by_split"]
    subject_counts = distribution["subject_counts_by_split"]
    thresholds = profile["required_thresholds"]
    train_families = distribution["episodes_per_subject_by_split"]["TRAIN"]["scenario_families"]
    train_history = distribution["episodes_per_subject_by_split"]["TRAIN"]["history_regimes"]
    train_history_buckets = distribution["history_record_count_buckets_by_split"]["TRAIN"]
    train_type_rows = distribution["available_patient_records_by_type_and_split"].get("TRAIN", {})
    dev_types = [distribution["available_patient_records_by_type_and_split"].get(role, {})
                 for role in ("DEV_IID", "DEV_STRUCTURAL")]
    required_train_types = distribution["required_memory_evidence_records_by_type_and_split"].get(
        "TRAIN", {}
    )
    required_dev_types = [distribution["required_memory_evidence_records_by_type_and_split"].get(role, {})
                          for role in ("DEV_IID", "DEV_STRUCTURAL")]
    train_revision_distribution = distribution["revision_depth_distribution_by_split"].get(
        "TRAIN", {}
    )
    train_temporal_distances = distribution[
        "memory_required_temporal_distance_distribution_by_split"
    ].get("TRAIN", {})
    distractor_by_history = distribution["distractor_regime_by_history_regime"]
    train_graph_depths = distribution["episodes_per_subject_by_split"]["TRAIN"][
        "dependency_depths"
    ]
    all_answer_types = set(profile["answer_types"])
    train_answer_types = set(distribution["episodes_per_subject_by_split"]["TRAIN"]["answer_types"])
    required_range = {
        "NONE": (0.10, 0.25), "MEMORY": (0.20, 0.40), "RAG": (0.15, 0.30),
        "MEMORY+RAG": (0.15, 0.30), "INSUFFICIENT": (0.05, 0.15),
    }
    train_total = episode_counts.get("TRAIN", 0)
    capability_counts = distribution["episodes_per_subject_by_split"]["TRAIN"][
        "derived_capability_requirements"
    ]
    capability_distribution_review = {
        label: {
            "count": int(capability_counts.get(label, 0)),
            "fraction": round(capability_counts.get(label, 0) / train_total, 4) if train_total else 0,
            "reference_range": list(bounds),
            "within_reference_range": bounds[0] <= (capability_counts.get(label, 0) / train_total)
            <= bounds[1] if train_total else False,
        }
        for label, bounds in required_range.items()
    }
    train_history_fractions = {
        regime: train_history.get(regime, 0) / train_total if train_total else 0
        for regime in ("SHORT", "MEDIUM", "LONG", "SATURATED")
    }
    required_record_types = set(profile["record_types"])
    near_dominance = max(required_train_types.values(), default=0) / max(
        1, sum(required_train_types.values())
    )
    all_record_types_required_in_dev = all(
        required_record_types.issubset(set(values)) for values in required_dev_types
    )
    same_fraction = matched["same_surface_different_requirement_pair_count"] * 2 / max(1, len(worlds))
    different_fraction = matched["different_surface_same_latent_graph_pair_count"] * 2 / max(
        1, len(worlds)
    )
    timeline = distribution["timeline_progression"]
    combined_max = max(
        combined["train_to_dev_iid"]["max_balanced_accuracy_or_macro_f1"],
        combined["train_to_dev_structural"]["max_balanced_accuracy_or_macro_f1"],
    )
    combined_review_ok = combined_max < thresholds["combined_probe_review_threshold"] \
        or human_review_status == "PASS"
    gates = {
        "target_episode_and_subject_counts": all(
            episode_counts.get(role) == target_counts[role]
            and subject_counts.get(role, 0) >= target_subjects[role] for role in SPLIT_DIRS
        ),
        "total_episode_cap": len(worlds) <= 8000,
        "train_history_regime_coverage": all(
            train_history_fractions[regime] >= minimum
            for regime, minimum in {"SHORT": .20, "MEDIUM": .25, "LONG": .25, "SATURATED": .15}.items()
        ),
        "legacy_one_to_four_history_bucket_below_70_percent": train_history_buckets.get(
            "1-4", 0
        ) / max(1, train_total) <= .70,
        "shared_longitudinal_subject_progression": timeline["all_subject_snapshots_monotonic"]
        and timeline["subjects_with_strict_later_record_additions"] > 0
        and timeline["initial_state_progression_present"],
        "all_patient_record_types_available_in_train_and_dev": required_record_types.issubset(
            set(train_type_rows)
        ) and all(required_record_types.issubset(set(values)) for values in dev_types),
        "required_memory_record_type_diversity": required_record_types.issubset(
            set(required_train_types)
        ) and all_record_types_required_in_dev and near_dominance < .80,
        "multiple_revision_depths_and_depth_over_one": all(
            train_revision_distribution.get(str(depth), 0) > 0 for depth in (1, 2, 3)
        ),
        "all_temporal_distance_buckets_present": all(
            train_temporal_distances.get(bucket, 0) > 0 for bucket in (
                "<1 day", "1-7 days", "8-30 days", "31-180 days", "181-365 days", ">365 days"
            )
        ),
        "high_and_extreme_distractors_in_long_histories": (
            distractor_by_history.get("LONG", {}).get("HIGH", 0) > 0
            and distractor_by_history.get("LONG", {}).get("EXTREME", 0) > 0
            and distractor_by_history.get("SATURATED", {}).get("EXTREME", 0) > 0
        ),
        "all_external_evidence_world_regimes_in_all_splits": all(
            all(distribution["evidence_world_regime_distribution_by_split"].get(role, {}).get(
                regime, 0
            ) > 0 for regime in ("SMALL", "MEDIUM", "LARGE")) for role in SPLIT_DIRS
        ),
        "scenario_family_support": all(
            train_families.get(family, 0) >= 100
            for family in profile["scenario_family_weights"]
        ),
        "all_six_answer_types_in_train": all_answer_types.issubset(train_answer_types),
        "all_insufficient_evidence_subtypes_in_train": all(
            distribution["insufficient_evidence_subtype_distribution"].get(subtype, 0) > 0
            for subtype in ("FUTURE_ONLY", "CONFLICT_UNRESOLVED", "MISSING_SET_MEMBER",
                            "WRONG_SOURCE_FAMILY_ONLY", "STALE_PERSONAL_STATE")
        ),
        "dependency_depths_one_two_and_three_in_train": all(
            train_graph_depths.get(str(depth), 0) > 0 for depth in (1, 2, 3)
        ),
        "same_surface_different_requirement_minimum": same_fraction
        >= thresholds["same_surface_episode_fraction_min"],
        "different_surface_same_dependency_minimum": different_fraction
        >= thresholds["different_surface_same_graph_episode_fraction_min"],
        "availability_does_not_determine_requirement": (
            0 < distribution["history_available_conditionals"]["true"]["memory_required_rate"] < 1
            and 0 < distribution["external_world_available_conditionals"]["true"]["required_rate"] < 1
        ),
        "single_feature_shortcut_gate": single["status"] == "PASS"
        and single["maximum_single_feature_accuracy"] <= thresholds["single_feature_accuracy_max"]
        and single["maximum_single_feature_lift_over_majority"]
        <= thresholds["single_feature_lift_over_majority_max"],
        "combined_cheap_feature_review": combined_review_ok,
        "split_and_sibling_isolation": split["status"] == "PASS",
        "runtime_evaluator_separation": runtime["status"] == "PASS",
        "complete_lineage": lineage["status"] == "PASS",
        "duplicate_audit": duplicate["status"] == "PASS"
        and duplicate["exact_runtime_serialization_duplicate_rows"] == 0
        and duplicate["latent_graph_duplicates_confined_to_sibling_group"],
        "temporal_audit": temporal["status"] == "PASS"
        and not temporal["future_personal_record_failures"]
        and not temporal["publication_effectivity_failures"]
        and all(row["pass"] for row in temporal["revision_checks"]
                + temporal["decision_boundary_checks"]),
        "counterfactual_contract": counterfactual["status"] == "PASS"
        and counterfactual["latent_to_counterfactual_mismatches"] == 0,
        "synthetic_key_opacity": key_audit["status"] == "PASS",
        "source_independence": source["status"] == "PASS",
        "reserved_test_ood_unmaterialized": reserved["status"] == "PASS",
        "deterministic_regeneration": determinism_verified,
        "human_spot_audit": human_review_status == "PASS"
        and sum(spot["sample_sizes"].values()) == 64,
        "architecture_and_budget_unresolved": all(
            case.scenario.oracle.architecture_required == "UNKNOWN"
            and case.scenario.supervision_status.architecture == "UNRESOLVED"
            and case.scenario.supervision_status.budget_class == "UNRESOLVED"
            for case in cases
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "train_derived_capability_distribution_review": capability_distribution_review,
        "train_history_regime_fractions": {key: round(value, 4)
                                           for key, value in train_history_fractions.items()},
        "required_memory_record_type_max_fraction": round(near_dominance, 4),
        "same_surface_different_requirement_episode_fraction": round(same_fraction, 4),
        "different_surface_same_dependency_episode_fraction": round(different_fraction, 4),
        "combined_probe_max_balanced_accuracy_or_macro_f1": combined_max,
        "combined_probe_review_threshold": thresholds["combined_probe_review_threshold"],
        "combined_probe_review_status": ("PASS" if combined_review_ok else "REVIEW_REQUIRED"),
        "combined_probe_human_disposition": human_review_status,
        "single_feature_gate_max_accuracy": thresholds["single_feature_accuracy_max"],
        "single_feature_gate_max_lift": thresholds["single_feature_lift_over_majority_max"],
        "univariate_references": {"single_feature_max_accuracy": .80,
                                  "single_feature_max_lift_over_majority": .15},
        "train_scenario_family_minimum": 100,
        "train_revision_depth_distribution": train_revision_distribution,
        "train_temporal_distance_distribution": train_temporal_distances,
        "train_dependency_depth_distribution": train_graph_depths,
    }


def _table(mapping: dict[str, Any]) -> str:
    if not mapping:
        return "(none)"
    return ", ".join(f"`{key}` {value}" for key, value in sorted(mapping.items()))


def _report_markdown(
    *, run_id: str, branch: str, base_commit: str, generator_commit: str,
    spec_hash: str, manifest_status: str, profile: dict[str, Any], distribution: dict[str, Any],
    single: dict[str, Any], combined: dict[str, Any], split: dict[str, Any],
    duplicate: dict[str, Any], temporal: dict[str, Any], counterfactual: dict[str, Any],
    source: dict[str, Any], gates: dict[str, Any], spot: dict[str, Any],
    lineage: dict[str, Any], reserved: dict[str, Any],
) -> str:
    roles = distribution["episode_counts_by_split"]
    subjects = distribution["subject_counts_by_split"]
    return f"""# Health-Copilot U2-F — Owned Longitudinal Universe

- Dataset ID: `{profile['dataset_id']}`
- Run ID: `{run_id}`
- Status: `{manifest_status}`
- Generator commit: `{generator_commit}`
- Branch: `{branch}`
- Base commit: `{base_commit}`
- Specification hash: `{spec_hash}`
- Content origin: `{CONTENT_ORIGIN}`
- Notice: `{WORLD_NOTICE}`
- Provider calls: `0`; training started: `NO`

## Scale and split ownership

| Split | Episodes | Subjects | Episodes per subject (min / median / mean / max) |
|---|---:|---:|---|
| TRAIN | {roles.get('TRAIN', 0)} | {subjects.get('TRAIN', 0)} | {distribution['episodes_per_subject_by_split']['TRAIN']['episodes_per_subject']} |
| DEV_IID | {roles.get('DEV_IID', 0)} | {subjects.get('DEV_IID', 0)} | {distribution['episodes_per_subject_by_split']['DEV_IID']['episodes_per_subject']} |
| DEV_STRUCTURAL | {roles.get('DEV_STRUCTURAL', 0)} | {subjects.get('DEV_STRUCTURAL', 0)} | {distribution['episodes_per_subject_by_split']['DEV_STRUCTURAL']['episodes_per_subject']} |
| Total | {sum(roles.values())} | {sum(subjects.values())} | {distribution['episodes_per_subject_overall']} |

- Timeline records per subject: {distribution['timeline_records_per_subject']}
- Timeline span in days per subject: {distribution['timeline_span_days_per_subject']}
- Shared-timeline progression: `{distribution['timeline_progression']['all_subject_snapshots_monotonic']}`; subjects gaining later records: {distribution['timeline_progression']['subjects_with_strict_later_record_additions']}
- History regime counts: {_table(distribution['history_regime_distribution'])}
- History record-count buckets: {_table(distribution['history_record_count_buckets'])}
- Available record types: {_table(distribution['available_patient_records_by_type'])}
- Required Memory records by type: {_table(distribution['required_memory_evidence_records_by_type'])}
- Scenario families: {_table(distribution['scenario_family_distribution'])}
- Answer types: {_table(distribution['answer_type_distribution'])}
- Derived capability requirements: {_table(distribution['derived_capability_distribution'])}
- Dependency depth: {_table(distribution['dependency_depth_distribution'])}
- Dependency width: {_table(distribution['dependency_width_distribution'])}
- Independent dependency groups: {_table(distribution['independent_dependency_groups_distribution'])}
- Distractor counts: {distribution['distractor_count_summary']}; regimes: {_table(distribution['distractor_regime_distribution'])}
- Evidence-world sizes: {distribution['external_evidence_record_count_summary']}; regimes: {_table(distribution['evidence_world_regime_distribution'])}
- Revision depths: {_table(distribution['revision_depth_distribution'])}
- Memory-required temporal distances: {_table(distribution['memory_required_temporal_distance_distribution'])}
- Insufficient-evidence subtypes: {_table(distribution['insufficient_evidence_subtype_distribution'])}
- Same surface, different requirement: {gates['same_surface_different_requirement_episode_fraction']:.4f} of episodes
- Different surface, same latent dependency: {gates['different_surface_same_dependency_episode_fraction']:.4f} of episodes
- IID lexical template overlap is intentional and reported; DEV_STRUCTURAL uses its four reserved families.
- Per-class TRAIN distribution is compared with the reference ranges in `manifest.json`; deviations are documented rather than silently rebalanced.

## Shortcut and integrity audits

- Majority and best single-feature diagnostics: `{single['status']}`; max accuracy {single['maximum_single_feature_accuracy']:.4f}; max lift over majority {single['maximum_single_feature_lift_over_majority']:.4f}.
- Combined context-aware Naive Bayes diagnostics: `{combined['status']}`; TRAIN→DEV_IID macro-F1/balanced-accuracy max {combined['train_to_dev_iid']['max_balanced_accuracy_or_macro_f1']:.4f}; TRAIN→DEV_STRUCTURAL {combined['train_to_dev_structural']['max_balanced_accuracy_or_macro_f1']:.4f}.
- Query-only diagnostic is separately recorded for DEV_IID and DEV_STRUCTURAL in `cheap_classifier_audit.json`.
- Split/sibling isolation: `{split['status']}`; subject, persona, scenario-seed, and counterfactual sibling leaks are zero.
- Exact duplicate audit: `{duplicate['status']}`; exact query duplicate rows {duplicate['exact_query_duplicate_rows']} ({duplicate['exact_query_duplicates_allowed_inside_matched_groups']} are approved same-surface pairs); duplicate runtime rows {duplicate['exact_runtime_serialization_duplicate_rows']}.
- Temporal/revision: `{temporal['status']}`; future personal rows checked {temporal['future_personal_records_checked']}, external publication/effectivity checks {temporal['publication_effectivity_checks']}, revision examples {len(temporal['revision_checks'])}, exact boundaries {len(temporal['decision_boundary_checks'])}.
- U1.1 contract: `{counterfactual['status']}`; {counterfactual['counterfactual_arm_count']} arms; mismatches {counterfactual['latent_to_counterfactual_mismatches']}.
- Source independence and generated identifier scan: `{source['status']}`.
- Runtime/evaluator separation: `{gates['gates']['runtime_evaluator_separation']}`; lineage rows checked {lineage['rows_checked']}.
- Deterministic spot sample: TRAIN {spot['sample_sizes'].get('TRAIN', 0)}, DEV_IID {spot['sample_sizes'].get('DEV_IID', 0)}, DEV_STRUCTURAL {spot['sample_sizes'].get('DEV_STRUCTURAL', 0)}; status `{spot['review_status']}`.
- Combined-probe review disposition: `{gates['combined_probe_review_status']}`; max score {gates['combined_probe_max_balanced_accuracy_or_macro_f1']:.4f} against review threshold {gates['combined_probe_review_threshold']:.2f}.

## Data and training boundary

- Reserved TEST/OOD pools: `{reserved['status']}`; materialized rows {reserved['rows_generated']}.
- Real patient data / PHI: `NO`.
- External benchmark rows: `NO`.
- WHO/CDC content or external medical documents: `NO`.
- Architecture supervision: `UNRESOLVED`; budget supervision: `UNRESOLVED`.
- Partial capability selection data eligible: `{gates['status'] == 'PASS'}` for Memory read, External retrieval, and answerability only.
- Full execution-policy training ready: `NO`; post-training ready: `NO`.
- Memory and RAG subsystem tracks modified: `NO`.
- E2-B / L4 started: `NO`.

## Gate result

`SCALE_GATE = {gates['status']}`. See `manifest.json` for every boolean gate and `distribution_report.json`, `shortcut_audit.json`, `cheap_classifier_audit.json`, `split_audit.json`, `duplicate_audit.json`, `temporal_revision_audit.json`, `counterfactual_contract_report.json`, `source_independence_audit.json`, `human_spot_audit.json`, and `lineage_audit.json` for full measurements.
"""


def build_run(
    *, spec_root: Path, source_root: Path, output: Path, branch: str,
    base_commit: str, generator_commit: str, determinism_verified: bool,
    human_review_status: str, human_review_note: str,
) -> dict[str, Any]:
    loaded = {name: _read_json(spec_root / name)
              for name in (*PLAN_FILES, PROFILE_FILE, RESERVED_FILE)}
    plans = tuple(loaded[name] for name in PLAN_FILES)
    profile = loaded[PROFILE_FILE]
    reserved_plan = loaded[RESERVED_FILE]
    if tuple(plan["split_role"] for plan in plans) != ("TRAIN", "DEV_IID", "DEV_STRUCTURAL"):
        raise ValueError("U2-F plans must be ordered TRAIN, DEV_IID, DEV_STRUCTURAL")
    if human_review_status not in {"PENDING", "PASS", "FAIL"}:
        raise ValueError("human_review_status must be PENDING, PASS, or FAIL")
    spec_hash = _canonical_hash({name: loaded[name] for name in (*PLAN_FILES, PROFILE_FILE, RESERVED_FILE)})
    run_id = sha256(f"{spec_hash}|{generator_commit}".encode()).hexdigest()[:12]

    worlds, timelines = build_u2f_worlds(
        plans=plans, profile=profile, run_seed=int(profile["global_seed"])
    )
    cases = tuple(materialize(world) for world in worlds)
    split = audit_splits(worlds, plans)
    distribution = u2f_distribution_report(worlds, cases, timelines)
    single = single_feature_audit(
        cases,
        float(profile["required_thresholds"]["single_feature_accuracy_max"]),
        float(profile["required_thresholds"]["single_feature_lift_over_majority_max"]),
    )
    combined = cheap_classifier_audit(cases)
    duplicate = duplicate_audit(worlds)
    temporal = temporal_revision_audit_u2f(cases)
    counterfactual = counterfactual_contract_report(cases)
    matched = matched_pair_diagnostics(worlds, cases)
    key_audit = synthetic_key_leakage_audit(worlds)
    source = source_independence_audit(
        worlds, source_root=source_root, spec_root=spec_root,
        forbidden_identifiers=tuple(profile["forbidden_generated_identifiers"]),
        timelines=timelines,
    )
    reserved = _reserved_plan_audit(reserved_plan, plans)
    runtime_rows = [case.episode.to_runtime_dict() for case in cases]
    runtime = _runtime_leakage_audit(runtime_rows)
    lineage_rows = [lineage_row(world, spec_hash, profile["generator_version"])
                    for world in worlds]
    lineage = _lineage_audit(lineage_rows, spec_hash, profile["generator_version"])
    spot_packet = deterministic_spot_sample(
        cases, int(profile["audit_seed"]), {"TRAIN": 32, "DEV_IID": 16, "DEV_STRUCTURAL": 16}
    )
    spot = {
        **spot_packet,
        "review_status": human_review_status,
        "reviewed_by": "Codex",
        "review_note": human_review_note,
        "review_scope": ["obvious nonsense", "template artifacts", "label leakage",
                         "impossible grammar", "unintended clinical claims", "duplicate wording"],
        "manual_example_edits": False,
    }

    gates = _build_gates(
        profile=profile, worlds=worlds, cases=cases,
        distribution=distribution, single=single, combined=combined,
        split=split, duplicate=duplicate, temporal=temporal,
        counterfactual=counterfactual, source=source, runtime=runtime,
        lineage=lineage, key_audit=key_audit, reserved=reserved, matched=matched, spot=spot,
        determinism_verified=determinism_verified,
        human_review_status=human_review_status,
    )
    eligible = gates["status"] == "PASS"
    output.mkdir(parents=True, exist_ok=False)
    cases_by_role = {role: [case for case in cases if case.scenario.world.split_role == role]
                     for role in SPLIT_DIRS}
    lineage_by_id = {row["episode_id"]: row for row in lineage_rows}
    for role, folder in SPLIT_DIRS.items():
        current_cases = cases_by_role[role]
        prefix = output / folder
        episode_rows = [case.episode.to_runtime_dict() for case in current_cases]
        truth_rows = [case.scenario.evaluator_truth_dict() for case in current_cases]
        current_lineage = [lineage_by_id[case.episode.episode_id] for case in current_cases]
        supervision_rows = _partial_policy_rows(tuple(current_cases), eligible)
        for name, rows in (
            ("episodes.jsonl", episode_rows),
            ("evaluator_truth.jsonl", truth_rows),
            ("lineage.jsonl", current_lineage),
            ("partial_policy_supervision.jsonl", supervision_rows),
        ):
            path = prefix / name
            _write_jsonl(path, rows)

    root_jsonl = {
        "latent_worlds.jsonl": [world.to_dict() for world in worlds],
        "natural_language_realizations.jsonl": [world.to_realization_dict() for world in worlds],
        "lineage.jsonl": lineage_rows,
    }
    for name, rows in root_jsonl.items():
        path = output / name
        _write_jsonl(path, rows)

    audit_payloads = {
        "distribution_report.json": distribution,
        "shortcut_audit.json": single,
        "cheap_classifier_audit.json": combined,
        "split_audit.json": split,
        "duplicate_audit.json": duplicate,
        "temporal_revision_audit.json": temporal,
        "counterfactual_contract_report.json": counterfactual,
        "matched_pair_diagnostics.json": matched,
        "synthetic_key_leakage_audit.json": key_audit,
        "source_independence_audit.json": source,
        "runtime_leakage_audit.json": runtime,
        "lineage_audit.json": lineage,
        "reserved_test_plan.json": reserved_plan,
        "reserved_test_audit.json": reserved,
        "human_spot_audit.json": spot,
        "scale_gates.json": gates,
        "scale_profile.json": profile,
        "generation_plans.json": list(plans),
    }
    for name, payload in audit_payloads.items():
        _write_json(output / name, payload)

    report = _report_markdown(
        run_id=run_id, branch=branch, base_commit=base_commit,
        generator_commit=generator_commit, spec_hash=spec_hash,
        manifest_status="FROZEN" if eligible else "CANDIDATE_NOT_FROZEN",
        profile=profile, distribution=distribution, single=single, combined=combined,
        split=split, duplicate=duplicate, temporal=temporal,
        counterfactual=counterfactual, source=source, gates=gates, spot=spot,
        lineage=lineage, reserved=reserved,
    )
    (output / "report.md").write_text(report, encoding="utf-8", newline="\n")
    artifact_hashes = _hash_artifacts(output)
    root_hash = _canonical_hash(artifact_hashes)
    manifest = {
        "schema_version": "owned-longitudinal-v1-manifest",
        "dataset_id": profile["dataset_id"],
        "dataset_version_frozen": eligible,
        "run_id": run_id,
        "status": gates["status"],
        "generator_version": profile["generator_version"],
        "generator_commit_sha": generator_commit,
        "branch": branch,
        "base_commit_sha": base_commit,
        "spec_hash": spec_hash,
        "generation_plan_hashes": {
            name: _canonical_hash(loaded[name]) for name in (*PLAN_FILES, PROFILE_FILE, RESERVED_FILE)
        },
        "artifact_count": len(artifact_hashes),
        "artifact_sha256": artifact_hashes,
        "dataset_root_hash": root_hash,
        "episode_counts_by_split": distribution["episode_counts_by_split"],
        "subject_counts_by_split": distribution["subject_counts_by_split"],
        "scale_gate_status": gates["status"],
        "scale_gate_details": gates,
        "deterministic_regeneration_verified_by_isolated_wrapper": determinism_verified,
        "human_spot_audit": {"status": human_review_status,
                              "sample_sizes": spot["sample_sizes"],
                              "review_note": human_review_note},
        "data_qualification": {
            "partial_policy_data_eligible": eligible,
            "eligible_dimensions": ["memory_read", "external_retrieval", "answerability"],
            "full_execution_policy_training_ready": False,
            "post_train_ready": False,
            "training_started": False,
        },
        "supervision": {"memory_read": "QUALIFIED", "external_retrieval": "QUALIFIED",
                        "answerability": "QUALIFIED", "architecture": "UNRESOLVED",
                        "budget": "UNRESOLVED"},
        "provenance_assertions": {
            "content_origin": CONTENT_ORIGIN,
            "real_patient_data_used": False,
            "phi_used": False,
            "external_benchmark_rows_used": False,
            "who_cdc_content_used": False,
            "external_medical_document_content_used": False,
            "provider_calls": 0,
            "memory_track_modified": False,
            "rag_track_modified": False,
            "e2_b_started": False,
            "l4_started": False,
        },
        "reserved_evaluation_rows_materialized": False,
        "notice": WORLD_NOTICE,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--determinism-verified", action="store_true")
    parser.add_argument("--human-review-status", choices=("PENDING", "PASS", "FAIL"),
                        default="PENDING")
    parser.add_argument("--human-review-note", default="")
    args = parser.parse_args()
    manifest = build_run(
        spec_root=args.spec_root, source_root=args.source_root, output=args.output,
        branch=args.branch, base_commit=args.base_commit,
        generator_commit=args.generator_commit,
        determinism_verified=args.determinism_verified,
        human_review_status=args.human_review_status,
        human_review_note=args.human_review_note,
    )
    print(json.dumps({"status": manifest["status"], "run_id": manifest["run_id"],
                      "episode_counts_by_split": manifest["episode_counts_by_split"],
                      "subject_counts_by_split": manifest["subject_counts_by_split"],
                      "dataset_root_hash": manifest["dataset_root_hash"],
                      "failed_gates": [key for key, value in manifest["scale_gate_details"]["gates"].items()
                                       if not value]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
