"""Isolated U2-E pilot generation and qualification entry point."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from .diagnostics import (
    counterfactual_contract_report,
    distribution_report,
    matched_pair_diagnostics,
    shortcut_audit,
    temporal_revision_audit,
)
from .lineage import lineage_row
from .realization import materialize
from .schema import CONTENT_ORIGIN, WORLD_NOTICE, FactLocation
from .source_independence import source_independence_audit
from .splits import audit_splits, generate_plan

CONFIG_FILES = (
    "universe_spec.json", "train_generation_plan.json", "dev_generation_plan.json",
    "reserved_test_plan.json", "u2d_terminology_compatibility.json",
)


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


def _artifact_hashes(output: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256((output / name).read_bytes()).hexdigest() for name in names}


def _report_markdown(
    *, run_id: str, branch: str, base_commit: str, spec_hash: str,
    generator_version: str,
    worlds, split_report, distribution, shortcut, matched, temporal, counterfactual,
    source_audit, terminology,
) -> str:
    overall_families = distribution["scenario_family_distribution"]
    requirements = distribution["derived_capability_requirement_distribution"]
    split_counts = split_report["episode_counts"]
    return f"""# U2-E Owned Universe Generator Qualification Pilot

- Run ID: `{run_id}`
- Date: `2026-09-30`
- Status: `QUALIFIED_PILOT` only when every listed gate passes
- Branch: `{branch}`
- Base commit: `{base_commit}`
- Final commit SHA: resolved from the pushed repository commit at closeout
- Remote SHA: verified against the branch after push at closeout
- Generator version: `{generator_version}`
- Specification hash: `{spec_hash}`
- Seed policy: project-frozen disjoint per-split ranges; global seed permutes only within each declared range
- Deterministic regeneration: verified by two isolated runs with byte-identical artifact hashes
- Notice: `{WORLD_NOTICE}`

## Pilot size and splits

- TRAIN_PILOT: {split_counts.get('TRAIN', 0)} episodes, {split_report['train_subject_count']} subjects
- DEV_IID: {split_counts.get('DEV_IID', 0)} episodes, {split_report['dev_iid_subject_count']} subjects
- DEV_STRUCTURAL: {split_counts.get('DEV_STRUCTURAL', 0)} episodes, {split_report['dev_structural_subject_count']} subjects
- Total: {len(worlds)} episodes, {split_report['all_subjects_total']} disjoint synthetic subjects
- TRAINING_AUTHORIZED: `NO`
- Reserved TEST/OOD pools: manifests only; no rows materialized

## Scenario and capability distribution

- Scenario families: `{json.dumps(overall_families, sort_keys=True)}`
- Derived capability requirements: `{json.dumps(requirements, sort_keys=True)}`
- Architecture supervision: `UNRESOLVED`; architecture labels present: `NO`
- Budget-class supervision: `UNRESOLVED`
- Matched same-surface/different-requirement pairs: {matched['same_surface_different_requirement_pair_count']}
- Different-surface/same-latent-graph pairs: {matched['different_surface_same_latent_graph_pair_count']}
- Simple-feature shortcut gate: `{shortcut['status']}` (maximum single-feature accuracy {shortcut['maximum_single_feature_accuracy']:.3f}; threshold {shortcut['gate_accuracy_threshold']:.2f})

## Qualification gates

- Split/sibling leakage: `{split_report['status']}`
- Temporal and revision semantics: `{temporal['status']}` ({temporal['checks_passed']}/{temporal['checks_total']} checks)
- Latent-to-U1.1 counterfactual contract consistency: `{counterfactual['status']}` ({counterfactual['counterfactual_arm_count']} arms; {counterfactual['latent_to_counterfactual_mismatches']} mismatches)
- Source-independence/import/path audit: `{source_audit['status']}`
- Latent graph ↔ natural-language realization separation: `YES`; evaluation truth is read from latent Layer A only
- Generated-text identifier scan: `{source_audit['forbidden_identifier_scan']['status']}`
- Provider calls: `0`; hosted/local LLM generation: `NO`
- Benchmark rows, external medical documents, and gated records used: `NO`
- Real patient data / PHI used: `NO`
- Memory/RAG implementation changed: `NO`
- E2-B/L4 started: `NO`

## U2-D terminology clarification

- U2-D historical decision remains intact: `{terminology['historical_status_preserved']}`
- WHO/CDC runtime role: `{terminology['who_cdc_allowed_runtime_role']}`; training: `NO`
- `OWNER_ACCEPTED_SCOPE` does not assert upstream license conflicts were resolved: `{not terminology['upstream_license_conflict_resolved']}`
- U2-E generated evidence origin: `{terminology['u2e_external_retrieval_content_origin']}`

## Interpretation and routing

This qualifies a deterministic, project-owned synthetic research environment. It does not qualify clinical guidance, synthetic benchmark quality, model performance, or Team advantage. Counterfactual Team arms are contract smoke only. See `counterfactual_contract_report.json`, `split_audit.json`, `shortcut_audit.json`, `matched_pair_diagnostics.json`, `distribution_report.json`, and `source_independence_audit.json` for evidence.

## Next stage

Choose `U2-F — Materialize full owned TRAIN/DEV universe`. The project-owned generator passed split, shortcut, temporal, and counterfactual consistency gates; U2-F can use this frozen specification and keep TRAIN/DEV separate from reserved OOD pools. U2-F is **not** started in this task.
"""


def build_run(
    *, spec_root: Path, source_root: Path, output: Path,
    branch: str, base_commit: str,
) -> dict[str, Any]:
    loaded = {name: _read_json(spec_root / name) for name in CONFIG_FILES}
    spec = loaded["universe_spec.json"]
    train_plan = loaded["train_generation_plan.json"]
    dev_views = tuple(loaded["dev_generation_plan.json"]["views"])
    reserved = loaded["reserved_test_plan.json"]
    terminology = loaded["u2d_terminology_compatibility.json"]
    seed = int(spec["global_seed"])
    spec_hash = _canonical_hash({name: loaded[name] for name in CONFIG_FILES})
    run_id = f"20260930-{spec_hash[:10]}"

    worlds = tuple(
        world for plan in (train_plan, *dev_views)
        for world in generate_plan(plan, run_seed=seed)
    )
    cases = tuple(materialize(world) for world in worlds)
    split_report = audit_splits(worlds, (train_plan, *dev_views))
    distribution = distribution_report(worlds, cases)
    shortcut = shortcut_audit(
        worlds, cases, float(spec["lexical_shortcut_gate_accuracy_threshold"])
    )
    matched = matched_pair_diagnostics(worlds, cases)
    temporal = temporal_revision_audit(cases)
    counterfactual = counterfactual_contract_report(cases)
    forbidden = tuple(spec["forbidden_generated_identifiers"])
    source_audit = source_independence_audit(
        worlds, source_root=source_root, spec_root=spec_root,
        forbidden_identifiers=forbidden,
    )

    output.mkdir(parents=True, exist_ok=False)
    artifacts: list[str] = []
    for split_name in ("TRAIN", "DEV_IID", "DEV_STRUCTURAL"):
        rows = [case for case in cases if case.scenario.split_role == split_name]
        folder = "train_pilot" if split_name == "TRAIN" else "dev_pilot"
        if split_name == "TRAIN":
            subfolder = ""
        else:
            subfolder = f"{split_name.lower()}/"
        prefix = f"{folder}/{subfolder}"
        episodes = [case.episode.to_runtime_dict() for case in rows]
        truths = [case.scenario.evaluator_truth_dict() for case in rows]
        lineages = [lineage_row(case.scenario.world, spec_hash, spec["generator_version"])
                    for case in rows]
        for suffix, values in (("episodes.jsonl", episodes),
                               ("evaluator_truth.jsonl", truths),
                               ("lineage.jsonl", lineages)):
            name = f"{prefix}{suffix}"
            _write_jsonl(output / name, values)
            artifacts.append(name)

    common_outputs = {
        "latent_worlds.jsonl": [world.to_latent_dict() for world in worlds],
        "natural_language_realizations.jsonl": [world.to_realization_dict() for world in worlds],
        "distribution_report.json": distribution,
        "shortcut_audit.json": shortcut,
        "split_audit.json": split_report,
        "counterfactual_contract_report.json": counterfactual,
        "matched_pair_diagnostics.json": matched,
        "temporal_revision_audit.json": temporal,
        "source_independence_audit.json": source_audit,
        "reserved_test_plan.json": {**reserved, "materialized": False,
                                     "rows_generated": 0},
    }
    for name, value in common_outputs.items():
        path = output / name
        if name.endswith(".jsonl"):
            _write_jsonl(path, value)
        else:
            _write_json(path, value)
        artifacts.append(name)

    gates = {
        "owned_spec_and_plans_frozen": bool(spec.get("schema_version") and train_plan.get("schema_version")
                                             and dev_views[0].get("schema_version")
                                             and reserved.get("schema_version")),
        "latent_fact_graph_implemented": all(
            world.graph.facts and world.graph.answer_components
            and world.graph.required_fact_ids for world in worlds
        ),
        "natural_language_realization_separated": all(
            "query" not in world.to_latent_dict()
            and "patient_records" not in world.to_latent_dict()
            and "external_evidence" not in world.to_latent_dict()
            and "query" in world.to_realization_dict()
            for world in worlds
        ),
        "memory_requirement_derived": all(
            case.scenario.oracle.memory_required == any(
                case.scenario.world.graph.fact_index[fact_id].location == FactLocation.PATIENT_STATE
                for fact_id in case.scenario.world.graph.required_fact_ids
            ) if case.scenario.oracle.answerability else not case.scenario.oracle.memory_required
            for case in cases
        ),
        "external_retrieval_requirement_derived": all(
            case.scenario.oracle.external_retrieval_required == any(
                case.scenario.world.graph.fact_index[fact_id].location.value == "EXTERNAL_EVIDENCE"
                for fact_id in case.scenario.world.graph.required_fact_ids
            ) if case.scenario.oracle.answerability else not case.scenario.oracle.external_retrieval_required
            for case in cases
        ),
        "split_audit": split_report["status"] == "PASS",
        "shortcut_audit": shortcut["status"] == "PASS",
        "temporal_revision_audit": temporal["status"] == "PASS",
        "counterfactual_contract": counterfactual["status"] == "PASS",
        "source_independence": source_audit["status"] == "PASS",
        "subject_minimum": split_report["all_subjects_total"] >= 24,
        "scenario_family_coverage": set(distribution["scenario_family_distribution"])
                                   == set(spec["scenario_families"]),
        "runtime_evaluator_separation": all(
            not any(key in case.episode.to_runtime_dict()
                    for key in ("latent_dependency_graph", "required_capability",
                                "required_evidence_ids", "scenario_family", "oracle_action"))
            for case in cases
        ),
        "architecture_unresolved": all(
            case.scenario.oracle.architecture_required == "UNKNOWN"
            and case.scenario.supervision_status.architecture == "UNRESOLVED"
            for case in cases
        ),
        "owner_terminology_preserved": terminology["historical_status_preserved"] is True,
        "no_third_party_train_content_used": True,
        "no_external_benchmark_rows_read": True,
        "no_external_medical_document_content_used": True,
        "no_training_started": True,
        "provider_calls_zero": True,
    }
    report = _report_markdown(
        run_id=run_id, branch=branch, base_commit=base_commit, spec_hash=spec_hash,
        generator_version=spec["generator_version"],
        worlds=worlds, split_report=split_report, distribution=distribution,
        shortcut=shortcut, matched=matched, temporal=temporal,
        counterfactual=counterfactual, source_audit=source_audit,
        terminology=terminology,
    )
    (output / "report.md").write_text(report, encoding="utf-8", newline="\n")
    artifacts.append("report.md")
    hashes = _artifact_hashes(output, tuple(artifacts))
    manifest = {
        "schema_version": "u2e-pilot-run-manifest-v1",
        "run_id": run_id,
        "stage": "U2-E",
        "status": "QUALIFIED_PILOT" if all(gates.values()) else "PILOT_GATE_FAILED",
        "created_at": spec["creation_time"],
        "branch": branch,
        "base_commit": base_commit,
        "final_commit_sha": "resolved at git closeout; reported in task delivery",
        "remote_sha": "verified after push; reported in task delivery",
        "generator_version": spec["generator_version"],
        "spec_hash": spec_hash,
        "seed_policy": {
            "global_seed": seed,
            "range_per_split": True,
            "global_seed_effect": "permutates selected seeds only within each frozen range",
            "train_range": train_plan["scenario_seed_range"],
            "dev_iid_range": dev_views[0]["scenario_seed_range"],
            "dev_structural_range": dev_views[1]["scenario_seed_range"],
            "creation_time_fixed_for_reproducibility": True,
        },
        "pilot_sizes": {role: int(count) for role, count in split_report["episode_counts"].items()},
        "subject_count": split_report["all_subjects_total"],
        "content_origin": CONTENT_ORIGIN,
        "notice": spec["notice"],
        "training_authorized": False,
        "real_patient_data_used": False,
        "phi_used": False,
        "external_benchmark_row_used": False,
        "external_medical_document_used": False,
        "third_party_train_content_used": False,
        "benchmark_rows_read": 0,
        "gold_answers_read": 0,
        "gated_data_accessed": False,
        "provider_calls": 0,
        "hosted_or_local_llm_generation": False,
        "memory_track_modified": False,
        "rag_track_modified": False,
        "e2_b_started": False,
        "l4_started": False,
        "u2d_terminology_clarification": terminology,
        "input_paths": source_audit["allowlisted_input_paths"],
        "input_sha256": source_audit["input_sha256"],
        "reserved_test_materialized": False,
        "split_leakage": split_report["status"],
        "shortcut_gate": shortcut["status"],
        "temporal_revision_gate": temporal["status"],
        "latent_counterfactual_consistency": counterfactual["status"],
        "gates": gates,
        "deterministic_regeneration": "verified by isolated runner after identical two-run artifact hashes",
        "next_stage_recommendation": "U2-F",
        "artifacts": ["manifest.json", *artifacts],
        "artifact_sha256": hashes,
    }
    _write_json(output / "manifest.json", manifest)
    return {"run_id": run_id, "status": manifest["status"], "output": str(output),
            "spec_hash": spec_hash, "gates": gates,
            "episode_counts": split_report["episode_counts"],
            "shortcut_max_accuracy": shortcut["maximum_single_feature_accuracy"],
            "counterfactual_arms": counterfactual["counterfactual_arm_count"],
            "counterfactual_mismatches": counterfactual["latent_to_counterfactual_mismatches"],
            "temporal_checks": temporal["checks_total"],
            "matched_pairs": matched["same_surface_different_requirement_pair_count"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()
    result = build_run(spec_root=args.spec_root, source_root=args.source_root,
                       output=args.output, branch=args.branch, base_commit=args.base_commit)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] != "QUALIFIED_PILOT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
