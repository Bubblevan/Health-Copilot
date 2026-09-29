"""Frozen seed/template ownership and leakage audits for U2-E pilot pools."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .scenarios import SCENARIO_FAMILIES, build_world
from .schema import LatentWorld


def generate_plan(plan: dict[str, Any], *, run_seed: int) -> tuple[LatentWorld, ...]:
    role = str(plan["split_role"])
    count = int(plan["episode_count"])
    subjects = int(plan["subject_count"])
    families = tuple(plan["scenario_families"])
    templates = tuple(plan["template_families"])
    budget_regimes = tuple(plan["budget_regimes"])
    deadline_regimes = tuple(plan["deadline_regimes"])
    if count % 2:
        raise ValueError(f"{role} count must be even so sibling groups stay intact")
    if subjects <= 0 or not templates or not families:
        raise ValueError(f"{role} plan requires subjects, templates, and scenario families")
    if set(families) != set(SCENARIO_FAMILIES):
        raise ValueError(f"{role} plan must cover the complete owned scenario grammar")
    if len(plan["scenario_seed_range"]) != 2 or count // 2 > (
        int(plan["scenario_seed_range"][1]) - int(plan["scenario_seed_range"][0]) + 1
    ):
        raise ValueError(f"{role} scenario seed range is too small")
    if len(plan["persona_seed_range"]) != 2 or subjects > (
        int(plan["persona_seed_range"][1]) - int(plan["persona_seed_range"][0]) + 1
    ):
        raise ValueError(f"{role} persona seed range is too small")

    scenario_start, scenario_end = (int(x) for x in plan["scenario_seed_range"])
    persona_start, persona_end = (int(x) for x in plan["persona_seed_range"])
    scenario_capacity = scenario_end - scenario_start + 1
    persona_capacity = persona_end - persona_start + 1
    worlds: list[LatentWorld] = []
    pairs = count // 2
    for pair_index in range(pairs):
        if pair_index in (3, 4):
            subject_index = 0
            identity_pair = 3
            group_id = f"CF-{role}-BOUNDARY-000"
            family = "TEMPORAL_BOUNDARY"
            mode = "boundary"
            diagnostic_mode = "boundary"
            template_index = 3 % len(templates)
        elif pair_index in (5, 6):
            subject_index = min(1, subjects - 1)
            identity_pair = 5
            group_id = f"CF-{role}-REVISION-000"
            family = "MEMORY_REVISION"
            mode = "revision"
            diagnostic_mode = "revision-before" if pair_index == 5 else "revision-after"
            template_index = 5 % len(templates)
        elif pair_index in (7, 8):
            subject_index = min(2, subjects - 1)
            identity_pair = 7
            group_id = f"CF-{role}-EVIDENCE-VERSION-000"
            family = "EXTERNAL_VERSIONED"
            mode = "evidence-version"
            diagnostic_mode = "evidence-before" if pair_index == 7 else "evidence-after"
            template_index = 7 % len(templates)
        else:
            subject_index = pair_index % subjects
            identity_pair = pair_index
            group_id = f"CF-{role}-{pair_index:04d}"
            if pair_index == 0:
                family, mode, diagnostic_mode = "MEMORY_LOOKUP", "memory-missing", "standard"
            elif pair_index == 1:
                family, mode, diagnostic_mode = "EXTERNAL_LOOKUP", "external-missing", "standard"
            elif pair_index == 2:
                family, mode, diagnostic_mode = "MEMORY_LOOKUP", "surface-pair", "standard"
            else:
                family = families[(pair_index - 9) % len(families)]
                mode, diagnostic_mode = "surface-pair", "standard"
            template_index = pair_index % len(templates)

        persona_seed = persona_start + (subject_index + run_seed) % persona_capacity
        pair_seed = scenario_start + (identity_pair + run_seed) % scenario_capacity
        base_budget_index = identity_pair % len(budget_regimes)
        budget_class = budget_regimes[base_budget_index]
        deadline_class = deadline_regimes[identity_pair % len(deadline_regimes)]
        selected_template = templates[template_index]

        for slot in range(2):
            episode_index = pair_index * 2 + slot
            if mode == "memory-missing":
                episode_family = "MEMORY_LOOKUP" if slot == 0 else "INSUFFICIENT_EVIDENCE"
                template_family = selected_template
                surface_seed = 0
            elif mode == "external-missing":
                episode_family = "EXTERNAL_LOOKUP" if slot == 0 else "INSUFFICIENT_EVIDENCE"
                template_family = selected_template
                surface_seed = 0
            elif mode == "boundary":
                episode_family = family
                boundary_slot = slot if pair_index == 3 else slot + 2
                template_family = selected_template
                surface_seed = 0
            else:
                episode_family = family
                boundary_slot = 0
                if mode == "revision" or mode == "evidence-version":
                    template_family = selected_template
                    surface_seed = 0
                elif mode == "surface-pair" and pair_index in (2,):
                    template_family = templates[(template_index + slot) % len(templates)]
                    surface_seed = slot
                else:
                    template_family = templates[(template_index + slot) % len(templates)]
                    surface_seed = slot

            if mode in {"revision", "evidence-version", "boundary"}:
                scenario_seed = pair_seed
            else:
                scenario_seed = pair_seed
            if mode == "boundary":
                diag_slot = boundary_slot
            else:
                diag_slot = slot

            actual_diagnostic_mode = diagnostic_mode
            if mode in {"memory-missing", "external-missing"}:
                actual_diagnostic_mode = "matched-abstain"
            if mode == "memory-missing" and slot == 1:
                episode_family = "INSUFFICIENT_EVIDENCE"
            if mode == "external-missing" and slot == 1:
                episode_family = "INSUFFICIENT_EVIDENCE"

            world = build_world(
                episode_id=f"U2E-{role}-{episode_index:04d}", split_role=role,
                subject_id=f"OWNED-{role}-S{subject_index:03d}",
                persona_seed=persona_seed, scenario_seed=scenario_seed,
                scenario_family=episode_family, template_family=template_family,
                surface_variant_seed=surface_seed,
                counterfactual_family_id=group_id, budget_class=budget_class,
                deadline_class=deadline_class, template_pool=templates,
                diagnostic_mode=actual_diagnostic_mode, diagnostic_slot=diag_slot,
            )
            worlds.append(world)
    return tuple(worlds)


def audit_splits(worlds: tuple[LatentWorld, ...], plans: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    plan_by_role = {str(plan["split_role"]): plan for plan in plans}
    subject_roles: dict[str, set[str]] = defaultdict(set)
    persona_roles: dict[int, set[str]] = defaultdict(set)
    scenario_roles: dict[int, set[str]] = defaultdict(set)
    sibling_roles: dict[str, set[str]] = defaultdict(set)
    template_roles: dict[str, set[str]] = defaultdict(set)
    template_iid_owners: dict[str, set[str]] = defaultdict(set)
    for world in worlds:
        subject_roles[world.subject_id].add(world.split_role)
        persona_roles[world.persona_seed].add(world.split_role)
        scenario_roles[world.scenario_seed].add(world.split_role)
        sibling_roles[world.counterfactual_family_id].add(world.split_role)
        template_roles[world.template_family].add(world.split_role)
        if plan_by_role[world.split_role].get("iid_lexical_augmentation", False):
            template_iid_owners[world.template_family].add(world.split_role)

    subject_leaks = sorted(key for key, roles in subject_roles.items() if len(roles) > 1)
    persona_leaks = sorted(key for key, roles in persona_roles.items() if len(roles) > 1)
    seed_leaks = sorted(key for key, roles in scenario_roles.items() if len(roles) > 1)
    sibling_leaks = sorted(key for key, roles in sibling_roles.items() if len(roles) > 1)
    template_leaks = []
    for template, roles in sorted(template_roles.items()):
        if len(roles) <= 1:
            continue
        if roles == {"TRAIN", "DEV_IID"} and template_iid_owners.get(template) == {"DEV_IID"}:
            continue
        template_leaks.append({"template_family": template, "roles": sorted(roles)})
    return {
        "status": "PASS" if not (subject_leaks or persona_leaks or seed_leaks
                                   or sibling_leaks or template_leaks) else "FAIL",
        "subject_split_leaks": subject_leaks,
        "persona_seed_split_leaks": persona_leaks,
        "scenario_seed_split_leaks": seed_leaks,
        "counterfactual_sibling_split_leaks": sibling_leaks,
        "template_family_split_leaks": template_leaks,
        "template_overlap_exception": {
            "families": sorted(set(template_roles) & {"COMMON_A", "COMMON_B"}),
            "basis": "explicit DEV_IID lexical augmentation; structural DEV templates are reserved",
        },
        "train_subject_count": len({world.subject_id for world in worlds if world.split_role == "TRAIN"}),
        "dev_iid_subject_count": len({world.subject_id for world in worlds if world.split_role == "DEV_IID"}),
        "dev_structural_subject_count": len({world.subject_id for world in worlds
                                             if world.split_role == "DEV_STRUCTURAL"}),
        "all_subjects_total": len(subject_roles),
        "episode_counts": {role: sum(world.split_role == role for world in worlds)
                            for role in sorted(plan_by_role)},
        "subject_seed_ranges": {role: plan["persona_seed_range"] for role, plan in plan_by_role.items()},
        "scenario_seed_ranges": {role: plan["scenario_seed_range"] for role, plan in plan_by_role.items()},
        "reserved_test_materialized": False,
    }
