"""Materialize the exact reserved seed pools using the frozen U2-F generator."""

from __future__ import annotations

import copy
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.rag_e6.data import E6Episode
from eval.u3r_rag_transfer import OwnedExternalCorpusAdapter
from health_ai_copilot.research.integration.owned_universe.realization import materialize
from health_ai_copilot.research.integration.owned_universe.scale import build_u2f_worlds
from health_ai_copilot.research.integration.owned_universe.scenarios import SCENARIO_FAMILIES

from .protocol import (
    PLAN_SHA256,
    POOL_SEEDS,
    RESERVED_COMPOSITIONS,
    RUN_ROOT_RELATIVE,
    U2F_DATASET_ROOT_SHA256,
    U2F_ROOT_RELATIVE,
    pool_counts,
    read_json,
    sha256_bytes,
    sha256_file,
    validate_reserved_plan,
)
from .protocol import (
    U2F_MANIFEST_SHA256 as LOCKED_U2F_MANIFEST_SHA256,
)
from .protocol import (
    canonical_json_bytes as canonical_bytes,
)

GENERATOR_FILES = (
    "src/health_ai_copilot/research/integration/owned_universe/scale.py",
    "src/health_ai_copilot/research/integration/owned_universe/realization.py",
    "src/health_ai_copilot/research/integration/owned_universe/schema.py",
    "src/health_ai_copilot/research/integration/contracts.py",
    "src/health_ai_copilot/research/integration/evidence_world.py",
    "src/health_ai_copilot/research/integration/owned_universe/facts.py",
    "src/health_ai_copilot/research/integration/owned_universe/realization_text.py",
    "src/health_ai_copilot/research/integration/owned_universe/scenarios.py",
    "src/health_ai_copilot/research/integration/owned_universe/timeline.py",
)


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def _method_freeze_commit(repository_root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = repository_root / RUN_ROOT_RELATIVE / "method_freeze.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("evaluator_truth_opened") is not False
        or manifest.get("reserved_rows_materialized") is not False
        or manifest.get("reserved_plan_sha256") != PLAN_SHA256
    ):
        raise ValueError("method freeze does not authorize the frozen reserved plan")
    frozen_sources = {
        **manifest.get("frozen_e6a_source_hashes", {}),
        **manifest.get("e6b_pipeline_source_hashes", {}),
        **manifest.get("retrieval_source_hashes", {}),
        **manifest.get("scoring_source_hashes", {}),
    }
    for relative, expected in frozen_sources.items():
        if sha256_file(repository_root / relative) != expected:
            raise ValueError(f"committed method-freeze source hash mismatch: {relative}")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(manifest_path.relative_to(repository_root))],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", str(manifest_path.relative_to(repository_root))],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )
    if tracked.returncode != 0 or dirty.stdout.strip():
        raise ValueError("method freeze must be committed and clean before materialization")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--diff-filter=A", "--format=%H", "--",
         str(manifest_path.relative_to(repository_root))],
        cwd=repository_root,
        text=True,
    ).strip()
    if not commit:
        raise ValueError("could not identify the committed method-freeze revision")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return manifest, commit


def _derive_generator_plans(
    reserved: dict[str, Any], source_profile: dict[str, Any],
    source_plans: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    train_plan = next(row for row in source_plans if row.get("split_role") == "TRAIN")
    base_pair_count = int(train_plan["scenario_seed_range"][1]) - int(
        train_plan["scenario_seed_range"][0]
    ) + 1
    base_same_surface = int(source_profile["same_surface_pair_count"]["TRAIN"])
    profile = copy.deepcopy(source_profile)
    profile["targets"] = {}
    profile["same_surface_pair_count"] = {}
    plans: list[dict[str, Any]] = []
    derivation: dict[str, Any] = {}
    for row in reserved["pools"]:
        role = str(row["split_role"])
        episodes, subjects, pairs = pool_counts(role)
        same_surface = math.floor(base_same_surface * pairs / base_pair_count + 0.5)
        profile["targets"][role] = {
            "episode_count": episodes,
            "subject_count": subjects,
        }
        profile["same_surface_pair_count"][role] = same_surface
        plan = {
            "schema_version": "u2f-split-plan-v1",
            "split_role": role,
            "episode_count": episodes,
            "subject_count": subjects,
            "persona_seed_range": list(row["persona_seed_range"]),
            "scenario_seed_range": list(row["scenario_seed_range"]),
            "same_surface_pair_count": same_surface,
            "iid_lexical_augmentation": False,
            "template_families": list(train_plan["template_families"]),
            "scenario_families": sorted(SCENARIO_FAMILIES),
        }
        plans.append(plan)
        derivation[role] = {
            "episodes": episodes,
            "subjects": subjects,
            "sibling_pairs": pairs,
            "same_surface_pairs": same_surface,
            "same_surface_rate_source": f"TRAIN {base_same_surface}/{base_pair_count}",
            "persona_seed_range": list(row["persona_seed_range"]),
            "scenario_seed_range": list(row["scenario_seed_range"]),
            "template_family_source": "frozen U2-F TRAIN plan; no template redefinition in reserved plan",
            "reserved_compositions": list(row.get("reserved_compositions", ())),
        }
    return plans, profile, derivation


def materialize_reserved(
    *, repository_root: Path, output_root: Path | None = None,
) -> dict[str, Any]:
    """Create all six pools exactly once; truth remains segregated from runtime files."""
    repository_root = repository_root.resolve()
    output_root = (output_root or repository_root / RUN_ROOT_RELATIVE / "reserved").resolve()
    _method_freeze, method_freeze_commit = _method_freeze_commit(repository_root)
    if output_root.exists():
        raise FileExistsError(f"reserved materialization is one-shot: {output_root}")

    u2f_root = repository_root / U2F_ROOT_RELATIVE
    source_manifest_path = u2f_root / "manifest.json"
    source_manifest = read_json(source_manifest_path)
    if (
        sha256_file(source_manifest_path) != LOCKED_U2F_MANIFEST_SHA256
        or source_manifest.get("dataset_root_hash") != U2F_DATASET_ROOT_SHA256
        or source_manifest.get("reserved_evaluation_rows_materialized") is not False
    ):
        raise ValueError("frozen U2-F source manifest identity/state mismatch")
    plan_path = u2f_root / "reserved_test_plan.json"
    plan_sha = sha256_file(plan_path)
    if (
        plan_sha != PLAN_SHA256
        or source_manifest["artifact_sha256"].get("reserved_test_plan.json") != plan_sha
    ):
        raise ValueError("reserved plan SHA differs from its frozen U2-F source manifest")
    reserved = read_json(plan_path)
    validate_reserved_plan(reserved)
    profile_path = u2f_root / "scale_profile.json"
    plans_path = u2f_root / "generation_plans.json"
    if sha256_file(profile_path) != source_manifest["artifact_sha256"]["scale_profile.json"]:
        raise ValueError("frozen U2-F scale profile hash mismatch")
    if sha256_file(plans_path) != source_manifest["artifact_sha256"]["generation_plans.json"]:
        raise ValueError("frozen U2-F plan family hash mismatch")
    source_profile = read_json(profile_path)
    source_plans = json_load_list(plans_path)
    generator_plans, profile, plan_derivation = _derive_generator_plans(
        reserved, source_profile, source_plans
    )

    # This is the first operation that creates reserved rows. It is guarded by the
    # committed METHOD FREEZE above and consumes only the unmaterialized seed plan.
    worlds, _timelines = build_u2f_worlds(
        plans=tuple(generator_plans),
        profile=profile,
        run_seed=int(profile["global_seed"]),
    )
    cases = tuple(materialize(world) for world in worlds)
    world_ids = {world.world_id.removeprefix("LW-") for world in worlds}
    if len(world_ids) != len(worlds) or len(cases) != len(worlds):
        raise ValueError("reserved generator produced duplicate or missing episode identities")

    output_root.mkdir(parents=True, exist_ok=False)
    runtime_hashes: dict[str, dict[str, str]] = {}
    truth_hashes: dict[str, dict[str, str]] = {}
    pool_summary: dict[str, dict[str, Any]] = {}
    all_episode_ids: list[str] = []
    for role in POOL_SEEDS:
        role_cases = [case for case in cases if case.scenario.world.split_role == role]
        role_cases.sort(key=lambda case: case.episode.episode_id)
        expected_episodes, expected_subjects, _ = pool_counts(role)
        subjects = {case.episode.subject_id for case in role_cases}
        if len(role_cases) != expected_episodes or len(subjects) != expected_subjects:
            raise ValueError(f"materialized {role} counts differ from seed-range derivation")

        runtime_rows: list[dict[str, Any]] = []
        truth_rows: list[dict[str, Any]] = []
        latent_rows: list[dict[str, Any]] = []
        corpus_rows: list[dict[str, Any]] = []
        for case in role_cases:
            world = case.scenario.world
            runtime_row = case.episode.to_runtime_dict()
            if runtime_row["episode_id"] != case.scenario.episode_id:
                raise ValueError("runtime episode ID differs from its scenario")
            runtime_rows.append(runtime_row)
            truth_row = case.scenario.evaluator_truth_dict()
            truth_row.update({
                "split_role": role,
                "scenario_family": world.scenario_family,
                "template_family": world.template_family,
                "counterfactual_family_id": world.counterfactual_family_id,
                "persona_seed": world.persona_seed,
                "scenario_seed": world.scenario_seed,
                "history_regime": world.history_regime,
                "structural_metadata": dict(world.structural_metadata),
            })
            truth_rows.append(truth_row)
            latent_rows.append(world.to_dict())

            episode = E6Episode(
                episode_id=runtime_row["episode_id"],
                subject_id=str(runtime_row["subject_id"]),
                partition=role,
                split=role,
                query=str(runtime_row["query"]),
                query_sha256=sha256_bytes(str(runtime_row["query"]).encode("utf-8")),
                decision_time=datetime.fromisoformat(runtime_row["decision_time"]),
                available_source_families=tuple(sorted(
                    runtime_row["observable_state"]["available_external_source_families"]
                )),
            )
            documents = OwnedExternalCorpusAdapter.documents_for_episode(episode, world.to_dict())
            corpus_rows.append({
                "episode_id": episode.episode_id,
                "split": role,
                "documents": [document.to_dict() for document in documents],
            })

        runtime_bytes = _jsonl(runtime_rows)
        corpus_bytes = _jsonl(corpus_rows)
        truth_bytes = _jsonl(truth_rows)
        latent_bytes = _jsonl(latent_rows)
        pool_runtime_dir = output_root / "runtime" / role
        pool_truth_dir = output_root / "evaluator_only" / role
        _write_new(pool_runtime_dir / "episodes.jsonl", runtime_bytes)
        _write_new(pool_runtime_dir / "runtime_corpus.jsonl", corpus_bytes)
        corpus_manifest = {
            "schema_version": "rag-e6b-runtime-corpus-v1",
            "pool": role,
            "episode_count": len(runtime_rows),
            "subject_count": len(subjects),
            "runtime_corpus_sha256": sha256_bytes(corpus_bytes),
            "evaluator_truth_opened": False,
        }
        _write_new(
            pool_runtime_dir / "runtime_corpus_manifest.json",
            canonical_bytes(corpus_manifest) + b"\n",
        )
        _write_new(pool_truth_dir / "evaluator_truth.jsonl", truth_bytes)
        _write_new(pool_truth_dir / "latent_worlds.jsonl", latent_bytes)
        runtime_hashes[role] = {
            "episodes_sha256": sha256_bytes(runtime_bytes),
            "runtime_corpus_sha256": sha256_bytes(corpus_bytes),
            "runtime_corpus_manifest_sha256": sha256_bytes(
                canonical_bytes(corpus_manifest) + b"\n"
            ),
        }
        truth_hashes[role] = {
            "evaluator_truth_sha256": sha256_bytes(truth_bytes),
            "latent_worlds_sha256": sha256_bytes(latent_bytes),
        }
        role_ids = [str(row["episode_id"]) for row in runtime_rows]
        all_episode_ids.extend(role_ids)
        pool_summary[role] = {
            "episodes": len(role_ids),
            "subjects": len(subjects),
            "capability_blind_episode_ids": role_ids,
            "persona_seeds": list(range(*_inclusive_range(POOL_SEEDS[role]["persona"]))),
            "scenario_seeds": list(range(*_inclusive_range(POOL_SEEDS[role]["scenario"]))),
            "visible_document_count": sum(len(row["documents"]) for row in corpus_rows),
            "reserved_compositions": list(
                RESERVED_COMPOSITIONS if role == "OOD_COMPOSITION" else ()
            ),
        }

    if len(all_episode_ids) != len(set(all_episode_ids)):
        raise ValueError("reserved episode IDs collide across frozen pools")
    source_files = {
        "manifest.json": sha256_file(source_manifest_path),
        "reserved_test_plan.json": plan_sha,
        "scale_profile.json": sha256_file(profile_path),
        "generation_plans.json": sha256_file(plans_path),
        "train_episodes.jsonl": sha256_file(u2f_root / "train" / "episodes.jsonl"),
    }
    generator_hashes = {
        relative: sha256_file(repository_root / relative) for relative in GENERATOR_FILES
    }
    manifest = {
        "schema_version": "rag-e6b-reserved-materialization-v1",
        "method_freeze_commit": method_freeze_commit,
        "method_freeze_sha256": sha256_file(
            repository_root / RUN_ROOT_RELATIVE / "method_freeze.json"
        ),
        "reserved_plan_sha256": plan_sha,
        "dataset_id": source_manifest["dataset_id"],
        "dataset_root_sha256": source_manifest["dataset_root_hash"],
        "source_hashes": source_files,
        "generator_code_sha256": generator_hashes,
        "pool_names": list(POOL_SEEDS),
        "pool_counts": pool_summary,
        "actual_subjects_per_pool": {name: item["subjects"] for name, item in pool_summary.items()},
        "actual_episodes_per_pool": {name: item["episodes"] for name, item in pool_summary.items()},
        "generation_plan_derivation": plan_derivation,
        "generator_semantics": {
            "source": "pinned in-repository U2-F build_u2f_worlds + materialize",
            "scenario_family_weights": "copied unchanged from frozen scale_profile.json",
            "template_families": "frozen TRAIN template set, since reserved plan defines no alternate templates",
            "same_surface_pair_rate": "frozen TRAIN ratio carried by nearest-integer pair count per pool",
            "ood_pool_definition": "disjoint persona/scenario seed pools exactly as named in the frozen plan",
            "composition_plan_note": (
                "The frozen plan lists four OOD_COMPOSITION themes but does not assign per-episode quotas; "
                "the generator distribution is not altered. Composition families are descriptive post-score slices."
            ),
            "seeds_changed": False,
            "capability_based_filtering": False,
        },
        "runtime_corpus_hashes": runtime_hashes,
        "truth_hashes": truth_hashes,
        "capability_blind_episode_ids": all_episode_ids,
        "evaluator_truth_opened": False,
        "score_started": False,
        "reserved_rows_materialized": True,
        "pool_subject_id_union_count": len({
            row["subject_id"] for role in POOL_SEEDS
            for row in _read_jsonl(output_root / "runtime" / role / "episodes.jsonl")
        }),
    }
    _write_new(output_root / "reserved_materialization_manifest.json", canonical_bytes(manifest) + b"\n")
    return manifest


def _inclusive_range(bounds: tuple[int, int]) -> tuple[int, int]:
    return bounds[0], bounds[1] + 1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_load_list(path: Path) -> list[dict[str, Any]]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise TypeError(f"expected JSON object array: {path}")
    return value
