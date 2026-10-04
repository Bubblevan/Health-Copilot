"""Post-freeze materialization of the reserved owned-universe MA-MVP2 test."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ma_mvp2_protocol import (
    RESERVED_TEST_RELATIVE,
    code_identity,
    file_sha256,
    prompt_hashes,
    validate_frozen_config,
)

from health_ai_copilot.multi_agent.data import (
    DATASET_ID,
    DATASET_ROOT_HASH,
    EvaluationRecord,
    _assert_unprivileged_runtime,
)
from health_ai_copilot.research.integration.owned_universe.realization import (
    materialize,
)
from health_ai_copilot.research.integration.owned_universe.scale import (
    STRUCTURAL_TEMPLATE_FAMILIES,
    TRAIN_TEMPLATE_FAMILIES,
    build_u2f_worlds,
)

U2F_PLAN_PATH = ROOT / "benchmarks" / "integration_owned_v1" / "u2f_reserved_test_plan.json"
PROFILE_PATH = ROOT / "benchmarks" / "integration_owned_v1" / "u2f_scale_profile.json"


def materialize_reserved_test(
    *, run_dir: Path, config_path: Path | None = None,
    repository_root: Path = ROOT,
) -> tuple[EvaluationRecord, ...]:
    """Generate test episodes and evaluator truth only after config freeze validates."""
    config_path = config_path or run_dir / "system_config.json"
    if not config_path.is_file():
        raise FileNotFoundError("MA_MVP2_FROZEN_CONFIG_REQUIRED_BEFORE_TEST_MATERIALIZATION")
    config = _read_json(config_path)
    validate_frozen_config(config)
    current_code = code_identity(repository_root)
    if current_code["code_sha256"] != config.get("code", {}).get("sha256"):
        raise ValueError("MA_MVP2_CODE_CHANGED_AFTER_FREEZE")
    if prompt_hashes() != config.get("prompts", {}).get("sha256"):
        raise ValueError("MA_MVP2_PROMPT_CHANGED_AFTER_FREEZE")

    plan_path = repository_root / RESERVED_TEST_RELATIVE
    if file_sha256(plan_path) != config.get("reserved_test", {}).get("plan_sha256"):
        raise ValueError("MA_MVP2_RESERVED_PLAN_CHANGED_AFTER_FREEZE")
    plan = _read_json(plan_path)
    if plan.get("materialized") is not False or plan.get("episode_count") != 1024:
        raise ValueError("MA_MVP2_RESERVED_PLAN_INVALID")
    reserved_pools = _read_json(U2F_PLAN_PATH)["pools"]
    reserved_by_role = {str(item["split_role"]): item for item in reserved_pools}
    profile = _read_json(PROFILE_PATH)
    test_profile = deepcopy(profile)
    plans: list[dict[str, Any]] = []
    allocations: dict[str, int] = {}
    for pool in plan.get("pools", []):
        role = str(pool["split_role"])
        if role not in reserved_by_role:
            raise ValueError("MA_MVP2_TEST_ROLE_NOT_RESERVED")
        source_pool = reserved_by_role[role]
        count = int(pool["episode_count"])
        subjects = int(pool["subject_count"])
        if count % 2 or subjects <= 0 or subjects > count:
            raise ValueError("MA_MVP2_TEST_POOL_SIZE_INVALID")
        pair_count = count // 2
        p_start, p_end = (int(value) for value in pool["persona_seed_range"])
        s_start, s_end = (int(value) for value in pool["scenario_seed_range"])
        source_p_start, source_p_end = (int(value) for value in source_pool["persona_seed_range"])
        source_s_start, source_s_end = (int(value) for value in source_pool["scenario_seed_range"])
        if (p_start < source_p_start or p_end > source_p_end
                or s_start < source_s_start or s_end > source_s_end
                or p_end - p_start + 1 != subjects
                or s_end - s_start + 1 != pair_count):
            raise ValueError("MA_MVP2_TEST_SEED_SELECTION_OUTSIDE_RESERVED_POOL")
        templates = tuple(str(value) for value in pool.get(
            "template_families",
            TRAIN_TEMPLATE_FAMILIES if role == "IID_TEST" else STRUCTURAL_TEMPLATE_FAMILIES,
        ))
        same_surface_pairs = round(pair_count * float(plan["same_surface_pair_fraction"]))
        plans.append({
            "schema_version": "u2f-split-plan-v1",
            "split_role": role,
            "episode_count": count,
            "subject_count": subjects,
            "persona_seed_range": [p_start, p_end],
            "scenario_seed_range": [s_start, s_end],
            "template_families": list(templates),
            "same_surface_pair_count": same_surface_pairs,
            "iid_lexical_augmentation": role == "IID_TEST",
            **({"scenario_family_weights": pool["scenario_family_weights"]}
               if "scenario_family_weights" in pool else {}),
        })
        test_profile["targets"][role] = {
            "episode_count": count,
            "subject_count": subjects,
        }
        test_profile["same_surface_pair_count"][role] = same_surface_pairs
        allocations[role] = count
    if sum(allocations.values()) != int(plan["episode_count"]):
        raise ValueError("MA_MVP2_RESERVED_TOTAL_COUNT_MISMATCH")
    worlds, _timelines = build_u2f_worlds(
        plans=tuple(plans),
        profile=test_profile,
        run_seed=int(test_profile["global_seed"]),
    )
    cases = tuple(materialize(world) for world in worlds)
    if len(cases) != int(plan["episode_count"]):
        raise ValueError("MA_MVP2_RESERVED_CASE_COUNT_MISMATCH")
    cases = tuple(sorted(cases, key=lambda item: (item.scenario.world.split_role,
                                                  item.episode.episode_id)))
    runtime_rows = [case.episode.to_runtime_dict() for case in cases]
    for row in runtime_rows:
        _assert_unprivileged_runtime(row)
    truth_rows = [case.scenario.evaluator_truth_dict() for case in cases]
    role_counts = Counter(case.scenario.world.split_role for case in cases)
    plan_sha = file_sha256(plan_path)
    runtime_sha = _jsonl_sha256(runtime_rows)
    truth_sha = _jsonl_sha256(truth_rows)
    manifest = {
        "schema_version": "ma-mvp2-reserved-materialization-v1",
        "dataset_id": DATASET_ID,
        "dataset_root_sha256": DATASET_ROOT_HASH,
        "config_version": config["config_version"],
        "config_sha256": config["config_sha256"],
        "code_sha256": config["code"]["sha256"],
        "reserved_plan_sha256": plan_sha,
        "reserved_test_materialized": True,
        "evaluator_truth_written_after_freeze": True,
        "episode_count": len(cases),
        "episode_counts_by_role": dict(sorted(role_counts.items())),
        "runtime_episodes_sha256": runtime_sha,
        "evaluator_truth_sha256": truth_sha,
        "lineage_status": "seed pools are disjoint from TRAIN and DEV",
    }
    known_files = {
        "episodes.jsonl", "evaluator_truth.jsonl", "materialization_manifest.json",
    }
    if run_dir.exists():
        unknown = {path.name for path in run_dir.iterdir()} - known_files
        if unknown:
            raise FileExistsError(
                f"MA_MVP2_RUN_DIR_UNEXPECTED_ARTIFACTS:{','.join(sorted(unknown))}"
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "materialization_manifest.json"
    if manifest_path.is_file():
        existing_manifest = _read_json(manifest_path)
        if (existing_manifest.get("config_sha256") != config["config_sha256"]
                or existing_manifest.get("code_sha256") != config["code"]["sha256"]
                or existing_manifest.get("reserved_plan_sha256") != plan_sha):
            raise ValueError("MA_MVP2_EXISTING_MATERIALIZATION_IDENTITY_MISMATCH")
    files_match = all(
        (run_dir / filename).is_file() and file_sha256(run_dir / filename) == expected_sha
        for filename, expected_sha in (
            ("episodes.jsonl", runtime_sha), ("evaluator_truth.jsonl", truth_sha),
        )
    )
    if not files_match:
        _write_jsonl(run_dir / "episodes.jsonl", runtime_rows)
        _write_jsonl(run_dir / "evaluator_truth.jsonl", truth_rows)
    _write_json(run_dir / "materialization_manifest.json", manifest)
    truth_by_id = {str(item["episode_id"]): item for item in truth_rows}
    return tuple(
        EvaluationRecord(
            split=case.scenario.world.split_role,
            case=case,
            runtime_episode=case.episode.to_runtime_dict(),
            truth=truth_by_id[case.episode.episode_id],
        )
        for case in cases
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")))
            stream.write("\n")


def _jsonl_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        digest.update(payload + b"\n")
    return digest.hexdigest()


__all__ = ["materialize_reserved_test"]
