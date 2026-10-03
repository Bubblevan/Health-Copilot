"""Leakage-separated loader for the frozen U2-F TRAIN_DEV and DEV splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..research.integration.owned_universe.realization import MaterializedCase, materialize
from ..research.integration.owned_universe.scale import build_u2f_worlds

DATASET_ID = "health-copilot-owned-longitudinal-v1"
DATASET_ROOT_HASH = "e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134"
RUN_RELATIVE = Path("runs/integration/u2f-owned-v1-55955b2eff38")
SPEC_ROOT_RELATIVE = Path("benchmarks/integration_owned_v1")
TRAIN_DEV_SIZE = 256
_FORBIDDEN_RUNTIME_KEYS = frozenset({
    "answer_fact_ids", "answer_values", "architecture_label", "architecture_requirement",
    "architecture_supervision", "capability_requirement_oracle", "counterfactual_family_id",
    "dependency_graph", "gold", "gold_answer", "history_regime", "latent_world_id",
    "optimal_architecture", "parent_latent_fact_ids", "required_capability",
    "required_evidence_ids", "required_external_evidence_ids", "required_fact_ids",
    "required_memory_facts", "required_memory_record_ids", "scenario_family",
    "scenario_seed", "split_role", "structural_evaluator_metadata", "template_family",
    "timeline_generation_id", "training_authorized", "training_target",
})


@dataclass(frozen=True)
class EvaluationRecord:
    split: str
    case: MaterializedCase
    runtime_episode: dict[str, Any]
    truth: dict[str, Any]

    @property
    def episode_id(self) -> str:
        return self.case.episode.episode_id

    @property
    def scenario_family(self) -> str:
        return self.case.scenario.world.scenario_family


def load_records(repository_root: Path, split: str) -> tuple[EvaluationRecord, ...]:
    """Load only TRAIN_DEV or DEV. Reserved TEST/OOD row files are never opened."""
    if split not in {"train_dev", "dev"}:
        raise ValueError("split must be train_dev or dev")
    root = repository_root / RUN_RELATIVE
    manifest = _read_json(root / "manifest.json")
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("OWNED_DATASET_ID_MISMATCH")
    if manifest.get("dataset_root_hash") != DATASET_ROOT_HASH:
        raise ValueError("OWNED_DATASET_ROOT_HASH_MISMATCH")
    if manifest.get("reserved_evaluation_rows_materialized") is not False:
        raise ValueError("RESERVED_ROWS_MUST_REMAIN_UNMATERIALIZED")
    if manifest.get("episode_counts_by_split") != {
        "DEV_IID": 512, "DEV_STRUCTURAL": 512, "TRAIN": 4096,
    }:
        raise ValueError("OWNED_DATASET_SPLIT_COUNTS_MISMATCH")

    if split == "train_dev":
        roles = ("TRAIN",)
        file_roles = (("TRAIN", "train"),)
    else:
        roles = ("DEV_IID", "DEV_STRUCTURAL")
        file_roles = (("DEV_IID", "dev_iid"), ("DEV_STRUCTURAL", "dev_structural"))

    plan_names = {
        "TRAIN": "u2f_train_plan.json",
        "DEV_IID": "u2f_dev_iid_plan.json",
        "DEV_STRUCTURAL": "u2f_dev_structural_plan.json",
    }
    spec_root = repository_root / SPEC_ROOT_RELATIVE
    plans = tuple(_read_json(spec_root / plan_names[role]) for role in roles)
    profile = _read_json(spec_root / "u2f_scale_profile.json")
    worlds, _timelines = build_u2f_worlds(
        plans=plans,
        profile=profile,
        run_seed=int(profile["global_seed"]),
    )
    cases_by_id = {materialized.episode.episode_id: materialized
                   for materialized in (materialize(world) for world in worlds)}
    if len(cases_by_id) != sum(
        4096 if role == "TRAIN" else 512 for role in roles
    ):
        raise ValueError("REGENERATED_OWNED_CASE_COUNT_MISMATCH")

    records: list[EvaluationRecord] = []
    for role, folder in file_roles:
        runtime_path = root / folder / "episodes.jsonl"
        truth_path = root / folder / "evaluator_truth.jsonl"
        _verify_artifact_hash(root, manifest, runtime_path)
        _verify_artifact_hash(root, manifest, truth_path)
        runtime_rows = _read_jsonl(runtime_path)
        truth_rows = _read_jsonl(truth_path)
        if len(runtime_rows) != (4096 if role == "TRAIN" else 512):
            raise ValueError(f"OWNED_RUNTIME_ROWS_COUNT_MISMATCH:{role}")
        if len(truth_rows) != len(runtime_rows):
            raise ValueError(f"OWNED_EVALUATOR_ROWS_COUNT_MISMATCH:{role}")
        truths = {str(row["episode_id"]): row for row in truth_rows}
        for runtime_row in runtime_rows:
            _assert_unprivileged_runtime(runtime_row)
            episode_id = str(runtime_row["episode_id"])
            case = cases_by_id.get(episode_id)
            truth = truths.get(episode_id)
            if case is None or truth is None:
                raise ValueError(f"OWNED_EPISODE_ID_UNMATCHED:{episode_id}")
            if _canonical_hash(runtime_row) != _canonical_hash(case.episode.to_runtime_dict()):
                raise ValueError(f"OWNED_RUNTIME_REGENERATION_MISMATCH:{episode_id}")
            records.append(EvaluationRecord(role, case, runtime_row, truth))

    if split == "train_dev":
        records.sort(key=lambda row: hashlib.sha256(
            f"MA_MVP1_TRAIN_DEV_V1|{row.episode_id}".encode()
        ).digest())
        records = records[:TRAIN_DEV_SIZE]
    if split == "dev":
        records.sort(key=lambda row: (row.split, row.episode_id))
    expected_count = TRAIN_DEV_SIZE if split == "train_dev" else 1024
    if len(records) != expected_count:
        raise ValueError(f"OWNED_SELECTED_ROWS_COUNT_MISMATCH:{split}")
    return tuple(records)


def _assert_unprivileged_runtime(row: dict[str, Any]) -> None:
    found: list[str] = []

    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                field = f"{path}.{key}" if path else key
                if key.casefold() in _FORBIDDEN_RUNTIME_KEYS:
                    found.append(field)
                visit(child, field)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(row)
    if found:
        raise ValueError("RUNTIME_ROW_CONTAINS_EVALUATOR_FIELDS:" + ",".join(found))


def _verify_artifact_hash(root: Path, manifest: dict[str, Any], path: Path) -> None:
    relative = path.relative_to(root).as_posix()
    expected = manifest.get("artifact_sha256", {}).get(relative)
    if not expected:
        raise ValueError(f"DATASET_ARTIFACT_HASH_MISSING:{relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"DATASET_ARTIFACT_HASH_MISMATCH:{relative}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
