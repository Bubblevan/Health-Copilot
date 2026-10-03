"""Versioned identities and gates shared by the MA-MVP2 research runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONFIG_VERSION = "MA_MVP2_CONFIG_V1"
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
DATASET_ROOT_HASH = "e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134"
MVP1_BASE_COMMIT = "4645735dcc8e9b6a66991e7ac421cdc77e184eea"
MVP1_COMPLEX_TEAM_SUCCESS = 57 / 248
MVP1_OVERALL_TEAM_FACT_COVERAGE = 0.6912
MVP1_EXACT_WORKER_SET = 371 / 1024
TRAIN_TUNE_SELECTION = "sha256(MA_MVP2_TRAIN_TUNE_V1|episode_id), first 512 of TRAIN"
RESERVED_TEST_RELATIVE = Path("benchmarks/integration_owned_v1/ma_mvp2_reserved_test_plan.json")

CODE_PATHS = (
    "pyproject.toml",
    "benchmarks/integration_owned_v1/ma_mvp2_reserved_test_plan.json",
    "benchmarks/integration_owned_v1/u2f_reserved_test_plan.json",
    "benchmarks/integration_owned_v1/u2f_scale_profile.json",
    "src/health_ai_copilot/multi_agent/__init__.py",
    "src/health_ai_copilot/multi_agent/contracts.py",
    "src/health_ai_copilot/multi_agent/data.py",
    "src/health_ai_copilot/multi_agent/evaluation.py",
    "src/health_ai_copilot/multi_agent/orchestration.py",
    "src/health_ai_copilot/multi_agent/providers.py",
    "src/health_ai_copilot/multi_agent/routing.py",
    "src/health_ai_copilot/multi_agent/runtime.py",
    "src/health_ai_copilot/multi_agent/shared_context.py",
    "src/health_ai_copilot/multi_agent/skills.py",
    "src/health_ai_copilot/routing/jev.py",
    "src/health_ai_copilot/research/integration/owned_universe/grammar.py",
    "src/health_ai_copilot/research/integration/owned_universe/realization.py",
    "src/health_ai_copilot/research/integration/owned_universe/scale.py",
    "src/health_ai_copilot/research/integration/owned_universe/schema.py",
    "tools/research/multi_agent/ma_mvp2_data.py",
    "tools/research/multi_agent/ma_mvp2_protocol.py",
    "tools/research/multi_agent/run_ma_mvp2.py",
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_identity(repository_root: Path) -> dict[str, Any]:
    file_hashes = {}
    for relative in CODE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"MA_MVP2_CODE_IDENTITY_FILE_MISSING:{relative}")
        file_hashes[relative] = file_sha256(path)
    return {
        "code_sha256": canonical_hash(file_hashes),
        "code_file_sha256": file_hashes,
    }


def prompt_hashes() -> dict[str, str]:
    import sys

    source = str(Path(__file__).resolve().parents[3] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from health_ai_copilot.multi_agent.routing import JEV_TRIAGE_QUESTIONS
    from health_ai_copilot.multi_agent.runtime import (
        _MVP2_COVERAGE_PROMPT,
        _MVP2_FINAL_PROMPT,
        _MVP2_LEAD_PLAN_PROMPT,
        _MVP2_ROLE_PROMPTS,
    )

    prompts = {
        "lead_planning": _MVP2_LEAD_PLAN_PROMPT,
        "coverage_judgment": _MVP2_COVERAGE_PROMPT,
        "finalizer": _MVP2_FINAL_PROMPT,
        "jev_triage_contract": json.dumps(
            JEV_TRIAGE_QUESTIONS, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        **{f"worker_{role.value}": prompt for role, prompt in _MVP2_ROLE_PROMPTS.items()},
    }
    return {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for name, value in sorted(prompts.items())
    }


def validate_frozen_config(payload: dict[str, Any]) -> None:
    if payload.get("config_version") != CONFIG_VERSION or payload.get("frozen") is not True:
        raise ValueError("MA_MVP2_CONFIG_NOT_FROZEN")
    expected = payload.get("config_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "config_sha256"}
    if expected != canonical_hash(unsigned):
        raise ValueError("MA_MVP2_CONFIG_HASH_MISMATCH")
    if payload.get("dataset", {}).get("root_sha256") != DATASET_ROOT_HASH:
        raise ValueError("MA_MVP2_DATASET_ROOT_HASH_MISMATCH")
    if payload.get("model", {}).get("artifact_sha256") != MODEL_SHA256:
        raise ValueError("MA_MVP2_MODEL_HASH_MISMATCH")
    if payload.get("reserved_test", {}).get("episode_count") != 1024:
        raise ValueError("MA_MVP2_RESERVED_TEST_PLAN_MISMATCH")


__all__ = [
    "CODE_PATHS",
    "CONFIG_VERSION",
    "DATASET_ROOT_HASH",
    "MODEL_SHA256",
    "MVP1_BASE_COMMIT",
    "MVP1_COMPLEX_TEAM_SUCCESS",
    "MVP1_EXACT_WORKER_SET",
    "MVP1_OVERALL_TEAM_FACT_COVERAGE",
    "RESERVED_TEST_RELATIVE",
    "TRAIN_TUNE_SELECTION",
    "canonical_hash",
    "code_identity",
    "file_sha256",
    "prompt_hashes",
    "validate_frozen_config",
]
