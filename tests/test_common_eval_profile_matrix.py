import json
from pathlib import Path

import pytest

from health_ai_copilot.evaluation.factorial import summarize_core_factorial


def test_profile_aliases_resolve_only_in_registry_and_keep_memory_off() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/eval/profile_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))["profiles"]
    assert set(registry) == {f"B{i}" for i in range(4)} | {f"P{i}" for i in range(4)} | {f"R{i}" for i in range(4)}
    assert all(item["memory_mode"] == "off" for item in registry.values())


def test_core_factorial_reports_paired_effects_and_interaction() -> None:
    report = summarize_core_factorial({
        "B0": {"a": False, "b": False, "c": True, "d": True},
        "B1": {"a": True, "b": False, "c": True, "d": True},
        "B2": {"a": False, "b": True, "c": True, "d": True},
        "B3": {"a": True, "b": True, "c": True, "d": True},
    })
    assert report["case_count"] == 4
    assert report["paired_comparisons"]["rag_without_adaptive"]["delta_pp"] == 25.0
    assert report["paired_comparisons"]["adaptive_without_rag"]["delta_pp"] == 25.0
    assert report["factor_effects_pp"]["rag_adaptive_interaction"] == 0.0
    assert "no significance test" in report["interpretation"]


def test_core_factorial_rejects_nonpaired_arms() -> None:
    with pytest.raises(ValueError, match="same non-empty case set"):
        summarize_core_factorial({
            "B0": {"a": True},
            "B1": {"a": True},
            "B2": {"b": True},
            "B3": {"a": True},
        })
