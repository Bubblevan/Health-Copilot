import importlib.util
import sys
from pathlib import Path

import pytest

_tools_path = Path(__file__).parents[1] / "tools" / "research" / "memory"
sys.path.insert(0, str(_tools_path))
_spec = importlib.util.spec_from_file_location(
    "mem1_local_judge_test", _tools_path / "run_mem1_local_judge.py"
)
local_judge = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = local_judge
_spec.loader.exec_module(local_judge)


@pytest.mark.parametrize(
    "category",
    [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
    ],
)
def test_local_judge_uses_category_specific_prompt_and_binary_contract(category):
    prompt = local_judge.build_longmemeval_judge_prompt(
        "What changed?", "The appointment moved to Friday.", "Friday", category, "q1"
    )

    assert "Answer yes or no only." in prompt
    assert "What changed?" in prompt
    assert "Friday" in prompt
    if category == "temporal-reasoning":
        assert "off-by-one errors" in prompt
    if category == "knowledge-update":
        assert "updated answer" in prompt
    if category == "single-session-preference":
        assert "rubric" in prompt.lower()


def test_local_judge_abstention_uses_id_suffix_not_category():
    prompt = local_judge.build_longmemeval_judge_prompt(
        "What is the unseen fact?", "The history does not say.", "None",
        "knowledge-update", "case_abs",
    )

    assert "unanswerable question" in prompt
    assert "Correct Answer:" not in prompt


def test_local_judge_refuses_unfrozen_prediction_artifact(tmp_path):
    with pytest.raises(RuntimeError, match="frozen predictions"):
        local_judge.run(tmp_path)
