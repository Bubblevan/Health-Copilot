from dataclasses import FrozenInstanceError

import pytest

from health_ai_copilot.research.integration.actions import ACTION_BY_KEY, ActionKey
from health_ai_copilot.research.integration.dataset_policy import DatasetSplit, SplitAssignment
from health_ai_copilot.research.integration.fixtures import build_synthetic_cases


def test_episode_runtime_serialization_has_no_evaluation_gold() -> None:
    case = next(item for item in build_synthetic_cases() if item.case_id == "U1-MEM")
    payload = case.episode.to_runtime_dict()
    assert "gold_answer" not in payload
    assert "required_facts" not in payload
    assert "counterfactual_outcomes" not in payload
    assert set(case.episode.observable_state.observability_flags().values()) == {"YES"}
    with pytest.raises(FrozenInstanceError):
        case.episode.query = "changed"  # type: ignore[misc]


def test_capability_action_surface_is_frozen_to_eight_arms() -> None:
    assert set(ACTION_BY_KEY) == set(ActionKey)
    assert len(ACTION_BY_KEY) == 8
    assert {item.external_retrieval.value for item in ACTION_BY_KEY.values()} == {"OFF", "STANDARD"}
    assert all("PARALLEL" not in item.action_id and "HETERO" not in item.action_id
               for item in ACTION_BY_KEY.values())


def test_public_benchmark_evaluation_defaults_to_external_transfer() -> None:
    with pytest.raises(ValueError, match="EXTERNAL_TRANSFER"):
        SplitAssignment("public_benchmark_evaluation", "synthetic-id", DatasetSplit.TRAIN, True)
