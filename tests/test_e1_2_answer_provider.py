import pytest

from eval.e1_2_answer_provider import verify_served_model


def test_served_model_identity_must_match_the_frozen_path():
    verify_served_model(
        {"served_model": r"F:\Health-Copilot-E1.2\models\Qwen3-8B-Q4_K_M.gguf"},
        expected_model_id=r"F:\Health-Copilot-E1.2\models\Qwen3-8B-Q4_K_M.gguf",
    )


def test_served_model_identity_rejects_unexpected_model():
    with pytest.raises(RuntimeError, match="outside the frozen E1.2 identity"):
        verify_served_model(
            {"served_model": "some-other-model"},
            expected_model_id=r"F:\Health-Copilot-E1.2\models\Qwen3-8B-Q4_K_M.gguf",
        )
