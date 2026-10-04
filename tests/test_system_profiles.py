from health_ai_copilot.harness.profiles import profile_from_dict


def test_profile_axes_are_independent_values() -> None:
    profile = profile_from_dict({
        "profile_id": "B3", "model_variant": "qwen3_8b_base",
        "retrieval_mode": "standard", "memory_mode": "off", "reasoning_mode": "adaptive_mdt",
    })
    assert profile.to_dict()["profile_id"] == "B3"
    assert profile.retrieval_mode.value == "standard"
    assert profile.memory_mode.value == "off"
