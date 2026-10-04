"""Orthogonal model, retrieval, memory, and reasoning choices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelVariant(StrEnum):
    QWEN3_8B_BASE = "qwen3_8b_base"
    MEDICAL_SFT = "medical_sft"
    MEDICAL_SFT_GSPO = "medical_sft_gspo"


class RetrievalMode(StrEnum):
    OFF = "off"
    STANDARD = "standard"


class MemoryMode(StrEnum):
    OFF = "off"
    READ = "read"


class ReasoningMode(StrEnum):
    SINGLE = "single"
    ADAPTIVE_MDT = "adaptive_mdt"


@dataclass(frozen=True)
class SystemProfile:
    profile_id: str
    model_variant: ModelVariant
    retrieval_mode: RetrievalMode
    memory_mode: MemoryMode
    reasoning_mode: ReasoningMode

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be empty")
        for name, enum_type in (
            ("model_variant", ModelVariant),
            ("retrieval_mode", RetrievalMode),
            ("memory_mode", MemoryMode),
            ("reasoning_mode", ReasoningMode),
        ):
            value = getattr(self, name)
            if not isinstance(value, enum_type):
                object.__setattr__(self, name, enum_type(value))

    def to_dict(self) -> dict[str, str]:
        return {
            "profile_id": self.profile_id,
            "model_variant": self.model_variant.value,
            "retrieval_mode": self.retrieval_mode.value,
            "memory_mode": self.memory_mode.value,
            "reasoning_mode": self.reasoning_mode.value,
        }


def profile_from_dict(value: dict[str, str]) -> SystemProfile:
    return SystemProfile(
        profile_id=value["profile_id"],
        model_variant=ModelVariant(value["model_variant"]),
        retrieval_mode=RetrievalMode(value["retrieval_mode"]),
        memory_mode=MemoryMode(value["memory_mode"]),
        reasoning_mode=ReasoningMode(value["reasoning_mode"]),
    )


# One product entry point. Static Single remains available only as an evaluation
# control; simple requests still take MDAgents' internal single-agent fast path.
DEFAULT_PRODUCT_PROFILE = SystemProfile(
    profile_id="product-adaptive-v1",
    model_variant=ModelVariant.QWEN3_8B_BASE,
    retrieval_mode=RetrievalMode.OFF,
    memory_mode=MemoryMode.OFF,
    reasoning_mode=ReasoningMode.ADAPTIVE_MDT,
)
