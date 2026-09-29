"""Prospective split and OOD contracts; this module does not assign real rows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DatasetSplit(StrEnum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    IID_TEST = "IID_TEST"
    OOD_TEST = "OOD_TEST"
    EXTERNAL_TRANSFER = "EXTERNAL_TRANSFER"


class OODCondition(StrEnum):
    OOD_PATIENT = "OOD_PATIENT"
    OOD_TASK = "OOD_TASK"
    OOD_TEMPORAL = "OOD_TEMPORAL"
    OOD_SOURCE = "OOD_SOURCE"
    OOD_COMPOSITION = "OOD_COMPOSITION"


@dataclass(frozen=True)
class SplitAssignment:
    dataset_id: str
    record_id: str
    split: DatasetSplit | str
    provenance_permits_use: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", DatasetSplit(self.split))
        if self.dataset_id == "public_benchmark_evaluation" and self.split != DatasetSplit.EXTERNAL_TRANSFER:
            raise ValueError("public benchmark evaluation content defaults to EXTERNAL_TRANSFER")
        if not self.provenance_permits_use:
            raise ValueError("split assignment requires explicit provenance permission")
