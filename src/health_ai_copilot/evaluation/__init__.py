"""Common, system-level evaluation contracts and adapters."""

from .contracts import CaseScore, DatasetAdapter, EvalCase
from .datasets import CMBAdapter, CMBCommon1024Adapter, DatasetNotReady, DiagnosisArenaAdapter
from .runner import run_dataset, summarize_records

__all__ = [
    "CMBAdapter",
    "CMBCommon1024Adapter",
    "CaseScore",
    "DatasetAdapter",
    "DatasetNotReady",
    "DiagnosisArenaAdapter",
    "EvalCase",
    "run_dataset",
    "summarize_records",
]
