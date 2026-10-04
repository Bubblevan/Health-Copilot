"""Stable metric definitions for common system evaluation."""

from __future__ import annotations

from .runner import summarize_records

METRIC_DEFINITIONS = {
    "accuracy": "exactly correct cases divided by all evaluated cases; parse failures count incorrect",
    "parse_rate": "cases with a deterministic parsed answer divided by all evaluated cases",
    "macro_accuracy_by_category": "unweighted mean exact accuracy across observed dataset categories",
    "p95_latency_ms": "nearest-lower empirical 95th percentile of per-case wall latency",
    "provider_calls_per_case": "Harness-accounted provider calls averaged across all cases",
}

__all__ = ["METRIC_DEFINITIONS", "summarize_records"]
