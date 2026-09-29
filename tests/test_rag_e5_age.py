from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from eval.rag_e5.age import (
    AGE_TEMPORAL_CONTRACT,
    AGE_TEMPORAL_CONTRACT_SHA256,
    age_bucket_at_decision,
)
from eval.rag_e5.longitudinal_state import build_longitudinal_state_packet
from eval.rag_e5.temporal import SourceRelativeDecisionBoundary

ROOT = Path(__file__).resolve().parents[1]


def test_profile_snapshot_age_not_used_directly() -> None:
    demographics = {"age": 63}
    boundary = SourceRelativeDecisionBoundary("user-a", "2026-06-01T12:00:00")

    assert age_bucket_at_decision(demographics, boundary) == "unknown"
    assert AGE_TEMPORAL_CONTRACT["excluded_dynamic_fields"] == ["profile.demographics.age"]
    assert len(AGE_TEMPORAL_CONTRACT_SHA256) == 64


def test_age_unknown_without_safe_birth_date() -> None:
    boundary = SourceRelativeDecisionBoundary("user-a", "2026-06-01T12:00:00")

    assert age_bucket_at_decision({"age": 40, "occupation": "x"}, boundary) == "unknown"


def test_age_computed_at_decision_if_birth_date_safe() -> None:
    boundary = SourceRelativeDecisionBoundary("user-a", "2026-06-01T12:00:00")

    assert age_bucket_at_decision({"date_of_birth": "1986-06-02"}, boundary) == "18_39"
    assert age_bucket_at_decision({"birth_date": "1986-06-01"}, boundary) == "40_59"


def test_future_profile_age_cannot_change_old_state_packet() -> None:
    boundary = SourceRelativeDecisionBoundary("user-a", "2026-06-01T12:00:00")
    timeline = {
        "user_id": "user-a",
        "entries": [
            {"time": "2026-05-31T12:00:00", "entry_type": "event", "event": "opaque"},
            {"time": boundary.naive_timestamp, "entry_type": "event", "event": "boundary"},
        ],
    }
    hashes = {"profile.json": "a" * 64, "timeline.json": "b" * 64, "exam_data.json": "c" * 64}

    def build(age: int):
        return build_longitudinal_state_packet(
            decision_boundary=boundary,
            profile={"demographics": {"age": age}},
            timeline=timeline,
            exam_data=[],
            source_file_hashes=hashes,
        )

    old_packet = build(40)
    future_updated_profile_packet = build(41)

    assert old_packet.age_bucket == "unknown"
    assert future_updated_profile_packet.age_bucket == "unknown"
    assert old_packet.packet_sha256 == future_updated_profile_packet.packet_sha256


def test_age_and_profile_temporal_manifests_match_canonical_contracts() -> None:
    age_manifest = json.loads((ROOT / "runs/rag_e5/age_temporal_contract.json").read_text())
    profile_manifest = json.loads((ROOT / "runs/rag_e5/profile_temporal_contract.json").read_text())

    def canonical_sha(value: object) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256(encoded.encode()).hexdigest()

    assert canonical_sha(age_manifest) == AGE_TEMPORAL_CONTRACT_SHA256
    assert age_manifest == AGE_TEMPORAL_CONTRACT
    assert profile_manifest["age_temporal_contract_sha256"] == AGE_TEMPORAL_CONTRACT_SHA256
