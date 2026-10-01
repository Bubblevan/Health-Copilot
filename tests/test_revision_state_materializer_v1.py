from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from tools.research.memory.revision_state_materializer_v1 import (
    RevisionAdmission,
    TemporalProposition,
    materialize,
)


SLOT = ("scope-a", "SELF", "RUNNING_PLAN", "PREFERRED_EXERCISE")


def _obs(
    memory_id: str,
    value: str | None,
    at: str,
    *,
    source: str | None = None,
    cardinality: str = "SINGLE_VALUE_AT_A_TIME",
    temporal_basis: str = "UNSPECIFIED",
    explicit_valid_at: str | None = None,
    action: str = "ASSERT",
    delete_evidence: str | None = None,
    deleted_value: str | None = None,
    scope: str = "scope-a",
    owner: str | None = "SELF",
    object_id: str | None = "RUNNING_PLAN",
    attribute: str | None = "PREFERRED_EXERCISE",
) -> TemporalProposition:
    text = source or (f"I prefer {value}." if value is not None else "I no longer use running.")
    return TemporalProposition(
        memory_id=memory_id,
        scope_id=scope,
        owner_id=owner,
        object_id=object_id,
        attribute_id=attribute,
        value_text=value,
        observed_at=datetime.fromisoformat(at.replace("Z", "+00:00")),
        source_text=text,
        source_session_id=f"session-{memory_id}",
        source_turn_ids=(f"turn-{memory_id}",),
        provenance_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        cardinality=cardinality,  # type: ignore[arg-type]
        temporal_basis=temporal_basis,  # type: ignore[arg-type]
        explicit_valid_at=(
            datetime.fromisoformat(explicit_valid_at.replace("Z", "+00:00"))
            if explicit_valid_at else None
        ),
        action=action,  # type: ignore[arg-type]
        delete_evidence_span=delete_evidence,
        deleted_value_text=deleted_value,
    )


def _revision(previous: str, current: str, at: str) -> RevisionAdmission:
    proof = f"validated:{previous}->{current}:{at}"
    return RevisionAdmission(
        previous_memory_id=previous,
        current_memory_id=current,
        effective_at=datetime.fromisoformat(at.replace("Z", "+00:00")),
        evidence_sha256=hashlib.sha256(proof.encode("utf-8")).hexdigest(),
    )


def test_update_closes_prior_interval_and_supports_current_as_of_change() -> None:
    old = _obs("m1", "running", "2026-01-01T00:00:00Z")
    new = _obs("m2", "swimming", "2026-03-01T00:00:00Z")
    result = materialize([new, old], [_revision("m1", "m2", "2026-03-01T00:00:00Z")])

    current = result.current(SLOT)
    assert [row.value_text for row in current] == ["swimming"]
    assert current[0].supersedes_id
    assert result.as_of(SLOT, datetime(2026, 2, 1, tzinfo=timezone.utc))[0].value_text == "running"
    assert result.as_of(SLOT, datetime(2026, 3, 1, tzinfo=timezone.utc))[0].value_text == "swimming"
    chain = result.change_chain(SLOT)
    assert [(row.value_text, row.status) for row in chain] == [
        ("running", "SUPERSEDED"),
        ("swimming", "ACTIVE"),
    ]
    assert chain[0].valid_until == "2026-03-01T00:00:00Z"


def test_different_singleton_values_without_admission_remain_visible_as_conflict() -> None:
    old = _obs(
        "m1", "500 followers", "2026-01-01T00:00:00Z",
        source="I have 500 followers.", object_id="INSTAGRAM_ACCOUNT", attribute="FOLLOWER_COUNT",
    )
    new = _obs(
        "m2", "600 followers", "2026-03-01T00:00:00Z",
        source="I have 600 followers.", object_id="INSTAGRAM_ACCOUNT", attribute="FOLLOWER_COUNT",
    )
    slot = ("scope-a", "SELF", "INSTAGRAM_ACCOUNT", "FOLLOWER_COUNT")
    result = materialize([old, new])

    assert {row.value_text for row in result.current(slot)} == {"500 followers", "600 followers"}
    assert all(row.status == "CONFLICT" for row in result.current(slot))
    assert len(result.as_of(slot, datetime(2026, 2, 1, tzinfo=timezone.utc))) == 1
    assert len(result.as_of(slot, datetime(2026, 4, 1, tzinfo=timezone.utc))) == 2


def test_current_singleton_snapshot_advances_state_without_free_form_update_label() -> None:
    slot = ("scope-a", "SELF", "INSTAGRAM_ACCOUNT", "FOLLOWER_COUNT")
    old = _obs(
        "m1", "500 followers", "2026-01-01T00:00:00Z",
        source="I currently have 500 followers.", temporal_basis="CURRENT_SNAPSHOT",
        object_id="INSTAGRAM_ACCOUNT", attribute="FOLLOWER_COUNT",
    )
    new = _obs(
        "m2", "600 followers", "2026-03-01T00:00:00Z",
        source="I currently have 600 followers.", temporal_basis="CURRENT_SNAPSHOT",
        object_id="INSTAGRAM_ACCOUNT", attribute="FOLLOWER_COUNT",
    )
    result = materialize([new, old])

    assert [row.value_text for row in result.current(slot)] == ["600 followers"]
    assert result.as_of(slot, datetime(2026, 2, 1, tzinfo=timezone.utc))[0].value_text == "500 followers"
    assert result.change_chain(slot)[0].status == "SUPERSEDED"
    assert result.change_chain(slot)[1].supersedes_id == result.change_chain(slot)[0].record_id


def test_explicit_historical_observation_is_ordered_by_valid_time_not_ingestion_time() -> None:
    slot = ("scope-a", "SELF", "INSTAGRAM_ACCOUNT", "FOLLOWER_COUNT")
    historical = _obs(
        "m1", "500 followers", "2026-03-01T00:00:00Z",
        source="On January 1, I had 500 followers.", temporal_basis="EXPLICIT_AS_OF",
        explicit_valid_at="2026-01-01T00:00:00Z",
        object_id="INSTAGRAM_ACCOUNT", attribute="FOLLOWER_COUNT",
    )
    current = _obs(
        "m2", "600 followers", "2026-03-02T00:00:00Z",
        source="I currently have 600 followers.", temporal_basis="CURRENT_SNAPSHOT",
        object_id="INSTAGRAM_ACCOUNT", attribute="FOLLOWER_COUNT",
    )
    result = materialize([current, historical])

    assert [row.value_text for row in result.current(slot)] == ["600 followers"]
    assert result.as_of(slot, datetime(2026, 2, 1, tzinfo=timezone.utc))[0].value_text == "500 followers"


def test_same_fact_adds_provenance_without_spurious_revision() -> None:
    first = _obs("m1", "running", "2026-01-01T00:00:00Z")
    repeat = _obs("m2", "running", "2026-02-01T00:00:00Z")
    result = materialize([first, repeat])

    assert len(result.records) == 1
    assert result.records[0].status == "ACTIVE"
    assert result.records[0].source_memory_ids == ("m1", "m2")
    assert result.records[0].supersedes_id is None


def test_delete_suppresses_current_but_preserves_historical_state_and_tombstone() -> None:
    old = _obs("m1", "running", "2026-01-01T00:00:00Z")
    tombstone = _obs(
        "m2", None, "2026-03-01T00:00:00Z",
        source="I no longer use running.",
        action="DELETE",
        delete_evidence="I no longer use running",
        deleted_value="running",
    )
    result = materialize([old, tombstone])

    assert result.current(SLOT) == ()
    assert result.as_of(SLOT, datetime(2026, 2, 1, tzinfo=timezone.utc))[0].value_text == "running"
    assert result.as_of(SLOT, datetime(2026, 4, 1, tzinfo=timezone.utc)) == ()
    assert [row.status for row in result.change_chain(SLOT)] == ["SUPERSEDED", "DELETED"]


def test_delete_requires_a_source_grounded_supported_cue() -> None:
    with pytest.raises(ValueError, match="supported first-person cue"):
        _obs(
            "m1", None, "2026-01-01T00:00:00Z",
            source="Please remove running from the list.",
            action="DELETE",
            delete_evidence="Please remove running",
            deleted_value="running",
        )


def test_delete_for_another_value_does_not_suppress_current_state() -> None:
    current = _obs("m1", "swimming", "2026-02-01T00:00:00Z")
    bad_delete = _obs(
        "m2", None, "2026-03-01T00:00:00Z",
        source="I no longer use running.", action="DELETE",
        delete_evidence="I no longer use running", deleted_value="running",
    )
    result = materialize([current, bad_delete])

    assert [row.value_text for row in result.current(SLOT)] == ["swimming"]
    assert any(row.status == "UNRESOLVED" for row in result.records)


def test_assertion_after_tombstone_starts_a_new_current_version() -> None:
    rows = [
        _obs("m1", "running", "2026-01-01T00:00:00Z"),
        _obs(
            "m2", None, "2026-02-01T00:00:00Z",
            source="I no longer use running.", action="DELETE",
            delete_evidence="I no longer use running",
            deleted_value="running",
        ),
        _obs("m3", "cycling", "2026-03-01T00:00:00Z"),
    ]
    result = materialize(rows)

    assert [row.value_text for row in result.current(SLOT)] == ["cycling"]
    assert result.current(SLOT)[0].supersedes_id == result.change_chain(SLOT)[1].record_id


def test_concurrent_values_coexist_without_revision_edges() -> None:
    rows = [
        _obs("m1", "hiking", "2026-01-01T00:00:00Z", cardinality="MULTI_VALUE_CONCURRENT"),
        _obs("m2", "reading", "2026-01-02T00:00:00Z", cardinality="MULTI_VALUE_CONCURRENT"),
    ]
    result = materialize(rows)

    assert {row.value_text for row in result.current(SLOT)} == {"hiking", "reading"}
    assert all(row.supersedes_id is None for row in result.current(SLOT))


def test_distinct_scopes_and_owners_never_share_state() -> None:
    rows = [
        _obs("m1", "running", "2026-01-01T00:00:00Z"),
        _obs("m2", "swimming", "2026-03-01T00:00:00Z", scope="scope-b"),
        _obs("m3", "cycling", "2026-04-01T00:00:00Z", owner="SISTER"),
    ]
    result = materialize(rows)

    assert len(result.current(SLOT)) == 1
    assert result.current(("scope-b", "SELF", "RUNNING_PLAN", "PREFERRED_EXERCISE"))[0].value_text == "swimming"
    assert result.current(("scope-a", "SISTER", "RUNNING_PLAN", "PREFERRED_EXERCISE"))[0].value_text == "cycling"


def test_same_time_conflicting_values_are_exposed_not_arbitrarily_ordered() -> None:
    rows = [
        _obs("m1", "running", "2026-01-01T00:00:00Z"),
        _obs("m2", "swimming", "2026-01-01T00:00:00Z"),
    ]
    result = materialize(rows)

    current = result.current(SLOT)
    assert {row.value_text for row in current} == {"running", "swimming"}
    assert all(row.status == "CONFLICT" for row in current)


def test_event_and_unknown_identity_do_not_enter_current_state() -> None:
    event = _obs("m1", "completed purchase", "2026-01-01T00:00:00Z", cardinality="EVENT_OR_NOT_STATE")
    unresolved = _obs("m2", "42", "2026-01-02T00:00:00Z", cardinality="UNKNOWN", owner=None)
    result = materialize([event, unresolved])

    assert result.current(SLOT) == ()
    assert [(row.status, row.reason) for row in result.records] == [
        ("EVENT", "event_or_non_state_excluded_from_revision_state"),
        ("UNRESOLVED", "unknown_cardinality_or_incomplete_slot_identity"),
    ]


def test_cardinality_disagreement_fails_closed_for_the_entire_slot() -> None:
    rows = [
        _obs("m1", "hiking", "2026-01-01T00:00:00Z"),
        _obs("m2", "reading", "2026-01-02T00:00:00Z", cardinality="MULTI_VALUE_CONCURRENT"),
    ]
    result = materialize(rows)

    assert result.current(SLOT) == ()
    assert all(row.status == "UNRESOLVED" for row in result.records)


def test_replay_is_order_independent_and_hash_stable() -> None:
    rows = [
        _obs("m1", "running", "2026-01-01T00:00:00Z"),
        _obs("m2", "swimming", "2026-03-01T00:00:00Z"),
        _obs("m3", "swimming", "2026-04-01T00:00:00Z"),
    ]
    first = materialize(rows)
    replay = materialize(list(reversed(rows)))

    assert first == replay
    assert first.input_sha256 == replay.input_sha256
    assert first.output_sha256 == replay.output_sha256


def test_memory_id_collision_with_different_content_fails_closed() -> None:
    first = _obs("m1", "running", "2026-01-01T00:00:00Z")
    different = _obs("m1", "swimming", "2026-03-01T00:00:00Z")

    with pytest.raises(ValueError, match="memory_id collision"):
        materialize([first, different])


def test_naive_time_is_rejected_without_reading_system_clock() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _obs("m1", "running", "2026-01-01T00:00:00")
