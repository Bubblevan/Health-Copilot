from __future__ import annotations

import copy

import pytest

from tools.research.memory import revision_safety_overlay as safety


def _identity(
    memory_id: str,
    *,
    kind: str = "SINGLETON_STATE",
    subject: str = "user",
    attribute: str = "preferred_activity",
    origin: str = "MODEL_VALIDATED",
    value: str = "swimming",
) -> dict[str, str | None]:
    return {
        "memory_id": memory_id,
        "revision_kind": kind,
        "subject_key": subject,
        "attribute_key": attribute,
        "value_text": value,
        "identity_origin": origin,
        "fallback_reason": None,
    }


def _source(
    memory_id: str,
    *,
    authority: str = "user",
    scope: str = "scope-a",
    proposition: str = "The user prefers swimming.",
) -> dict[str, str]:
    return {
        "memory_id": memory_id,
        "source_authority": authority,
        "scope_id": scope,
        "proposition_text": proposition,
        "observed_at": "2030-01-01T00:00:00Z",
        "question_id": "benchmark-label-must-not-project",
        "gold": "must-not-project",
    }


def test_assistant_only_proposition_never_becomes_user_revision_eligible() -> None:
    gate = safety.derive_record_eligibility(
        _identity("a"), _source("a", authority="assistant")
    )
    assert gate["authority_namespace"] == "ASSISTANT_ORIGIN"
    assert gate["revision_eligibility"] == "INELIGIBLE_ASSISTANT_ORIGIN"


def test_user_proposition_may_be_eligible() -> None:
    gate = safety.derive_record_eligibility(_identity("u"), _source("u"))
    assert gate["authority_namespace"] == "USER_ASSERTED"
    assert gate["revision_eligibility"] == "ELIGIBLE_USER_STATE"


def test_mixed_proposition_may_be_eligible_but_verifier_still_required() -> None:
    gate = safety.derive_record_eligibility(
        _identity("m"), _source("m", authority="mixed")
    )
    assert gate["authority_namespace"] == "USER_MIXED"
    assert gate["revision_eligibility"] == "ELIGIBLE_USER_STATE"


@pytest.mark.parametrize(
    ("identity", "authority", "eligibility"),
    [
        (_identity("fallback", origin="HARNESS_UNKNOWN_FALLBACK"), "user", "REVISION_INELIGIBLE"),
        (_identity("unknown", kind="UNKNOWN"), "user", "INELIGIBLE_REVISION_KIND"),
        (_identity("set", kind="SET_STATE"), "user", "INELIGIBLE_REVISION_KIND"),
        (_identity("assistant-ns", subject="assistant/recommender"), "user", "INELIGIBLE_SUBJECT_NAMESPACE"),
    ],
)
def test_fallback_non_singleton_and_assistant_namespace_are_excluded(
    identity: dict[str, str | None], authority: str, eligibility: str
) -> None:
    assert safety.derive_record_eligibility(
        identity, _source(identity["memory_id"], authority=authority)
    )["revision_eligibility"] == eligibility


def test_exact_grouping_does_not_merge_different_scope_subject_or_attribute() -> None:
    identities = [
        _identity("a", attribute="workout_days"),
        _identity("b", attribute="gym_frequency_per_week"),
        _identity("c", attribute="workout_days"),
        _identity("d", subject="partner", attribute="workout_days"),
        _identity("e", attribute="workout_days"),
    ]
    sources = [_source(row["memory_id"], scope="scope-b" if row["memory_id"] == "c" else "scope-a") for row in identities]
    universe, stats = safety.build_candidate_universe(identities, sources)
    repeated = universe["repeated_candidate_groups"]
    assert len(repeated) == 1
    assert repeated[0]["member_memory_ids"] == ["a", "e"]
    assert stats["repeated_candidate_groups"] == 1


@pytest.mark.parametrize(
    ("raw", "finish_reason", "verdict", "failure"),
    [
        ('{"verdict":"COHERENT_SINGLETON_SLOT"}', "stop", "COHERENT_SINGLETON_SLOT", None),
        ('{"verdict":"INCOHERENT_GROUP"}', "stop", "INCOHERENT_GROUP", None),
        ('{"verdict":"UNKNOWN"}', "stop", "UNKNOWN", None),
        ('{"verdict":"COHERENT_SINGLETON_SLOT","reason":"extra"}', "stop", None, "UNEXPECTED_OUTPUT_FIELDS"),
        ('{"verdict":"WRONG"}', "stop", None, "ILLEGAL_VERDICT"),
        ("not json", "stop", None, "MALFORMED_ASSISTANT_JSON"),
        ('{"verdict":"UNKNOWN"}', "length", None, "COMPLETION_TRUNCATED"),
    ],
)
def test_verifier_output_is_strict_and_failures_are_conservative(
    raw: str, finish_reason: str, verdict: str | None, failure: str | None
) -> None:
    parsed_verdict, parsed_failure = safety.parse_verifier_output(raw, finish_reason)
    assert (parsed_verdict, parsed_failure) == (verdict, failure)
    if failure:
        assert safety.materialization_status(parsed_verdict, failure) == "MATERIALIZATION_BLOCKED_VERIFIER_FAILURE"
    elif verdict == "COHERENT_SINGLETON_SLOT":
        assert safety.materialization_status(verdict) == "MATERIALIZATION_SAFE"
    elif verdict == "INCOHERENT_GROUP":
        assert safety.materialization_status(verdict) == "MATERIALIZATION_BLOCKED_SEMANTIC_COLLISION"
    elif verdict == "UNKNOWN":
        assert safety.materialization_status(verdict) == "MATERIALIZATION_BLOCKED_UNKNOWN"


def test_verifier_projection_excludes_timestamps_and_benchmark_labels() -> None:
    identities = [_identity("a"), _identity("b", value="running")]
    sources = [
        _source("a", proposition="The user prefers swimming."),
        _source("b", proposition="The user prefers running."),
    ]
    source_before = copy.deepcopy(sources)
    universe, _ = safety.build_candidate_universe(identities, sources)
    request = safety.build_verifier_request(universe["repeated_candidate_groups"][0])
    assert request["propositions"]
    assert safety.FORBIDDEN_INPUT_FIELDS.isdisjoint(request)
    assert "observed_at" not in str(request)
    assert "question_id" not in str(request)
    assert "gold" not in str(request)
    assert sources == source_before


def test_candidate_join_rejects_missing_and_duplicate_source_ids() -> None:
    with pytest.raises(ValueError, match="identity_source_coverage_mismatch"):
        safety.build_candidate_universe([_identity("a")], [])
    with pytest.raises(ValueError, match="duplicate_source_memory_id"):
        safety.build_candidate_universe(
            [_identity("a")], [_source("a"), _source("a")]
        )


def test_safety_overlay_does_not_materialize_store_operations() -> None:
    universe, _ = safety.build_candidate_universe(
        [_identity("a"), _identity("b", value="running")],
        [_source("a"), _source("b", proposition="The user prefers running.")],
    )
    group = universe["repeated_candidate_groups"][0]
    record = safety.safety_overlay_record(group, "INCOHERENT_GROUP")
    assert record["materialization_status"] == "MATERIALIZATION_BLOCKED_SEMANTIC_COLLISION"
    assert "operation" not in record
    assert "supersedes_id" not in record
