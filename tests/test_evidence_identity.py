from health_ai_copilot.providers.retrieval import RetrievalResult, RetrievedEvidence


def test_evidence_payload_hash_is_stable_and_changes_with_observed_bytes() -> None:
    row = RetrievedEvidence("e1", "s1", "source", "text", 0.5)
    one = RetrievalResult((row,))
    same = RetrievalResult((row,))
    changed = RetrievalResult((RetrievedEvidence("e1", "s1", "source", "changed", 0.5),))
    assert one.evidence_sha256 == same.evidence_sha256
    assert one.evidence_sha256 != changed.evidence_sha256
