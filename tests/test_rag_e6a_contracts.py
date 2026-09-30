from __future__ import annotations

import json

from eval.rag_e6.reader import (
    assign_requirement_ids,
    claims_used_evidence,
    composer_prompt,
    evidence_identity_sha256,
    issue_evidence_aliases,
    parse_claims,
    parse_last_final,
    parse_requirements,
    resolve_aliases,
    vanilla_prompt,
)
from eval.rag_e6.split import assign_subjects, build_split_manifest


def test_subject_split_is_deterministic_and_exactly_sized() -> None:
    counts = {f"SUBJ-{index:03d}": 12 + index % 3 for index in range(320)}
    first = assign_subjects(counts)
    second = assign_subjects(dict(reversed(list(counts.items()))))
    assert first == second
    assert {key: len(value) for key, value in first.items()} == {
        "BUILD": 64,
        "FROZEN_DEV": 128,
        "FUTURE_TRAIN": 128,
    }
    partition_subjects = [
        row["subject_id"] for rows in first.values() for row in rows
    ]
    assert len(partition_subjects) == len(set(partition_subjects)) == 320


def test_split_manifest_reads_runtime_ids_without_opening_truth(tmp_path, monkeypatch) -> None:
    episodes = tmp_path / "episodes.jsonl"
    rows = []
    episode_index = 0
    for subject_index in range(320):
        count = 13 if subject_index < 256 else 12
        for _ in range(count):
            rows.append(json.dumps({
                "episode_id": f"EP-{episode_index}",
                "subject_id": f"SUBJ-{subject_index}",
            }))
            episode_index += 1
    episodes.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    original_open = type(episodes).open

    def guarded_open(path, *args, **kwargs):
        if "evaluator_truth" in str(path):
            raise AssertionError("TRAIN evaluator truth must remain unopened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(type(episodes), "open", guarded_open)
    result = build_split_manifest(
        episodes_path=episodes, source_manifest_path=manifest, verify_pins=False
    )
    assert result["evaluator_truth_opened"] is False
    assert result["partition_subject_counts"] == {
        "BUILD": 64,
        "FROZEN_DEV": 128,
        "FUTURE_TRAIN": 128,
    }


def test_vanilla_and_cfec_receive_identical_issued_evidence() -> None:
    docs = [
        {"doc_id": "doc-a", "text": "Blood pressure evidence A."},
        {"doc_id": "doc-b", "text": "Blood pressure evidence B."},
    ]
    vanilla_evidence = issue_evidence_aliases(docs)
    cfec_evidence = issue_evidence_aliases(docs)
    assert vanilla_evidence == cfec_evidence
    assert evidence_identity_sha256(vanilla_evidence) == evidence_identity_sha256(cfec_evidence)
    assert "doc-a" not in vanilla_prompt("Question?", vanilla_evidence)


def test_last_final_marker_controls_answer_and_citations() -> None:
    parsed = parse_last_final("Draft [E8]\nFINAL: wrong [E2]\nFINAL: right [E1]")
    assert parsed.answer == "right [E1]"
    assert parsed.cited_aliases == ("[E1]",)
    assert not parsed.contract_failure
    assert parse_last_final("No final marker").contract_failure


def test_requirement_ids_are_harness_owned_and_capped() -> None:
    requirements = parse_requirements("- one\n- two\n- three\n- four\n- five")
    assert requirements == ("one", "two", "three", "four")
    assert assign_requirement_ids(requirements) == (
        ("req_1", "one"),
        ("req_2", "two"),
        ("req_3", "three"),
        ("req_4", "four"),
    )


def test_claims_only_resolve_issued_aliases_and_provenance_is_harness_owned() -> None:
    evidence = issue_evidence_aliases([{"doc_id": "doc-a", "text": "A"}])
    claims, unknown, failed = parse_claims(
        "- Claim supported here [E1] [E11]",
        requirement_id="req_1",
        evidence=evidence,
    )
    assert not failed
    assert unknown == ("[E11]",)
    assert claims[0].evidence_ids == ("doc-a",)
    assert claims_used_evidence(claims) == ("doc-a",)
    resolved, unknown_final = resolve_aliases(("[E1]", "[E11]"), evidence)
    assert resolved == ("doc-a",)
    assert unknown_final == ("[E11]",)


def test_final_composer_receives_claims_but_not_raw_evidence() -> None:
    claim = parse_claims(
        "Supported claim [E1]",
        requirement_id="req_1",
        evidence=issue_evidence_aliases([{"doc_id": "secret-doc-id", "text": "RAW DOCUMENT BODY"}]),
    )[0][0]
    prompt = composer_prompt("Question?", (claim,))
    assert "Supported claim" in prompt
    assert "[E1]" in prompt
    assert "RAW DOCUMENT BODY" not in prompt
    assert "secret-doc-id" not in prompt


def test_no_requirements_or_invalid_claims_fail_closed_without_new_ids() -> None:
    assert assign_requirement_ids(()) == ()
    evidence = issue_evidence_aliases([{"doc_id": "doc-a", "text": "A"}])
    claims, _unknown, failed = parse_claims(
        "Unsupported statement without citation",
        requirement_id="req_1",
        evidence=evidence,
    )
    assert claims == ()
    assert failed
