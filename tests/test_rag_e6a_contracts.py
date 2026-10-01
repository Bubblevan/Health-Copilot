from __future__ import annotations

import json
from pathlib import Path

from eval.rag_e6.data import RUNTIME_EPISODE_FIELDS, E6Episode
from eval.rag_e6.llm import (
    COMMON_SYSTEM_PROMPT,
    MODEL_NAME,
    CallJournal,
    LocalLlamaCppClient,
    PromptContextExceeded,
)
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
from eval.rag_e6.reader_executor import (
    _retrieval_identity,
    _verify_inherited_u3r_runtime_source,
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


def test_reader_prompts_align_citation_location_with_final_parser() -> None:
    vanilla = vanilla_prompt(
        "Question?", issue_evidence_aliases([{"doc_id": "doc-a", "text": "Evidence."}])
    )
    assert "same FINAL line" in vanilla
    assert "Never write a bare alias" in vanilla
    assert "citations only before the FINAL line" in vanilla
    composer = composer_prompt("Question?", ())
    assert "aliases on the FINAL line" in composer


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


def test_claim_parser_rejects_parenthetical_aliases() -> None:
    evidence = issue_evidence_aliases([{"doc_id": "doc-a", "text": "A"}])
    claims, unknown, failed = parse_claims(
        "Claim with a non-contract citation (E1)",
        requirement_id="req_1",
        evidence=evidence,
    )
    assert claims == ()
    assert unknown == ()
    assert failed


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


def test_runtime_episode_contract_rejects_teacher_fields() -> None:
    row = {
        "episode_id": "EP-1",
        "environment_version": "env-v1",
        "source_provenance": "synthetic",
        "decision_time": "2026-01-01T00:00:00+00:00",
        "subject_id": "SUBJ-1",
        "query": "Question?",
        "observable_state": {"available_external_source_families": ["PUBLIC_HEALTH"]},
        "patient_state_ref": None,
        "external_world_ref": {
            "world_id": "EXT-1",
            "version": "owned-evidence-v1",
            "source_families": ["PUBLIC_HEALTH"],
        },
        "tool_surface_ref": None,
        "budget": {},
        "evaluator_ref": {},
    }
    assert set(row) == RUNTIME_EPISODE_FIELDS
    episode = E6Episode.from_runtime_row(row, partition="BUILD")
    assert episode.partition == "BUILD"
    assert episode.split == "TRAIN"
    contaminated = {**row, "required_external_evidence_ids": ["GOLD"]}
    try:
        E6Episode.from_runtime_row(contaminated, partition="BUILD")
    except ValueError:
        pass
    else:
        raise AssertionError("runtime loader accepted an evaluator-only field")


def test_retrieval_identity_matches_frozen_u3r_configuration() -> None:
    config = _retrieval_identity()
    assert config["retrieval_config_changed"] is False
    assert config["standard"]["bm25"] == {"analyzer": "Lucene", "k1": 0.9, "b": 0.4}
    assert config["standard"]["rrf_k"] == 60
    assert config["standard"]["weights"] == [1, 1]
    assert config["standard"]["top_k"] == 10
    assert config["strong"]["method"] == "pinned R2MED LameR-MV"
    assert config["strong"]["rrf_k"] == 20
    assert config["strong"]["weights"] == [1, 2, 1, 2]
    assert config["strong"]["top_k"] == 10


def test_inherited_u3r_runtime_modules_match_frozen_hashes() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    identity = _verify_inherited_u3r_runtime_source(repository_root)
    assert identity["code_commit"] == "2d2d3ca2de48dc5fb5a5f011e8fa5c5aa3003d0d"
    assert len(identity["verified_source_sha256"]) == 6


def test_call_journal_reuses_completed_calls_without_retry(tmp_path) -> None:
    class FakeClient:
        model_name = MODEL_NAME

        def __init__(self):
            self.attempts = 0

        def complete(self, _prompt, *, system_prompt):
            assert system_prompt == COMMON_SYSTEM_PROMPT
            self.attempts += 1
            return {
                "text": "FINAL: synthetic",
                "finish_reason": "stop",
                "input_tokens": 3,
                "output_tokens": 2,
            }

    journal_path = tmp_path / "calls.jsonl"
    client = FakeClient()
    journal = CallJournal(journal_path)
    first = journal.call_once(
        call_id="EP-1|VANILLA_OFF|reader",
        prompt="synthetic prompt",
        system_prompt=COMMON_SYSTEM_PROMPT,
        client=client,
    )
    resumed = CallJournal(journal_path).call_once(
        call_id="EP-1|VANILLA_OFF|reader",
        prompt="synthetic prompt",
        system_prompt=COMMON_SYSTEM_PROMPT,
        client=client,
    )
    assert first == resumed
    assert client.attempts == 1


def test_interrupted_call_is_not_retried(tmp_path) -> None:
    class FakeClient:
        model_name = MODEL_NAME
        attempts = 0

        def complete(self, _prompt, *, system_prompt):
            self.attempts += 1
            return {"text": "unexpected", "finish_reason": "stop"}

    journal_path = tmp_path / "calls.jsonl"
    first_client = FakeClient()
    CallJournal(journal_path).call_once(
        call_id="EP-2|VANILLA_OFF|reader",
        prompt="synthetic prompt",
        system_prompt=COMMON_SYSTEM_PROMPT,
        client=first_client,
    )
    start_record = journal_path.read_text(encoding="utf-8").splitlines()[0]
    journal_path.write_text(start_record + "\n", encoding="utf-8")
    resumed_client = FakeClient()
    result = CallJournal(journal_path).call_once(
        call_id="EP-2|VANILLA_OFF|reader",
        prompt="synthetic prompt",
        system_prompt=COMMON_SYSTEM_PROMPT,
        client=resumed_client,
    )
    assert result["status"] == "interrupted_no_retry"
    assert resumed_client.attempts == 0


def test_oversized_prompt_is_rejected_before_generation_request() -> None:
    client = LocalLlamaCppClient(
        "http://127.0.0.1:8092/v1", effective_context_size=16384
    )
    try:
        client.complete("x" * 8000, system_prompt="system")
    except PromptContextExceeded as exc:
        assert exc.prompt_bytes > exc.prompt_byte_limit
    else:
        raise AssertionError("context guard did not reject an oversized prompt")
