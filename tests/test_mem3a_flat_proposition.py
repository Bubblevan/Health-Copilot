import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tools.research.memory import run_mem3a_flat_proposition as mem3a


def _session():
    return {
        "session_identity_sha256": "a" * 64,
        "source_turns_sha256": "b" * 64,
        "session_date": "2023-05-28T22:57:00",
        "turns": [
            {
                "turn_index": 0,
                "role": "user",
                "content": "I'm now at 600 followers on Instagram.",
            }
        ],
    }


def _proposition():
    return {
        "proposition_text": "The user has 600 Instagram followers.",
        "entity_key_candidate": "user",
        "attribute_key_candidate": "instagram_follower_count",
        "value_text": "600",
        "source_role": "user",
        "source_turn_indices": [0],
        "evidence_quotes": ["600 followers on Instagram"],
    }


def test_writer_input_excludes_question_and_benchmark_labels():
    session = _session()
    baseline = mem3a._writer_messages(session, "frozen system prompt")
    with_benchmark_metadata = {
        **session,
        "question_id": "question-a",
        "question": "What is the current follower count?",
        "question_date": "2023-05-29",
        "answer": "600",
        "answer_session_ids": ["source-1"],
        "question_type": "knowledge-update",
        "has_answer": True,
    }

    messages = mem3a._writer_messages(with_benchmark_metadata, "frozen system prompt")
    assert messages == baseline
    assert set(json.loads(messages[1]["content"])) == {"session_date", "ordered_turns"}
    assert "question" not in messages[1]["content"]


def test_session_identity_uses_only_date_and_exact_ordered_turns():
    turns = _session()["turns"]
    identity = mem3a._session_identity_sha256("2023-05-28T22:57:00", turns)
    assert identity == mem3a._session_identity_sha256("2023-05-28T22:57:00", turns)
    assert identity != mem3a._session_identity_sha256(
        "2023-05-28T22:57:00", [{**turns[0], "content": "I'm now at 601 followers."}]
    )
    assert identity != mem3a._session_identity_sha256("2023-05-29T22:57:00", turns)


def test_extraction_cache_identity_has_no_question_fields():
    session = _session()
    session.update({"question_id": "q", "question": "changed", "answer": "changed"})
    identity = mem3a._session_cache_identity(session, "c" * 64, "d" * 64, "e" * 64)
    assert not mem3a.FORBIDDEN_WRITER_KEYS.intersection(identity["identity"])
    assert identity["identity_sha256"] == mem3a._sha_json(identity["identity"])


def test_exact_quote_schema_and_source_role_are_enforced():
    packet = {"propositions": [_proposition()]}
    validated = mem3a._validate_packet(json.dumps(packet), _session())
    assert validated == packet["propositions"]

    bad_quote = {"propositions": [{**_proposition(), "evidence_quotes": ["601 followers"]}]}
    with pytest.raises(ValueError, match="evidence_quote_not_exact_source_substring"):
        mem3a._validate_packet(json.dumps(bad_quote), _session())

    bad_role = {"propositions": [{**_proposition(), "source_role": "assistant"}]}
    with pytest.raises(ValueError, match="source_role_mismatch"):
        mem3a._validate_packet(json.dumps(bad_role), _session())


def test_dense_document_is_exactly_proposition_text():
    row = {
        **_proposition(),
        "memory_id": "opaque-id",
        "key": "diagnostic-key",
        "evidence_quotes": ["secret quote"],
        "question_id": "benchmark-question",
        "valid_from": "2023-05-28T22:57:00",
    }
    assert mem3a._proposition_retrieval_document(row) == row["proposition_text"]
    with pytest.raises(ValueError, match="retrieval_document_must_be_nonempty"):
        mem3a._proposition_retrieval_document({"proposition_text": "  "})


def test_candidate_group_does_not_assign_revision_semantics():
    common = {
        "entity_key_candidate": "user",
        "attribute_key_candidate": "instagram_follower_count",
        "question_id": mem3a.QUESTION_IDS[0],
        "source_session_id": "source-1",
    }
    groups = mem3a._candidate_groups(
        [
            {
                **common,
                "value_text": "500",
                "valid_from": "2023-05-27",
                "proposition_text": "The user has 500 Instagram followers.",
            },
            {
                **common,
                "value_text": "600",
                "valid_from": "2023-05-28",
                "source_session_id": "source-2",
                "proposition_text": "The user has 600 Instagram followers.",
            },
        ]
    )
    candidate = groups["groups"][0]
    assert candidate["candidate_revision_group"] is True
    assert candidate["values"] == ["500", "600"]
    assert groups["revision_semantics"] is False
    assert "stale" not in candidate and "current" not in candidate


def test_answer_session_recall_is_fractional_and_ranked():
    rows = [
        {"rank": 1, "source_session_id": "answer-a"},
        {"rank": 2, "source_session_id": "other"},
        {"rank": 3, "source_session_id": "answer-b"},
    ]
    metrics = mem3a._retrieval_session_metrics(rows, {"answer-a", "answer-b"})
    assert metrics == {
        "answer_session_recall_at_5": 1.0,
        "answer_session_recall_at_8": 1.0,
        "mrr": 1.0,
    }
    assert (
        mem3a._retrieval_session_metrics(rows[:1], {"answer-a", "answer-b"})[
            "answer_session_recall_at_8"
        ]
        == 0.5
    )


def test_main_stack_is_local_without_openai_credentials():
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        assert urlsplit(mem3a.READER_ENDPOINT).hostname == "127.0.0.1"
        runner_source = Path(mem3a.__file__).read_text(encoding="utf-8")
        adapter_source = (Path(mem3a.__file__).with_name("local_qwen3_embedding.py")).read_text(
            encoding="utf-8"
        )
        assert "OPENAI_API_KEY" not in runner_source
        assert "api.openai.com" not in runner_source.casefold()
        assert "OPENAI_API_KEY" not in adapter_source
        assert "api.openai.com" not in adapter_source.casefold()
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old


def test_freeze_verifier_accepts_both_existing_sidecar_conventions(tmp_path):
    artifact = tmp_path / "contract.json"
    artifact.write_text("frozen\n", encoding="utf-8")
    digest = mem3a.sha256_file(artifact)
    appended = artifact.with_name(f"{artifact.name}.sha256")
    appended.write_text(f"{digest}  {artifact.name}\n", encoding="ascii")
    assert mem3a._verify_frozen(artifact)

    suffix = artifact.with_suffix(".sha256")
    suffix.write_text(f"{digest}  {artifact.name}\n", encoding="ascii")
    assert mem3a._verify_frozen(artifact)
    appended.write_text(f"{'0' * 64}  {artifact.name}\n", encoding="ascii")
    assert not mem3a._verify_frozen(artifact)
