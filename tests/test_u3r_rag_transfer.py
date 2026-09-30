from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from eval.u3r_rag_scoring import _cost_reduction, _evaluate_arm
from eval.u3r_rag_transfer import (
    ACTION_ORDER,
    ANSWER_CONTEXT_K,
    U2F_ROOT,
    OwnedExternalCorpusAdapter,
    RankedDocument,
    RetrievalView,
    U3RDocument,
    U3RRetriever,
    fixed_arm_runtime_row,
    load_dev_runtime_episodes,
    materialize_runtime_corpus,
    select_runtime_world_fields,
)


def test_runtime_world_projection_never_decodes_hidden_gold_fields() -> None:
    line = json.dumps({
        "latent_world_id": "LW-dev-1",
        "split_role": "DEV_IID",
        "dependency_graph": {"required_fact_ids": ["DO-NOT-PARSE"]},
        "answer": "DO-NOT-PARSE-EITHER",
        "external_evidence": [{
            "source_id": "doc-1",
            "source_family": "PUBLIC_HEALTH",
            "publication_time": "2026-01-01T00:00:00+00:00",
            "effective_time": None,
            "effective_until": None,
            "natural_language_content": "approved runtime-visible text",
            "retrieval_terms": ["not part of the U3-R corpus view"],
            "latent_fact_ids": ["DO-NOT-PARSE"],
        }],
    })
    selected = select_runtime_world_fields(line)
    assert set(selected) == {"latent_world_id", "split_role", "external_evidence"}
    assert set(selected["external_evidence"][0]) == {
        "source_id", "source_family", "publication_time", "effective_time",
        "effective_until", "natural_language_content",
    }
    assert "dependency_graph" not in selected
    assert "latent_fact_ids" not in selected["external_evidence"][0]


def test_adapter_never_indexes_forbidden_evidence_fields() -> None:
    episode = load_dev_runtime_episodes()[0]
    forbidden = {
        "latent_fact_ids", "gold_answer", "required_fact_ids", "answer_values",
        "retrieval_terms",
    }

    class GuardedMapping(dict):
        def __getitem__(self, key):
            if key in forbidden:
                raise AssertionError(f"forbidden field accessed: {key}")
            return super().__getitem__(key)

        def __contains__(self, key):
            if key in forbidden:
                raise AssertionError(f"forbidden field checked: {key}")
            return super().__contains__(key)

        def get(self, key, default=None):
            if key in forbidden:
                raise AssertionError(f"forbidden field read: {key}")
            return super().get(key, default)

    family = episode.available_source_families[0]
    publication = (episode.decision_time - timedelta(days=1)).isoformat()
    evidence = GuardedMapping({
        "source_id": "runtime-doc-1",
        "source_family": family,
        "publication_time": publication,
        "effective_time": None,
        "effective_until": None,
        "natural_language_content": "only approved evidence text",
        "retrieval_terms": ["never-access"],
        "latent_fact_ids": ["never-access"],
        "gold_answer": "never-access",
    })
    projection = GuardedMapping({
        "latent_world_id": f"LW-{episode.episode_id}",
        "split_role": episode.split,
        "external_evidence": [evidence],
        "evaluator_truth": {"gold_answer": "never-access"},
    })
    docs = OwnedExternalCorpusAdapter.documents_for_episode(episode, projection)
    assert [item.doc_id for item in docs] == ["runtime-doc-1"]
    assert docs[0].text == "only approved evidence text"


def test_runtime_arms_keep_llm_text_out_of_deterministic_answer() -> None:
    episode = load_dev_runtime_episodes()[0]
    family = episode.available_source_families[0]
    document = U3RDocument(
        doc_id="doc-selected-by-frozen-ranker",
        source_family=family,
        publication_time=episode.decision_time - timedelta(days=1),
        effective_time=None,
        effective_until=None,
        text="SYNVAL-1234567890 approved synthetic evidence",
    )
    ranked = RetrievalView(
        final_ids=(document.doc_id,),
        candidate_union_ids=(document.doc_id,),
        channel_ids=((document.doc_id,),),
        search_invocations=1,
    )
    bridge = {"generated_text": "LLM MUST NOT WRITE THE ANSWER", "provider_calls": 1}
    row = fixed_arm_runtime_row(
        episode,
        (document, U3RDocument(
            doc_id="not-in-frozen-top10",
            source_family=family,
            publication_time=episode.decision_time - timedelta(days=1),
            effective_time=None,
            effective_until=None,
            text="This document was in the visible corpus but not in the frozen retrieval result.",
        )),
        standard=ranked,
        strong=ranked,
        bridge=bridge,
    )
    assert set(row["actions"]) == set(ACTION_ORDER)
    assert "LLM MUST NOT WRITE THE ANSWER" not in row["actions"]["STRONG"]["answer"]
    assert row["actions"]["STANDARD"]["used_evidence_ids"] == [document.doc_id]
    assert row["actions"]["STRONG"]["used_evidence_ids"] == [document.doc_id]
    assert row["actions"]["STRONG"]["deterministic_execution"]["outcome"]["provider_calls"] == 0
    assert row["actions"]["STRONG"]["bridge_sha256"]


def test_strong_four_view_uses_the_prefetched_original_bm25_channel() -> None:
    episode = load_dev_runtime_episodes()[0]
    retriever = object.__new__(U3RRetriever)
    calls: list[tuple[str, str]] = []
    original = [RankedDocument("original", 1.0)]

    def bm25(_docs, query):
        calls.append(("bm25", query))
        return [RankedDocument("bridge-bm25", 1.0)]

    def dense(_docs, query):
        calls.append(("dense", query))
        return [RankedDocument(f"dense-{len(calls)}", 1.0)]

    retriever._bm25 = bm25
    retriever._dense = dense
    view = retriever.strong(episode, (), "clinical bridge", original)
    assert len(view.channel_ids) == 4
    assert view.channel_ids[0] == ("original",)
    assert calls == [
        ("bm25", f"{episode.query} clinical bridge"),
        ("dense", episode.query),
        ("dense", "clinical bridge"),
    ]
    assert view.search_invocations == 4


def test_runtime_corpus_materializer_never_opens_evaluator_or_train_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = Path.open
    forbidden_fragments = (
        "evaluator_truth", "partial_policy_supervision", "train/", "test/", "ood/",
    )
    opened: list[str] = []

    def guarded_open(path: Path, *args, **kwargs):
        normalized = path.as_posix().casefold()
        if any(fragment in normalized for fragment in forbidden_fragments):
            raise AssertionError(f"runtime attempted forbidden file access: {path.name}")
        opened.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    manifest = materialize_runtime_corpus(
        u2f_root=Path(U2F_ROOT).resolve(), run_root=tmp_path / "u3r-corpus"
    )
    assert manifest["episode_count"] == 1024
    assert manifest["train_world_rows_parsed"] == 0
    assert manifest["evaluator_truth_opened"] is False
    assert manifest["reserved_test_ood_opened"] is False
    assert "latent_worlds.jsonl" in opened


@pytest.mark.parametrize("raw", [
    '{"latent_world_id":"LW-x","split_role":"DEV_IID","external_evidence":[],}',
    '{"latent_world_id":"LW-x","latent_world_id":"LW-y","split_role":"DEV_IID","external_evidence":[]}',
])
def test_runtime_world_projection_rejects_malformed_or_ambiguous_json(raw: str) -> None:
    with pytest.raises((ValueError, json.JSONDecodeError)):
        select_runtime_world_fields(raw)


def test_answer_context_budget_is_fixed() -> None:
    assert ANSWER_CONTEXT_K == 10


def test_cost_gate_measurement_counts_actual_retrieval_searches() -> None:
    always_strong = {"retrieval_search_invocations": 400, "retrieval_activations": 100}
    minimal = {"retrieval_search_invocations": 240, "retrieval_activations": 80}
    assert _cost_reduction(always_strong, minimal, "retrieval_search_invocations") == 0.4
    assert _cost_reduction(always_strong, minimal, "retrieval_activations") == 0.2


def test_cost_reduction_with_zero_baseline_is_defined_as_zero() -> None:
    assert _cost_reduction({"calls": 0}, {"calls": 0}, "calls") == 0.0


def test_scoring_uses_gold_only_as_a_post_execution_evaluator_input() -> None:
    truth = {
        "answer_values": ["SYNVAL-1234567890"],
        "answer_type": "EXACT_TOKEN",
        "capability_requirement_oracle": {"answerability": True},
        "required_memory_record_ids": [],
        "required_external_evidence_ids": ["required-doc"],
    }
    arm = {
        "action": "STANDARD",
        "answer": "SYNVAL-1234567890",
        "used_evidence_ids": ["required-doc"],
        "ranked_evidence_ids": ["required-doc"],
    }
    result = _evaluate_arm(truth=truth, runtime_arm=arm, visible_doc_ids={"required-doc"})
    assert result["task_success"] is True
    assert result["grounding_pass"] is True
    assert result["required_external_fact_coverage"] == 1.0


def test_insufficient_retrieval_is_not_mistaken_for_a_successful_abstention() -> None:
    truth = {
        "answer_values": [],
        "answer_type": "ABSTAIN",
        "capability_requirement_oracle": {"answerability": False},
        "required_memory_record_ids": [],
        "required_external_evidence_ids": [],
    }
    arm = {
        "action": "STRONG",
        "answer": "SYNVAL-1234567890",
        "used_evidence_ids": ["distractor-doc"],
        "ranked_evidence_ids": ["distractor-doc"],
    }
    result = _evaluate_arm(truth=truth, runtime_arm=arm, visible_doc_ids={"distractor-doc"})
    assert result["abstention_correct"] is False
    assert result["grounded_success"] is False
