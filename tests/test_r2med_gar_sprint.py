from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from eval.r2med_crb import (
    BGE_QUERY_PREFIX,
    bm25_query_text,
    crb_dense_vector,
    crb_lexical_query,
    dense_gar_vector,
)
from eval.r2med_crb_data import (
    PARTITIONS,
    UPSTREAM_PROMPT_FAMILY,
    Document,
    R2MedSubset,
    _subset_manifest,
)
from eval.r2med_crb_evaluator import (
    METRICS,
    paired_stratified_bootstrap,
    select_best_fusion,
    summarize_query_metrics,
)
from eval.r2med_gar_generation import (
    CRB_JSON_SCHEMA,
    METHODS,
    GeneratedView,
    R2MedGARGenerator,
)
from eval.r2med_multiview import (
    FUSION_CONFIGS,
    RankedDocument,
    collapse_duplicate_document_scores,
    dense_search,
    dense_search_many,
    make_four_channels,
    weighted_rrf,
)
from tools.generate_r2med_gar import DEFAULT_SERVER
from tools.run_r2med_crb_dev import DEFAULT_LLAMA_SERVER
from tools.run_r2med_crb_test import assert_frozen_weights, validate_test_lock


class StaticClient:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[tuple[str, dict | None]] = []

    def complete(self, prompt: str, *, json_schema=None):
        self.calls.append((prompt, json_schema))
        return {"text": self.text, "finish_reason": "stop", "output_tokens": 17}


def test_dev_runner_uses_shared_llama_server_default():
    assert DEFAULT_LLAMA_SERVER == Path(DEFAULT_SERVER)


def _deny_qrels_path_reads(monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name.lower() == "qrels.jsonl":
            raise AssertionError("generation/ranking attempted to open qrels")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


def test_gar_generator_never_reads_qrels(monkeypatch):
    _deny_qrels_path_reads(monkeypatch)
    client = StaticClient("hypothetical evidence paragraph")
    prompts = {method: {"family": "Query: {TEXT} Passage:"} for method in ("hyde", "query2doc", "lamer")}
    generated = R2MedGARGenerator("family", client, prompts).generate(
        "q1", "query text", "hyde"
    )
    assert generated.generated_text == "hypothetical evidence paragraph"
    assert len(client.calls) == 1


def test_crb_generator_never_reads_qrels(monkeypatch):
    _deny_qrels_path_reads(monkeypatch)
    valid = {
        "canonical_query": "canonical medical query",
        "key_concepts": ["condition"],
        "disambiguating_terms": ["mechanism"],
        "pseudo_evidence": "A hypothetical evidence passage.",
    }
    client = StaticClient(json.dumps(valid))
    prompts = {method: {"family": "unused"} for method in ("hyde", "query2doc", "lamer")}
    output = R2MedGARGenerator("family", client, prompts).generate("q1", "query", "crb_q")
    assert output.valid
    assert set(output.structured) == set(CRB_JSON_SCHEMA["required"])
    assert client.calls[0][1] == CRB_JSON_SCHEMA
    assert len(client.calls) == 1


def test_ranking_never_reads_qrels(monkeypatch):
    _deny_qrels_path_reads(monkeypatch)
    channels = [[RankedDocument("a", 2), RankedDocument("b", 1)] for _ in range(4)]
    assert [row.doc_id for row in weighted_rrf(channels, rrf_k=60, weights=(1, 1, 1, 1))] == ["a", "b"]


def test_duplicate_corpus_ids_match_upstream_dense_and_bm25_loading():
    subset = R2MedSubset(
        name="PMC-Treatment",
        upstream_prompt_family="PMC-Treat",
        directory="fixture",
        queries=(),
        documents=(Document("dup", "same text"), Document("other", "text"), Document("dup", "same text")),
    )
    assert [document.doc_id for document in subset.dense_documents] == ["dup", "other"]
    collapsed = collapse_duplicate_document_scores(["dup", "other", "dup"], [0.1, 0.5, 0.5], top_k=10)
    assert collapsed == [RankedDocument("dup", 0.5), RankedDocument("other", 0.5)]


def test_evaluator_is_only_new_sprint_qrels_consumer():
    root = Path(__file__).resolve().parents[1]
    evaluator = (root / "eval/r2med_crb_evaluator.py").read_text(encoding="utf-8")
    assert '"qrels.jsonl"' in evaluator
    non_evaluators = (
        "eval/r2med_crb_data.py",
        "eval/r2med_gar_generation.py",
        "eval/r2med_crb.py",
        "eval/r2med_multiview.py",
        "tools/generate_r2med_gar.py",
        "tools/run_r2med_baselines.py",
    )
    for relative in non_evaluators:
        source = (root / relative).read_text(encoding="utf-8")
        assert '"qrels.jsonl"' not in source
        assert "qrels.jsonl" not in source


@pytest.mark.parametrize(
    ("method", "query", "generated", "expected"),
    [
        ("hyde", "Q", "H", "H"),
        ("query2doc", "Q", "D", "Q Q D"),
        ("lamer", "Q", "L", "Q L"),
    ],
)
def test_hyde_query2doc_lamer_bm25_semantics(method, query, generated, expected):
    assert bm25_query_text(method, query, generated) == expected


def test_hyde_dense_semantics_averages_original_and_generated():
    seen = []

    def embed(texts):
        seen.extend(texts)
        return np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)

    result = dense_gar_vector("hyde", "Q", "H", embed)
    assert seen == [BGE_QUERY_PREFIX + "Q", BGE_QUERY_PREFIX + "H"]
    np.testing.assert_array_equal(result, np.array([4.0, 6.0], dtype=np.float32))


def test_lamer_dense_semantics_averages_original_and_generated():
    seen = []

    def embed(texts):
        seen.extend(texts)
        return np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)

    result = dense_gar_vector("lamer", "Q", "L", embed)
    assert seen == [BGE_QUERY_PREFIX + "Q", BGE_QUERY_PREFIX + "L"]
    np.testing.assert_array_equal(result, np.array([3.0, 5.0], dtype=np.float32))


def test_query2doc_dense_semantics_uses_single_sep_query():
    seen = []

    def embed(texts):
        seen.extend(texts)
        return np.array([[1.0, 2.0]], dtype=np.float32)

    dense_gar_vector("query2doc", "Q", "D", embed)
    assert seen == [BGE_QUERY_PREFIX + "Q[SEP]D"]


def test_crb_json_schema_and_no_answer_fields():
    assert set(CRB_JSON_SCHEMA["properties"]) == {
        "canonical_query",
        "key_concepts",
        "disambiguating_terms",
        "pseudo_evidence",
    }
    assert CRB_JSON_SCHEMA["additionalProperties"] is False
    assert "answer" not in CRB_JSON_SCHEMA["properties"]
    assert "gold" not in inspect.signature(R2MedGARGenerator.generate).parameters
    assert "qrels" not in inspect.signature(R2MedGARGenerator.generate).parameters


def test_crb_lexical_text_uses_only_declared_bridge_fields():
    structured = {
        "canonical_query": "canonical",
        "key_concepts": ["condition", "therapy"],
        "disambiguating_terms": ["duration"],
        "pseudo_evidence": "not a lexical bridge",
    }
    assert crb_lexical_query("original", structured) == "canonical condition therapy duration"


def test_crb_dense_channel_embeds_only_pseudo_evidence():
    seen = []

    def embed(texts):
        seen.extend(texts)
        return np.array([[1.0, 2.0]], dtype=np.float32)

    structured = {
        "canonical_query": "canonical must not be used",
        "key_concepts": ["concept must not be used"],
        "disambiguating_terms": [],
        "pseudo_evidence": "only this evidence passage",
    }
    crb_dense_vector("original query", structured, embed)
    assert seen == [BGE_QUERY_PREFIX + "only this evidence passage"]


def test_dense_search_many_matches_single_query_search():
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    docs = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    ids = ["a", "b", "c"]
    batch = dense_search_many(queries, docs, ids)
    singles = [dense_search(query, docs, ids) for query in queries]
    assert batch == singles


def test_multiview_same_channel_count_and_same_weight_grid():
    channels = [[RankedDocument(str(i), 1.0)] for i in range(4)]
    assert len(make_four_channels(
        bm25_original=channels[0],
        bm25_bridge=channels[1],
        bge_original=channels[2],
        bge_generated=channels[3],
    )) == 4
    assert len(FUSION_CONFIGS) == 10
    assert {entry["rrf_k"] for entry in FUSION_CONFIGS} == {20, 60}
    assert {tuple(entry["weights"]) for entry in FUSION_CONFIGS} == {
        (1, 1, 1, 1), (2, 1, 2, 1), (1, 2, 1, 2), (2, 2, 1, 1), (1, 1, 2, 2)
    }


def test_rrf_is_deterministic():
    channels = [
        [RankedDocument("doc-b", 0.2), RankedDocument("doc-a", 0.1)],
        [RankedDocument("doc-a", 0.9), RankedDocument("doc-b", 0.4)],
        [RankedDocument("doc-c", 0.8)],
        [RankedDocument("doc-a", 0.2)],
    ]
    first = weighted_rrf(channels, rrf_k=20, weights=(2, 1, 2, 1))
    second = weighted_rrf(channels, rrf_k=20, weights=(2, 1, 2, 1))
    assert first == second


def test_dev_selection_only_uses_dev():
    manifest = {"datasets": {"DEV": []}}
    with pytest.raises(ValueError, match="not part of partition"):
        _subset_manifest(manifest, "DEV", "MedQA-Diag")
    assert PARTITIONS["DEV"] == ("PMC-Treatment", "PMC-Clinical", "IIYi-Clinical")
    assert "PMC-Treatment" in UPSTREAM_PROMPT_FAMILY


def test_test_runner_requires_final_lock():
    with pytest.raises(FileNotFoundError, match="committed"):
        validate_test_lock(None, committed=False)
    with pytest.raises(ValueError, match="negative"):
        validate_test_lock({"status": "FROZEN_BEFORE_TEST", "test_status": "PUBLIC_BENCHMARK_REUSED", "crb_variant": "crb_q", "dev_gate": {"signal": "NEGATIVE", "delta": -0.01, "positive_subsets": 0}}, committed=True)


def test_test_weights_are_frozen():
    lock = {"rrf": {"k": 60, "weights": [2, 1, 2, 1]}}
    assert_frozen_weights(lock, rrf_k=60, weights=(2, 1, 2, 1))
    with pytest.raises(ValueError, match="differ"):
        assert_frozen_weights(lock, rrf_k=20, weights=(1, 1, 1, 1))


def test_metric_macro_equal_subset_weight():
    rows = [
        {"subset": "A", **{metric: 1.0 for metric in METRICS}},
        {"subset": "A", **{metric: 1.0 for metric in METRICS}},
        {"subset": "B", **{metric: 0.0 for metric in METRICS}},
    ]
    summary = summarize_query_metrics(rows)
    assert summary["macro_equal_subset_weight"]["ndcg@10"] == pytest.approx(0.5)
    assert summary["query_count"] == 3


def test_bootstrap_is_paired_and_subset_stratified():
    baseline = {f"q{i}": {"ndcg@10": 0.1} for i in range(6)}
    candidate = {f"q{i}": {"ndcg@10": 0.2} for i in range(6)}
    subsets = {"q0": "A", "q1": "A", "q2": "B", "q3": "B", "q4": "C", "q5": "C"}
    result = paired_stratified_bootstrap(candidate, baseline, subsets)
    assert result["resamples"] == 10_000
    assert result["strata"] == 3
    assert result["mean_delta"] == pytest.approx(0.1)
    assert result["lower_95"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="identical"):
        paired_stratified_bootstrap(candidate, baseline, {"q0": "A"})


def test_fusion_selection_stays_within_predeclared_grid():
    one = FUSION_CONFIGS[0]["config_id"]
    summary = {
        "macro_equal_subset_weight": {"ndcg@10": 0.1, "mrr@10": 0.2, "recall@10": 0.3}
    }
    selected, _ = select_best_fusion({one: summary})
    assert selected == one
    with pytest.raises(ValueError, match="outside"):
        select_best_fusion({"invented": summary})


def test_all_required_generator_methods_are_present():
    assert METHODS == ("hyde", "query2doc", "lamer", "crb_q", "crb_prf")
    assert GeneratedView.__dataclass_fields__["query_id"]
