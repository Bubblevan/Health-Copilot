from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from tools.research.memory import local_qwen3_embedding as embedding
from tools.research.memory import run_mem2d_semantic_retrieval as runner


def _memory_row(memory_id: str, doc_sha: str) -> dict:
    return {
        "record": SimpleNamespace(
            source_session_id=f"session-{memory_id}", key=f"span-{memory_id}"
        ),
        "inventory": {"source_turn_index": 2, "source_span_index": 1, "parent_turn_key": "turn-1"},
        "retrieval_document_sha256": doc_sha,
    }


def _rank_row(memory_id: str, rank: int, score: float) -> dict:
    return {
        "memory_id": memory_id,
        "rank": rank,
        "score": score,
        "cosine_similarity": score,
        "source_session_id": f"session-{memory_id}",
        "source_turn_id": "turn-1",
        "source_span_id": f"span-{memory_id}",
        "source_turn_index": 2,
        "source_span_index": 1,
        "retrieval_document_sha256": hashlib.sha256(memory_id.encode()).hexdigest(),
    }


def test_retrieval_document_view_matches_m10_json_bytes() -> None:
    value = {"content": "心率", "role": "user"}
    expected = 'raw_span:key {"content": "心率", "role": "user"}'
    assert embedding.retrieval_document_text("raw_span:key", value) == expected
    assert embedding.retrieval_document_text("key", "raw value") == "key raw value"


def test_frozen_query_instruction_is_query_only() -> None:
    adapter = object.__new__(embedding.LocalQwen3Embedding)
    query = adapter.prepare_texts(["What changed?"], "query")
    document = adapter.prepare_texts(["Memory text"], "document")
    assert query == [
        (
            "Instruct: Given a conversation-history question, retrieve memory passages that provide "
            "evidence needed to answer the question.\nQuery:What changed?"
        )
    ]
    assert document == ["Memory text"]
    assert embedding.DIMENSIONS == 1024
    assert embedding.DEVICE == "cuda:0"
    assert embedding.DTYPE == "float16"


def test_local_model_identity_fails_closed_on_weights_revision_and_tree() -> None:
    with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        model = Path(temporary) / "model"
        model.mkdir()
        for name, content in {
            "model.safetensors": "weights",
            "config.json": "{}",
            "tokenizer.json": "{}",
            "tokenizer_config.json": "{}",
        }.items():
            (model / name).write_text(content, encoding="utf-8")
        metadata = model / ".hfd" / "repo_metadata.json"
        metadata.parent.mkdir()
        metadata.write_text(
            json.dumps({"id": embedding.MODEL_ID, "sha": embedding.MODEL_REVISION}),
            encoding="utf-8",
        )
        tree_sha = embedding.model_tree_sha256(model)
        weights_sha = embedding.sha256_file(model / "model.safetensors")
        identity = embedding.verify_local_model(
            model,
            expected_tree_sha256=tree_sha,
            expected_weights_sha256=weights_sha,
        )
        assert identity["model_tree_sha256"] == tree_sha
        with pytest.raises(RuntimeError, match="weights"):
            embedding.verify_local_model(
                model,
                expected_tree_sha256=tree_sha,
                expected_weights_sha256="0" * 64,
            )
        metadata.write_text(
            json.dumps({"id": embedding.MODEL_ID, "sha": "new-revision"}),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="revision"):
            embedding.verify_local_model(
                model,
                expected_tree_sha256=tree_sha,
                expected_weights_sha256=weights_sha,
            )


def test_dense_ties_break_by_memory_id() -> None:
    np = pytest.importorskip("numpy")
    vectors = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    rows = runner._rank_dense(
        question_id="qid",
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        eligible_ids={"z", "a"},
        memory_by_id={"z": _memory_row("z", "doc-z"), "a": _memory_row("a", "doc-a")},
        doc_vectors=vectors,
        doc_index_by_sha={"doc-z": 0, "doc-a": 1},
        top_k=2,
    )
    assert [row["memory_id"] for row in rows] == ["a", "z"]
    assert [row["rank"] for row in rows] == [1, 2]


def test_rrf_exact_formula_and_tie_break() -> None:
    lexical = [_rank_row("z", 1, 0.5)]
    dense = [_rank_row("a", 1, 0.9), _rank_row("z", 2, 0.8)]
    fused = runner._fuse_rrf(lexical, dense)
    assert fused[0]["memory_id"] == "z"
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    tied = runner._fuse_rrf([_rank_row("b", 1, 0.0)], [_rank_row("a", 1, 0.0)])
    assert [row["memory_id"] for row in tied] == ["a", "b"]
    assert tied[0]["rrf_score"] == tied[1]["rrf_score"]


def test_historical_retrieval_results_are_flattened_without_field_changes() -> None:
    rows = [
        {
            "question_id": "qid-a",
            "query": "first",
            "results": [
                {"memory_id": "a1", "rank": 1, "source_session_id": "s1"},
                {"memory_id": "a2", "rank": 2, "source_session_id": "s2"},
            ],
        },
        {
            "question_id": "qid-b",
            "query": "second",
            "results": [{"memory_id": "b1", "rank": 1, "source_session_id": "s3"}],
        },
    ]
    assert runner._flatten_historical_retrieval(rows) == [
        {
            "memory_id": "a1",
            "rank": 1,
            "source_session_id": "s1",
            "question_id": "qid-a",
        },
        {
            "memory_id": "a2",
            "rank": 2,
            "source_session_id": "s2",
            "question_id": "qid-a",
        },
        {
            "memory_id": "b1",
            "rank": 1,
            "source_session_id": "s3",
            "question_id": "qid-b",
        },
    ]


def test_reader_and_embedding_locality_are_pinned_without_provider_fallback() -> None:
    source = Path(embedding.__file__).read_text(encoding="utf-8")
    runner_source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "local_files_only=True" in source
    assert "trust_remote_code=False" in source
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in source
    assert "openai" not in source.lower()
    assert "OPENAI_API_KEY" not in runner_source
    assert 'READER_ENDPOINT = mem2c.READER_ENDPOINT' in runner_source


def test_retrieval_identities_bind_question_and_contracts_without_labels() -> None:
    upstream = {
        "rawspan_segmenter_contract_sha256": "a" * 64,
        "retrieval_view_contract_sha256": "b" * 64,
        "rrf_contract_sha256": "c" * 64,
    }
    sources = {"embedding_adapter": "d" * 64, "memory_store": "e" * 64}
    dense = runner._retrieval_identity(
        question_id="frozen-qid", upstream=upstream, source_hashes=sources, hybrid=False
    )
    hybrid = runner._retrieval_identity(
        question_id="frozen-qid", upstream=upstream, source_hashes=sources, hybrid=True
    )
    assert dense["identity"]["question_id"] == "frozen-qid"
    assert dense["identity"]["projection_contract_sha256"] == runner.PROJECTION_SHA256
    assert dense["identity"]["final_reader_contract_sha256"] == runner.FINAL_READER_SHA256
    assert "question_type" not in dense["identity"]
    assert "answer_session_ids" not in dense["identity"]
    assert hybrid["identity"]["rrf_contract_sha256"] == upstream["rrf_contract_sha256"]
    assert hybrid["identity"]["lexical_scorer_source_sha256"] == sources["memory_store"]


def test_final_token_pooling_and_normalization_are_explicit() -> None:
    torch = pytest.importorskip("torch")
    from torch.nn import functional

    hidden = torch.tensor(
        [
            [[90.0, 90.0], [1.0, 0.0], [0.0, 1.0]],
            [[80.0, 80.0], [70.0, 70.0], [3.0, 4.0]],
        ]
    )
    attention = torch.tensor([[0, 1, 1], [0, 0, 1]])
    pooled = embedding.pool_final_nonpadding(hidden, attention, torch)
    normalized = embedding.normalize_l2_float32(pooled, functional)
    assert pooled.tolist() == [[0.0, 1.0], [3.0, 4.0]]
    assert normalized.dtype == torch.float32
    assert torch.allclose(torch.linalg.norm(normalized, dim=1), torch.ones(2))


def test_complete_embedding_cache_skips_model_inference(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")

    class FakeAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def encode_batches(self, texts, input_type, *, token_counts=None):
            self.calls += 1
            for start in range(0, len(texts), 2):
                end = min(start + 2, len(texts))
                vectors = np.zeros((end - start, embedding.DIMENSIONS), dtype=np.float32)
                vectors[:, 0] = 1.0
                yield SimpleNamespace(
                    start=start,
                    end=end,
                    vectors=vectors,
                    latency_ms=1.0,
                )

    identity = {"identity": {"test": True}, "identity_sha256": "frozen-cache-test"}
    doc_shas = ["a" * 64, "b" * 64]
    texts = ["first", "second"]
    first = FakeAdapter()
    vectors, first_stats = runner._embed_document_corpus(
        adapter=first,
        unique_doc_shas=doc_shas,
        unique_texts=texts,
        token_counts=[1, 1],
        cache_identity=identity,
        cache_root=tmp_path,
    )
    assert first.calls == 1
    assert first_stats["cache_misses"] == 2
    vectors._mmap.close()

    class NoInferenceAdapter:
        def encode_batches(self, *args, **kwargs):
            raise AssertionError("a complete frozen cache must not invoke the model")

    cached, second_stats = runner._embed_document_corpus(
        adapter=NoInferenceAdapter(),
        unique_doc_shas=doc_shas,
        unique_texts=texts,
        token_counts=[1, 1],
        cache_identity=identity,
        cache_root=tmp_path,
    )
    assert second_stats["cache_hits"] == 2
    assert second_stats["cache_misses"] == 0
    assert second_stats["batches_computed"] == 0
    assert np.allclose(cached[:, 0], 1.0)
    cached._mmap.close()


def test_malformed_reader_choice_is_an_infra_failure_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [None]}

    class FakeClient:
        def post(self, *args, **kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)
    bundle = {
        "question_id": "qid",
        "arm": "dense",
        "context_bundle_sha256": "bundle-sha",
        "reader_messages": [{"role": "user", "content": "question"}],
        "reader_prompt_sha256": "prompt-sha",
        "rendered_prompt_sha256": "rendered-sha",
        "reader_prompt_tokens_preflight": 10,
        "output_reserve": 256,
    }
    prediction, call = runner._reader_call(
        arm="dense",
        bundle_row=bundle,
        question={"question": "question", "question_date": "date"},
        client=FakeClient(),
        runtime_identity={"runtime_props_sha256": "runtime-sha"},
        pre_reader_sha="pre-reader-sha",
    )
    assert call["success"] is False
    assert call["quality_status"] == "INFRA_FAILURE"
    assert call["error_type"] == "TypeError"
    assert prediction["predicted"] is None


def test_label_join_requires_both_frozen_prediction_arms_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)
    dataset_reads = []
    monkeypatch.setattr(
        runner.mem2c,
        "iter_json_array",
        lambda _path, include_gold: dataset_reads.append(include_gold)
        or [
            {
                "question_id": question_id,
                "answer": "answer",
                "answer_session_ids": [],
                "question_type": "temporal",
                "has_answer": True,
            }
            for question_id in runner.QUESTION_IDS
        ],
    )

    predictions_by_arm = {}
    ledger = []
    for arm, artifact_arm in (("dense", "dense"), ("hybrid_rrf", "hybrid")):
        bundle_rows = []
        prediction_rows = []
        for question_id in runner.QUESTION_IDS:
            bundle_sha = f"bundle-{arm}-{question_id}"
            call_identity = hashlib.sha256(f"{arm}:{question_id}".encode()).hexdigest()
            bundle_rows.append(
                {"question_id": question_id, "context_bundle_sha256": bundle_sha}
            )
            prediction_rows.append(
                {
                    "system": f"rawspan_{arm}",
                    "question_id": question_id,
                    "quality_status": "OK",
                    "final_reader_contract_sha256": runner.FINAL_READER_SHA256,
                    "context_bundle_sha256": bundle_sha,
                    "cache_identity": {"identity_sha256": call_identity},
                }
            )
            ledger.append(
                {
                    "role": "reader_answer",
                    "provider": "local_qwen",
                    "loopback_only": True,
                    "hosted_call": False,
                    "success": True,
                    "quality_status": "OK",
                    "arm": arm,
                    "question_id": question_id,
                    "context_bundle_sha256": bundle_sha,
                    "cache_identity_sha256": call_identity,
                }
            )
        runner._write_jsonl_atomic(
            tmp_path / f"{artifact_arm}_context_bundles.jsonl", bundle_rows
        )
        predictions_by_arm[arm] = prediction_rows
        if arm == "dense":
            runner._write_jsonl_atomic(tmp_path / "dense_predictions.jsonl", prediction_rows)

    runner._write_jsonl_atomic(tmp_path / "call_ledger.jsonl", ledger)
    with pytest.raises(RuntimeError, match="Prediction and call ledger SHA freeze"):
        runner._load_labels_after_freeze()
    assert dataset_reads == []

    runner._write_jsonl_atomic(
        tmp_path / "hybrid_predictions.jsonl", predictions_by_arm["hybrid_rrf"]
    )
    labels = runner._load_labels_after_freeze()
    assert set(labels) == set(runner.QUESTION_IDS)
    assert dataset_reads == [True]
