from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval.rag_e5.qualification import (
    bind_active_corpus_identity,
    exact_dense_rankings,
    validate_dense_matrix,
)


def _dense_rows() -> tuple[np.ndarray, list[str]]:
    matrix = np.eye(3, dtype=np.float32)
    return matrix, ["doc-a", "doc-b", "doc-c"]


def test_dense_integrity_does_not_depend_on_relevance_rank() -> None:
    matrix, ids = _dense_rows()
    diagnostic_hits_at_10 = 8  # Semantic quality is deliberately not an integrity input.

    report = validate_dense_matrix(
        matrix,
        expected_rows=3,
        expected_dimension=3,
        document_ids=ids,
        manifest_document_order_sha256="order-sha",
        observed_document_order_sha256="order-sha",
    )

    assert diagnostic_hits_at_10 == 8
    assert report["document_count"] == 3
    assert report["finite_vector_count"] == 3


def test_dense_embedding_row_count_exact() -> None:
    matrix, ids = _dense_rows()
    with pytest.raises(ValueError, match="shape"):
        validate_dense_matrix(
            matrix[:2],
            expected_rows=3,
            expected_dimension=3,
            document_ids=ids,
            manifest_document_order_sha256="order-sha",
            observed_document_order_sha256="order-sha",
        )


def test_dense_embeddings_all_finite() -> None:
    matrix, ids = _dense_rows()
    matrix[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_dense_matrix(
            matrix,
            expected_rows=3,
            expected_dimension=3,
            document_ids=ids,
            manifest_document_order_sha256="order-sha",
            observed_document_order_sha256="order-sha",
        )


def test_dense_embeddings_normalized_and_nonzero() -> None:
    matrix, ids = _dense_rows()
    matrix[0] *= 2
    with pytest.raises(ValueError, match="normalization"):
        validate_dense_matrix(
            matrix,
            expected_rows=3,
            expected_dimension=3,
            document_ids=ids,
            manifest_document_order_sha256="order-sha",
            observed_document_order_sha256="order-sha",
        )
    matrix[0] = 0
    with pytest.raises(ValueError, match="normalization|zero vector"):
        validate_dense_matrix(
            matrix,
            expected_rows=3,
            expected_dimension=3,
            document_ids=ids,
            manifest_document_order_sha256="order-sha",
            observed_document_order_sha256="order-sha",
        )


def test_dense_document_order_bound() -> None:
    matrix, ids = _dense_rows()
    with pytest.raises(ValueError, match="document order"):
        validate_dense_matrix(
            matrix,
            expected_rows=3,
            expected_dimension=3,
            document_ids=ids,
            manifest_document_order_sha256="expected-order",
            observed_document_order_sha256="other-order",
        )


def test_dense_search_replay_deterministic() -> None:
    matrix, ids = _dense_rows()
    queries = [np.array([0.8, 0.6, 0], dtype=np.float32)]
    first = exact_dense_rankings(matrix, queries, ids)
    second = exact_dense_rankings(matrix, queries, ids)

    assert first == second
    assert first[0][0][0] == "doc-a"
    assert all(np.isfinite(score) for _, score in first[0])


def test_semantic_smoke_retained_as_diagnostic() -> None:
    report = json.loads(
        Path("runs/rag_e5/e5a31_retrieval_diagnostic.json").read_text(encoding="utf-8")
    )

    public_health = report["views"]["PUBLIC_HEALTH_ONLY"]
    assert public_health["dense"]["hits_at_10"] == 8
    assert public_health["dense"]["query_count"] == 10
    assert [row["target_rank"] for row in public_health["dense"]["misses"]] == [18, 15]
    assert report["diagnostic_only"] is True
    assert report["controls_index_readiness"] is False


def test_a3_smoke_results_immutable() -> None:
    report = json.loads(
        Path("runs/rag_e5/e5a3_activation_report.json").read_text(encoding="utf-8")
    )
    diagnostic = json.loads(
        Path("runs/rag_e5/e5a31_retrieval_diagnostic.json").read_text(encoding="utf-8")
    )

    assert report["A3_FROZEN_PUBLIC_HEALTH_BGE_HITS_AT_10"] == 8
    assert report["A3_FROZEN_MISS_RANKS"] == [18, 15]
    assert report["A3_FROZEN_PUBLIC_HEALTH_BGE_HITS_AT_10"] == diagnostic["views"][
        "PUBLIC_HEALTH_ONLY"
    ]["dense"]["hits_at_10"]


def test_active_corpus_equals_candidate_corpus() -> None:
    corpus = json.loads(Path("runs/rag_e5/external_corpus_manifest.json").read_text())
    index = json.loads(Path("runs/rag_e5/external_index_manifest.json").read_text())

    assert corpus["active_external_corpus_identity"] == corpus[
        "candidate_external_corpus_identity"
    ]
    assert index["external_corpus_identity"] == index["candidate_external_corpus_identity"]
    assert index["external_corpus_identity"] == corpus["active_external_corpus_identity"]


def test_method_hash_unchanged_after_binding() -> None:
    identity = "a" * 64
    expected = {"STANDARD": "b" * 64, "STRONG": "c" * 64}
    registry = {
        "external_corpus_identity": None,
        "profiles": [
            {"action": "STANDARD", "config_sha256": expected["STANDARD"]},
            {"action": "STRONG", "config_sha256": expected["STRONG"]},
        ],
    }

    bound = bind_active_corpus_identity(
        registry, corpus_identity=identity, expected_config_sha256=expected
    )

    assert bound["external_corpus_identity"] == identity
    assert [row["config_sha256"] for row in bound["profiles"]] == list(expected.values())
    assert registry["external_corpus_identity"] is None
