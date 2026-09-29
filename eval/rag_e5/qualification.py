"""Pure integrity helpers for frozen E5-A3.1 retrieval artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from itertools import pairwise
from typing import Any

import numpy as np


def validate_dense_matrix(
    matrix: np.ndarray,
    *,
    expected_rows: int,
    expected_dimension: int,
    document_ids: Sequence[str],
    manifest_document_order_sha256: str,
    observed_document_order_sha256: str,
) -> dict[str, Any]:
    """Validate dense artifact structure without consulting relevance labels."""
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("dense embedding matrix must be two-dimensional")
    if values.shape != (expected_rows, expected_dimension):
        raise ValueError("dense embedding matrix shape does not match the corpus view")
    if len(document_ids) != expected_rows or len(set(document_ids)) != expected_rows:
        raise ValueError("document IDs must be unique and match the expected row count")
    finite_rows = np.isfinite(values).all(axis=1)
    norms = np.linalg.norm(values.astype(np.float64, copy=False), axis=1)
    normalized_rows = np.abs(norms - 1.0) <= 1e-5
    zero_rows = norms == 0.0
    if manifest_document_order_sha256 != observed_document_order_sha256:
        raise ValueError("dense document order differs from its index manifest")
    if not finite_rows.all():
        raise ValueError("dense matrix contains a non-finite value")
    if not normalized_rows.all():
        raise ValueError("dense matrix contains a vector outside the normalization tolerance")
    if zero_rows.any():
        raise ValueError("dense matrix contains a zero vector")
    return {
        "document_count": len(document_ids),
        "embedding_row_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "finite_vector_count": int(finite_rows.sum()),
        "normalized_vector_count": int(normalized_rows.sum()),
        "zero_vector_count": int(zero_rows.sum()),
        "document_order_sha_match": True,
    }


def exact_dense_rankings(
    matrix: np.ndarray,
    query_vectors: Sequence[np.ndarray],
    document_ids: Sequence[str],
) -> list[list[tuple[str, float]]]:
    """Run deterministic exact-flat cosine ranking over normalized vectors."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(document_ids):
        raise ValueError("dense matrix rows must match document IDs")
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("document IDs must be unique")
    rankings: list[list[tuple[str, float]]] = []
    for query in query_vectors:
        vector = np.asarray(query, dtype=np.float32)
        if vector.shape != (values.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("query vector must be finite and match the index dimension")
        norm = float(np.linalg.norm(vector.astype(np.float64, copy=False)))
        if abs(norm - 1.0) > 1e-5:
            raise ValueError("query vector must be normalized")
        scores = values @ vector
        if not np.isfinite(scores).all():
            raise ValueError("exact dense search returned non-finite scores")
        indices = sorted(
            range(len(document_ids)), key=lambda index: (-float(scores[index]), document_ids[index])
        )
        ranking = [(document_ids[index], float(scores[index])) for index in indices]
        if len({item[0] for item in ranking}) != len(document_ids):
            raise ValueError("dense search returned duplicate document IDs")
        if any(left[1] < right[1] for left, right in pairwise(ranking)):
            raise ValueError("dense search scores are not sorted descending")
        rankings.append(ranking)
    return rankings


def bind_active_corpus_identity(
    profile_registry: dict[str, Any],
    *,
    corpus_identity: str,
    expected_config_sha256: dict[str, str],
) -> dict[str, Any]:
    """Bind a validated corpus without changing any frozen action config."""
    result = deepcopy(profile_registry)
    profiles = result.get("profiles")
    if not isinstance(profiles, list):
        raise TypeError("profile registry must contain a profile list")
    found: set[str] = set()
    for row in profiles:
        if not isinstance(row, dict) or row.get("action") not in expected_config_sha256:
            continue
        action = str(row["action"])
        if action in found:
            raise ValueError(f"duplicate frozen action profile: {action}")
        found.add(action)
        if row.get("config_sha256") != expected_config_sha256[action]:
            raise ValueError(f"frozen {action} method config identity changed")
    if found != set(expected_config_sha256):
        raise ValueError("profile registry is missing a frozen external action")
    result["external_corpus_identity"] = corpus_identity
    return result
