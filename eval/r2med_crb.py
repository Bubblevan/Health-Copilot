"""Clinical Reasoning Bridge view construction and BGE query semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eval.r2med_gar_generation import GeneratedView

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def bm25_query_text(method: str, query_text: str, generated_text: str) -> str:
    """R2MED's method-specific sparse-query semantics, with the sprint's Q2D repeat."""
    if method == "hyde":
        return generated_text
    if method == "query2doc":
        return f"{query_text} {query_text} {generated_text}"
    if method == "lamer":
        return f"{query_text} {generated_text}"
    raise ValueError(f"not a single-view GAR method: {method}")


def dense_query_text(method: str, query_text: str, generated_text: str) -> str:
    """R2MED's dense GAR semantics; HyDE/LameR use vector averaging elsewhere."""
    if method in {"hyde", "lamer"}:
        return generated_text
    if method == "query2doc":
        return f"{query_text}[SEP]{generated_text}"
    raise ValueError(f"not a single-view GAR method: {method}")


def crb_lexical_text(structured: Mapping[str, Any]) -> str:
    values = [structured.get("canonical_query", "")]
    for key in ("key_concepts", "disambiguating_terms"):
        values.extend(structured.get(key, []))
    return " ".join(str(value).strip() for value in values if str(value).strip())


def crb_dense_text(structured: Mapping[str, Any]) -> str:
    return str(structured.get("pseudo_evidence", "")).strip()


def dense_gar_vector(method: str, query_text: str, generated_text: str, embed) -> Any:
    """Create the one dense query vector using separate BGE query encodings."""
    if method == "query2doc":
        return embed([BGE_QUERY_PREFIX + f"{query_text}[SEP]{generated_text}"])[0]
    if method not in {"hyde", "lamer"}:
        raise ValueError(f"not a single-view GAR method: {method}")
    query_vector, generated_vector = embed(
        [BGE_QUERY_PREFIX + query_text, BGE_QUERY_PREFIX + generated_text]
    )
    return (query_vector + generated_vector) / 2.0


def crb_dense_vector(query_text: str, structured: Mapping[str, Any], embed) -> Any:
    return embed([BGE_QUERY_PREFIX + crb_dense_text(structured)])[0]


def crb_lexical_query(query_text: str, structured: Mapping[str, Any]) -> str:
    lexical = crb_lexical_text(structured)
    return lexical or query_text


def generated_view_for_method(method: str, query_text: str, view: GeneratedView) -> str:
    if view.query_id == "" or not query_text:
        raise ValueError("generated view and query must be identified")
    return view.generated_text
