"""Chunk-level lexical baseline reusing Health-Copilot's jieba BM25 contract."""

from __future__ import annotations

from typing import Any

from ..retrieval.bm25 import BM25Retriever
from ..retrieval.documents import RetrievalDocument

BM25_K1 = 1.5
BM25_B = 0.75


class HuiyiBM25:
    def __init__(self, chunks: list[dict[str, Any]], *, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.chunks = {row["chunk_id"]: row for row in chunks}
        documents = [
            RetrievalDocument(
                id=row["chunk_id"],
                title=row["title"],
                text=row["text"],
                metadata={
                    "source_url": row["source_url"],
                    "source_id": row["source_id"],
                    "document_type": row["document_type"],
                    "department": row.get("department"),
                    "topic": row.get("topic"),
                    "tags": list(row.get("section_path", [])),
                },
            )
            for row in chunks
        ]
        self.retriever = BM25Retriever(documents, k1=k1, b=b)
        self.k1 = k1
        self.b = b

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [
            {**self.chunks[item.source_id], "score": item.score}
            for item in self.retriever.search(query, top_k=top_k)
        ]

    def manifest(self, *, corpus_sha256: str) -> dict[str, Any]:
        import jieba

        return {
            "retriever": "Health-Copilot BM25Retriever",
            "tokenizer": "jieba",
            "tokenizer_version": getattr(jieba, "__version__", "unknown"),
            "tokenizer_contract": "NFKC, lowercase, jieba.lcut(cut_all=False), meaningful tokens only",
            "k1": self.k1,
            "b": self.b,
            "chunk_count": len(self.chunks),
            "corpus_sha256": corpus_sha256,
        }
