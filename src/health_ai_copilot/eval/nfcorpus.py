"""Adapter and offline metrics for the BEIR NFCorpus retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import KnowledgeCard
from ..retrieval.bm25 import BM25Retriever


@dataclass(frozen=True)
class RetrievalEvalCase:
    """A benchmark query with its external graded relevance judgments."""

    id: str
    question: str
    relevance: dict[str, int]


@dataclass(frozen=True)
class NFCorpusDataset:
    """The three BEIR artifacts after parsing, without changing their semantics."""

    documents: tuple[KnowledgeCard, ...]
    cases: tuple[RetrievalEvalCase, ...]
    split: str


def _find_file(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"could not find {filename} below {root}")
    return matches[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} line {line_number}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"expected an object in {path} line {line_number}")
        records.append(value)
    return records


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    relevance: dict[str, dict[str, int]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if fields[0].lower() in {"query-id", "query_id"}:
            continue
        if len(fields) != 3:
            raise ValueError(f"expected 3 tab-separated fields in {path} line {line_number}")
        query_id, document_id, score_text = fields
        try:
            score = int(score_text)
        except ValueError as exc:
            raise ValueError(f"invalid relevance score in {path} line {line_number}") from exc
        relevance.setdefault(query_id, {})[document_id] = score
    return relevance


def _as_eval_card(record: dict[str, Any]) -> KnowledgeCard:
    document_id = record.get("_id")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("NFCorpus corpus record needs a non-empty _id")
    title = record.get("title", "")
    text = record.get("text", "")
    if not isinstance(title, str) or not isinstance(text, str) or not (title or text):
        raise ValueError(f"NFCorpus document {document_id} needs title or text")
    return KnowledgeCard(
        id=document_id,
        title=title,
        content=text,
        source_url="https://github.com/beir-cellar/beir",
        publisher="BEIR NFCorpus",
        published_at=None,
        collected_at="external-benchmark",
        reviewed_at=None,
        reviewer="external-benchmark",
        version="beir-nfcorpus",
        expires_at=None,
        audience=["retrieval_evaluation"],
        tags=["benchmark", "biomedical", "nfcorpus"],
    )


def load_nfcorpus(data_dir: str | Path, split: str = "test") -> NFCorpusDataset:
    """Load BEIR corpus/queries/qrels into an explicit retrieval-eval adapter."""
    root = Path(data_dir)
    corpus_records = _read_jsonl(_find_file(root, "corpus.jsonl"))
    query_records = _read_jsonl(_find_file(root, "queries.jsonl"))
    qrels_path = _find_file(root, f"{split}.tsv")
    qrels = _load_qrels(qrels_path)

    documents = tuple(_as_eval_card(record) for record in corpus_records)
    cases: list[RetrievalEvalCase] = []
    for record in query_records:
        query_id = record.get("_id")
        question = record.get("text")
        if not isinstance(query_id, str) or not isinstance(question, str):
            raise TypeError("NFCorpus query records need string _id and text")
        if query_id in qrels:
            cases.append(
                RetrievalEvalCase(id=query_id, question=question, relevance=qrels[query_id])
            )
    return NFCorpusDataset(documents=documents, cases=tuple(cases), split=split)


def _ndcg_at_k(retrieved_ids: list[str], relevance: dict[str, int], k: int) -> float:
    def gain(score: int) -> float:
        return float(2**score - 1)

    dcg = sum(
        gain(relevance.get(document_id, 0)) / math.log2(rank + 2)
        for rank, document_id in enumerate(retrieved_ids[:k])
    )
    ideal_scores = sorted((score for score in relevance.values() if score > 0), reverse=True)[:k]
    ideal_dcg = sum(gain(score) / math.log2(rank + 2) for rank, score in enumerate(ideal_scores))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_nfcorpus(dataset: NFCorpusDataset, top_k: int = 10) -> dict[str, float | int]:
    """Run the current BM25 implementation and report standard IR metrics."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    retriever = BM25Retriever(dataset.documents)
    recall_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    evaluated = 0
    for case in dataset.cases:
        relevant = {document_id for document_id, score in case.relevance.items() if score > 0}
        if not relevant:
            continue
        retrieved = retriever.search(case.question, top_k=top_k)
        retrieved_ids = [item.source_id for item in retrieved]
        recall_sum += len(set(retrieved_ids).intersection(relevant)) / len(relevant)
        for rank, document_id in enumerate(retrieved_ids, 1):
            if document_id in relevant:
                mrr_sum += 1 / rank
                break
        ndcg_sum += _ndcg_at_k(retrieved_ids, case.relevance, top_k)
        evaluated += 1
    return {
        "dataset": "nfcorpus",
        "split": dataset.split,
        "num_documents": len(dataset.documents),
        "num_cases": len(dataset.cases),
        "evaluated_cases": evaluated,
        f"recall_at_{top_k}": recall_sum / evaluated if evaluated else 0.0,
        "mrr": mrr_sum / evaluated if evaluated else 0.0,
        f"ndcg_at_{top_k}": ndcg_sum / evaluated if evaluated else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate BM25 on BEIR NFCorpus")
    parser.add_argument("--data-dir", default="artifacts/benchmarks/nfcorpus")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    dataset = load_nfcorpus(args.data_dir, split=args.split)
    print(json.dumps(evaluate_nfcorpus(dataset, top_k=args.top_k), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
