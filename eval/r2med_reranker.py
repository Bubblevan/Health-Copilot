"""Qrels-blind cross-encoder scoring and depth-limited reranking for R2MED."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from eval.r2med_candidate_union import Candidate, source_agreement

MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MAX_PAIR_TOKENS = 512
INFERENCE_BATCH_SIZE = 16
RERANK_DEPTHS = (20, 30, 50)
AAR_ALPHAS = (0.0, 0.1, 0.2)


def aar_is_eligible(dual_source_macro_ndcg: float, lamer_reranked_macro_ndcg: float) -> bool:
    """AAR is allowed only after the frozen DualSource+BGE parent has positive signal."""
    return dual_source_macro_ndcg > lamer_reranked_macro_ndcg


def select_best_dev_arm(
    summaries: Mapping[str, Mapping[str, Any]],
    configs: Mapping[str, Mapping[str, Any]],
    *,
    candidates: Sequence[str] | None = None,
) -> str:
    """Select using the frozen macro-nDCG, MRR, Recall, depth, and lambda tie-breaks."""
    names = list(candidates) if candidates is not None else list(summaries)
    if not names or any(name not in summaries or name not in configs for name in names):
        raise ValueError("DEV selection requires summaries and configs for every candidate")

    def key(name: str) -> tuple[float, float, float, int, float, float, str]:
        metrics = summaries[name]["macro_equal_subset_weight"]
        config = configs[name]
        lambda_value = config.get("lambda")
        lambda_value = 1.0 if lambda_value is None else float(lambda_value)
        return (
            float(metrics["ndcg@10"]),
            float(metrics["mrr@10"]),
            float(metrics["recall@10"]),
            -int(config.get("depth", 0)),
            -abs(lambda_value - 1.0),
            -float(config.get("alpha", 0.0)),
            name,
        )

    return max(names, key=key)


@dataclass(frozen=True)
class RerankedEntry:
    candidate: Candidate
    reranker_score: float | None
    final_score: float | None


class PairScorer(Protocol):
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


def rerank_top_k(
    candidates: Sequence[Candidate],
    scores_by_doc: Mapping[str, float],
    *,
    depth: int,
    agreement_alpha: float = 0.0,
) -> list[RerankedEntry]:
    """Rerank exactly the first K candidates; preserve the untouched tail order."""
    if depth not in RERANK_DEPTHS:
        raise ValueError(f"rerank depth must be one of {RERANK_DEPTHS}")
    if depth > len(candidates):
        raise ValueError("rerank depth exceeds candidate count")
    if agreement_alpha not in AAR_ALPHAS:
        raise ValueError(f"agreement alpha must be one of {AAR_ALPHAS}")
    if len({candidate.doc_id for candidate in candidates}) != len(candidates):
        raise ValueError("reranking candidates must be deduplicated")

    head = list(candidates[:depth])
    missing = [candidate.doc_id for candidate in head if candidate.doc_id not in scores_by_doc]
    if missing:
        raise KeyError(f"missing reranker scores for {len(missing)} top-K candidates")
    scored: list[tuple[int, Candidate, float, float]] = []
    for original_rank, candidate in enumerate(head):
        score = float(scores_by_doc[candidate.doc_id])
        if not math.isfinite(score):
            raise ValueError(f"non-finite reranker score for {candidate.doc_id}")
        final_score = score + agreement_alpha * source_agreement(candidate)
        scored.append((original_rank, candidate, score, final_score))
    reranked = sorted(scored, key=lambda item: (-item[3], item[0], item[1].doc_id))
    entries = [RerankedEntry(candidate, score, final) for _, candidate, score, final in reranked]
    entries.extend(RerankedEntry(candidate, None, None) for candidate in candidates[depth:])
    return entries


def pair_cache_key(
    *,
    query_id: str,
    doc_id: str,
    query_text: str,
    document_text: str,
    model_revision: str = MODEL_REVISION,
    max_pair_tokens: int = MAX_PAIR_TOKENS,
) -> str:
    identity = {
        "query_id": query_id,
        "doc_id": doc_id,
        "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
        "document_sha256": hashlib.sha256(document_text.encode("utf-8")).hexdigest(),
        "model_revision": model_revision,
        "max_pair_tokens": max_pair_tokens,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_model_files(model_dir: Path) -> dict[str, dict[str, str | int]]:
    """Hash the exact local weight and tokenizer/config files used for inference."""
    required = (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
    )
    files: dict[str, dict[str, str | int]] = {}
    for name in required:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"required pinned reranker file is missing: {path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        files[name] = {"bytes": size, "sha256": digest.hexdigest()}
    return files


class BGEReranker:
    """Offline-only half-precision CUDA inference wrapper."""

    def __init__(
        self,
        model_dir: Path,
        *,
        batch_size: int = INFERENCE_BATCH_SIZE,
        max_pair_tokens: int = MAX_PAIR_TOKENS,
    ) -> None:
        if batch_size != INFERENCE_BATCH_SIZE or max_pair_tokens != MAX_PAIR_TOKENS:
            raise ValueError("inference settings differ from the frozen protocol")
        self.model_files = hash_model_files(model_dir)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("the frozen reranker protocol requires CUDA")
        self.torch = torch
        self.batch_size = batch_size
        self.max_pair_tokens = max_pair_tokens
        self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if any(not query or not document for query, document in pairs):
            raise ValueError("reranker inputs must contain non-empty query/document text")
        all_scores: list[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                tokenized = self.tokenizer(
                    [query for query, _ in batch],
                    [document for _, document in batch],
                    max_length=self.max_pair_tokens,
                    truncation="longest_first",
                    padding=True,
                    return_tensors="pt",
                )
                tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
                logits = self.model(**tokenized).logits.reshape(-1).float().cpu().tolist()
                if len(logits) != len(batch) or any(not math.isfinite(float(score)) for score in logits):
                    raise RuntimeError("reranker returned invalid scores")
                all_scores.extend(float(score) for score in logits)
        return all_scores


def load_score_cache(path: Path) -> dict[str, float]:
    """Read append-only pair-score cache; this function never opens qrels."""
    scores: dict[str, float] = {}
    if not path.exists():
        return scores
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            cache_key, score = row.get("cache_key"), row.get("reranker_score")
            if not isinstance(cache_key, str) or isinstance(score, bool) or not isinstance(score, (float, int)):
                raise TypeError(f"invalid reranker cache record at {path}:{line_number}")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError(f"non-finite cached score at {path}:{line_number}")
            previous = scores.get(cache_key)
            if previous is not None and previous != numeric_score:
                raise ValueError(f"conflicting cached score for {cache_key}")
            scores[cache_key] = numeric_score
    return scores


def append_score_cache(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    known_scores: Mapping[str, float] | None = None,
) -> None:
    """Append only unseen score identities; never replace an existing cache."""
    existing = dict(known_scores) if known_scores is not None else load_score_cache(path)
    additions: list[Mapping[str, Any]] = []
    seen: dict[str, float] = dict(existing)
    for row in rows:
        key, score = row.get("cache_key"), row.get("reranker_score")
        if not isinstance(key, str) or isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("invalid reranker cache row")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("reranker cache rows must contain finite scores")
        if key in seen:
            if seen[key] != value:
                raise ValueError(f"refusing to change cached score for {key}")
            continue
        seen[key] = value
        additions.append(row)
    if not additions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in additions:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
