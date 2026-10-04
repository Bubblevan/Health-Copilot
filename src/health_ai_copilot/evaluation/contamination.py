"""Deterministic contamination candidate discovery; confirmation stays human-reviewed."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from typing import Any


@dataclass(frozen=True)
class ContaminationCandidate:
    train_id: str
    eval_id: str
    kind: str
    score: float
    confirmed: bool | None = None


def normalize_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    lines = normalized.splitlines()
    option_lines: list[str] = []
    question_lines: list[str] = []
    for line in lines:
        match = re.match(r"\s*[a-z][.)、:：]\s*(.+)\s*$", line)
        if match:
            option_lines.append(" ".join(re.findall(r"\w+", match.group(1), flags=re.UNICODE)))
        else:
            question_lines.append(" ".join(re.findall(r"\w+", line, flags=re.UNICODE)))
    body = " ".join(item for item in question_lines if item)
    options = " <options> " + " <option> ".join(sorted(option_lines)) if option_lines else ""
    return body + options


def audit_contamination(
    training_rows: Iterable[Mapping[str, Any]],
    evaluation_rows: Iterable[Mapping[str, Any]],
    *,
    lexical_threshold: float = 0.8,
    semantic_threshold: float = 0.9,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict[str, Any]:
    training = [(str(row["id"]), normalize_question(str(row["question"]))) for row in training_rows]
    evaluation = [(str(row["id"]), normalize_question(str(row["question"]))) for row in evaluation_rows]
    by_hash: dict[str, list[str]] = {}
    for train_id, question in training:
        by_hash.setdefault(sha256(question.encode("utf-8")).hexdigest(), []).append(train_id)
    candidates: list[ContaminationCandidate] = []
    for eval_id, question in evaluation:
        digest = sha256(question.encode("utf-8")).hexdigest()
        candidates.extend(
            ContaminationCandidate(train_id, eval_id, "exact", 1.0, None)
            for train_id in by_hash.get(digest, ())
        )

    train_shingles = {train_id: _shingles(text) for train_id, text in training}
    for eval_id, question in evaluation:
        eval_shingles = _shingles(question)
        if not eval_shingles:
            continue
        for train_id, _ in training:
            other = train_shingles[train_id]
            if not other:
                continue
            score = len(eval_shingles & other) / len(eval_shingles | other)
            if score >= lexical_threshold and score < 1.0:
                candidates.append(ContaminationCandidate(train_id, eval_id, "lexical", score))

    semantic_pairs: list[tuple[str, str, float]] = []
    if embed is not None and training and evaluation:
        vectors = embed([text for _, text in training] + [text for _, text in evaluation])
        if len(vectors) != len(training) + len(evaluation):
            raise ValueError("embed must return one vector per question")
        train_vectors = vectors[:len(training)]
        eval_vectors = vectors[len(training):]
        for (eval_id, _), eval_vector in zip(evaluation, eval_vectors, strict=True):
            for (train_id, _), train_vector in zip(training, train_vectors, strict=True):
                score = _cosine(train_vector, eval_vector)
                if score >= semantic_threshold:
                    semantic_pairs.append((train_id, eval_id, score))
                    candidates.append(ContaminationCandidate(train_id, eval_id, "semantic", score))

    counts = {
        kind: sum(item.kind == kind for item in candidates)
        for kind in ("exact", "lexical", "semantic")
    }
    return {
        "status": "COMPLETE" if training and evaluation else "BLOCKED_MISSING_ROWS",
        "training_rows": len(training),
        "evaluation_rows": len(evaluation),
        "normalized_question_sha256": {
            "training": sorted({sha256(text.encode("utf-8")).hexdigest() for _, text in training}),
            "evaluation": sorted({sha256(text.encode("utf-8")).hexdigest() for _, text in evaluation}),
        },
        "candidate_counts": counts,
        "confirmed_overlaps": 0,
        "removed_training_rows": [],
        "semantic_candidates_require_human_review": True,
        "pretraining_absence_claim": "NOT_ASSESSED",
        "candidates": [item.__dict__ for item in candidates],
    }


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    tokens = text.split()
    return {tuple(tokens[index:index + n]) for index in range(max(0, len(tokens) - n + 1))}


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    norm_left = sqrt(sum(item * item for item in left))
    norm_right = sqrt(sum(item * item for item in right))
    if not norm_left or not norm_right:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (norm_left * norm_right)
