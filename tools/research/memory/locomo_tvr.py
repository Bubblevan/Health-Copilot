"""Deterministic query-time temporal projection over retrieved PropMem propositions."""

from __future__ import annotations

import calendar
import re
import time
from datetime import date, datetime, timedelta
from typing import Any

SAME_TOPIC_COSINE_THRESHOLD = 0.85

_CHANGE_CUES = re.compile(
    r"\b(?:change|changes|changed|changing|increase|increased|increases|"
    r"decrease|decreased|decreases|used\s+to|difference\s+over\s+time)\b|"
    r"\bfrom\b.{1,80}\bto\b",
    re.IGNORECASE,
)
_CURRENT_CUES = re.compile(
    r"\b(?:current|currently|now|latest|most\s+recent|at\s+present|"
    r"these\s+days|still)\b",
    re.IGNORECASE,
)
_EXPLICIT_DATE = re.compile(
    r"\b(?P<prefix>as\s+of|by|before|on|in|during)\s+"
    r"(?P<date>(?:\d{4}-\d{1,2}-\d{1,2})|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,\s*|\s+)\d{4}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{4}|\d{4})\b",
    re.IGNORECASE,
)

TEMPORAL_READER_INSTRUCTIONS = (
    "A temporal validity view may appear before the ordinary evidence.\n"
    "For CURRENT questions, prefer the current candidate when older history conflicts.\n"
    "For AS-OF questions, answer from the candidate valid at the requested time.\n"
    "For CHANGE questions, use the ordered history.\n"
    "Older history remains evidence of past state and must not be treated as current."
)


def _parse_explicit_date(value: str) -> date | None:
    value = " ".join(value.strip().replace(",", " ").split())
    for fmt in ("%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%B %Y", "%b %Y", "%Y"):
        try:
            parsed = datetime.strptime(value, fmt).date()
        except ValueError:
            continue
        if fmt in {"%B %Y", "%b %Y"}:
            return date(parsed.year, parsed.month, calendar.monthrange(parsed.year, parsed.month)[1])
        if fmt == "%Y":
            return date(parsed.year, 12, 31)
        return parsed
    return None


def detect_temporal_intent(question: str) -> dict[str, Any]:
    """Use question text only; ambiguous or unparseable as-of targets fail closed."""
    if _CHANGE_CUES.search(question):
        return {"mode": "CHANGE", "target_date": None}

    matches = list(_EXPLICIT_DATE.finditer(question))
    if matches:
        if len(matches) != 1:
            return {"mode": "NONE", "target_date": None}
        match = matches[0]
        target = _parse_explicit_date(match.group("date"))
        if target is None:
            return {"mode": "NONE", "target_date": None}
        if match.group("prefix").lower() == "before":
            target -= timedelta(days=1)
        return {"mode": "AS_OF", "target_date": target.isoformat()}

    if _CURRENT_CUES.search(question):
        return {"mode": "CURRENT", "target_date": None}
    return {"mode": "NONE", "target_date": None}


def _connected_components(
    propositions: list[dict[str, Any]],
    embeddings: list[list[float]],
    threshold: float,
) -> list[list[int]]:
    if len(propositions) != len(embeddings):
        raise ValueError("Each retrieved proposition must have one embedding")
    import numpy as np

    parent = list(range(len(propositions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    vectors = [np.asarray(vector, dtype=np.float64) for vector in embeddings]
    norms = [float(np.linalg.norm(vector)) for vector in vectors]
    for left in range(len(propositions)):
        for right in range(left + 1, len(propositions)):
            if str(propositions[left].get("entity", "")).casefold() != str(
                propositions[right].get("entity", "")
            ).casefold():
                continue
            denom = norms[left] * norms[right]
            cosine = float(np.dot(vectors[left], vectors[right]) / denom) if denom else 0.0
            if cosine >= threshold:
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    parent[max(left_root, right_root)] = min(left_root, right_root)

    groups: dict[int, list[int]] = {}
    for index in range(len(propositions)):
        groups.setdefault(find(index), []).append(index)
    return [groups[key] for key in sorted(groups)]


def project_temporal_view(
    question: str,
    propositions: list[dict[str, Any]],
    embeddings: list[list[float]],
    *,
    threshold: float = SAME_TOPIC_COSINE_THRESHOLD,
) -> dict[str, Any]:
    """Return reader-only state without mutating the retrieved propositions or store."""
    started = time.perf_counter()
    intent = detect_temporal_intent(question)
    mode = intent["mode"]
    original_ids = [str(item["memory_id"]) for item in propositions]
    if mode == "NONE":
        return {
            **intent,
            "view_text": "",
            "visible_memory_ids": original_ids,
            "component_memory_ids": [],
            "resolver_latency_ms": (time.perf_counter() - started) * 1000,
        }

    components = _connected_components(propositions, embeddings, threshold)
    target_ordinal = date.fromisoformat(intent["target_date"]).toordinal() if intent["target_date"] else None
    lines = [f"[TEMPORAL {mode} VIEW]"]
    visible_ids = list(original_ids)
    projected_groups: list[list[str]] = []

    for component in components:
        if len(component) < 2:
            continue
        timed = [i for i in component if int(propositions[i].get("date_ordinal") or 0) > 0]
        dates = {int(propositions[i]["date_ordinal"]) for i in timed}
        if len(dates) < 2:
            continue
        ordered = sorted(timed, key=lambda i: (int(propositions[i]["date_ordinal"]), i))
        projected_groups.append([original_ids[i] for i in ordered])
        lines.append(f"Entity: {propositions[ordered[0]].get('entity') or 'unknown'}")

        if mode == "CURRENT":
            newest = max(int(propositions[i]["date_ordinal"]) for i in ordered)
            candidates = [i for i in ordered if int(propositions[i]["date_ordinal"]) == newest]
            history = [i for i in reversed(ordered) if int(propositions[i]["date_ordinal"]) < newest]
            lines.append("Current candidate:")
            lines.extend(_format_proposition(propositions[i]) for i in candidates)
            lines.append("Earlier history:")
            lines.extend(_format_proposition(propositions[i]) for i in history)
        elif mode == "AS_OF":
            assert target_ordinal is not None
            valid = [i for i in ordered if int(propositions[i]["date_ordinal"]) <= target_ordinal]
            future = [i for i in ordered if int(propositions[i]["date_ordinal"]) > target_ordinal]
            if valid:
                latest = max(int(propositions[i]["date_ordinal"]) for i in valid)
                candidates = [i for i in valid if int(propositions[i]["date_ordinal"]) == latest]
                history = [i for i in reversed(valid) if int(propositions[i]["date_ordinal"]) < latest]
                lines.append(f"AS-OF candidate valid at {intent['target_date']}:")
                lines.extend(_format_proposition(propositions[i]) for i in candidates)
                lines.append("Earlier history:")
                lines.extend(_format_proposition(propositions[i]) for i in history)
            else:
                lines.append(f"No retrieved proposition is dated on or before {intent['target_date']}.")
            if future:
                lines.append("Later history - do not use for the as-of answer:")
                lines.extend(_format_proposition(propositions[i]) for i in future)
                future_ids = {original_ids[i] for i in future}
                visible_ids = [memory_id for memory_id in visible_ids if memory_id not in future_ids]
        else:
            lines.append("Ordered history:")
            lines.extend(_format_proposition(propositions[i]) for i in ordered)

    if not projected_groups:
        lines.append("No same-topic multi-date proposition group was retrieved.")

    return {
        **intent,
        "view_text": "\n".join(lines),
        "visible_memory_ids": visible_ids,
        "component_memory_ids": projected_groups,
        "resolver_latency_ms": (time.perf_counter() - started) * 1000,
    }


def _format_proposition(proposition: dict[str, Any]) -> str:
    return f"- [{proposition.get('date') or 'unknown date'}] {proposition.get('text', '')}"


def inject_temporal_view(prompt: str, view_text: str) -> str:
    if not view_text:
        return prompt
    marker = re.search(r"(?m)^Known facts(?: about .+| from conversation)?:", prompt)
    if marker is None:
        marker = re.search(r"(?m)^Known facts: \(none extracted\)", prompt)
    if marker is None:
        raise ValueError("Could not find PropMem evidence boundary for temporal projection")
    return (
        prompt[: marker.start()]
        + TEMPORAL_READER_INSTRUCTIONS
        + "\n\n"
        + view_text
        + "\n\n"
        + prompt[marker.start() :]
    )
