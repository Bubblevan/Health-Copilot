"""Build zero-model-call MEM-1D2 provenance and reflection artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MEMEVAL_ROOT = ROOT.parent / "external" / "memory" / "MemEval"
RUN_DIR = ROOT / "runs/memory/mem1/mem1d1-frozen-10-20260927"
DATASET_PATH = ROOT / "data/longmemeval/longmemeval_s_cleaned.json"
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
MEMEVAL_SHA = "807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4"
QUESTION_IDS = (
    "1cea1afa",
    "1c549ce4",
    "778164c6",
    "fca70973",
    "a82c026e",
    "gpt4_e061b84g",
    "gpt4_f420262c",
    "8550ddae",
    "06878be2",
    "c4ea545c",
)
SYSTEMS = ("fullcontext", "openclaw", "mem0", "simplemem", "propmem")
FROZEN_ARTIFACTS = (
    "context_bundles.jsonl",
    "predictions.jsonl",
    "call_ledger.jsonl",
    "baseline_warnings.jsonl",
)
OUTPUT_DIR = ROOT / "docs/research/memory"
OVERLAY_PATH = OUTPUT_DIR / "mem_1d2_provenance_overlay.json"
METRICS_PATH = OUTPUT_DIR / "mem_1d2_retrieval_metrics.json"
PACKET_PATH = OUTPUT_DIR / "mem_1d2_reflection_packet.json"
REPORT_PATH = OUTPUT_DIR / "mem_1d2_reflection_and_provenance.md"
QUESTION_ID_PREFIX = re.compile(rb'^\s*\{\s*"question_id"\s*:\s*"([A-Za-z0-9_-]+)"')


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selected_json_array(path: Path, wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Decode only requested top-level rows from the frozen JSON array.

    Non-target rows are scanned for object boundaries and their first question_id
    field only; their question, answer, and conversation fields are not decoded.
    """
    selected: dict[str, dict[str, Any]] = {}
    with path.open("rb") as source:  # noqa: SIM117
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
            length = len(data)
            index = 0
            while index < length and data[index] in b" \t\r\n":
                index += 1
            if index >= length or data[index] != ord("["):
                raise ValueError("Frozen LongMemEval file must be a top-level JSON array")
            index += 1

            while True:
                while index < length and data[index] in b" \t\r\n,":
                    index += 1
                if index >= length:
                    raise ValueError("Frozen LongMemEval array is unterminated")
                if data[index] == ord("]"):
                    index += 1
                    break
                if data[index] != ord("{"):
                    raise ValueError("Expected a question object in frozen LongMemEval array")

                start = index
                depth = 0
                in_string = False
                escaped = False
                while index < length:
                    byte = data[index]
                    if in_string:
                        if escaped:
                            escaped = False
                        elif byte == ord("\\"):
                            escaped = True
                        elif byte == ord('"'):
                            in_string = False
                    elif byte == ord('"'):
                        in_string = True
                    elif byte == ord("{"):
                        depth += 1
                    elif byte == ord("}"):
                        depth -= 1
                        if depth == 0:
                            index += 1
                            break
                    index += 1
                else:
                    raise ValueError("Frozen LongMemEval contains an unterminated question object")

                prefix_end = min(index, start + 256)
                match = QUESTION_ID_PREFIX.match(data[start:prefix_end])
                if match is None:
                    raise ValueError("Frozen LongMemEval row no longer starts with question_id")
                question_id = match.group(1).decode("ascii")
                if question_id in wanted_ids:
                    record = json.loads(data[start:index].decode("utf-8"))
                    if record.get("question_id") != question_id:
                        raise ValueError("Selected LongMemEval row ID mismatch")
                    if question_id in selected:
                        raise ValueError(f"Duplicate frozen question ID: {question_id}")
                    selected[question_id] = record

            while index < length and data[index] in b" \t\r\n":
                index += 1
            if index != length:
                raise ValueError("Unexpected trailing bytes after LongMemEval array")

    missing = wanted_ids - set(selected)
    if missing:
        raise ValueError(f"Frozen DEV question IDs missing from LongMemEval-S: {sorted(missing)}")
    return selected


def format_with_physical_session_map(
    dialogues: list[dict[str, Any]],
    original_formatter,
) -> tuple[str, str, list[str | None], list[str | None]]:
    """Return byte-identical Markdown plus legacy and physical-line provenance maps."""
    legacy_lines: list[str] = []
    legacy_sources: list[str | None] = []
    physical_lines: list[str] = []
    physical_sources: list[str | None] = []
    current_date = None

    for dialogue in dialogues:
        timestamp = dialogue.get("timestamp", "")
        date_part = timestamp.split(" ")[0] if timestamp else ""
        if date_part and date_part != current_date:
            current_date = date_part
            legacy_lines.append(f"\n## {current_date}\n")
            legacy_sources.extend([None, None, None])
            date_lines = f"\n## {current_date}\n".split("\n")
            physical_lines.extend(date_lines)
            physical_sources.extend([None] * len(date_lines))

        speaker = dialogue.get("speaker", "Unknown")
        text = dialogue.get("text", "")
        rendered = f"**{speaker}** ({timestamp}): {text}" if timestamp else f"**{speaker}**: {text}"
        session_id = dialogue.get("session_id") or None
        legacy_lines.append(rendered)
        legacy_sources.append(session_id)
        turn_lines = rendered.split("\n")
        physical_lines.extend(turn_lines)
        physical_sources.extend([session_id] * len(turn_lines))

    legacy_markdown = "\n".join(legacy_lines)
    corrected_markdown = "\n".join(physical_lines)
    expected_markdown = original_formatter(dialogues)
    if legacy_markdown != expected_markdown:
        raise AssertionError("Legacy renderer does not match pinned upstream Markdown bytes")
    if corrected_markdown != expected_markdown:
        raise AssertionError("Physical-line rendering changed Markdown bytes")
    if len(physical_sources) != len(corrected_markdown.split("\n")):
        raise AssertionError(
            "Corrected source sidecar is not aligned to physical Markdown lines: "
            f"map={len(physical_sources)}, parts={len(physical_lines)}, "
            f"rendered={len(corrected_markdown.split(chr(10)))}"
        )
    return legacy_markdown, corrected_markdown, legacy_sources, physical_sources


def source_sessions_for_chunk(chunk, source_map: list[str | None]) -> list[str]:
    start = max(1, int(chunk.start_line))
    end = max(start, int(chunk.end_line))
    return list(dict.fromkeys(
        source_map[line_number - 1]
        for line_number in range(start, end + 1)
        if line_number <= len(source_map) and source_map[line_number - 1]
    ))


def _configure_dataset_turns(record: dict[str, Any], normalize) -> dict[str, Any]:
    normalized = normalize(record)
    conversation = normalized["conversation"]
    session_ids = record.get("haystack_session_ids", [])
    for session_index, session_id in enumerate(session_ids, 1):
        for turn in conversation.get(f"session_{session_index}", []):
            turn["session_id"] = str(session_id)
    return normalized


def _get_dialogues(normalized: dict[str, Any], extract_dialogues) -> list[dict[str, Any]]:
    conversation = normalized["conversation"]
    by_turn_id = {
        str(turn.get("dia_id", "")): str(turn.get("session_id", ""))
        for key, turns in conversation.items()
        if key.startswith("session_") and not key.endswith("_date_time")
        for turn in turns
        if isinstance(turn, dict)
    }
    dialogues = extract_dialogues(normalized)
    for dialogue in dialogues:
        dialogue["session_id"] = by_turn_id.get(str(dialogue.get("dia_id", "")), "")
    return dialogues


def _chunk_identity(chunk) -> dict[str, Any]:
    text_hash = sha256_bytes(chunk.text.encode("utf-8"))
    if getattr(chunk, "hash", text_hash) != text_hash:
        raise AssertionError("Pinned chunk hash differs from SHA256(text UTF-8)")
    return {
        "text": chunk.text,
        "text_sha256": text_hash,
        "start_line": int(chunk.start_line),
        "end_line": int(chunk.end_line),
    }


def _ordered_chunk_digest(chunks: list[dict[str, Any]]) -> str:
    identities = [
        [row["text_sha256"], row["start_line"], row["end_line"]]
        for row in chunks
    ]
    return sha256_bytes(json.dumps(identities, separators=(",", ":")).encode("utf-8"))


def _dataset_chunk_index(records, upstream_tools) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    normalize, extract_dialogues, original_formatter, chunk_markdown = upstream_tools
    by_question: dict[str, list[dict[str, Any]]] = {}
    parity_rows = []
    for question_id in QUESTION_IDS:
        normalized = _configure_dataset_turns(records[question_id], normalize)
        dialogues = _get_dialogues(normalized, extract_dialogues)
        old_markdown, corrected_markdown, legacy_map, corrected_map = format_with_physical_session_map(
            dialogues, original_formatter
        )
        old_chunks = chunk_markdown(old_markdown, tokens=400, overlap=80)
        corrected_chunks = chunk_markdown(corrected_markdown, tokens=400, overlap=80)
        old_identity = [_chunk_identity(chunk) for chunk in old_chunks]
        corrected_identity = [_chunk_identity(chunk) for chunk in corrected_chunks]
        if old_identity != corrected_identity:
            raise AssertionError(f"Chunk text/hash/order changed for {question_id}")

        indexed = []
        for chunk, identity in zip(corrected_chunks, corrected_identity, strict=True):
            indexed.append({
                **identity,
                "chunk_index": len(indexed) + 1,
                "legacy_source_session_ids": source_sessions_for_chunk(chunk, legacy_map),
                "corrected_source_session_ids": source_sessions_for_chunk(chunk, corrected_map),
            })
        by_question[question_id] = indexed
        parity_rows.append({
            "question_id": question_id,
            "physical_markdown_line_count": len(corrected_markdown.split("\n")),
            "legacy_sidecar_entry_count": len(legacy_map),
            "corrected_sidecar_entry_count": len(corrected_map),
            "old_markdown_sha256": sha256_bytes(old_markdown.encode("utf-8")),
            "corrected_markdown_sha256": sha256_bytes(corrected_markdown.encode("utf-8")),
            "old_equals_corrected_markdown_bytes": old_markdown == corrected_markdown,
            "chunk_count": len(old_identity),
            "old_ordered_chunk_identity_sha256": _ordered_chunk_digest(old_identity),
            "corrected_ordered_chunk_identity_sha256": _ordered_chunk_digest(corrected_identity),
            "chunk_text_hash_and_order_parity": True,
        })
    return by_question, parity_rows


def _overlay_context_items(
    bundle_rows, chunk_index
) -> tuple[dict[str, Any], dict[tuple[str, str, int], list[str]]]:
    overlay_rows = []
    corrected_by_item: dict[tuple[str, str, int], list[str]] = {}
    for row in bundle_rows:
        system = row["system"]
        question_id = row["question_id"]
        if system not in {"openclaw", "propmem"}:
            continue
        for index, item in enumerate(row["context_bundle"]["items"], 1):
            affected = system == "openclaw" and item["kind"] == "chunk"
            affected = affected or (system == "propmem" and item["kind"] == "fallback_chunk")
            if not affected:
                continue
            text = item["text"]
            text_hash = sha256_bytes(text.encode("utf-8"))
            original_ids = list(item.get("source_session_ids", []))
            matches = [
                chunk for chunk in chunk_index[question_id]
                if chunk["text_sha256"] == text_hash and chunk["text"] == text
            ]
            source_matches = [
                chunk for chunk in matches
                if chunk["legacy_source_session_ids"] == original_ids
            ]
            if not source_matches:
                raise AssertionError(
                    f"No exact legacy chunk/provenance match for {system}/{question_id}/item-{index}"
                )
            corrected_options = [chunk["corrected_source_session_ids"] for chunk in source_matches]
            if all(value == corrected_options[0] for value in corrected_options):
                corrected_ids = corrected_options[0]
                resolution = "EXACT_TEXT_HASH_AND_LEGACY_PROVENANCE"
            else:
                shared = set(corrected_options[0])
                for values in corrected_options[1:]:
                    shared.intersection_update(values)
                corrected_ids = [value for value in corrected_options[0] if value in shared]
                resolution = "AMBIGUOUS_IDENTICAL_CHUNK_CONSERVATIVE_INTERSECTION"
            corrected_by_item[(system, question_id, index)] = corrected_ids
            overlay_rows.append({
                "system": system,
                "question_id": question_id,
                "context_bundle_sha256": row["context_bundle"]["context_bundle_sha256"],
                "context_item_index": index,
                "item_rank": item["rank"],
                "item_kind": item["kind"],
                "item_text_sha256": text_hash,
                "item_text_characters": len(text),
                "original_source_session_ids": original_ids,
                "corrected_source_session_ids": corrected_ids,
                "candidate_corrected_source_session_ids": corrected_options,
                "matched_chunk_indices": [chunk["chunk_index"] for chunk in source_matches],
                "derivation_method": "exact text and SHA256 match to deterministic 400/80 re-chunk; replace logical-turn sidecar lookup with physical-Markdown-line mapping",
                "resolution": resolution,
            })

    overlay = {
        "artifact_version": "mem1d2-provenance-overlay-v1",
        "run_id": "mem1d1-frozen-10-20260927",
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA256,
        "frozen_context_bundle_sha256": sha256_file(RUN_DIR / "context_bundles.jsonl"),
        "derivation": {
            "source_session_map_unit": "physical Markdown line",
            "date_header_and_blank_line_provenance": None,
            "chunker": "pinned MemEval OpenClaw chunk_markdown(tokens=400, overlap=80)",
            "memeval_commit": MEMEVAL_SHA,
            "memeval_patch_sha256": json.loads((RUN_DIR / "run_manifest.json").read_text(encoding="utf-8")).get("code_patch_sha256"),
            "no_embeddings_or_models": True,
        },
        "chunk_parity_by_question": chunk_index["parity_rows"],
        "rows": overlay_rows,
    }
    return overlay, corrected_by_item


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+", normalized, flags=re.UNICODE)


def _gold_coverage(gold: str, context: str) -> tuple[float | None, bool | None, int]:
    gold_tokens = _tokens(gold)
    if not gold_tokens:
        return None, None, 0
    context_tokens = set(_tokens(context))
    covered = sum(token in context_tokens for token in set(gold_tokens))
    denominator = len(set(gold_tokens))
    return covered / denominator, _contains_token_sequence(_tokens(context), gold_tokens), denominator


def _contains_token_sequence(context_tokens: list[str], sequence: list[str]) -> bool:
    width = len(sequence)
    return any(context_tokens[index:index + width] == sequence for index in range(len(context_tokens) - width + 1))


def _retrieval_metrics_and_context(
    bundle_rows,
    predictions,
    records,
    corrected_by_item,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    bundle_lookup = {(row["system"], row["question_id"]): row["context_bundle"] for row in bundle_rows}
    rows = []
    diagnostics: dict[tuple[str, str], dict[str, Any]] = {}
    validity = {
        "fullcontext": "TRIVIAL_FULL_HISTORY_COVERAGE",
        "openclaw": "CORRECTED_CHUNK_PROVENANCE",
        "propmem": "DIRECT_PROPOSITION_PLUS_CORRECTED_FALLBACK_PROVENANCE",
        "mem0": "UNAVAILABLE",
        "simplemem": "UNAVAILABLE",
    }
    for system in SYSTEMS:
        for question_id in QUESTION_IDS:
            bundle = bundle_lookup[(system, question_id)]
            prediction = predictions[(system, question_id)]
            answer_ids = list(prediction.get("answer_session_ids") or [])
            items = sorted(bundle["items"], key=lambda item: item["rank"])
            corrected_groups = []
            all_text = []
            item_types = Counter()
            item_session_counts = {str(session): 0 for session in answer_ids}
            first_rank = {str(session): None for session in answer_ids}
            distinct_sources = set()
            for index, item in enumerate(items, 1):
                all_text.append(item["text"])
                item_types[item["kind"]] += 1
                if system == "openclaw" or (
                    system == "propmem" and item["kind"] == "fallback_chunk"
                ):
                    source_ids = corrected_by_item[(system, question_id, index)]
                else:
                    source_ids = list(item.get("source_session_ids", []))
                corrected_groups.append(source_ids)
                distinct_sources.update(source_ids)
                for session_id in set(source_ids):
                    if session_id in item_session_counts:
                        item_session_counts[session_id] += 1
                        if first_rank[session_id] is None:
                            first_rank[session_id] = item["rank"]

            provenance_available = system in {"fullcontext", "openclaw", "propmem"}
            recall_at_5 = recall_at_10 = mrr = None
            if provenance_available and answer_ids:
                expected = {str(session) for session in answer_ids}
                hit_at_5 = set().union(*(set(group) for group in corrected_groups[:5])) if corrected_groups[:5] else set()
                hit_at_10 = set().union(*(set(group) for group in corrected_groups[:10])) if corrected_groups[:10] else set()
                recall_at_5 = len(expected & hit_at_5) / len(expected)
                recall_at_10 = len(expected & hit_at_10) / len(expected)
                first_matching_rank = next((
                    item["rank"] for item, group in zip(items, corrected_groups, strict=True)
                    if expected.intersection(group)
                ), None)
                mrr = 1 / first_matching_rank if first_matching_rank else 0.0

            serialized_context = "\n".join(all_text)
            coverage, exact_present, gold_token_count = _gold_coverage(
                prediction.get("ground_truth", ""), serialized_context
            )
            diagnostics_row = {
                "system": system,
                "question_id": question_id,
                "context_bundle_sha256": bundle["context_bundle_sha256"],
                "validity_label": validity[system],
                "answer_session_recall_at_5": recall_at_5,
                "answer_session_recall_at_10": recall_at_10,
                "mrr": mrr,
                "answer_session_item_counts": item_session_counts if provenance_available else None,
                "first_rank_by_answer_session": first_rank if provenance_available else None,
                "normalized_gold_token_coverage": coverage,
                "normalized_gold_token_count": gold_token_count,
                "exact_normalized_gold_substring_present": exact_present,
                "context_reader_tokens": bundle["context_reader_tokens"],
                "distinct_source_session_count": len(distinct_sources) if provenance_available else None,
                "item_type_counts": dict(sorted(item_types.items())),
            }
            diagnostics[(system, question_id)] = diagnostics_row
            rows.append(diagnostics_row)

    aggregates = {}
    category_aggregates = {}
    for system in SYSTEMS:
        selected = [diagnostics[(system, qid)] for qid in QUESTION_IDS]
        aggregates[system] = _mean_diagnostics(selected)
        category_aggregates[system] = {}
        for category in sorted({predictions[(system, qid)]["category"] for qid in QUESTION_IDS}):
            category_rows = [row for row in selected if predictions[(system, row["question_id"])]["category"] == category]
            category_aggregates[system][category] = _mean_diagnostics(category_rows)

    return {
        "artifact_version": "mem1d2-retrieval-metrics-v1",
        "run_id": "mem1d1-frozen-10-20260927",
        "source_context_bundle_sha256": sha256_file(RUN_DIR / "context_bundles.jsonl"),
        "source_prediction_sha256": sha256_file(RUN_DIR / "predictions.jsonl"),
        "metrics_are_diagnostic_only": True,
        "original_mem1d1_openclaw_propmem_metrics": "SUPERSEDED_FOR_DIAGNOSTIC_INTERPRETATION; original MEM-1D1 files remain unchanged",
        "validity_labels": validity,
        "by_system": aggregates,
        "by_system_and_category": category_aggregates,
        "rows": rows,
    }, diagnostics


def _mean_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"n": len(rows)}
    for key in (
        "answer_session_recall_at_5",
        "answer_session_recall_at_10",
        "mrr",
        "normalized_gold_token_coverage",
        "context_reader_tokens",
        "distinct_source_session_count",
    ):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        output[key] = sum(values) / len(values) if values else None
    return output


def _session_turn_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    ids = record.get("haystack_session_ids", [])
    dates = record.get("haystack_dates", [])
    output = []
    for session_index, session in enumerate(record.get("haystack_sessions", [])):
        session_id = str(ids[session_index]) if session_index < len(ids) else f"session_{session_index + 1}"
        date = dates[session_index] if session_index < len(dates) else None
        for turn_index, turn in enumerate(session):
            if not isinstance(turn, dict):
                continue
            output.append({
                "session_id": session_id,
                "date": date,
                "turn_index": turn_index,
                "speaker": turn.get("role", "user"),
                "text": str(turn.get("content", "")),
            })
    return output


def _gold_match_spans(text: str, gold_tokens: list[str]) -> list[tuple[int, int]]:
    if not gold_tokens:
        return []
    matches = list(re.finditer(r"\w+", unicodedata.normalize("NFKC", text), flags=re.UNICODE))
    tokens = [match.group(0).casefold() for match in matches]
    width = len(gold_tokens)
    spans = []
    for index in range(len(tokens) - width + 1):
        if tokens[index:index + width] == gold_tokens:
            spans.append((matches[index].start(), matches[index + width - 1].end()))
    return spans


def _excerpt(text: str, span: tuple[int, int] | None = None, radius: int = 180) -> str:
    if span is None:
        start, end = 0, min(len(text), radius * 2)
    else:
        start = max(0, span[0] - radius)
        end = min(len(text), span[1] + radius)
    value = text[start:end]
    return ("..." if start else "") + value + ("..." if end < len(text) else "")


def _answer_evidence_snippets(record: dict[str, Any], gold: str, answer_ids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gold_tokens = _tokens(gold)
    turns = _session_turn_rows(record)
    answer_set = {str(value) for value in answer_ids}
    answer_snippets = []
    match_windows = []
    for session_id in answer_ids:
        session_turns = [turn for turn in turns if turn["session_id"] == str(session_id)]
        chosen = []
        for turn in session_turns:
            spans = _gold_match_spans(turn["text"], gold_tokens)
            if spans:
                chosen.extend((turn, span) for span in spans[:2])
        if not chosen and session_turns:
            chosen = [(session_turns[0], None)]
        for turn, span in chosen[:3]:
            answer_snippets.append({
                "session_id": turn["session_id"],
                "date": turn["date"],
                "turn_index": turn["turn_index"],
                "speaker": turn["speaker"],
                "excerpt": _excerpt(turn["text"], span),
                "normalized_gold_match_in_turn": span is not None,
                "source": "selected frozen LongMemEval-S DEV record",
            })
    for turn in turns:
        spans = _gold_match_spans(turn["text"], gold_tokens)
        for span in spans[:1]:
            match_windows.append({
                "session_id": turn["session_id"],
                "date": turn["date"],
                "turn_index": turn["turn_index"],
                "speaker": turn["speaker"],
                "excerpt": _excerpt(turn["text"], span),
                "answer_session": turn["session_id"] in answer_set,
            })
            if len(match_windows) >= 8:
                break
        if len(match_windows) >= 8:
            break
    return answer_snippets, match_windows


def _item_correction(
    system: str,
    question_id: str,
    index: int,
    item: dict[str, Any],
    corrected_by_item: dict[tuple[str, str, int], list[str]],
) -> list[str]:
    if system == "openclaw" or (system == "propmem" and item["kind"] == "fallback_chunk"):
        return list(corrected_by_item[(system, question_id, index)])
    return list(item.get("source_session_ids", []))


def _compact_items(system, question_id, bundle, corrected_by_item) -> list[dict[str, Any]]:
    items = sorted(bundle["items"], key=lambda item: item["rank"])
    output = []
    for index, item in enumerate(items[:5], 1):
        text = item["text"]
        row = {
            "context_item_index": index,
            "rank": item["rank"],
            "kind": item["kind"],
            "text_sha256": sha256_bytes(text.encode("utf-8")),
            "text_characters": len(text),
            "source_session_ids": _item_correction(system, question_id, index, item, corrected_by_item),
        }
        if item["kind"] == "full_history":
            row["excerpt"] = None
            row["omitted_full_history"] = True
        else:
            row["excerpt"] = _excerpt(text, radius=210)
            row["omitted_full_history"] = False
        output.append(row)
    return output


def _build_packet(bundle_rows, predictions, records, diagnostics, corrected_by_item) -> dict[str, Any]:
    bundle_lookup = {(row["system"], row["question_id"]): row["context_bundle"] for row in bundle_rows}
    rows = []
    for system in SYSTEMS:
        for question_id in QUESTION_IDS:
            prediction = predictions[(system, question_id)]
            bundle = bundle_lookup[(system, question_id)]
            record = records[question_id]
            answer_ids = list(prediction.get("answer_session_ids") or [])
            answer_snippets, gold_windows = _answer_evidence_snippets(
                record, prediction.get("ground_truth", ""), answer_ids
            )
            row = {
                "system": system,
                "question_id": question_id,
                "category": prediction["category"],
                "case_status": "KNOWN_GATE_CASE" if question_id == "1cea1afa" else "FRESH_DIAGNOSTIC_CASES",
                "question": prediction["question"],
                "gold_answer": prediction["ground_truth"],
                "prediction": prediction["predicted"],
                "token_metrics": {
                    "precision": prediction["token_precision"],
                    "recall": prediction["token_recall"],
                    "f1": prediction["f1"],
                    "normalized_exact_match": prediction["normalized_exact_match"],
                },
                "context_bundle_sha256": bundle["context_bundle_sha256"],
                "corrected_retrieval_metrics": diagnostics[(system, question_id)],
                "context_item_types": dict(sorted(Counter(item["kind"] for item in bundle["items"]).items())),
                "top_retrieved_items": _compact_items(system, question_id, bundle, corrected_by_item),
                "answer_session_evidence_snippets": answer_snippets,
                "normalized_gold_match_windows": gold_windows,
                "answer_session_ids": answer_ids,
                "corrected_source_session_ids_by_context_item": [
                    {
                        "context_item_index": index,
                        "rank": item["rank"],
                        "kind": item["kind"],
                        "source_session_ids": _item_correction(system, question_id, index, item, corrected_by_item),
                    }
                    for index, item in enumerate(bundle["items"], 1)
                ],
                "context_reader_tokens": bundle["context_reader_tokens"],
                "baseline_warning_types": sorted({
                    warning.get("warning_type")
                    for warning in prediction.get("baseline_warnings", [])
                }),
                "deterministic_lexical_hints": list(prediction.get("failure_attribution_hint", [])),
                "lexical_hints_are_heuristic_not_causal": True,
            }
            rows.append(row)
    return {
        "artifact_version": "mem1d2-reflection-packet-v1",
        "run_id": "mem1d1-frozen-10-20260927",
        "dataset_revision": DATASET_REVISION,
        "dataset_sha256": DATASET_SHA256,
        "source_prediction_sha256": sha256_file(RUN_DIR / "predictions.jsonl"),
        "source_context_bundle_sha256": sha256_file(RUN_DIR / "context_bundles.jsonl"),
        "no_causal_failure_labels_assigned": True,
        "rows": rows,
    }


def _cost_review(predictions, calls, warnings) -> dict[str, Any]:
    by_system = {}
    for system in SYSTEMS:
        selected = [predictions[(system, qid)] for qid in QUESTION_IDS]
        calls_for_system = [row for row in calls if row.get("system") == system]
        embeddings = [row for row in calls_for_system if row.get("role") == "embedding"]
        memory_calls = [
            row for row in calls_for_system
            if row.get("role") in {"memory_ingest", "memory_reasoning"}
        ]
        warning_rows = [row for row in warnings if row.get("system") == system]
        warning_queries = sum(bool(row.get("baseline_warnings")) for row in selected)
        by_system[system] = {
            "n_questions": len(selected),
            "average_ingestion_wall_ms": _mean([row.get("ingestion_latency_ms") for row in selected]),
            "average_retrieval_wall_ms": _mean([row.get("retrieval_latency_ms") for row in selected]),
            "average_context_reader_tokens": _mean([row.get("context_reader_tokens") for row in selected]),
            "memory_internal_llm_calls": len(memory_calls),
            "embedding_calls": len(embeddings),
            "embedding_input_tokens": sum(row.get("prompt_tokens") or 0 for row in embeddings),
            "baseline_warning_ledger_rows": len(warning_rows),
            "baseline_warning_events": sum(int(row.get("count") or 1) for row in warning_rows),
            "questions_with_warnings": warning_queries,
            "warning_rate": warning_queries / len(selected) if selected else None,
        }
    return {
        "artifact_version": "mem1d2-comparator-cost-review-v1",
        "source_call_ledger_sha256": sha256_file(RUN_DIR / "call_ledger.jsonl"),
        "source_warning_ledger_sha256": sha256_file(RUN_DIR / "baseline_warnings.jsonl"),
        "quality_cost_winner_calculated": False,
        "by_system": by_system,
    }


def _mean(values) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return sum(selected) / len(selected) if selected else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_upstream_tools():
    head = subprocess.check_output(
        ["git", "-C", str(MEMEVAL_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(MEMEVAL_ROOT), "status", "--porcelain"], text=True
    )
    if head != MEMEVAL_SHA or status.strip():
        raise RuntimeError("MemEval must be clean at the pinned commit for deterministic re-chunking")
    source_root = str(MEMEVAL_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    import importlib

    importlib.invalidate_caches()
    from agents_memory.benchmarks.longmemeval import _normalize
    from agents_memory.locomo import extract_dialogues, format_as_markdown
    from agents_memory.openclaw import chunk_markdown

    return _normalize, extract_dialogues, format_as_markdown, chunk_markdown


def _write_report(overlay, metrics, packet, costs, parity_rows, frozen_hashes) -> None:
    lines = [
        "# MEM-1D2 Reflection And Provenance Review",
        "",
        "- Gate: `MEM1D2_REFLECTION_PACKET_READY=YES`",
        "- Scope: frozen MEM-1D1 ten DEV cases only; zero model, embedding, reader, judge, or inference-provider calls.",
        f"- Dataset revision/hash: `{DATASET_REVISION}` / `{DATASET_SHA256}`; only the ten frozen DEV records were decoded.",
        "- No TEST records were decoded or analyzed; no causal failure labels were assigned.",
        "- The four MEM-1D1 evidence JSONL files remain byte-identical to their frozen SHA sidecars.",
        "",
        "## Frozen Evidence",
        "",
        "| Artifact | SHA256 |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | `{digest}` |" for name, digest in frozen_hashes.items())
    lines.extend([
        "",
        "## ContextBundle Status Semantics",
        "",
        "`context_bundle_content_valid` means each canonical bundle and prediction binding validates. `context_bundle_hash_frozen` is true only when `context_bundles.sha256` exists and verifies. This post-hoc field cleanup changes reporting semantics only; it does not change any frozen MEM-1D1 evidence row.",
        f"For this run: content valid = `{metrics['context_bundle_content_valid']}`; hash frozen = `{metrics['context_bundle_hash_frozen']}`. The historical `deterministic_metrics.json` was not overwritten; these unambiguous fields live in the D2 diagnostics artifact.",
        "",
        "## Provenance Root Cause And Parity",
        "",
        "The historical adapter attached one session ID per logical turn even when a turn rendered to multiple physical Markdown lines. The chunker indexes physical line numbers, so after the first embedded newline later line-to-session lookups could shift. Date/header/blank lines correctly remain null in the repaired map.",
        "",
        "| Question | Markdown bytes equal | Chunk count | Ordered text/hash/range digest |",
        "|---|---:|---:|---|",
    ])
    lines.extend(
        f"| {row['question_id']} | {row['old_equals_corrected_markdown_bytes']} | {row['chunk_count']} | `{row['corrected_ordered_chunk_identity_sha256']}` |"
        for row in parity_rows
    )
    ambiguous = sum(row["resolution"].startswith("AMBIGUOUS") for row in overlay["rows"])
    lines.extend([
        "",
        f"All ten old/corrected Markdown byte comparisons and all reconstructed OpenClaw chunk text/hash/order comparisons pass. The overlay contains {len(overlay['rows'])} affected OpenClaw/PropMem fallback items; {ambiguous} identical-text matches require conservative provenance intersection.",
        "",
        "OpenClaw retrieved chunks and PropMem fallback chunks are matched by exact text and SHA256. PropMem proposition session IDs are retained directly and are not rewritten. FullContext remains trivial full-history coverage; Mem0 and SimpleMem provenance remains unavailable.",
        "",
        "## Corrected Retrieval Diagnostics",
        "",
        "Original OpenClaw/PropMem MEM-1D1 recall/MRR values are `SUPERSEDED_FOR_DIAGNOSTIC_INTERPRETATION`; original artifacts remain unchanged. These corrected values are session-level diagnostics, not answer-quality claims.",
        "",
        "| System | Validity | Recall@5 | Recall@10 | MRR | Gold token coverage | Context reader tokens |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for system, row in metrics["by_system"].items():
        lines.append(
            f"| {system} | {metrics['validity_labels'][system]} | "
            f"{_format(row['answer_session_recall_at_5'])} | {_format(row['answer_session_recall_at_10'])} | "
            f"{_format(row['mrr'])} | {_format(row['normalized_gold_token_coverage'])} | "
            f"{_format(row['context_reader_tokens'])} |"
        )
    lines.extend([
        "",
        "## Category Separation",
        "",
        "| Category | LongMemEval type(s) | Questions | Interpretation boundary |",
        "|---|---|---:|---|",
        "| Single-session factual recall | single-session-user, single-session-assistant | 3 | Direct fact surfacing; not preference abstraction. |",
        "| Multi-session | multi-session | 1 | Cross-session composition; not reducible to temporal ordering. |",
        "| Temporal reasoning | temporal-reasoning | 2 | Event ordering/relations. |",
        "| Knowledge update | knowledge-update | 2 | Current versus prior state. |",
        "| Preference | single-session-preference | 2 | Synthesis/abstraction; not evidence for temporal revision by itself. |",
        "",
        "The fixed sample is descriptive and sparse by category; no rebalancing or causal classification was performed.",
        "",
        "## Comparator Cost",
        "",
        "| System | Avg ingestion ms | Avg retrieval ms | Avg context reader tokens | Memory-internal LLM calls | Embedding calls/tokens | Warning rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for system, row in costs["by_system"].items():
        lines.append(
            f"| {system} | {_format(row['average_ingestion_wall_ms'])} | {_format(row['average_retrieval_wall_ms'])} | "
            f"{_format(row['average_context_reader_tokens'])} | {row['memory_internal_llm_calls']} | "
            f"{row['embedding_calls']}/{row['embedding_input_tokens']} | {row['warning_rate']:.2%} |"
        )
    lines.extend([
        "",
        "This is descriptive cost accounting only. No quality/cost winner is computed.",
        "",
        "## Reflection Packet",
        "",
        f"The packet contains {len(packet['rows'])} rows with frozen answers, compact context items, answer-session evidence excerpts, corrected provenance, context diagnostics, warning types, and heuristic lexical hints. Long FullContext items are represented by hashes and local evidence windows, never copied wholesale.",
        "No Reflection-causal labels (for example `RETRIEVAL_EVIDENCE_MISS` or `TEMPORAL_ORDERING_FAIL`) were assigned.",
        "",
        "## M10 And RevMem Boundary",
        "",
        "The current M10 substrate already has typed `MemoryRecord`, ADD/UPDATE/DELETE/NOOP, versions, SUPERSEDED status, `supersedes_id`, `valid_from`/`valid_until`/expiry, an active-state materialized view, scope/objective/intent filtering, and deterministic lexical retrieval (see `src/health_ai_copilot/runtime/memory.py` and `src/health_ai_copilot/runtime/context_manager.py`).",
        "Retire the provisional name `M10-Flat`; the future public-benchmark adapter is `M10-Base`. No M10-Base code is implemented in MEM-1D2.",
        "Any earlier locked protocol occurrence of `M10-Flat` is retained as historical terminology only and is superseded by this naming decision.",
        "",
        "Candidate hypotheses only: H1 compact context may reduce attention dilution; H2 session hits may miss answer-bearing items; H3 knowledge updates may need explicit current/history resolution; H4 CURRENT/AS_OF/CHANGE may be distinct intents; H5 temporal revision alone may not solve multi-session composition; H6 preference may need separate synthesis policy. All remain untested until Reflection review.",
        "",
        "## Artifacts",
        "",
        f"- Provenance overlay: `{OVERLAY_PATH.relative_to(ROOT).as_posix()}`",
        f"- Retrieval/context metrics: `{METRICS_PATH.relative_to(ROOT).as_posix()}`",
        f"- Reflection packet: `{PACKET_PATH.relative_to(ROOT).as_posix()}`",
        "",
        "STOP: no M10-Base, RevMem, 102 DEV, TEST, judge, embedding, SFT, or RL stage was started.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _format(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def build(run_dir: Path = RUN_DIR, dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    if run_dir.resolve() != RUN_DIR.resolve():
        raise ValueError("MEM-1D2 accepts only the canonical frozen MEM-1D1 run directory")
    if sha256_file(dataset_path) != DATASET_SHA256:
        raise RuntimeError("LongMemEval-S bytes differ from the frozen MEM-1D1 dataset")

    from mem1_artifacts import read_jsonl, verify_hash_sidecar

    frozen_hashes = {}
    for name in FROZEN_ARTIFACTS:
        artifact = run_dir / name
        sidecar = run_dir / name.replace(".jsonl", ".sha256")
        if not verify_hash_sidecar(artifact, sidecar):
            raise RuntimeError(f"Frozen MEM-1D1 evidence hash failed before D2: {name}")
        frozen_hashes[name] = sha256_file(artifact)

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset", {}).get("sha256") != DATASET_SHA256 or manifest.get("test_access") is not False:
        raise RuntimeError("MEM-1D1 manifest is not the frozen DEV-only source")
    records = load_selected_json_array(dataset_path, set(QUESTION_IDS))
    if set(records) != set(QUESTION_IDS):
        raise RuntimeError("Selected data differs from the exact frozen ten DEV IDs")

    upstream_tools = _load_upstream_tools()
    chunk_index, parity_rows = _dataset_chunk_index(records, upstream_tools)
    bundle_rows = read_jsonl(run_dir / "context_bundles.jsonl")
    prediction_rows = read_jsonl(run_dir / "predictions.jsonl")
    predictions = {}
    for row in prediction_rows:
        predictions[(row["system"], row["question_id"])] = row
    expected = {(system, question_id) for system in SYSTEMS for question_id in QUESTION_IDS}
    if set(predictions) != expected or len(predictions) != 50:
        raise RuntimeError("MEM-1D2 requires exactly 50 frozen predictions")
    if {(row["system"], row["question_id"]) for row in bundle_rows} != expected:
        raise RuntimeError("MEM-1D2 requires exactly 50 frozen ContextBundles")

    from context_bundle import verify_context_bundle

    content_valid = all(
        verify_context_bundle(row["context_bundle"])
        and predictions[(row["system"], row["question_id"])].get("context_bundle_sha256")
        == row["context_bundle"].get("context_bundle_sha256")
        for row in bundle_rows
    )
    context_sidecar = run_dir / "context_bundles.sha256"
    hash_frozen = context_sidecar.is_file() and verify_hash_sidecar(
        run_dir / "context_bundles.jsonl", context_sidecar
    )

    overlay_index = {"parity_rows": parity_rows, **chunk_index}
    overlay, corrected_by_item = _overlay_context_items(bundle_rows, overlay_index)
    retrieval_metrics, diagnostics = _retrieval_metrics_and_context(
        bundle_rows, predictions, records, corrected_by_item
    )
    retrieval_metrics["context_bundle_content_valid"] = content_valid
    retrieval_metrics["context_bundle_hash_frozen"] = hash_frozen
    packet = _build_packet(bundle_rows, predictions, records, diagnostics, corrected_by_item)
    calls = read_jsonl(run_dir / "call_ledger.jsonl")
    warnings = read_jsonl(run_dir / "baseline_warnings.jsonl")
    costs = _cost_review(predictions, calls, warnings)

    _write_json(OVERLAY_PATH, overlay)
    _write_json(METRICS_PATH, retrieval_metrics)
    _write_json(PACKET_PATH, packet)
    _write_report(overlay, retrieval_metrics, packet, costs, parity_rows, frozen_hashes)

    for name, digest in frozen_hashes.items():
        if sha256_file(run_dir / name) != digest:
            raise RuntimeError(f"MEM-1D2 unexpectedly modified frozen evidence: {name}")
    return {
        "overlay_rows": len(overlay["rows"]),
        "packet_rows": len(packet["rows"]),
        "markdown_parity_cases": sum(row["old_equals_corrected_markdown_bytes"] for row in parity_rows),
        "chunk_parity_cases": sum(row["chunk_text_hash_and_order_parity"] for row in parity_rows),
        "frozen_hashes": frozen_hashes,
        "costs": costs,
        "retrieval_metrics": retrieval_metrics["by_system"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    args = parser.parse_args()
    summary = build(args.run_dir.resolve(), args.dataset.resolve())
    print("MEM1D2_REFLECTION_PACKET_READY=YES")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
