"""One-pass, DEV-only compact-schema variant for the frozen CRB-Q queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.r2med_crb_data import (
    SOURCE_MANIFEST_PATH,
    load_partition_inputs,
    load_source_manifest,
    read_jsonl,
    sha256_file,
)
from eval.r2med_gar_generation import GENERATION_CONFIG, LocalLlamaCppClient
from tools.generate_r2med_gar import (
    DEFAULT_SERVER,
    PORT,
    _check_qwen,
    _llama_version,
    _server_ready,
    _start_server,
)

DEFAULT_SOURCE_ROOT = Path(r"E:\Health-Copilot-E1.2\sources")
PARENT_GENERATION_ROOT = ROOT / "runs/rag_r2med_crb/generation/dev/crb_q"
REPAIR_ROOT = ROOT / "runs/rag_r2med_crb/compact_repair"
GENERATION_ROOT = REPAIR_ROOT / "generation"
MANIFEST_PATH = GENERATION_ROOT / "generation_manifest.json"

COMPACT_PROMPT_TEMPLATE = """You create a compact retrieval bridge for biomedical literature search.

Do NOT answer the question, choose an option, or summarize the patient case.
Return only one JSON object with exactly these keys:
- q: a concise medical evidence-search query, 8-30 words; preserve the actual clinical question.
- t: up to 4 short, discriminative biomedical search terms; each should be at most 4 words.
- e: one hypothetical evidence sentence, at most 30 words, describing what a useful passage would discuss.

Do not copy the full case narrative, timeline, demographics, or lab list. Use only details needed to retrieve evidence. Do not invent diagnostic certainty.

Question:
{QUERY}"""

COMPACT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["q", "t", "e"],
    "properties": {
        "q": {"type": "string"},
        "t": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
        "e": {"type": "string"},
    },
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact_prompt(query: str) -> str:
    if not query.strip():
        raise ValueError("query must be non-empty")
    return COMPACT_PROMPT_TEMPLATE.format(QUERY=query)


def normalize_compact_payload(value: Any) -> dict[str, Any] | None:
    """Validate and normalize the compact JSON without reading labels or answers."""
    if not isinstance(value, dict) or set(value) != {"q", "t", "e"}:
        return None
    canonical_query, terms, evidence = value["q"], value["t"], value["e"]
    if not isinstance(canonical_query, str) or not canonical_query.strip():
        return None
    if len(canonical_query.split()) > 40:
        return None
    if not isinstance(terms, list) or len(terms) > 4:
        return None
    if any(not isinstance(term, str) or not term.strip() or len(term.split()) > 4 for term in terms):
        return None
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence.split()) > 40:
        return None
    return {
        "canonical_query": canonical_query.strip(),
        "key_concepts": [term.strip() for term in terms],
        "disambiguating_terms": [],
        "pseudo_evidence": evidence.strip(),
    }


def _fallback(query: str) -> dict[str, Any]:
    return {
        "canonical_query": query,
        "key_concepts": [],
        "disambiguating_terms": [],
        "pseudo_evidence": query,
    }


def _generation_record(query_id: str, query: str, client: LocalLlamaCppClient) -> dict[str, Any]:
    truncated = False
    completed = False
    output_tokens = 0
    finish_reason: str | None = None
    try:
        response = client.complete(compact_prompt(query), json_schema=COMPACT_JSON_SCHEMA)
        raw_text = str(response.get("text", ""))
        finish_reason = response.get("finish_reason")
        truncated = finish_reason == "length"
        completed = not truncated
        output_tokens = int(response.get("output_tokens", 0))
        try:
            compact_value = json.loads(raw_text)
        except json.JSONDecodeError:
            compact_value = None
        structured = normalize_compact_payload(compact_value)
        valid = structured is not None
        error = None if valid else "invalid_compact_json_or_constraints"
    except (OSError, RuntimeError, TimeoutError, ValueError, TypeError) as exc:
        raw_text = ""
        structured = None
        valid = False
        error = type(exc).__name__

    normalized = structured if structured is not None else _fallback(query)
    return {
        "query_id": query_id,
        "method": "crb_q_compact_repair",
        "generated_text": json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
        "raw_output": raw_text,
        "valid": valid,
        "completed": completed,
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "fallback_original": not valid,
        "structured": normalized,
        "error": error,
        "raw_output_sha256": _sha256_text(raw_text),
    }


def _validate_parent_generation(inputs) -> str:
    parent_manifest_path = PARENT_GENERATION_ROOT / "generation_manifest.json"
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(parent_manifest_path)
    parent_manifest_bytes = parent_manifest_path.read_bytes()
    parent_manifest = json.loads(parent_manifest_bytes)
    if parent_manifest.get("partition") != "DEV" or parent_manifest.get("method") != "crb_q":
        raise ValueError("compact repair only accepts the pinned CRB-Q DEV parent generation")
    inputs_by_subset = {subset.name: subset for subset in inputs}
    inputs_by_subset = {subset.name: subset for subset in inputs}
    for entry in parent_manifest["subsets"]:
        subset = entry["subset"]
        path = Path(entry["artifact"])
        if sha256_file(path) != entry["artifact_sha256"]:
            raise ValueError(f"parent CRB-Q artifact hash mismatch: {path}")
        input_ids = {query.query_id for query in inputs_by_subset[subset].queries}
        rows = read_jsonl(path)
        if {row["query_id"] for row in rows} != input_ids:
            raise ValueError(f"parent CRB-Q query IDs differ from DEV inputs: {subset}")
    return hashlib.sha256(parent_manifest_bytes).hexdigest()


def _append_record(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--llama-server", type=Path, default=Path(DEFAULT_SERVER))
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    diagnostic_path = ROOT / "runs/rag_r2med_crb/dev/valid_fallback_diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if diagnostic["partition"] != "DEV" or diagnostic["test_accessed"]:
        raise ValueError("compact recovery requires the offline DEV-only diagnostic")
    if diagnostic["variants"]["crb_q"]["valid_stratum_paired_delta_vs_fallback"]["signal"] != "POSITIVE":
        raise ValueError("compact recovery is forbidden without a positive CRB-Q valid-stratum signal")
    if MANIFEST_PATH.exists():
        raise FileExistsError(f"refusing a second compact-schema repair: {MANIFEST_PATH}")
    if not args.llama_server.is_file():
        raise FileNotFoundError(f"llama.cpp server executable not found: {args.llama_server}")

    load_source_manifest(SOURCE_MANIFEST_PATH)
    source_sha = hashlib.sha256(SOURCE_MANIFEST_PATH.read_bytes()).hexdigest()
    inputs = load_partition_inputs("DEV", source_root=args.source_root)
    parent_manifest_sha = _validate_parent_generation(inputs)
    query_by_subset = {
        subset.name: {query.query_id: query.text for query in subset.queries}
        for subset in inputs
    }
    prompt_sha = _sha256_text(COMPACT_PROMPT_TEMPLATE)
    GENERATION_ROOT.mkdir(parents=True, exist_ok=True)
    query_ids_by_subset = {
        subset.name: [query.query_id for query in subset.queries]
        for subset in inputs
    }
    final_paths = {subset: GENERATION_ROOT / f"{subset}.jsonl" for subset in query_ids_by_subset}
    partial_paths = {subset: path.with_suffix(path.suffix + ".partial") for subset, path in final_paths.items()}
    if any(path.exists() for path in final_paths.values()):
        raise FileExistsError("compact repair artifacts already exist; refusing duplicate model calls")

    pending_by_subset: dict[str, list[dict[str, Any]]] = {}
    recovered_rows: dict[str, list[dict[str, Any]]] = {}
    already_written = 0
    for subset, target_query_ids in query_ids_by_subset.items():
        partial = partial_paths[subset]
        existing_rows = read_jsonl(partial) if partial.exists() else []
        targets = set(target_query_ids)
        if any(
            row.get("method") != "crb_q_compact_repair"
            or row.get("repair_prompt_sha256") != prompt_sha
            or row.get("query_id") not in targets
            for row in existing_rows
        ):
            raise ValueError(f"partial compact-repair checkpoint is incompatible: {partial}")
        if len({row["query_id"] for row in existing_rows}) != len(existing_rows):
            raise ValueError(f"duplicate query in compact-repair checkpoint: {partial}")
        recovered_rows[subset] = existing_rows
        already_written += len(existing_rows)
        written_ids = {row["query_id"] for row in existing_rows}
        pending_by_subset[subset] = [query_id for query_id in target_query_ids if query_id not in written_ids]

    qwen_identity = _check_qwen()
    llama_version = _llama_version(args.llama_server)
    log_path = Path(tempfile.gettempdir()) / "health-copilot-crb-compact-repair-llama.log"
    process = None
    log_handle = None
    calls_this_invocation = 0
    started = time.monotonic()
    try:
        process, log_handle = _start_server(args.llama_server, args.port, log_path)
        _server_ready(args.port)
        client = LocalLlamaCppClient(f"http://127.0.0.1:{args.port}/v1")
        total_pending = sum(map(len, pending_by_subset.values()))
        print(f"compact repair: {total_pending} local Qwen requests pending", flush=True)
        for subset, target_query_ids in query_ids_by_subset.items():
            pending = pending_by_subset[subset]
            for query_id in pending:
                record = _generation_record(query_id, query_by_subset[subset][query_id], client)
                record["subset"] = subset
                record["repair_prompt_sha256"] = prompt_sha
                _append_record(partial_paths[subset], record)
                recovered_rows[subset].append(record)
                calls_this_invocation += 1
                if calls_this_invocation % 20 == 0 or calls_this_invocation == total_pending:
                    valid_count = sum(row["valid"] is True for rows in recovered_rows.values() for row in rows)
                    print(
                        f"compact repair progress: {calls_this_invocation}/{total_pending}; "
                        f"valid={valid_count}",
                        flush=True,
                    )
            recovered_rows[subset].sort(key=lambda row: target_query_ids.index(row["query_id"]))
            partial_paths[subset].replace(final_paths[subset])
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_handle is not None:
            log_handle.close()

    output_entries = []
    total_valid = 0
    total_truncated = 0
    total_failures = 0
    for subset, path in final_paths.items():
        rows = read_jsonl(path)
        if len(rows) != len(query_ids_by_subset[subset]):
            raise ValueError(f"compact recovery count mismatch: {subset}")
        total_valid += sum(row["valid"] is True for row in rows)
        total_truncated += sum(row["truncated"] is True for row in rows)
        total_failures += sum(row["valid"] is not True for row in rows)
        output_entries.append(
            {
                "subset": subset,
                "query_count": len(rows),
                "artifact": str(path.resolve()),
                "artifact_sha256": sha256_file(path),
                "query_ids_sha256": _sha256_text("\n".join(row["query_id"] for row in rows)),
            }
        )

    manifest = {
        "schema_version": "r2med-crb-compact-repair-generation-v1",
        "partition": "DEV",
        "method": "crb_q_compact_repair",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "single compact-schema generation pass over every DEV query; no selective fallback-only regeneration",
        "test_accessed": False,
        "generator": {**qwen_identity, "llama_cpp_version": llama_version, "endpoint_bind": "127.0.0.1"},
        "generation_config": GENERATION_CONFIG,
        "compact_prompt_sha256": prompt_sha,
        "parent_crb_q_manifest_sha256": parent_manifest_sha,
        "dataset_source_manifest_sha256": source_sha,
        "attempted_query_count": sum(map(len, query_ids_by_subset.values())),
        "calls_this_invocation": calls_this_invocation,
        "calls_resumed_from_checkpoints": already_written,
        "valid_output_count": total_valid,
        "failure_count": total_failures,
        "truncation_count": total_truncated,
        "valid_rate": total_valid / sum(map(len, query_ids_by_subset.values())),
        "elapsed_seconds": time.monotonic() - started,
        "subsets": output_entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "attempted_query_count", "calls_this_invocation", "valid_output_count", "failure_count",
        "truncation_count", "valid_rate", "elapsed_seconds"
    )}, indent=2))
    print(f"generation manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
