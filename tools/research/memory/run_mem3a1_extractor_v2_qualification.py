"""MEM-3A.1 frozen 12-session local writer qualification; no benchmark scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
TOOLS_DIR = ROOT / "tools" / "research" / "memory"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_mem1d3_reader as reader_runtime
import run_mem2b_rank_aware_projection as mem2b
import run_mem2c_rawspan as mem2c
from flat_proposition_writer_v2 import (
    WriterQualificationFailure,
    WriterRecoveryError,
    atomic_write_bytes,
    build_source_span_catalog,
    canonical_json,
    catalog_sha256,
    execute_or_resume,
    sha256_bytes,
    writer_request,
)
from mem1_artifacts import read_jsonl

RUN_ID = "mem3a1-extractor-v2-qualification-20260928"
RUN_DIR = ROOT / "runs" / "memory" / "mem3" / RUN_ID
CACHE_ROOT = ROOT / ".cache" / "health-copilot" / "mem3a1-writer"
MEM2_ROOT = ROOT / "runs" / "memory" / "mem2"
MEM2A_DIR = MEM2_ROOT / "mem2a-m10-base-10-20260928"
MEM2B_DIR = MEM2_ROOT / "mem2b-rank-aware-projection-10-20260928"
MEM2C_DIR = MEM2_ROOT / "mem2c-rawspan-10-20260928"
MEM2D_DIR = MEM2_ROOT / "mem2d-semantic-retrieval-10-20260928"
INVENTORY_PATH = MEM2C_DIR / "memory_inventory.jsonl"
MEM2C_MANIFEST_PATH = MEM2C_DIR / "run_manifest.json"
MEM2D_MANIFEST_PATH = MEM2D_DIR / "run_manifest.json"
READER_CONTRACT_PATH = ROOT / "docs" / "research" / "memory" / "final_reader_contract.json"
READER_PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem_1d4_final_reader_contract.md"
PROMPT_PATH = ROOT / "docs" / "research" / "memory" / "flat_proposition_extractor_v2.txt"
CONTRACT_PATH = ROOT / "docs" / "research" / "memory" / "flat_proposition_extractor_v2_contract.json"
READER_ENDPOINT = "http://127.0.0.1:8081/v1"
READER_MODEL = "health-memory-qwen3-8b"
PINNED_BASE_COMMIT = "5e4f48109800f1d23a843ea88fdcf4834310a5e7"
PINNED_READER_SHA256 = "57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3"
PINNED_RAWSPAN_SHA256 = "93414251555c003e3acaa51968f5cf07a85ba3a189743a66301dd20ed7e09c26"
PINNED_WRITER_MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
HISTORICAL_FAILED_IDENTITY = "25267ea359060788693f6abf37b1731af749145149afb1927a3f39f538ff19a1"
CONTEXT_LIMIT = 131072
OUTPUT_RESERVE = 4096
REQUIRED_QUALIFICATION_COUNT = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(".sha256")


def sidecar_candidates(path: Path) -> tuple[Path, Path]:
    return sidecar_path(path), Path(str(path) + ".sha256")


def write_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = sidecar_path(path)
    content = f"{digest}  {path.name}\n".encode("ascii")
    atomic_write_bytes(sidecar, content)
    return digest


def verify_sidecar(path: Path) -> str:
    candidates = [candidate for candidate in sidecar_candidates(path) if candidate.is_file()]
    if not path.is_file() or not candidates:
        raise RuntimeError(f"missing frozen artifact or SHA sidecar: {path}")
    actual = sha256_file(path)
    for sidecar in candidates:
        expected = sidecar.read_text(encoding="ascii").split()[0]
        if expected != actual:
            raise RuntimeError(f"SHA sidecar mismatch: {path}")
    return actual


def write_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)
    return write_sidecar(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    atomic_write_bytes(path, payload)
    return write_sidecar(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def _verify_source_freeze(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"required v2 contract file missing: {path}")
    digest = sha256_file(path)
    sidecars = [candidate for candidate in sidecar_candidates(path) if candidate.exists()]
    if sidecars:
        for sidecar in sidecars:
            expected = sidecar.read_text(encoding="ascii").split()[0]
            if digest != expected:
                raise RuntimeError(f"frozen v2 source artifact drift: {path}")
    else:
        atomic_write_bytes(sidecar_path(path), f"{digest}  {path.name}\n".encode("ascii"))
    return digest


def _verify_upstream() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != PINNED_BASE_COMMIT:
        raise RuntimeError(f"MEM-3A.1 requires base {PINNED_BASE_COMMIT}; found {head}")
    reader_protocol = READER_PROTOCOL_PATH.read_text(encoding="utf-8")
    if "MEM1D4_FINAL_READER_CONTRACT_FROZEN=YES" not in reader_protocol:
        raise RuntimeError("MEM-1D4 final reader gate marker missing")
    reader_sha = verify_sidecar(READER_CONTRACT_PATH)
    if reader_sha != PINNED_READER_SHA256:
        raise RuntimeError("MEM-1D4 reader contract hash differs from the frozen identity")

    m2a_manifest = read_json(MEM2A_DIR / "run_manifest.json")
    m2b_manifest = read_json(MEM2B_DIR / "run_manifest.json")
    m2c_manifest = read_json(MEM2C_MANIFEST_PATH)
    m2d_manifest = read_json(MEM2D_MANIFEST_PATH)
    for name, manifest in (("MEM-2A", m2a_manifest), ("MEM-2B", m2b_manifest)):
        if manifest.get("status") != "COMPLETE" or manifest.get("gate", {}).get("passed") is not True:
            raise RuntimeError(f"{name} required frozen diagnostic gate failed")
        if manifest.get("test_access") is not False:
            raise RuntimeError(f"{name} TEST access is not explicitly false")
    if (
        m2b_manifest.get("question_ids") != list(mem2c.QUESTION_IDS)
        or m2b_manifest.get("gate", {}).get("102_dev_not_run") is not True
        or m2b_manifest.get("final_reader_contract_sha256") != reader_sha
    ):
        raise RuntimeError("MEM-2B frozen-ten gate identity mismatch")
    if (
        m2c_manifest.get("status") != "COMPLETE"
        or m2c_manifest.get("completion_gate_marker") != "MEM2C_RAWSPAN_FROZEN_10_DIAGNOSTIC=YES"
        or m2c_manifest.get("gate", {}).get("passed") is not True
        or m2c_manifest.get("test_access") is not False
        or m2c_manifest.get("gate", {}).get("102_dev_run") is not False
        or m2c_manifest.get("final_reader_contract_sha256") != reader_sha
    ):
        raise RuntimeError("MEM-2C RawSpan gate failed")
    inventory_sha = sha256_file(INVENTORY_PATH)
    if inventory_sha != PINNED_RAWSPAN_SHA256 or m2c_manifest.get("artifacts_sha256", {}).get("memory_inventory.jsonl") != inventory_sha:
        raise RuntimeError("Frozen MEM-2C RawSpan inventory identity mismatch")
    if (
        m2d_manifest.get("status") != "COMPLETE"
        or m2d_manifest.get("completion_gate_marker") != "MEM2D_SEMANTIC_RETRIEVAL_FROZEN_10_DIAGNOSTIC=YES"
        or m2d_manifest.get("gate", {}).get("passed") is not True
        or m2d_manifest.get("test_access") is not False
        or m2d_manifest.get("102_dev_run") is not False
    ):
        raise RuntimeError("MEM-2D semantic retrieval gate failed")
    if m2d_manifest.get("mem2c_inventory_sha256") != inventory_sha:
        raise RuntimeError("MEM-2D did not bind the frozen MEM-2C inventory")
    return {
        "base_commit_sha": head,
        "reader_contract_sha256": reader_sha,
        "mem2b_gate": "MEM2B_RANK_AWARE_PROJECTION_FROZEN_10_DIAGNOSTIC=YES",
        "mem2b_manifest_sha256": sha256_file(MEM2B_DIR / "run_manifest.json"),
        "mem2c_gate": m2c_manifest["completion_gate_marker"],
        "mem2c_manifest_sha256": sha256_file(MEM2C_MANIFEST_PATH),
        "mem2c_inventory_sha256": inventory_sha,
        "mem2d_gate": m2d_manifest["completion_gate_marker"],
        "mem2d_manifest_sha256": sha256_file(MEM2D_MANIFEST_PATH),
        "labels_loaded": False,
        "test_access": False,
    }


def _read_source_sessions() -> dict[str, dict[str, Any]]:
    scoped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(INVENTORY_PATH):
        qid = row.get("question_id")
        if qid not in mem2c.QUESTION_IDS or row.get("scope_id") != f"longmemeval:{qid}":
            raise RuntimeError("MEM-2C inventory escaped the frozen ten DEV scopes")
        value = row.get("value")
        if not isinstance(value, dict) or set(value) != {
            "role",
            "session_date",
            "content",
            "source_turn_index",
            "source_span_index",
        }:
            raise RuntimeError("Frozen RawSpan value schema changed")
        sid = row.get("source_session_id")
        if not isinstance(sid, str) or not sid:
            raise RuntimeError("RawSpan source session ID missing")
        key = (qid, sid)
        item = scoped.setdefault(
            key,
            {"session_date": value["session_date"], "spans": []},
        )
        if item["session_date"] != value["session_date"]:
            raise RuntimeError("RawSpan session has inconsistent date")
        item["spans"].append(
            {
                "source_turn_index": value["source_turn_index"],
                "source_span_index": value["source_span_index"],
                "char_start": row.get("char_start"),
                "char_end": row.get("char_end"),
                "role": value["role"],
                "content": value["content"],
            }
        )

    sessions: dict[str, dict[str, Any]] = {}
    for (qid, source_session_id), raw in scoped.items():
        catalog = build_source_span_catalog(raw["spans"])
        by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for span in catalog:
            by_turn[span["source_turn_index"]].append(span)
        turns = []
        for turn_index, turn_spans in sorted(by_turn.items()):
            roles = {span["role"] for span in turn_spans}
            if len(roles) != 1:
                raise RuntimeError("RawSpan roles disagree within a source turn")
            turns.append(
                {
                    "turn_index": turn_index,
                    "role": next(iter(roles)),
                    "content": "".join(span["content"] for span in turn_spans),
                }
            )
        identity = sha256_bytes(
            canonical_json(
                {
                    "session_date": raw["session_date"],
                    "turns": turns,
                }
            )
        )
        catalog_sha = catalog_sha256(catalog)
        existing = sessions.get(identity)
        if existing is not None:
            if (
                existing["session_date"] != raw["session_date"]
                or existing["catalog_sha256"] != catalog_sha
            ):
                raise RuntimeError("Source identity collision in frozen RawSpan inventory")
            existing["source_session_ids"].append(source_session_id)
            existing["scope_count"] += 1
            continue
        sessions[identity] = {
            "session_identity_sha256": identity,
            "session_date": raw["session_date"],
            "turns": turns,
            "catalog": catalog,
            "catalog_sha256": catalog_sha,
            "source_session_ids": [source_session_id],
            "scope_count": 1,
        }
    if len(scoped) != 477 or len(sessions) != 477:
        raise RuntimeError(f"Expected 477 scoped/unique sessions; found {len(scoped)}/{len(sessions)}")
    return dict(sorted(sessions.items()))


def _session_diagnostics(session: dict[str, Any], prompt_tokens: int) -> dict[str, Any]:
    catalog = session["catalog"]
    char_counts = defaultdict(int)
    for span in catalog:
        char_counts[span["role"]] += len(span["content"])
    total_chars = sum(char_counts.values())
    roles = set(char_counts)
    mixed = roles == {"user", "assistant"}
    return {
        "session_identity_sha256": session["session_identity_sha256"],
        "session_date": session["session_date"],
        "span_count": len(catalog),
        "turn_count": len(session["turns"]),
        "prompt_tokens": prompt_tokens,
        "prompt_fits_with_output_reserve": prompt_tokens + OUTPUT_RESERVE <= CONTEXT_LIMIT,
        "truncated": False,
        "catalog_sha256": session["catalog_sha256"],
        "user_characters": char_counts.get("user", 0),
        "assistant_characters": char_counts.get("assistant", 0),
        "mixed_roles": mixed,
        "user_character_fraction": char_counts.get("user", 0) / total_chars if total_chars else 0.0,
        "assistant_character_fraction": char_counts.get("assistant", 0) / total_chars if total_chars else 0.0,
    }


def _select_qualification(preflight: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["session_identity_sha256"]: row for row in preflight}
    if HISTORICAL_FAILED_IDENTITY not in by_id:
        raise RuntimeError("Historical v1 failed session missing from v2 source catalog")
    ordered: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)

    def add(identity: str, reason: str) -> None:
        if identity not in ordered:
            ordered.append(identity)
        reasons[identity].append(reason)

    add(HISTORICAL_FAILED_IDENTITY, "historical_v1_failed_session")

    def choose_maximum(rows: list[dict[str, Any]], field: str, reason: str) -> None:
        selected = min(rows, key=lambda row: (-row[field], row["session_identity_sha256"]))
        add(selected["session_identity_sha256"], reason)

    choose_maximum(preflight, "prompt_tokens", "maximum_v2_prompt_tokens")
    choose_maximum(preflight, "span_count", "maximum_source_span_count")
    choose_maximum(preflight, "turn_count", "maximum_turn_count")
    mixed = [row for row in preflight if row["mixed_roles"]]
    if not mixed:
        raise RuntimeError("No mixed-role sessions for structural selection")
    choose_maximum(mixed, "assistant_character_fraction", "maximum_assistant_character_fraction_mixed")
    choose_maximum(mixed, "user_character_fraction", "maximum_user_character_fraction_mixed")

    remaining = [row for row in preflight if row["session_identity_sha256"] not in ordered]
    hash_order = sorted(
        remaining,
        key=lambda row: (
            hashlib.sha256(
                ("mem3a1-qualification-v1:" + row["session_identity_sha256"]).encode("ascii")
            ).hexdigest(),
            row["session_identity_sha256"],
        ),
    )
    for row in hash_order:
        if len(ordered) >= REQUIRED_QUALIFICATION_COUNT:
            break
        add(row["session_identity_sha256"], "deterministic_hash_fill")
    if len(ordered) != REQUIRED_QUALIFICATION_COUNT:
        raise RuntimeError(f"Qualification selection has {len(ordered)} sessions, expected 12")
    return {
        "selection_version": "mem3a1-qualification-v1",
        "selection_rule": "historical failure, five structural maxima with identity-ascending tie-break, then deterministic SHA256 fill",
        "session_identities": [
            {
                "session_identity_sha256": identity,
                "reasons": reasons[identity],
                "prompt_tokens": by_id[identity]["prompt_tokens"],
                "catalog_sha256": by_id[identity]["catalog_sha256"],
            }
            for identity in ordered
        ],
        "labels_used": False,
        "question_ids_persisted": False,
    }


def _runtime_manifest(client: httpx.Client, mem2a_manifest: dict[str, Any]) -> dict[str, Any]:
    runtime = mem2b._verify_reader(client, mem2a_manifest)
    if runtime.get("endpoint") != READER_ENDPOINT or runtime.get("model_sha256") != PINNED_WRITER_MODEL_SHA256:
        raise RuntimeError("Live writer/reader runtime does not match the frozen local model")
    return runtime


def _tokenize_preflight(client: httpx.Client, sessions: dict[str, dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    rows = []
    for identity, session in sessions.items():
        request = writer_request(
            session_date=session["session_date"],
            catalog=session["catalog"],
            system_prompt=prompt,
            model_alias=READER_MODEL,
        )
        messages = request["messages"]
        prompt_tokens, rendered_sha = reader_runtime._render_and_tokenize(client, messages)
        row = _session_diagnostics(session, prompt_tokens)
        row["rendered_prompt_sha256"] = rendered_sha
        row["request_sha256"] = sha256_bytes(canonical_json(request))
        row["dynamic_schema_sha256"] = sha256_bytes(
            canonical_json(request["response_format"]["json_schema"]["schema"])
        )
        rows.append(row)
        if prompt_tokens + OUTPUT_RESERVE > CONTEXT_LIMIT:
            raise RuntimeError(f"v2 prompt exceeds context reserve: {identity} ({prompt_tokens})")
    if len(rows) != 477 or any(row["truncated"] for row in rows):
        raise RuntimeError("All 477 v2 tokenizer preflights must pass without truncation")
    return rows


def _provider(client: httpx.Client):
    def call(request: dict[str, Any]) -> tuple[int, bytes, str | None]:
        response = client.post(f"{READER_ENDPOINT}/chat/completions", json=request)
        return response.status_code, response.content, response.headers.get("content-type")

    return call


def _review_packet(
    session: dict[str, Any], normalized: dict[str, Any]
) -> dict[str, Any]:
    return {
        "session_identity_sha256": session["session_identity_sha256"],
        "session_date": session["session_date"],
        "review": {
            "durable": None,
            "semantically_supported": None,
            "overgeneralized": None,
            "transient_request": None,
            "assistant_hypothetical_leak": None,
            "notes": None,
        },
        "propositions": [
            {
                **proposition,
                "review": {
                    "durable": None,
                    "semantically_supported": None,
                    "overgeneralized": None,
                    "transient_request": None,
                    "assistant_hypothetical_leak": None,
                    "notes": None,
                },
            }
            for proposition in normalized["propositions"]
        ],
    }


def _diagnostics(packets: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    propositions = [prop for packet in packets for prop in packet["propositions"]]
    kind_counts = Counter(prop["memory_kind"] for prop in propositions)
    refs = [len(prop["evidence_refs"]) for prop in propositions]
    candidate_reuse = Counter(
        (prop["entity_key_candidate"], prop["attribute_key_candidate"])
        for prop in propositions
    )
    return {
        "qualified_sessions": len(packets),
        "sessions_with_zero_propositions": sum(not packet["propositions"] for packet in packets),
        "total_propositions": len(propositions),
        "propositions_per_session": len(propositions) / len(packets) if packets else 0.0,
        "memory_kind_distribution": dict(sorted(kind_counts.items())),
        "mean_evidence_refs_per_proposition": sum(refs) / len(refs) if refs else 0.0,
        "candidate_key_reuse": {
            f"{entity}.{attribute}": count
            for (entity, attribute), count in sorted(candidate_reuse.items())
        },
        "gate_thresholds_applied": False,
    }


def _report(gate: str, preflight_count: int, selection: dict[str, Any], manifest: dict[str, Any], diagnostics: dict[str, Any] | None, failure: dict[str, Any] | None = None) -> str:
    lines = [
        "# MEM-3A.1 - Provenance-by-Reference Extractor v2 Qualification",
        "",
        f"Gate: {gate}",
        "",
        "This is a writer-contract qualification, not a benchmark run. It performs no proposition materialization, embedding, retrieval, reader calls, label access, answer scoring, revision grouping, MEM-3B, or RevMem work.",
        "",
        "## Frozen Protocol",
        "",
        f"- Source: frozen MEM-2C raw-span-v1 inventory; SHA256 {manifest['upstream']['mem2c_inventory_sha256']}.",
        f"- Source-span catalog sessions and tokenizer preflight: {preflight_count}/477; prompt + 4096 output reserve fits the 131072 context; truncations: 0.",
        f"- Frozen qualification identities: {len(selection['session_identities'])}; selection SHA256 {manifest['selection_sha256']}.",
        f"- Prompt SHA256: {manifest['prompt_sha256']}; contract SHA256: {manifest['contract_sha256']}.",
        f"- Local model: {manifest['writer_runtime']['model_alias']}; model SHA256 {manifest['writer_runtime']['model_sha256']}; endpoint {manifest['writer_runtime']['endpoint']}.",
        "- Hosted calls: 0; TEST access: false; benchmark-label access: false.",
        "- Writer reproducibility mode: ARTIFACT_FROZEN_NOT_BITWISE_REPLAY.",
        "",
        "## Qualification Results",
        "",
        f"- Initial writer calls attempted: {manifest['writer_calls_attempted']}; retries: 0.",
        f"- Successful frozen packets: {manifest['successful_packets']}/12.",
    ]
    if diagnostics is not None:
        lines.extend(
            [
                f"- Sessions with zero propositions: {diagnostics['sessions_with_zero_propositions']}.",
                f"- Total propositions: {diagnostics['total_propositions']}; propositions/session: {diagnostics['propositions_per_session']:.3f}.",
                f"- Memory-kind distribution: {json.dumps(diagnostics['memory_kind_distribution'], sort_keys=True)}.",
                f"- Mean evidence references/proposition: {diagnostics['mean_evidence_refs_per_proposition']:.3f}.",
                f"- Candidate-key reuse: {json.dumps(diagnostics['candidate_key_reuse'], ensure_ascii=False, sort_keys=True)}.",
            ]
        )
    if failure:
        lines.extend(["", "## First Failure", "", f"- Failure detail: {json.dumps(failure, ensure_ascii=False, sort_keys=True)}."])
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "Counts above are descriptive only and have no pass thresholds. Provenance roles, exact source quotes, turn/span indices, offsets, and content hashes are harness-derived from the frozen span catalog. Human semantic-review fields remain null. No quality or benchmark-ranking claim is made.",
            "",
            "Raw model responses remain only in the git-ignored local forensic cache; committed artifacts contain response hashes and normalized review packets, not raw response bodies.",
            "",
            "STOP: do not run the remaining 465 sessions, full MEM-3A, revision materialization, embedding, retrieval, reader evaluation, TEST, or MEM-3B.",
        ]
    )
    return "\n".join(lines) + "\n"


def _freeze_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if path.exists() or sidecar_path(path).exists():
        if verify_sidecar(path) != sha256_bytes(payload):
            raise RuntimeError(f"frozen preparation artifact drift: {path}")
        return sha256_file(path)
    atomic_write_bytes(path, payload)
    return write_sidecar(path)


def _frozen_preparation(
    sessions: dict[str, dict[str, Any]], prompt: str, prompt_sha: str, contract_sha: str, inventory_sha: str
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    preflight_path = RUN_DIR / "writer_preflight_v2.json"
    selection_path = RUN_DIR / "qualification_selection.json"
    verify_sidecar(preflight_path)
    verify_sidecar(selection_path)
    preflight = read_json(preflight_path)
    selection = read_json(selection_path)
    if (
        preflight.get("artifact_version") != "mem3a1-writer-preflight-v2"
        or preflight.get("rawspan_inventory_sha256") != inventory_sha
        or preflight.get("prompt_sha256") != prompt_sha
        or preflight.get("contract_sha256") != contract_sha
        or preflight.get("sessions_preflighted") != 477
        or preflight.get("truncations") != 0
        or preflight.get("all_fit") is not True
        or selection.get("selection_version") != "mem3a1-qualification-v1"
    ):
        raise RuntimeError("frozen 477-session preflight or selection metadata mismatch")
    rows = preflight.get("session_summaries")
    if not isinstance(rows, list) or len(rows) != 477:
        raise RuntimeError("frozen preflight does not contain exactly 477 sessions")
    for row in rows:
        identity = row.get("session_identity_sha256")
        session = sessions.get(identity)
        if session is None or row.get("catalog_sha256") != session["catalog_sha256"]:
            raise RuntimeError("frozen preflight source catalog differs from RawSpan")
        request = writer_request(
            session_date=session["session_date"],
            catalog=session["catalog"],
            system_prompt=prompt,
            model_alias=READER_MODEL,
        )
        if row.get("request_sha256") != sha256_bytes(canonical_json(request)):
            raise RuntimeError("frozen preflight writer request identity mismatch")
        schema_sha = sha256_bytes(canonical_json(request["response_format"]["json_schema"]["schema"]))
        if row.get("dynamic_schema_sha256") != schema_sha:
            raise RuntimeError("frozen preflight dynamic schema identity mismatch")
    expected_selection = _select_qualification(rows)
    if canonical_json(selection) != canonical_json(expected_selection):
        raise RuntimeError("frozen qualification selection no longer matches the deterministic rule")
    return rows, selection, verify_sidecar(preflight_path), verify_sidecar(selection_path)


def run(*, prepare_only: bool) -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    upstream = _verify_upstream()
    prompt_sha = _verify_source_freeze(PROMPT_PATH)
    contract_sha = _verify_source_freeze(CONTRACT_PATH)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    contract = read_json(CONTRACT_PATH)
    if contract.get("contract_id") != "flat-proposition-extractor-v2" or contract.get("prompt_file") != PROMPT_PATH.name:
        raise RuntimeError("v2 prompt/contract identity mismatch")

    m2a_manifest = read_json(MEM2A_DIR / "run_manifest.json")
    timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        writer_runtime = _runtime_manifest(client, m2a_manifest)
        sessions = _read_source_sessions()
        if prepare_only:
            preflight_rows = _tokenize_preflight(client, sessions, prompt)
            preflight = {
                "artifact_version": "mem3a1-writer-preflight-v2",
                "rawspan_inventory_sha256": upstream["mem2c_inventory_sha256"],
                "prompt_sha256": prompt_sha,
                "contract_sha256": contract_sha,
                "context_limit": CONTEXT_LIMIT,
                "output_reserve_tokens": OUTPUT_RESERVE,
                "sessions_expected": 477,
                "sessions_preflighted": len(preflight_rows),
                "truncations": 0,
                "all_fit": all(row["prompt_fits_with_output_reserve"] for row in preflight_rows),
                "session_summaries": preflight_rows,
                "labels_loaded": False,
                "test_access": False,
            }
            selection = _select_qualification(preflight_rows)
            preflight_sha = _freeze_json(RUN_DIR / "writer_preflight_v2.json", preflight)
            selection_sha = _freeze_json(RUN_DIR / "qualification_selection.json", selection)
            prepared = {
                "run_id": RUN_ID,
                "stage": "MEM-3A.1 Provenance-by-Reference Extractor v2 Qualification",
                "status": "PREPARED",
                "completion_gate_marker": "MEM3A1_EXTRACTOR_V2_QUALIFIED=PENDING",
                "base_commit_sha": upstream["base_commit_sha"],
                "prompt_sha256": prompt_sha,
                "contract_sha256": contract_sha,
                "preflight_sha256": preflight_sha,
                "selection_sha256": selection_sha,
                "upstream": upstream,
                "writer_runtime": writer_runtime,
                "writer_calls_attempted": 0,
                "writer_calls_current_process": 0,
                "writer_call_count_expected": 12,
                "retries": 0,
                "successful_packets": 0,
                "qualification_identities": [row["session_identity_sha256"] for row in selection["session_identities"]],
                "reproducibility_mode": "ARTIFACT_FROZEN_NOT_BITWISE_REPLAY",
                "hosted_calls": 0,
                "labels_loaded": False,
                "materialization": False,
                "embedding": False,
                "retrieval": False,
                "reader_calls": 0,
                "test_access": False,
                "failure": None,
                "diagnostics": None,
            }
            write_json(RUN_DIR / "run_manifest.json", prepared)
            report_path = RUN_DIR / "report.md"
            atomic_write_bytes(report_path, _report(prepared["completion_gate_marker"], len(preflight_rows), selection, prepared, None).encode("utf-8"))
            write_sidecar(report_path)
            return prepared

        preflight_rows, selection, preflight_sha, selection_sha = _frozen_preparation(
            sessions, prompt, prompt_sha, contract_sha, upstream["mem2c_inventory_sha256"]
        )
        selected_ids = [row["session_identity_sha256"] for row in selection["session_identities"]]

        packets: list[dict[str, Any]] = []
        reviews: list[dict[str, Any]] = []
        ledger_rows: list[dict[str, Any]] = []
        failure: dict[str, Any] | None = None
        writer_calls_attempted = 0
        for call_index, identity in enumerate(selected_ids, start=1):
            session = sessions[identity]
            print(f"QUALIFICATION {call_index}/12 START {identity}", flush=True)
            request = writer_request(
                session_date=session["session_date"],
                catalog=session["catalog"],
                system_prompt=prompt,
                model_alias=READER_MODEL,
            )

            def call(req: dict[str, Any]) -> tuple[int, bytes, str | None]:
                nonlocal writer_calls_attempted
                writer_calls_attempted += 1
                return _provider(client)(req)

            calls_before = writer_calls_attempted
            try:
                normalized, ledger = execute_or_resume(
                    request=request,
                    catalog=session["catalog"],
                    session_identity_sha256=identity,
                    prompt_sha256=prompt_sha,
                    contract_sha256=contract_sha,
                    local_cache_root=CACHE_ROOT,
                    provider=call,
                )
            except WriterQualificationFailure as exc:
                ledger_rows.append(exc.ledger)
                failure = exc.error or {"code": "WRITER_FAILURE", "detail": str(exc)}
                write_jsonl(RUN_DIR / "writer_call_ledger.jsonl", ledger_rows)
                break
            except (WriterRecoveryError, OSError) as exc:
                failure = {"code": "WRITER_RECOVERY_OR_CAPTURE_FAILURE", "detail": str(exc)}
                calls_this_identity = writer_calls_attempted - calls_before
                ledger_rows.append(
                    {
                        "session_identity_sha256": identity,
                        "validation": "failed",
                        "failure_code": failure["code"],
                        "provider_calls": calls_this_identity,
                        "provider_call_outcome": "unknown" if calls_this_identity else "not_issued",
                        "local_response_retained": False,
                    }
                )
                write_jsonl(RUN_DIR / "writer_call_ledger.jsonl", ledger_rows)
                break
            packet = {
                "session_identity_sha256": identity,
                "session_date": session["session_date"],
                "catalog_sha256": session["catalog_sha256"],
                **normalized,
            }
            packets.append(packet)
            reviews.append(_review_packet(session, normalized))
            ledger_rows.append(ledger)
            print(
                f"QUALIFICATION {call_index}/12 VALIDATED propositions={len(normalized['propositions'])} response_sha256={ledger['response_sha256']}",
                flush=True,
            )
            write_jsonl(RUN_DIR / "qualification_packets.jsonl", packets)
            write_jsonl(RUN_DIR / "writer_call_ledger.jsonl", ledger_rows)
            write_json(RUN_DIR / "qualification_review.json", {
                "review_schema": {
                    "durable": None,
                    "semantically_supported": None,
                    "overgeneralized": None,
                    "transient_request": None,
                    "assistant_hypothetical_leak": None,
                    "notes": None,
                },
                "sessions": reviews,
                "human_review_complete": False,
            })

        diagnostics = _diagnostics(packets) if len(packets) == 12 and failure is None else None
        total_writer_calls = sum(row.get("provider_calls", 0) for row in ledger_rows)
        gate = "MEM3A1_EXTRACTOR_V2_QUALIFIED=YES" if (
            len(packets) == 12 and total_writer_calls == 12 and failure is None
        ) else "MEM3A1_EXTRACTOR_V2_QUALIFIED=NO"
        manifest = {
            "run_id": RUN_ID,
            "stage": "MEM-3A.1 Provenance-by-Reference Extractor v2 Qualification",
            "status": "COMPLETE" if gate.endswith("YES") else "FAILED_OR_INCOMPLETE",
            "completion_gate_marker": gate,
            "base_commit_sha": upstream["base_commit_sha"],
            "prompt_sha256": prompt_sha,
            "contract_sha256": contract_sha,
            "preflight_sha256": preflight_sha,
            "selection_sha256": selection_sha,
            "upstream": upstream,
            "writer_runtime": writer_runtime,
            "writer_calls_attempted": total_writer_calls,
            "writer_calls_current_process": writer_calls_attempted,
            "writer_call_count_expected": 12,
            "retries": 0,
            "successful_packets": len(packets),
            "qualification_identities": selected_ids,
            "reproducibility_mode": "ARTIFACT_FROZEN_NOT_BITWISE_REPLAY",
            "hosted_calls": 0,
            "labels_loaded": False,
            "materialization": False,
            "embedding": False,
            "retrieval": False,
            "reader_calls": 0,
            "test_access": False,
            "failure": failure,
            "diagnostics": diagnostics,
        }
        if packets:
            write_jsonl(RUN_DIR / "qualification_packets.jsonl", packets)
        if not (RUN_DIR / "writer_call_ledger.jsonl").exists():
            write_jsonl(RUN_DIR / "writer_call_ledger.jsonl", ledger_rows)
        write_json(RUN_DIR / "qualification_review.json", {
            "review_schema": {
                "durable": None,
                "semantically_supported": None,
                "overgeneralized": None,
                "transient_request": None,
                "assistant_hypothetical_leak": None,
                "notes": None,
            },
            "sessions": reviews,
            "human_review_complete": False,
        })
        write_json(RUN_DIR / "run_manifest.json", manifest)
        report_path = RUN_DIR / "report.md"
        atomic_write_bytes(report_path, _report(gate, len(preflight_rows), selection, manifest, diagnostics, failure).encode("utf-8"))
        write_sidecar(report_path)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--prepare-only", action="store_true", help="tokenize all 477 and freeze the 12-session selection")
    stages.add_argument("--qualify", action="store_true", help="validate frozen preparation and execute the 12-session qualification")
    args = parser.parse_args()
    try:
        manifest = run(prepare_only=args.prepare_only)
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        failure = {"code": type(exc).__name__, "detail": str(exc)}
        marker = "MEM3A1_EXTRACTOR_V2_QUALIFIED=NO"
        try:
            current = read_json(RUN_DIR / "run_manifest.json") if (RUN_DIR / "run_manifest.json").exists() else {}
        except (OSError, ValueError):
            current = {}
        current.update(
            {
                "run_id": RUN_ID,
                "stage": "MEM-3A.1 Provenance-by-Reference Extractor v2 Qualification",
                "status": "FAILED_OR_INCOMPLETE",
                "completion_gate_marker": marker,
                "failure": failure,
                "hosted_calls": 0,
                "labels_loaded": False,
                "materialization": False,
                "embedding": False,
                "retrieval": False,
                "reader_calls": 0,
                "test_access": False,
            }
        )
        write_json(RUN_DIR / "run_manifest.json", current)
        report_path = RUN_DIR / "report.md"
        report = (
            "# MEM-3A.1 - Provenance-by-Reference Extractor v2 Qualification\n\n"
            "Gate: MEM3A1_EXTRACTOR_V2_QUALIFIED=NO\n\n"
            "No benchmark scoring or downstream memory pipeline was run.\n\n"
            f"First failure: {json.dumps(failure, ensure_ascii=False, sort_keys=True)}\n\n"
            "Raw writer responses, if any, remain only in the git-ignored local forensic cache.\n"
        ).encode()
        atomic_write_bytes(report_path, report)
        write_sidecar(report_path)
        print(f"{marker}: {failure}", flush=True)
        raise
    print(manifest["completion_gate_marker"], flush=True)
    if manifest.get("failure"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
