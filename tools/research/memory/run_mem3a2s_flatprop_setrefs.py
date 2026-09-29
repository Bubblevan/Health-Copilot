"""Run MEM-3A.2S: exact evidence-ref set normalization plus frozen-ten FlatProp."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.memory import flat_proposition_writer_v2 as writer_v2
from tools.research.memory import flat_proposition_writer_v3 as writer_v3
from tools.research.memory import run_mem3a2_minimal_flatprop as base
from tools.research.memory import run_mem3a2r_writer_capacity as capacity

ROUTING_COMMIT = "5f666b241fefc0dbcd7524bd4ea6c85bbeaa6bb9"
MEM3A2R_COMMIT = "73332535a7aef74d6a93d39092f5f596d00ec767"
SOURCE_ROOT = ROOT.parent / "Health-Copilot"
SOURCE_DATASET_PATH = SOURCE_ROOT / "data/longmemeval/longmemeval_s_cleaned.json"
RUN_ID = "mem3a2s-flatprop-setrefs-frozen-10-20260929-final"
RUN_DIR = ROOT / "runs/memory/mem3" / RUN_ID
PREVIOUS_ROOT = ROOT.parent / "Health-Copilot-mem3a2r"
PREVIOUS_RUN = PREVIOUS_ROOT / "runs/memory/mem3/mem3a2r-minimal-flatprop-16k-frozen-10-20260929"
PREVIOUS_CACHE = PREVIOUS_ROOT / ".cache/health-copilot/mem3a2r-minimal-flatprop-v3-16k-writer"
NORMALIZATION_CONTRACT = ROOT / "docs/research/memory/evidence_ref_set_normalization_v1.json"
RAWSPAN_CONTRACT = ROOT / "docs/research/memory/raw_span_segmenter_contract.json"
STAGE_PROTOCOL = ROOT / "docs/research/memory/mem_3a2s_evidence_ref_set_canonicalization.md"
VALIDATOR_IDENTITY_PATH = RUN_DIR / "packet_validator_identity.json"
NORMALIZATION_COPY_PATH = RUN_DIR / "evidence_ref_normalization_contract.json"
IMPORT_MANIFEST_PATH = RUN_DIR / "historical_response_import_manifest.json"
NORMALIZATION_DIAGNOSTICS_PATH = RUN_DIR / "evidence_ref_normalization_diagnostics.json"
NORMALIZATION_SHA256 = "d718ae5895976e4736fd4aed63019d656cf60b863a1632c4b15631fc55b5862e"
EXTRACTOR_PROMPT_SHA256 = "0c0211bac51cf48e8e01663bbee7d9f9db69815afc7d40c1e8d6877f4a0f2376"
EXTRACTOR_CONTRACT_SHA256 = "708da9fce6990c3b410f1bc610009e76dc39a993a786f59bfeac9f5798fd1545"
RAWSPAN_CONTRACT_SHA256 = "e89f199a1da9ded83dbc4b07e0581661dc06451a284efdcd5f0548f9193177d3"
HISTORICAL_ENVELOPE_102_SHA256 = "1789f1ffecdf67f83e726a3c6b44382c39e811747dfdd955f3e66eec11d65d98"
HISTORICAL_IDENTITY_102 = "545f7c68c72c33db6a9a523c6d82f070b8827da187ad8d199a91d0b59adc4902"
HISTORICAL_SOURCE_SESSION_102 = "35201d43"
HISTORICAL_FAILURE_CODE_102 = "DUPLICATE_EVIDENCE_REF"
GATE = "MEM3A2S_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC"
GATE_YES = f"{GATE}=YES"
GATE_NO = f"{GATE}=NO"
CONTEXT_LIMIT = 131072
MAX_TOKENS = 16384
OUTPUT_RESERVE = 16384

_ORIGINAL_VERIFY_UPSTREAM = base._verify_upstream
_ORIGINAL_DOWNSTREAM = base._downstream
_ORIGINAL_BASE_RENDER = base._render_report
_ORIGINAL_CASE_REVIEW = base._case_review
_ORIGINAL_CAP_FINALIZE = capacity._finalize_success
_ORIGINAL_CAP_PREFLIGHT = capacity._preflight
_ORIGINAL_FLAT_VERIFY_FROZEN = base.flat._verify_frozen
_ORIGINAL_FREEZE_SOURCE = base._freeze_source


def _sha256_file(path: Path) -> str:
    return writer_v2.sha256_bytes(path.read_bytes())


def _verify_sidecar(path: Path) -> bool:
    sidecars = [path.with_suffix(".sha256"), path.with_name(path.name + ".sha256")]
    sidecars = list(dict.fromkeys(sidecar for sidecar in sidecars if sidecar.is_file()))
    if not path.is_file() or not sidecars:
        return False
    return all(
        (fields := sidecar.read_text(encoding="ascii").strip().split(None, 1))
        and len(fields) == 2
        and fields[0].lower() == _sha256_file(path)
        and fields[1] == path.name
        for sidecar in sidecars
    )


def _verify_frozen(path: Path) -> bool:
    resolved = path.resolve()
    if resolved == base.PROMPT_PATH.resolve():
        return _sha256_file(path) == EXTRACTOR_PROMPT_SHA256
    if resolved == base.CONTRACT_PATH.resolve():
        return _sha256_file(path) == EXTRACTOR_CONTRACT_SHA256
    return _ORIGINAL_FLAT_VERIFY_FROZEN(path)


def _freeze_source(path: Path) -> str:
    resolved = path.resolve()
    if resolved == base.PROMPT_PATH.resolve():
        digest = _sha256_file(path)
        if digest != EXTRACTOR_PROMPT_SHA256:
            raise RuntimeError("Frozen v3 writer prompt SHA changed")
        return digest
    if resolved == base.CONTRACT_PATH.resolve():
        digest = _sha256_file(path)
        if digest != EXTRACTOR_CONTRACT_SHA256:
            raise RuntimeError("Frozen v3 writer contract SHA changed")
        return digest
    return _ORIGINAL_FREEZE_SOURCE(path)


def _write_json(path: Path, value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    return capacity._write_frozen(path, value) if path.parent == RUN_DIR else _write_bytes_frozen(path, data)


def _write_bytes_frozen(path: Path, data: bytes) -> str:
    if path.exists():
        if not base.flat._verify_frozen(path) or path.read_bytes() != data:
            raise RuntimeError(f"Frozen MEM-3A.2S artifact differs: {path}")
        return _sha256_file(path)
    writer_v2.atomic_write_bytes(path, data)
    return base.flat._freeze(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = b"".join(writer_v2.canonical_json(row) + b"\n" for row in rows)
    return _write_bytes_frozen(path, payload)


def _verify_routing_artifacts() -> dict[str, str]:
    names = (
        "benchmark_routing_contract.md",
        "memory_benchmark_roadmap.md",
        "medmemorybench_schema_audit.json",
        "medmemorybench_query_taxonomy.json",
        "medmemorybench_provenance_compatibility.json",
        "medmemorybench_dataset_manifest.json",
        "medmemorybench_compatibility_audit.md",
        "medmemorybench_baseline_protocol_matrix.json",
    )
    directory = ROOT / "docs/research/memory"
    hashes = {}
    for name in names:
        path = directory / name
        if not _verify_sidecar(path):
            raise RuntimeError(f"MEM-B0 routing artifact sidecar failed: {name}")
        hashes[name] = _sha256_file(path)
    audit = (directory / "medmemorybench_compatibility_audit.md").read_text(encoding="utf-8")
    if "MEDMEMORYBENCH_COMPATIBILITY_AUDITED=YES" not in audit:
        raise RuntimeError("MEM-B0 MedMemoryBench compatibility gate changed")
    if "LONGMEM_MEMORY_MECHANISM_FROZEN=NO" not in audit:
        raise RuntimeError("MEM-B0 LongMem mechanism gate changed")
    return hashes


def _verify_prior_run_sidecars() -> dict[str, str]:
    if not PREVIOUS_RUN.is_dir():
        raise RuntimeError("Historical MEM-3A.2R run directory is unavailable")
    verified: dict[str, str] = {}
    for sidecar in sorted(PREVIOUS_RUN.glob("*.sha256")):
        fields = sidecar.read_text(encoding="ascii").strip().split(None, 1)
        if len(fields) != 2 or Path(fields[1]).name != fields[1]:
            raise RuntimeError(f"Malformed historical sidecar: {sidecar.name}")
        target = sidecar.parent / fields[1]
        if not target.is_file() or _sha256_file(target) != fields[0].lower():
            raise RuntimeError(f"Historical MEM-3A.2R artifact SHA failed: {sidecar.name}")
        verified[target.name] = fields[0].lower()
    required = {
        "protocol_v3_16k.json",
        "writer_preflight_v3_16k.json",
        "writer_extraction_identity.json",
        "writer_runtime_contract.json",
        "writer_call_ledger.partial.jsonl",
    }
    if not required.issubset(verified):
        raise RuntimeError("Historical MEM-3A.2R sidecar coverage is incomplete")
    return verified


def _verify_upstream() -> dict[str, Any]:
    head = base.subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    for commit in (ROUTING_COMMIT, MEM3A2R_COMMIT):
        result = base.subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=ROOT,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Required historical commit is not an ancestor: {commit}")
    base.flat.PINNED_BASE_COMMIT = head
    prior_run_hashes = _verify_prior_run_sidecars()
    upstream = base.flat._verify_upstream()
    historical = base._verify_historical_gates()
    routing_hashes = _verify_routing_artifacts()
    if base.flat._verify_frozen(NORMALIZATION_CONTRACT):
        if _sha256_file(NORMALIZATION_CONTRACT) != NORMALIZATION_SHA256:
            raise RuntimeError("Evidence-ref normalization contract SHA changed")
    else:
        raise RuntimeError("Evidence-ref normalization contract has no valid SHA sidecar")
    if _sha256_file(RAWSPAN_CONTRACT) != RAWSPAN_CONTRACT_SHA256:
        raise RuntimeError("RawSpan catalog contract SHA changed")
    if not _verify_sidecar(STAGE_PROTOCOL):
        raise RuntimeError("MEM-3A.2S protocol sidecar failed")
    return {
        **upstream,
        "historical_gates": historical,
        "routing_commit_sha256": ROUTING_COMMIT,
        "mem3a2r_commit_sha256": MEM3A2R_COMMIT,
        "convergence_head_sha256": head,
        "routing_artifact_sha256": routing_hashes,
        "historical_mem3a2r_artifact_sha256": prior_run_hashes,
        "labels_loaded": False,
        "test_access": False,
        "102_dev_access": False,
    }


def _freeze_normalization_identity() -> tuple[dict[str, Any], str]:
    if not base.flat._verify_frozen(NORMALIZATION_CONTRACT):
        raise RuntimeError("Normalization contract must be SHA-frozen before any response read")
    if _sha256_file(NORMALIZATION_CONTRACT) != NORMALIZATION_SHA256:
        raise RuntimeError("Normalization contract SHA mismatch")
    if _sha256_file(base.CONTRACT_PATH) != EXTRACTOR_CONTRACT_SHA256:
        raise RuntimeError("Frozen v3 extractor contract changed")
    if _sha256_file(base.PROMPT_PATH) != EXTRACTOR_PROMPT_SHA256:
        raise RuntimeError("Frozen v3 extractor prompt changed")
    identity = {
        "validator_id": "flatprop-packet-validator-v3-set-v1",
        "extractor_v3_contract_sha256": EXTRACTOR_CONTRACT_SHA256,
        "evidence_ref_normalization_contract_sha256": NORMALIZATION_SHA256,
        "validator_source_sha256": _sha256_file(Path(writer_v3.__file__)),
        "rawspan_catalog_contract_sha256": RAWSPAN_CONTRACT_SHA256,
        "rawspan_catalog_builder_source_sha256": _sha256_file(Path(writer_v2.__file__)),
        "writer_prompt_sha256": EXTRACTOR_PROMPT_SHA256,
        "normalization_order": "exact set; source catalog ordinal ascending",
    }
    identity_sha = writer_v2.sha256_bytes(writer_v2.canonical_json(identity))
    _write_json(VALIDATOR_IDENTITY_PATH, {"identity": identity, "identity_sha256": identity_sha})
    _write_bytes_frozen(NORMALIZATION_COPY_PATH, NORMALIZATION_CONTRACT.read_bytes())
    return identity, identity_sha


def _preflight_16k(*, save: bool) -> dict[str, Any]:
    current = base._build_preflight(tokenize=True)
    if (
        current["extractor_prompt_sha256"] != EXTRACTOR_PROMPT_SHA256
        or current["extractor_contract_sha256"] != EXTRACTOR_CONTRACT_SHA256
        or current.get("output_reserve") != OUTPUT_RESERVE
        or len(current["requests"]) != 477
    ):
        raise RuntimeError("Frozen v3 writer source or 16K request count changed")
    for row in current["requests"]:
        row["max_tokens"] = MAX_TOKENS
        if row["prompt_tokens"] + MAX_TOKENS > CONTEXT_LIMIT or row["truncated"]:
            raise RuntimeError(f"MEM-3A.2S 16K prompt reserve does not fit: {row['session_identity_sha256']}")
    current["writer_max_tokens"] = MAX_TOKENS
    current["output_reserve"] = OUTPUT_RESERVE
    current["context_limit"] = CONTEXT_LIMIT
    current["max_reserved_prompt_tokens"] = max(
        row["prompt_tokens"] + OUTPUT_RESERVE for row in current["requests"]
    )
    sessions, _, _ = base._source_sessions_with_catalog()
    order = capacity._execution_order(sessions)
    current["execution_order_session_identities"] = order
    current["execution_order_selected_for_failure_efficiency_not_model_quality"] = True
    current["historical_failure_session_identity_sha256"] = capacity.HISTORICAL_SENTINEL
    current["requests_are_canonical_order_independent_of_execution_order"] = True
    runtime_contract = capacity._runtime_contract(current)
    runtime_sha = capacity._write_frozen(RUN_DIR / "writer_runtime_contract.json", runtime_contract)
    identity = current["extraction_identity"]["identity"]
    identity["writer_max_tokens"] = MAX_TOKENS
    identity["output_reserve"] = OUTPUT_RESERVE
    identity["runtime_contract_sha256"] = runtime_sha
    identity["execution_order_sha256"] = writer_v2.sha256_bytes(
        ("\n".join(order) + "\n").encode("ascii")
    )
    current["extraction_identity"]["identity_sha256"] = writer_v2.sha256_bytes(
        writer_v2.canonical_json(identity)
    )
    path = RUN_DIR / "writer_preflight_v3_16k.json"
    identity_path = RUN_DIR / "writer_extraction_identity.json"
    threshold_contract = {
        "schema_version": 1,
        "stage": capacity.RUN_ID,
        "metric": "lowercase word-token set Jaccard similarity",
        "threshold_inclusive": capacity.SEMANTIC_SIMILARITY_THRESHOLD,
        "diagnostic_only": True,
        "applies_after_writer_packets_freeze": True,
        "no_merge_drop_or_retry": True,
    }
    protocol = {
        "schema_version": 1,
        "stage": capacity.RUN_ID,
        "base_commit_sha": capacity.BASE_COMMIT,
        "sole_method_change": "writer max_tokens 4096 -> 16384; MEM-3A.2S adds post-response exact evidence-ref set canonicalization",
        "prompt_sha256": EXTRACTOR_PROMPT_SHA256,
        "extractor_contract_sha256": EXTRACTOR_CONTRACT_SHA256,
        "normalization_contract_sha256": NORMALIZATION_SHA256,
        "validator_identity_id": "flatprop-packet-validator-v3-set-v1",
        "writer_runtime_contract_sha256": runtime_sha,
        "preflight_requests": 477,
        "all_prompt_tokens_plus_16384_fit": current["all_prompt_tokens_plus_reserve_fit"],
        "max_reserved_prompt_tokens": current["max_reserved_prompt_tokens"],
        "execution_order_selected_for_failure_efficiency_not_model_quality": True,
        "first_execution_identity": capacity.HISTORICAL_SENTINEL,
        "execution_order_sha256": identity["execution_order_sha256"],
        "retries": 0,
        "hosted_calls": 0,
        "semantic_noise_threshold": threshold_contract,
        "label_barrier": "labels remain unloaded through writer, inventory, retrieval, bundles, predictions and reader ledger freeze",
    }
    if save:
        cache_root = capacity.LOCAL_CACHE_ROOT
        if cache_root.exists() and any(cache_root.rglob("journal.jsonl")):
            raise RuntimeError("Cannot update preflight identity after MEM-3A.2S cache journals exist")
        if path.exists() and base._preflight_core(base._json(path)) != base._preflight_core(current):
            attempt = 1
            while (RUN_DIR / f"prepare_attempt_{attempt:03d}_writer_preflight.json").exists():
                attempt += 1
            _write_bytes_frozen(
                RUN_DIR / f"prepare_attempt_{attempt:03d}_writer_preflight.json", path.read_bytes()
            )
            if identity_path.exists():
                _write_bytes_frozen(
                    RUN_DIR / f"prepare_attempt_{attempt:03d}_writer_extraction_identity.json",
                    identity_path.read_bytes(),
                )
            old_protocol = RUN_DIR / "protocol_v3_16k.json"
            if old_protocol.exists():
                _write_bytes_frozen(
                    RUN_DIR / f"prepare_attempt_{attempt:03d}_protocol_v3_16k.json",
                    old_protocol.read_bytes(),
                )
        base._write_json(path, current)
        base._write_json(identity_path, current["extraction_identity"])
        base._write_json(RUN_DIR / "semantic_noise_threshold.json", threshold_contract)
        base._write_json(RUN_DIR / "protocol_v3_16k.json", protocol)
        return current
    for frozen_path, expected in (
        (path, current),
        (identity_path, current["extraction_identity"]),
        (RUN_DIR / "writer_runtime_contract.json", runtime_contract),
    ):
        if not frozen_path.exists() or not base.flat._verify_frozen(frozen_path):
            raise RuntimeError(f"Required frozen 16K protocol artifact missing: {frozen_path.name}")
        if base._json(frozen_path) != expected:
            raise RuntimeError(f"Live 16K protocol differs from frozen artifact: {frozen_path.name}")
    return current


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _historical_response_rows(
    preflight: dict[str, Any], ordered_sessions: dict[str, dict[str, Any]],
    identity_sha: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    old_preflight_path = PREVIOUS_RUN / "writer_preflight_v3_16k.json"
    old_ledger_path = PREVIOUS_RUN / "writer_call_ledger.partial.jsonl"
    for path in (old_preflight_path, old_ledger_path):
        if not _verify_sidecar(path):
            raise RuntimeError("HISTORICAL_FROZEN_RESPONSE_UNAVAILABLE: historical ledger/preflight SHA failed")
    old_preflight = json.loads(old_preflight_path.read_text(encoding="utf-8"))
    old_protocol = json.loads((PREVIOUS_RUN / "protocol_v3_16k.json").read_text(encoding="utf-8"))
    if (
        old_protocol.get("prompt_sha256") != EXTRACTOR_PROMPT_SHA256
        or old_protocol.get("extractor_contract_sha256") != EXTRACTOR_CONTRACT_SHA256
        or old_protocol.get("writer_runtime_contract_sha256")
        != _sha256_file(PREVIOUS_RUN / "writer_runtime_contract.json")
    ):
        raise RuntimeError("Historical 16K writer contract/runtime provenance differs")
    old_order = old_preflight.get("execution_order_session_identities", [])
    if len(old_order) != 477 or old_order != list(capacity._execution_order(ordered_sessions)):
        raise RuntimeError("Historical 16K execution order differs from the frozen source sessions")
    old_rows = _read_jsonl(old_ledger_path)
    if len(old_rows) != 102 or len({row.get("session_identity_sha256") for row in old_rows}) != 102:
        raise RuntimeError("Historical 16K ledger does not contain 102 unique captured outcomes")
    if [row["session_identity_sha256"] for row in old_rows] != old_order[:102]:
        raise RuntimeError("Historical 16K captured outcomes are not the exact first 102 frozen identities")
    failures = [row for row in old_rows if not row.get("success")]
    if len(failures) != 1 or failures[0].get("failure_code") != HISTORICAL_FAILURE_CODE_102:
        raise RuntimeError("Historical writer ledger contains an additional structural failure")
    if any(row.get("retry_count", 0) != 0 or row.get("hosted_call") for row in old_rows):
        raise RuntimeError("Historical writer ledger contains retry/hosted-call provenance")
    old_request = {row["session_identity_sha256"]: row for row in old_preflight["requests"]}
    new_request = {row["session_identity_sha256"]: row for row in preflight["requests"]}
    captured_by_id: dict[str, dict[str, Any]] = {}
    import_records = []
    for index, row in enumerate(old_rows, 1):
        identity = row["session_identity_sha256"]
        frozen = old_request.get(identity)
        current = new_request.get(identity)
        if (
            frozen is None
            or current is None
            or frozen.get("request_sha256") != row.get("request_sha256")
            or current.get("request_sha256") != row.get("request_sha256")
            or frozen.get("max_tokens") != MAX_TOKENS
            or frozen.get("prompt_tokens", -1) + MAX_TOKENS > CONTEXT_LIMIT
        ):
            raise RuntimeError("Historical 16K request identity/capacity mismatch")
        source_session_ids = frozen.get("source_session_ids", [])
        source_session_id = source_session_ids[0] if len(source_session_ids) == 1 else None
        old_raw_path = Path(row.get("response_cache_path", ""))
        old_content_path = Path(row.get("assistant_content_cache_path", ""))
        if not old_raw_path.is_file() or not old_content_path.is_file():
            raise RuntimeError("HISTORICAL_FROZEN_RESPONSE_UNAVAILABLE: captured response bytes are missing")
        raw_body = old_raw_path.read_bytes()
        if writer_v2.sha256_bytes(raw_body) != row.get("http_envelope_sha256"):
            raise RuntimeError("HISTORICAL_FROZEN_RESPONSE_UNAVAILABLE: envelope SHA mismatch")
        unwrapped = writer_v2.unwrap_chat_completion_response(raw_body)
        if (
            unwrapped.assistant_content_sha256 != row.get("assistant_content_sha256")
            or old_content_path.read_bytes() != unwrapped.assistant_content_bytes
            or unwrapped.finish_reason != "stop"
            or unwrapped.prompt_tokens != frozen.get("prompt_tokens")
        ):
            raise RuntimeError("HISTORICAL_FROZEN_RESPONSE_UNAVAILABLE: content/envelope metadata mismatch")
        if index == 102:
            if (
                identity != HISTORICAL_IDENTITY_102
                or source_session_id != HISTORICAL_SOURCE_SESSION_102
                or row.get("http_envelope_sha256") != HISTORICAL_ENVELOPE_102_SHA256
            ):
                raise RuntimeError("Historical session-102 exact envelope identity/SHA mismatch")
        captured_by_id[identity] = {
            "http_status": row.get("http_status", 200),
            "raw_http_body": raw_body,
            "content_type": "application/json",
            "imported_from_stage": "MEM-3A.2R",
            "import_reason": "historical frozen 16K response; revalidated under set-semantic packet validator",
        }
        import_records.append(
            {
                "session_identity_sha256": identity,
                "source_session_id": source_session_id,
                "request_sha256": row["request_sha256"],
                "http_envelope_sha256": row["http_envelope_sha256"],
                "assistant_content_sha256": row["assistant_content_sha256"],
                "historical_finish_reason": row.get("finish_reason"),
                "historical_prompt_tokens": row.get("prompt_tokens"),
                "historical_completion_tokens": row.get("completion_tokens"),
                "historical_validation": row.get("validation"),
                "historical_failure_code": row.get("failure_code"),
                "import_index": index,
                "provider_replay_this_stage": False,
            }
        )
    return captured_by_id, old_request, {
        "schema_version": 1,
        "stage": RUN_ID,
        "historical_stage": "MEM-3A.2R",
        "historical_writer_max_tokens": MAX_TOKENS,
        "writer_prompt_sha256": EXTRACTOR_PROMPT_SHA256,
        "writer_contract_sha256": EXTRACTOR_CONTRACT_SHA256,
        "normalization_contract_sha256": NORMALIZATION_SHA256,
        "packet_validator_identity_sha256": identity_sha,
        "historical_outcomes": len(import_records),
        "historical_successes": sum(row["historical_validation"] == "passed" for row in import_records),
        "historical_failure_canonicalized": sum(
            row["historical_failure_code"] == HISTORICAL_FAILURE_CODE_102 for row in import_records
        ),
        "provider_replays": 0,
        "records": import_records,
    }


def _normalization_diagnostics(packets: list[dict[str, Any]]) -> dict[str, Any]:
    propositions = [prop for packet in packets for prop in packet["propositions"]]
    session_102 = next(
        packet for packet in packets
        if packet["session_identity_sha256"] == HISTORICAL_IDENTITY_102
    )
    session_102_duplicates = [
        prop for prop in session_102["propositions"] if prop["duplicate_evidence_ref_count"]
    ]
    session_102_s0070 = [
        item
        for prop in session_102_duplicates
        for item in prop["duplicate_evidence_refs"]
        if item["evidence_ref"] == "S0070"
    ]
    duplicate_props = [prop for prop in propositions if prop["duplicate_evidence_ref_count"]]
    duplicate_sessions = {
        packet["session_identity_sha256"]
        for packet in packets
        if any(prop["duplicate_evidence_ref_count"] for prop in packet["propositions"])
    }


def _case_review(*args: Any, **kwargs: Any) -> dict[str, Any]:
    review = _ORIGINAL_CASE_REVIEW(*args, **kwargs)
    bundles = {
        row["question_id"]: row["context_bundle"]["items"]
        for row in _read_jsonl(base.flat.CONTEXT_BUNDLES_PATH)
    }
    for case in review["cases"]:
        selected = {item["memory_id"] for item in bundles[case["question_id"]]}
        for item in case["retrieved_top8"]:
            item["retrieval_rank"] = item["rank"]
            item["canonical_evidence_refs"] = [
                evidence["evidence_ref"] for evidence in item["exact_harness_evidence"]
            ]
            item["projection_status"] = (
                "included_in_shared_reader_context"
                if item["memory_id"] in selected
                else "not_projected"
            )
    checkpoint = review["instagram_500_to_600_checkpoint"]
    observations = checkpoint["detected_observations"]
    expected = {"500 Instagram followers", "600 Instagram followers"}
    target_observations = [row for row in observations if row["expected_observation"] in expected]
    checkpoint["independent_observations"] = (
        len({row["memory_id"] for row in target_observations}) >= 2
        and len({row["source_session_id"] for row in target_observations}) >= 2
    )
    checkpoint["dense_projects_500"] = any(
        row["expected_observation"] == "500 Instagram followers" and row["retrieved_top8"]
        for row in observations
    )
    checkpoint["dense_projects_600"] = any(
        row["expected_observation"] == "600 Instagram followers" and row["retrieved_top8"]
        for row in observations
    )
    return review
    return {
        "schema_version": 1,
        "stage": RUN_ID,
        "packet_validator_id": "flatprop-packet-validator-v3-set-v1",
        "source_sessions": len(packets),
        "proposition_count": len(propositions),
        "sessions_with_duplicate_evidence_refs": len(duplicate_sessions),
        "propositions_with_duplicate_evidence_refs": len(duplicate_props),
        "total_raw_evidence_refs": sum(prop["raw_evidence_ref_count"] for prop in propositions),
        "total_canonical_evidence_refs": sum(prop["canonical_evidence_ref_count"] for prop in propositions),
        "refs_removed_by_exact_dedup": sum(prop["duplicate_evidence_ref_count"] for prop in propositions),
        "maximum_duplicate_multiplicity": max(
            (item["multiplicity"] for prop in duplicate_props for item in prop["duplicate_evidence_refs"]),
            default=1,
        ),
        "session_102_regression": {
            "source_session_ids": session_102["source_session_ids"],
            "duplicate_proposition_count": len(session_102_duplicates),
            "duplicate_proposition_indices": [prop["proposition_index"] for prop in session_102_duplicates],
            "duplicate_ref_identities": sum(len(prop["duplicate_evidence_refs"]) for prop in session_102_duplicates),
            "raw_refs_in_duplicate_propositions": sum(prop["raw_evidence_ref_count"] for prop in session_102_duplicates),
            "canonical_refs_in_duplicate_propositions": sum(prop["canonical_evidence_ref_count"] for prop in session_102_duplicates),
            "duplicate_entries_removed": sum(prop["duplicate_evidence_ref_count"] for prop in session_102_duplicates),
            "s0070_wire_multiplicity": max((item["multiplicity"] for item in session_102_s0070), default=0),
            "s0070_canonical_occurrences": sum(
                prop["evidence_refs"].count("S0070") for prop in session_102["propositions"]
            ),
            "only_exact_duplicate_refs_required_normalization": True,
        },
        "details": [
            {
                "session_identity_sha256": packet["session_identity_sha256"],
                "proposition_index": prop["proposition_index"],
                "raw_evidence_ref_count": prop["raw_evidence_ref_count"],
                "canonical_evidence_ref_count": prop["canonical_evidence_ref_count"],
                "duplicate_evidence_ref_count": prop["duplicate_evidence_ref_count"],
                "duplicate_evidence_refs": prop["duplicate_evidence_refs"],
            }
            for packet in packets
            for prop in packet["propositions"]
            if prop["duplicate_evidence_ref_count"]
        ],
        "normalization_is_infrastructure_not_retrieval_or_reasoning": True,
    }


def _extract_all(preflight: dict[str, Any], sessions: dict[str, dict[str, Any]]):
    _, validator_identity_sha = _freeze_normalization_identity()
    ordered_ids = capacity._execution_order(sessions)
    ordered_sessions = {identity: sessions[identity] for identity in ordered_ids}
    captures, old_preflight_rows, import_manifest = _historical_response_rows(
        preflight, ordered_sessions, validator_identity_sha
    )
    import_manifest_path_sha = _write_json(IMPORT_MANIFEST_PATH, import_manifest)
    preflight_by_id = {row["session_identity_sha256"]: row for row in preflight["requests"]}
    prompt = base.PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha = preflight["extractor_prompt_sha256"]
    contract_sha = preflight["extractor_contract_sha256"]
    runtime = preflight["writer_runtime_identity"]
    outcomes: dict[str, dict[str, Any]] = {}
    ledger_by_id: dict[str, dict[str, Any]] = {}
    current_stage_calls = 0
    provider_durations = []
    base._write_json(
        RUN_DIR / "run_manifest.json",
        {
            "schema_version": 1,
            "stage": RUN_ID,
            "status": "WRITER_IN_PROGRESS",
            "base_commit_sha": capacity.BASE_COMMIT,
            "completion_gate_marker": f"{GATE}=PENDING",
            "expected_unique_source_sessions": 477,
            "completed_unique_source_sessions": 0,
            "historical_outcomes_imported": 102,
            "historical_provider_replays": 0,
            "provider_calls_maximum_remaining": 375,
            "provider_calls_this_process": 0,
            "retries": 0,
            "hosted_calls": 0,
            "labels_loaded": False,
            "test_access": False,
            "102_dev_access": False,
        },
    )
    with httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=10.0),
        trust_env=False,
    ) as client:
        for index, (identity, session) in enumerate(ordered_sessions.items(), 1):
            frozen = preflight_by_id.get(identity)
            if frozen is None or frozen.get("catalog_sha256") != session["catalog_sha256"]:
                raise RuntimeError(f"MEM-3A.2S preflight identity mismatch: {identity}")
            request = writer_v3.writer_request(
                session_date=session["session_date"],
                catalog=session["catalog"],
                system_prompt=prompt,
                model_alias=base.flat.READER_MODEL,
                max_tokens=MAX_TOKENS,
            )
            request_sha = writer_v2.sha256_bytes(writer_v2.canonical_json(request))
            if request_sha != frozen.get("request_sha256"):
                raise RuntimeError(f"Frozen v3 16K writer request changed: {identity}")

            def provider(req: dict[str, Any]) -> tuple[int, bytes, str | None]:
                nonlocal current_stage_calls
                current_stage_calls += 1
                started = time.perf_counter()
                response = client.post(f"{base.flat.READER_ENDPOINT}/chat/completions", json=req)
                provider_durations.append(round((time.perf_counter() - started) * 1000, 3))
                return response.status_code, response.content, response.headers.get("content-type")

            imported = captures.get(identity)
            normalized, call = writer_v2.execute_or_resume(
                request=request,
                catalog=session["catalog"],
                session_identity_sha256=identity,
                prompt_sha256=prompt_sha,
                contract_sha256=contract_sha,
                local_cache_root=capacity.LOCAL_CACHE_ROOT,
                provider=provider,
                stage_identity="MEM-3A.2S/flatprop-v3-setrefs-v1",
                model_sha256=runtime["model_sha256"],
                dynamic_schema_sha256=frozen["dynamic_schema_sha256"],
                unwrap_source_sha256=_sha256_file(Path(writer_v2.__file__)),
                packet_validator=writer_v3.validate_packet,
                packet_validator_sha256=validator_identity_sha,
                imported_capture=imported,
            )
            packet = base._normalized_session_packet(session, normalized)
            outcomes[identity] = packet
            call.update(
                {
                    "session_identity_sha256": identity,
                    "source_session_id": session["source_session_ids"][0],
                    "request_sha256": request_sha,
                    "dynamic_schema_sha256": frozen["dynamic_schema_sha256"],
                    "writer_prompt_sha256": prompt_sha,
                    "writer_contract_sha256": contract_sha,
                    "validator_id": "flatprop-packet-validator-v3-set-v1",
                    "validator_identity_sha256": validator_identity_sha,
                    "validator_source_sha256": _sha256_file(Path(writer_v3.__file__)),
                    "normalization_contract_sha256": NORMALIZATION_SHA256,
                    "source_provider_calls": 1,
                    "provider_calls": 0 if imported else call.get("provider_calls", 0),
                    "provider_calls_this_resume": 0 if imported else call.get("provider_calls_this_resume", 0),
                    "imported_from_stage": "MEM-3A.2R" if imported else None,
                    "validation": "passed",
                    "success": True,
                    "provider": "local_llama_cpp",
                    "endpoint": base.flat.READER_ENDPOINT,
                    "loopback_only": True,
                    "writer_role": "memory_write_extract",
                    "hosted_call": False,
                    "retry_count": 0,
                    "latency_ms": call.get("provider_duration_ms"),
                    "latency_measured_this_process": not imported and call.get("provider_calls_this_resume") == 1,
                    "prompt_tokens_preflight": frozen["prompt_tokens"],
                    "prompt_tokens_match_preflight": call.get("prompt_tokens") == frozen["prompt_tokens"],
                    "completion_tokens": call.get("completion_tokens"),
                }
            )
            # Keep machine-local ignored-cache paths out of committed ledgers.
            for path_key in ("response_cache_path", "assistant_content_cache_path"):
                value = call.get(path_key)
                if value:
                    try:
                        call[path_key] = Path(value).resolve().relative_to(capacity.LOCAL_CACHE_ROOT.resolve()).as_posix()
                    except ValueError:
                        call[path_key] = "historical_local_ignored_cache"
            if imported:
                call["historical_http_envelope_sha256"] = captures[identity]["raw_http_body"] and next(
                    row["http_envelope_sha256"]
                    for row in import_manifest["records"]
                    if row["session_identity_sha256"] == identity
                )
            ledger_by_id[identity] = call
            if index % 10 == 0 or index == 477:
                base._write_json(
                    RUN_DIR / "run_manifest.json",
                    {
                        "schema_version": 1,
                        "stage": RUN_ID,
                        "status": "WRITER_IN_PROGRESS",
                        "base_commit_sha": capacity.BASE_COMMIT,
                        "completion_gate_marker": f"{GATE}=PENDING",
                        "expected_unique_source_sessions": 477,
                        "completed_unique_source_sessions": index,
                        "historical_outcomes_imported": 102,
                        "historical_provider_replays": 0,
                        "provider_calls_this_process": current_stage_calls,
                        "retries": 0,
                        "hosted_calls": 0,
                        "labels_loaded": False,
                        "test_access": False,
                        "102_dev_access": False,
                    },
                )
                partial = b"".join(
                    writer_v2.canonical_json(row) + b"\n" for row in ledger_by_id.values()
                )
                writer_v2.atomic_write_bytes(RUN_DIR / "writer_call_ledger.partial.jsonl", partial)
                print(
                    f"MEM-3A.2S writer outcomes {index}/477; imported=102; new_local_calls={current_stage_calls}",
                    flush=True,
                )

            if index == 102:
                imported_packet = packet
                duplicate_props = [
                    prop
                    for prop in imported_packet["propositions"]
                    if prop["duplicate_evidence_ref_count"]
                ]
                s0070_occurrences = [
                    (prop, item)
                    for prop in duplicate_props
                    for item in prop["duplicate_evidence_refs"]
                    if item["evidence_ref"] == "S0070"
                ]
                has_expected_duplicate = any(item["multiplicity"] >= 2 for _, item in s0070_occurrences)
                s0070_is_canonical_once = any(
                    prop["evidence_refs"].count("S0070") == 1 for prop, _ in s0070_occurrences
                )
                source_match = session["source_session_ids"] == [HISTORICAL_SOURCE_SESSION_102]
                if (
                    identity != HISTORICAL_IDENTITY_102
                    or not source_match
                    or not has_expected_duplicate
                    or not s0070_is_canonical_once
                ):
                    raise RuntimeError("Session-102 expected exact S0070 set normalization was not observed")
    packets = [outcomes[identity] for identity in ordered_ids]
    ledger = [ledger_by_id[identity] for identity in ordered_ids]
    if len(packets) != 477 or len(ledger) != 477 or any(not row["success"] for row in ledger):
        raise RuntimeError("MEM-3A.2S normalized packet set is incomplete")
    if current_stage_calls > 375 or sum(row.get("retry_count", 0) for row in ledger) != 0:
        raise RuntimeError("MEM-3A.2S writer call/retry cap was violated")
    diagnostics = _normalization_diagnostics(packets)
    _write_json(NORMALIZATION_DIAGNOSTICS_PATH, diagnostics)
    _write_jsonl(RUN_DIR / "session_extractions.jsonl", packets)
    _write_jsonl(RUN_DIR / "writer_call_ledger.jsonl", ledger)
    extraction_manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "contract_id": "flat-proposition-extractor-v3-minimal",
        "validator_id": "flatprop-packet-validator-v3-set-v1",
        "validator_identity_sha256": validator_identity_sha,
        "status": "WRITER_COMPLETE",
        "unique_source_sessions": len(packets),
        "structurally_valid_outcomes": len(packets),
        "historical_imported_outcomes": 102,
        "historical_provider_replays": 0,
        "new_provider_calls_this_stage": current_stage_calls,
        "provider_calls_unique_lineage": 477,
        "provider_calls_this_process": current_stage_calls,
        "retries": 0,
        "hosted_calls": 0,
        "labels_loaded": False,
        "test_access": False,
        "102_dev_access": False,
        "failure": None,
        "normalization_diagnostics_sha256": _sha256_file(NORMALIZATION_DIAGNOSTICS_PATH),
        "historical_import_manifest_sha256": import_manifest_path_sha,
        "session_packets": [
            {
                "session_identity_sha256": packet["session_identity_sha256"],
                "source_session_ids": packet["source_session_ids"],
                "session_date": packet["session_date"],
                "catalog_sha256": packet["catalog_sha256"],
                "proposition_count": len(packet["propositions"]),
                "packet_sha256": writer_v2.sha256_bytes(writer_v2.canonical_json(packet)),
            }
            for packet in packets
        ],
    }
    _write_json(RUN_DIR / "session_extraction_manifest.json", extraction_manifest)
    current_stage_calls = sum(row.get("provider_calls", 0) for row in ledger)
    writer_manifest = {
        **extraction_manifest,
        "provider_calls_unique": 477,
        "provider_calls_this_process": current_stage_calls,
        "new_provider_calls_this_stage": current_stage_calls,
        "historical_imported_outcomes": 102,
        "historical_provider_replays": 0,
    }
    capacity._write_frozen(
        RUN_DIR / "writer_scale_diagnostics.json",
        capacity._writer_scale(packets, ledger, ordered_sessions),
    )
    capacity._write_frozen(
        RUN_DIR / "semantic_noise_diagnostics.json", capacity._semantic_noise(packets)
    )
    writer_manifest["historical_import_manifest_sha256"] = _sha256_file(IMPORT_MANIFEST_PATH)
    writer_manifest["normalization_diagnostics_sha256"] = _sha256_file(NORMALIZATION_DIAGNOSTICS_PATH)
    return packets, ledger, writer_manifest


def _extra_gate_values(manifest: dict[str, Any], writer_diag: dict[str, Any]) -> dict[str, bool]:
    imported = json.loads(IMPORT_MANIFEST_PATH.read_text(encoding="utf-8"))
    diag = json.loads(NORMALIZATION_DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    identity = json.loads(VALIDATOR_IDENTITY_PATH.read_text(encoding="utf-8"))
    artifacts = (IMPORT_MANIFEST_PATH, NORMALIZATION_DIAGNOSTICS_PATH, VALIDATOR_IDENTITY_PATH, NORMALIZATION_COPY_PATH)
    reader_rows = _read_jsonl(base.flat.FLAT_PROPOSITIONS_PATH)
    bundles = _read_jsonl(base.flat.CONTEXT_BUNDLES_PATH) if base.flat.CONTEXT_BUNDLES_PATH.exists() else []
    case_review = (
        json.loads(base.flat.CASE_REVIEW_PATH.read_text(encoding="utf-8"))
        if base.flat.CASE_REVIEW_PATH.exists()
        else {"cases": []}
    )

    def contains_diagnostic_key(value: Any) -> bool:
        diagnostic_keys = {
            "raw_evidence_ref_count",
            "canonical_evidence_ref_count",
            "duplicate_evidence_ref_count",
            "duplicate_evidence_refs",
        }
        if isinstance(value, dict):
            return bool(set(value).intersection(diagnostic_keys)) or any(
                contains_diagnostic_key(item) for item in value.values()
            )
        if isinstance(value, list):
            return any(contains_diagnostic_key(item) for item in value)
        return False

    return {
        "historical_102_envelopes_revalidated_without_replay": imported.get("historical_outcomes") == 102
        and imported.get("historical_successes") == 101
        and imported.get("historical_failure_canonicalized") == 1
        and imported.get("provider_replays") == 0,
        "historical_session_102_exact_envelope_and_duplicate_regression": any(
            row.get("session_identity_sha256") == HISTORICAL_IDENTITY_102
            and row.get("http_envelope_sha256") == HISTORICAL_ENVELOPE_102_SHA256
            and row.get("source_session_id") == HISTORICAL_SOURCE_SESSION_102
            for row in imported["records"]
        )
        and diag.get("refs_removed_by_exact_dedup", 0) >= 1,
        "exactly_375_new_writer_calls_and_zero_retry": manifest.get("writer_calls_this_process", 0) == 375
        and manifest.get("retries") == 0,
        "normalization_identity_sha_frozen_and_bound": identity.get("identity", {}).get("validator_id")
        == "flatprop-packet-validator-v3-set-v1"
        and identity.get("identity", {}).get("evidence_ref_normalization_contract_sha256")
        == NORMALIZATION_SHA256
        and identity.get("identity", {}).get("validator_source_sha256")
        == _sha256_file(Path(writer_v3.__file__))
        and identity.get("identity_sha256")
        == writer_v2.sha256_bytes(writer_v2.canonical_json(identity["identity"])),
        "normalization_artifacts_sha_frozen": all(base.flat._verify_frozen(path) for path in artifacts),
        "stage_protocol_artifact_sha_frozen": base.flat._verify_frozen(RUN_DIR / "mem_3a2s_protocol.md")
        and _verify_sidecar(STAGE_PROTOCOL),
        "v3_schema_still_unique_items_and_writer_contract_unchanged": (
            writer_v3.dynamic_output_schema([{"evidence_ref": "S0000"}])
            ["properties"]["propositions"]["items"]["properties"]["evidence_refs"]["uniqueItems"]
            is True
            and _sha256_file(base.CONTRACT_PATH) == EXTRACTOR_CONTRACT_SHA256
            and _sha256_file(base.PROMPT_PATH) == EXTRACTOR_PROMPT_SHA256
        ),
        "diagnostic_metadata_excluded_from_reader_value": all(
            not set(row.get("reader_value", {})).intersection(
                {"raw_evidence_ref_count", "canonical_evidence_ref_count", "duplicate_evidence_ref_count", "duplicate_evidence_refs"}
            )
            and row.get("retrieval_document_sha256") == row.get("proposition_text_sha256")
            for row in reader_rows
        ) and not any(contains_diagnostic_key(bundle) for bundle in bundles),
        "critical_case_review_has_refs_and_projection_status": len(case_review["cases"]) == 8
        and all(
            "canonical_evidence_refs" in row
            and "retrieval_rank" in row
            and row.get("projection_status") in {
                "included_in_shared_reader_context",
                "not_projected",
            }
            for case in case_review["cases"]
            for row in case["retrieved_top8"]
        ),
        "benchmark_routing_gate_preserved": "MEDMEMORYBENCH_COMPATIBILITY_AUDITED=YES"
        in (ROOT / "docs/research/memory/medmemorybench_compatibility_audit.md").read_text(encoding="utf-8"),
        "memory_mechanism_gate_remains_not_frozen": "LONGMEM_MEMORY_MECHANISM_FROZEN=NO"
        in (ROOT / "docs/research/memory/medmemorybench_compatibility_audit.md").read_text(encoding="utf-8"),
        "writer_diagnostics_cover_477_normalized_packets": diag.get("source_sessions") == 477
        and writer_diag.get("source_sessions") == 477,
    }


def _render_report(gate: str, writer_diag: dict[str, Any], embedding: dict[str, Any], metrics: dict[str, Any], comparison: dict[str, Any], manifest: dict[str, Any]) -> str:
    extra_gates = _extra_gate_values(manifest, writer_diag)
    manifest["gate"].update(extra_gates)
    manifest["gate"]["flat_no_revision_invariant"] = manifest["gate"].get(
        "flat_add_active_v1_no_revision"
    ) is True
    final_gate = "YES" if all(manifest["gate"].values()) else "NO"
    report = _ORIGINAL_BASE_RENDER(final_gate, writer_diag, embedding, metrics, comparison, manifest)
    diag = json.loads(NORMALIZATION_DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    report = report.replace("# MEM-3A.2 - Minimal FlatProp v3 Frozen-Ten Diagnostic", "# MEM-3A.2S - Evidence-Ref Set Canonicalization and FlatProp Diagnostic")
    report = report.replace("MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC", GATE)
    report = report.replace("MEM-3A.2 downstream", "MEM-3A.2S downstream")
    report += "\n## Evidence-Ref Set Normalization\n\n"
    report += f"- Validator: `flatprop-packet-validator-v3-set-v1`; identity SHA `{json.loads(VALIDATOR_IDENTITY_PATH.read_text(encoding='utf-8'))['identity_sha256']}`.\n"
    report += f"- Historical outcomes revalidated: 102 (101 prior successes + the session-102 exact-duplicate case); provider replay: 0. New local writer calls: {manifest.get('writer_calls_this_process', 'see extraction manifest')} / 375 maximum.\n"
    report += f"- Sessions / propositions with duplicate refs: {diag['sessions_with_duplicate_evidence_refs']} / {diag['propositions_with_duplicate_evidence_refs']}; raw/canonical/removed refs: {diag['total_raw_evidence_refs']} / {diag['total_canonical_evidence_refs']} / {diag['refs_removed_by_exact_dedup']}; maximum multiplicity: {diag['maximum_duplicate_multiplicity']}.\n"
    report += "- Only exact duplicate ref strings inside one proposition are canonicalized; every raw ref is checked against the session catalog first. Duplicate propositions remain intact.\n"
    report += "- Canonicalizer diagnostics remain evaluator metadata; MemoryRecord reader-visible values, embedding documents, and query prompts exclude those fields.\n"
    regression = diag["session_102_regression"]
    report += f"- Session 102 exact regression: envelope SHA verified with no provider request; `S0070` wire multiplicity {regression['s0070_wire_multiplicity']} became {regression['s0070_canonical_occurrences']} canonical ref. The duplicate-bearing proposition had {regression['raw_refs_in_duplicate_propositions']} raw refs, {regression['canonical_refs_in_duplicate_propositions']} canonical refs, and {regression['duplicate_entries_removed']} redundant entries across {regression['duplicate_ref_identities']} repeated ref identities; all passed current-session provenance reconstruction.\n"
    review_path = base.flat.CASE_REVIEW_PATH
    if review_path.exists():
        checkpoint = json.loads(review_path.read_text(encoding="utf-8"))["instagram_500_to_600_checkpoint"]
        report += f"- Instagram 500/600: detected={checkpoint['has_500']}/{checkpoint['has_600']}; independent historical sessions={checkpoint.get('independent_observations')}; Dense top-8 projects 500/600={checkpoint.get('dense_projects_500')}/{checkpoint.get('dense_projects_600')}; no reconciliation applied.\n"
    report += "\n## Scope\n\n- Diagnostic-only frozen ten; no LongMemEval 102 DEV, TEST, MedMemoryBench scoring, ESL ingestion, or MEM-3B work.\n- The resulting proposition inventory is FlatProp only: ADD / SESSION_NOTE / SESSION_DERIVED / ACTIVE / version 1 / no supersession.\n"
    report += f"- `MEM3A2S_FLAT_NO_REVISION={'YES' if manifest['gate']['flat_no_revision_invariant'] else 'NO'}`.\n"
    report += f"\nCompletion gate: `{GATE}={final_gate}`.\n"
    return report


def _downstream(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _ORIGINAL_DOWNSTREAM(*args, **kwargs)
    manifest = result
    manifest["historical_routing_commit_sha256"] = ROUTING_COMMIT
    manifest["historical_mem3a2r_commit_sha256"] = MEM3A2R_COMMIT
    manifest["convergence_head_sha256"] = capacity.BASE_COMMIT
    manifest["validator_identity_sha256"] = json.loads(VALIDATOR_IDENTITY_PATH.read_text(encoding="utf-8"))["identity_sha256"]
    manifest["flat_no_revision_gate_marker"] = (
        "MEM3A2S_FLAT_NO_REVISION=YES"
        if manifest.get("gate", {}).get("flat_no_revision_invariant") is True
        else "MEM3A2S_FLAT_NO_REVISION=NO"
    )
    manifest["gate"].update(_extra_gate_values(manifest, {"source_sessions": 477}))
    manifest["completion_gate_marker"] = f"{GATE}={'YES' if all(manifest['gate'].values()) else 'NO'}"
    manifest["stage"] = RUN_ID
    result.update(manifest)
    return result


def _finalize_success(manifest: dict[str, Any]) -> dict[str, Any]:
    alias = RUN_DIR / "comparison_mem2d_vs_mem3a2r.json"
    current_comparison = RUN_DIR / "comparison_mem2d_vs_mem3a2s.json"
    if current_comparison.exists() and not alias.exists():
        _write_bytes_frozen(alias, current_comparison.read_bytes())
    finalized = _ORIGINAL_CAP_FINALIZE(manifest)
    gate = finalized["gate"]
    gate["normalization_contract_sha_frozen"] = base.flat._verify_frozen(NORMALIZATION_CONTRACT)
    gate["normalization_run_artifacts_sha_frozen"] = all(
        base.flat._verify_frozen(path)
        for path in (NORMALIZATION_COPY_PATH, VALIDATOR_IDENTITY_PATH, IMPORT_MANIFEST_PATH, NORMALIZATION_DIAGNOSTICS_PATH)
    )
    gate["all_required_stage_gates_passed"] = all(value for key, value in gate.items() if key != "all_required_stage_gates_passed")
    passed = all(gate.values())
    finalized["stage"] = RUN_ID
    finalized["completion_gate_marker"] = f"{GATE}={'YES' if passed else 'NO'}"
    finalized["status"] = "COMPLETE" if passed else "FAILED"
    finalized["artifact_sha256"].update(
        {
            path.name: _sha256_file(path)
            for path in (
                NORMALIZATION_COPY_PATH,
                VALIDATOR_IDENTITY_PATH,
                IMPORT_MANIFEST_PATH,
                NORMALIZATION_DIAGNOSTICS_PATH,
                RUN_DIR / "mem_3a2s_protocol.md",
                RUN_DIR / "writer_preflight.json",
                RUN_DIR / "comparison_mem2d_vs_mem3a2s.json",
            )
            if path.exists()
        }
    )
    if not passed:
        finalized["status"] = "FAILED"
        raise RuntimeError(GATE_NO)
    report_path = RUN_DIR / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(f"{GATE}=PENDING", GATE_YES)
    if f"Completion gate: `{GATE_YES}`." not in report:
        report = report.replace(f"Completion gate: `{GATE}=NO`.", f"Completion gate: `{GATE_YES}`.")
    writer_v2.atomic_write_bytes(report_path, report.encode("utf-8"))
    base.flat._freeze(report_path)
    finalized["artifact_sha256"][report_path.name] = _sha256_file(report_path)
    finalized["historical_routing_commit_sha256"] = ROUTING_COMMIT
    finalized["historical_mem3a2r_commit_sha256"] = MEM3A2R_COMMIT
    finalized["convergence_head_sha256"] = capacity.BASE_COMMIT
    finalized["validator_identity_sha256"] = json.loads(VALIDATOR_IDENTITY_PATH.read_text(encoding="utf-8"))["identity_sha256"]
    base._write_json(RUN_DIR / "run_manifest.json", finalized)
    print(GATE_YES, flush=True)
    return finalized


def run(*, prepare_only: bool = False) -> dict[str, Any]:
    capacity.BASE_COMMIT = base.subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    capacity.RUN_ID = RUN_ID
    capacity.RUN_DIR = RUN_DIR
    capacity.LOCAL_CACHE_ROOT = ROOT / ".cache/health-copilot/mem3a2s-flatprop-setrefs-v1-final"
    capacity.GATE = GATE
    capacity.GATE_NO = GATE_NO
    capacity.GATE_YES = GATE_YES
    capacity._patch_runner()
    capacity._preflight = _preflight_16k
    base._preflight = _preflight_16k
    base.__file__ = str(Path(__file__).resolve())
    base.flat._verify_frozen = _verify_frozen
    base._freeze_source = _freeze_source
    base.flat.DATASET_PATH = SOURCE_DATASET_PATH
    base.flat.mem2c.DATASET_PATH = SOURCE_DATASET_PATH
    base._verify_upstream = _verify_upstream
    base._downstream = _downstream
    base._render_report = _render_report
    base._case_review = _case_review
    capacity._finalize_success = _finalize_success
    base._configure_flat_paths()
    flat = base.flat
    flat.COMPARISON_PATH = RUN_DIR / "comparison_mem2d_vs_mem3a2s.json"
    flat.PROTOCOL_PATH = RUN_DIR / "protocol_v3_16k_setrefs.md"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _write_bytes_frozen(RUN_DIR / "mem_3a2s_protocol.md", STAGE_PROTOCOL.read_bytes())
    preflight = capacity._preflight(save=True)
    if (
        len(preflight["requests"]) != 477
        or preflight["truncated_requests"] != 0
        or any(row["prompt_tokens"] + MAX_TOKENS > CONTEXT_LIMIT for row in preflight["requests"])
        or not preflight["all_prompt_tokens_plus_reserve_fit"]
    ):
        raise RuntimeError("MEM-3A.2S full-corpus 16K local writer preflight failed")
    _freeze_normalization_identity()
    preflight_path = RUN_DIR / "writer_preflight_v3_16k.json"
    base._write_json(RUN_DIR / "writer_preflight.json", preflight)
    if prepare_only:
        upstream = _verify_upstream()
        sessions, _, inventory = base._source_sessions_with_catalog()
        if len(sessions) != 477 or len(inventory) != 52703:
            raise RuntimeError("MEM-3A.2S frozen source corpus count changed")
        print(f"preflighted_sessions={len(preflight['requests'])}", flush=True)
        print(f"source_sessions={len(sessions)} rawspan_rows={len(inventory)}", flush=True)
        print(f"convergence_head={upstream['convergence_head_sha256']}", flush=True)
        print("NORMALIZATION_CONTRACT_SHA_FROZEN=YES", flush=True)
        return preflight
    upstream = _verify_upstream()
    sessions, refs, inventory = base._source_sessions_with_catalog()
    packets, writer_ledger, writer_manifest = _extract_all(preflight, sessions)
    if writer_manifest.get("failure") or len(packets) != 477:
        raise RuntimeError(f"{GATE_NO}; downstream not started")
    result = base._downstream(
        upstream, preflight, packets, writer_ledger, sessions, refs, inventory, writer_manifest
    )
    if not all(result["gate"].values()):
        result["completion_gate_marker"] = GATE_NO
        base._write_json(RUN_DIR / "run_manifest.json", result)
        raise RuntimeError(GATE_NO)
    return _finalize_success(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        run(prepare_only=args.prepare_only)
    except Exception as exc:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        message = str(exc)
        failure_code = (
            "WRITER_UNBOUNDED_OUTPUT_FAILURE"
            if "COMPLETION_TRUNCATED" in message or "finish_reason=length" in message
            else "HISTORICAL_FROZEN_RESPONSE_UNAVAILABLE"
            if "HISTORICAL_FROZEN_RESPONSE_UNAVAILABLE" in message
            else type(exc).__name__
        )
        failure = {
            "schema_version": 1,
            "stage": RUN_ID,
            "status": "FAILED",
            "failure": message,
            "failure_classification": failure_code,
            "completion_gate_marker": GATE_NO,
            "hosted_calls": 0,
            "labels_loaded": False,
            "test_access": False,
            "102_dev_access": False,
        }
        base._write_json(RUN_DIR / "run_failure.json", failure)
        print(f"{GATE_NO} ({failure_code}): {exc}", file=sys.stderr, flush=True)
        raise
