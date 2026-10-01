"""One-case LongMemEval current-snapshot diagnostic over frozen FlatProp artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.research.memory import final_reader_contract
from tools.research.memory.revision_state_materializer_v1 import (
    TemporalProposition,
    materialize,
)


SOURCE_RUN = ROOT / "runs/memory/mem3/mem3a3r-recursive-flatprop-frozen-10-20260929"
OUTPUT_RUN = ROOT / "runs/memory/mem3/mem3b1-instagram-snapshot-current-diagnostic-v1"
OUTPUT_RUN_STRUCTURED = ROOT / "runs/memory/mem3/mem3b1-instagram-structured-current-diagnostic-v1"
ENDPOINT = "http://127.0.0.1:8081/v1"
MODEL = "health-memory-qwen3-8b"
MODEL_PATH = r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf"
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
SERVER_PATH = r"C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
SERVER_SHA256 = "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"
QUESTION_ID = "1cea1afa"
DATASET_MANIFEST_PATH = ROOT / "docs/research/memory/dataset_manifest.json"
DATASET_PATH_CANDIDATES = (
    ROOT / "data/longmemeval/longmemeval_s_cleaned.json",
    ROOT.parent / "data/longmemeval/longmemeval_s_cleaned.json",
)
CURRENT_ID = "m10flat3-07ac40706e41096e3d9fd02724b63bebc480cdc607e8e661d8889d757d89c124"
HISTORICAL_ID = "m10flat3-33043d264ad7e8aca606b5998811aa46713126de5958e035196791146989a33a"
CURRENT_SLOT = ("longmemeval:1cea1afa", "SELF", "INSTAGRAM_ACCOUNT", "FOLLOWER_COUNT")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _iter_json_array(path: Path):
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    with path.open("r", encoding="utf-8") as source:
        buffer = ""
        eof = False
        started = False
        finished = False
        while not finished:
            if not eof and (len(buffer) < chunk_size or not started):
                buffer += source.read(chunk_size)
                eof = source.tell() == path.stat().st_size
            buffer = buffer.lstrip()
            if not started:
                if not buffer and eof:
                    raise ValueError("empty LongMemEval JSON array")
                if buffer:
                    if buffer[0] != "[":
                        raise ValueError("LongMemEval source must be a top-level JSON array")
                    buffer = buffer[1:]
                    started = True
                continue
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                finished = True
                continue
            if not buffer:
                if eof:
                    raise ValueError("unterminated LongMemEval JSON array")
                buffer += source.read(chunk_size)
                eof = source.tell() == path.stat().st_size
                continue
            try:
                row, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                buffer += source.read(chunk_size)
                eof = source.tell() == path.stat().st_size
                continue
            if not isinstance(row, dict):
                raise TypeError("LongMemEval array members must be objects")
            yield row
            buffer = buffer[end:]


_REFUSAL_PHRASES = {
    "none", "unknown", "not mentioned", "not stated", "cannot be determined",
    "cannot determine", "not specified", "not available", "no information", "no info",
    "no evidence", "not found", "not provided", "not addressed", "no relevant information",
    "no memory", "no record", "no data", "i dont know", "i do not know",
    "i dont remember", "i do not remember",
}


def _answer_metrics(predicted: str, expected: str) -> dict[str, float]:
    pred_tokens = set(re.findall(r"\w+", predicted.lower()))
    gold_tokens = set(re.findall(r"\w+", expected.lower()))
    if not gold_tokens:
        normalized = " ".join(re.findall(r"[a-z0-9]+", predicted.lower()))
        score = float(normalized in _REFUSAL_PHRASES)
        return {
            "token_precision": score,
            "token_recall": score,
            "f1": score,
            "normalized_exact_match": score,
        }
    if not pred_tokens:
        return {
            "token_precision": 0.0,
            "token_recall": 0.0,
            "f1": 0.0,
            "normalized_exact_match": 0.0,
        }
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    normalized_prediction = " ".join(re.findall(r"\w+", predicted.lower()))
    normalized_expected = " ".join(re.findall(r"\w+", expected.lower()))
    return {
        "token_precision": precision,
        "token_recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "normalized_exact_match": float(normalized_prediction == normalized_expected),
    }


def _verify_sidecar(path: Path) -> str:
    sidecars = (path.with_suffix(".sha256"), path.with_suffix(path.suffix + ".sha256"))
    sidecar = next((candidate for candidate in sidecars if candidate.is_file()), None)
    if not path.is_file() or sidecar is None:
        raise RuntimeError(f"missing frozen source artifact or SHA sidecar: {path.name}")
    expected, filename = sidecar.read_text(encoding="ascii").strip().split()
    actual = _sha_bytes(path.read_bytes())
    if filename != path.name or expected != actual:
        raise RuntimeError(f"frozen source artifact SHA mismatch: {path.name}")
    return actual


def _write_new(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return _sha_bytes(payload)


def _freeze_json(name: str, value: Any, *, output_run: Path = OUTPUT_RUN) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    return _freeze_artifact(output_run / name, payload)


def _freeze_artifact(path: Path, payload: bytes) -> str:
    digest = _write_new(path, payload)
    _write_new(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return digest


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _runtime_preflight(client: httpx.Client) -> dict[str, Any]:
    if urlsplit(ENDPOINT).hostname != "127.0.0.1":
        raise RuntimeError("reader endpoint is not loopback")
    response = client.get(f"{ENDPOINT}/models")
    response.raise_for_status()
    payload = response.json()
    model_ids = [row.get("id") for row in payload.get("data", []) if isinstance(row, dict)]
    props_response = client.get("http://127.0.0.1:8081/props")
    props_response.raise_for_status()
    props = props_response.json()
    if (
        props.get("build_info") != "b10068-571d0d540"
        or props.get("model_path", "").replace("/", "\\").casefold() != MODEL_PATH.replace("/", "\\").casefold()
        or props.get("model_ftype") != "Q4_K - Medium"
        or props.get("default_generation_settings", {}).get("n_ctx") != 131072
    ):
        raise RuntimeError("active llama.cpp endpoint does not expose the frozen reader identity")

    matching_processes = []
    for process in psutil.process_iter(["name", "exe", "cmdline"]):
        try:
            info = process.info
        except psutil.Error:
            continue
        args = info.get("cmdline") or []
        if (
            (info.get("name") or "").casefold() == "llama-server.exe"
            and "--host" in args
            and args[args.index("--host") + 1 : args.index("--host") + 2] == ["127.0.0.1"]
            and "--port" in args
            and args[args.index("--port") + 1 : args.index("--port") + 2] == ["8081"]
        ):
            matching_processes.append((process.pid, info, args))
    if len(matching_processes) != 1:
        raise RuntimeError(f"expected exactly one loopback llama-server on port 8081; found {len(matching_processes)}")
    process_id, info, args = matching_processes[0]
    process_identity = {
        "process_id": process_id,
        "executable_path": info.get("exe"),
        "model_path": MODEL_PATH,
        "command_line": " ".join(args),
    }
    process_identity["model_sha256"] = _sha_file(Path(process_identity["model_path"]))
    try:
        process_identity["executable_sha256"] = _sha_file(Path(process_identity["executable_path"]))
        process_identity["executable_hash_readable"] = True
    except PermissionError:
        process_identity["executable_sha256"] = None
        process_identity["executable_hash_readable"] = False
    process_identity["version"] = props["build_info"]
    required_args = {
        "--host": "127.0.0.1",
        "--port": "8081",
        "--ctx-size": "131072",
        "--rope-scaling": "yarn",
        "--rope-scale": "4",
        "--yarn-orig-ctx": "32768",
        "--cache-type-k": "q4_0",
        "--cache-type-v": "q4_0",
        "--n-gpu-layers": "99",
        "--flash-attn": "on",
        "--parallel": "1",
    }
    observed_args = {
        args[index]: args[index + 1]
        for index in range(len(args) - 1)
        if args[index].startswith("--")
    }
    if (
        process_identity.get("model_sha256") != MODEL_SHA256
        or str(process_identity.get("executable_path", "")).casefold() != SERVER_PATH.casefold()
        or (
            process_identity.get("executable_hash_readable")
            and process_identity.get("executable_sha256") != SERVER_SHA256
        )
        or "10068" not in process_identity.get("version", "")
        or "571d0d540" not in process_identity.get("version", "")
        or args[1:3] != ["-m", MODEL_PATH]
        or any(observed_args.get(key) != value for key, value in required_args.items())
    ):
        raise RuntimeError("active llama-server process differs from frozen model/runtime configuration")
    served_paths = [value for value in model_ids if isinstance(value, str)]
    if len(served_paths) != 1 or MODEL_PATH.casefold() not in served_paths[0].replace("/", "\\").casefold():
        raise RuntimeError(f"local endpoint serves an unexpected model ID: {model_ids}")
    return {
        "endpoint": ENDPOINT,
        "loopback_only": True,
        "served_model_ids": model_ids,
        "reader_model": MODEL,
        "reader_model_role": "answer model",
        "served_model_path": process_identity["model_path"],
        "reader_model_sha256": process_identity["model_sha256"],
        "llama_server_sha256_expected": SERVER_SHA256,
        "llama_server_sha256_active": process_identity["executable_sha256"],
        "llama_server_sha256_active_verified": process_identity["executable_hash_readable"],
        "llama_server_hash_note": (
            "active binary SHA256 matched" if process_identity["executable_hash_readable"]
            else "protected WinGet binary unreadable in sandbox; expected SHA256 carried from frozen startup pin"
        ),
        "llama_server_version": process_identity["version"],
        "llama_server_process_id": process_identity["process_id"],
        "runtime_args_verified": required_args,
        "hosted_api_calls": 0,
        "required_api_key": "NONE",
    }


def _load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    names = (
        "context_bundles.jsonl",
        "predictions.jsonl",
        "reader_call_ledger.jsonl",
        "flatprop_inventory.jsonl",
        "materialization_ledger.jsonl",
        "run_manifest.json",
    )
    source_hashes = {name: _verify_sidecar(SOURCE_RUN / name) for name in names[:-1]}
    manifest = _read_json(SOURCE_RUN / "run_manifest.json")
    if manifest.get("stage") != SOURCE_RUN.name or manifest.get("status") != "COMPLETE":
        raise RuntimeError("source FlatProp run is not a completed frozen diagnostic")
    contexts = _read_jsonl(SOURCE_RUN / "context_bundles.jsonl")
    predictions = _read_jsonl(SOURCE_RUN / "predictions.jsonl")
    calls = _read_jsonl(SOURCE_RUN / "reader_call_ledger.jsonl")
    inventory = _read_jsonl(SOURCE_RUN / "flatprop_inventory.jsonl")
    materialization = _read_jsonl(SOURCE_RUN / "materialization_ledger.jsonl")
    context_row = next(row for row in contexts if row["question_id"] == QUESTION_ID)
    baseline = next(row for row in predictions if row["question_id"] == QUESTION_ID)
    baseline_call = next(row for row in calls if row["question_id"] == QUESTION_ID)
    if baseline_call.get("success") is not True or baseline_call.get("hosted_call") is not False:
        raise RuntimeError("frozen baseline reader call is incomplete or not local")
    baseline_model_sha = baseline.get("cache_identity", {}).get("identity", {}).get("reader_model_sha256")
    baseline_runtime_sha = baseline.get("cache_identity", {}).get("identity", {}).get("reader_runtime_props_sha256")
    if (
        baseline_model_sha != "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
        or baseline_runtime_sha != "3842a26b19d18343b3a0525f15d01e80228a2b2a07e020414cfb41c5d2bd355b"
    ):
        raise RuntimeError("frozen baseline is not bound to the pinned local Qwen3-8B runtime")
    selected = {CURRENT_ID, HISTORICAL_ID}
    proposition_rows = [row for row in inventory if row.get("memory_id") in selected]
    ledger_rows = [row for row in materialization if row.get("memory_id") in selected]
    if {row["memory_id"] for row in proposition_rows} != selected or {row["memory_id"] for row in ledger_rows} != selected:
        raise RuntimeError("Instagram control facts do not resolve to exactly two frozen source rows")
    eligible_path = ROOT / "runs/memory/mem3/mem3b0q-factorized-admission-20260930/eligible_records.jsonl"
    eligible_sha = _verify_sidecar(eligible_path)
    eligible_rows = _read_jsonl(eligible_path)
    eligible_by_id = {row["memory_id"]: row for row in eligible_rows if row.get("memory_id") in {CURRENT_ID, HISTORICAL_ID}}
    if set(eligible_by_id) != {CURRENT_ID, HISTORICAL_ID} or any(
        row.get("scope_id") != "longmemeval:1cea1afa"
        or row.get("subject_key") != "user"
        or row.get("attribute_key") != "instagram_followers"
        for row in eligible_by_id.values()
    ):
        raise RuntimeError("frozen identity audit does not support one exact Instagram follower slot")
    return context_row, baseline, proposition_rows, {
        "source_hashes": source_hashes,
        "baseline_call": baseline_call,
        "materialization_rows": ledger_rows,
        "eligible_records_sha256": eligible_sha,
        "run_manifest_sha256": _sha_bytes((SOURCE_RUN / "run_manifest.json").read_bytes()),
        "audited_slot_keys": {memory_id: eligible_by_id[memory_id] for memory_id in sorted(eligible_by_id)},
    }


def _make_observations(
    proposition_rows: list[dict[str, Any]], materialization_rows: list[dict[str, Any]]
) -> list[TemporalProposition]:
    ledger_by_id = {row["memory_id"]: row for row in materialization_rows}
    result = []
    for row in proposition_rows:
        text = row["proposition_text"]
        quote = row["evidence"][0]["evidence_quote"]
        if not quote or not isinstance(text, str):
            raise RuntimeError("frozen user-source witness is missing")
        if "500 followers" in text:
            value = "500 followers"
            expected_cue = "500 followers"
        elif "600 followers" in text:
            value = "600 followers"
            expected_cue = "600 followers"
        else:
            raise RuntimeError("Instagram control value is outside the frozen expected pair")
        if expected_cue not in quote:
            raise RuntimeError("proposition value is not present in its user-source quote")
        ledger = ledger_by_id[row["memory_id"]]
        if row.get("source_authority") != "user" or ledger.get("source_type") != "session_derived":
            raise RuntimeError("non-user source cannot create the snapshot control")
        observed_at = ledger.get("valid_from")
        if not isinstance(observed_at, str):
            raise RuntimeError("frozen source observation time is missing")
        turns = row.get("source_turn_indices") or []
        result.append(
            TemporalProposition(
                memory_id=row["memory_id"],
                scope_id=row["scope_id"],
                owner_id="SELF",
                object_id="INSTAGRAM_ACCOUNT",
                attribute_id="FOLLOWER_COUNT",
                value_text=value,
                observed_at=_dt(observed_at),
                source_text=text,
                source_session_id=row["source_session_id"],
                source_turn_ids=tuple(f"{row['source_session_id']}:{turn}" for turn in turns),
                provenance_hash=_sha_bytes(text.encode("utf-8")),
                cardinality="SINGLE_VALUE_AT_A_TIME",
                temporal_basis="CURRENT_SNAPSHOT",
            )
        )
    return result


def _load_gold() -> str:
    dataset_manifest = _read_json(DATASET_MANIFEST_PATH)
    dataset_entry = next(
        row for row in dataset_manifest["datasets"] if row.get("dataset_id") == "longmemeval_s"
    )
    dataset_path = next((path for path in DATASET_PATH_CANDIDATES if path.is_file()), None)
    if dataset_path is None:
        raise RuntimeError("manifest-pinned LongMemEval-S dataset is not available")
    if dataset_path.stat().st_size != dataset_entry.get("expected_size_bytes"):
        raise RuntimeError("LongMemEval-S file size differs from the frozen dataset manifest")
    dataset_sha = _sha_file(dataset_path)
    if dataset_sha != dataset_entry.get("expected_sha256"):
        raise RuntimeError("LongMemEval-S SHA256 differs from the frozen dataset manifest")
    rows = _iter_json_array(dataset_path)
    target = next(row for row in rows if row.get("question_id") == QUESTION_ID)
    answer = target.get("answer")
    if not isinstance(answer, str) or not answer:
        raise RuntimeError("frozen LongMemEval gold answer is missing")
    return answer


def _serialize_context(items: list[dict[str, Any]]) -> str:
    blocks = []
    for index, item in enumerate(items, 1):
        blocks.append(f"[Context item {index} | memory]\n{item['text']}")
    return "\n\n".join(blocks)


def _render_current_state(
    items: list[dict[str, Any]],
    current_record: Any,
    observations: list[TemporalProposition],
) -> list[dict[str, Any]]:
    rendered = [dict(item) for item in items]
    source_by_id = {row.memory_id: row for row in observations}
    for item in rendered:
        if item.get("memory_id") != CURRENT_ID:
            continue
        if current_record.source_memory_ids != (CURRENT_ID,):
            raise RuntimeError("structured projection requires the exact materialized current value")
        metadata = json.loads(item["text"])
        proposition = metadata["value"]["proposition"]
        source = source_by_id[CURRENT_ID]
        item["text"] = "\n".join(
            (
                "[CURRENT STATE]",
                "Subject: SELF (the user)",
                "Object: Instagram account",
                "Attribute: follower count",
                f"Value: {current_record.value_text}",
                f"Valid from: {current_record.valid_from}",
                f"Source: user-reported session {source.source_session_id}",
                f"Source statement: {proposition}",
            )
        )
        item["reader_projection_format"] = "structured_current_state_v1"
    if not any(item.get("reader_projection_format") == "structured_current_state_v1" for item in rendered):
        raise RuntimeError("current state was not rendered into the shared-reader context")
    return rendered


def _preflight_token_count(client: httpx.Client, text: str) -> int:
    response = client.post(
        "http://127.0.0.1:8081/tokenize",
        json={"content": text, "add_special": False},
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("frozen Qwen llama.cpp tokenizer returned no token IDs")
    return len(tokens)


def run_once(
    *, projection_format: str = "raw", output_run: Path | None = None
) -> dict[str, Any]:
    if projection_format not in {"raw", "structured-current"}:
        raise ValueError(f"unsupported projection format: {projection_format}")
    output_run = output_run or (
        OUTPUT_RUN if projection_format == "raw" else OUTPUT_RUN_STRUCTURED
    )
    if output_run.exists():
        raise RuntimeError(f"append-never output already exists: {output_run}")
    context_row, baseline, proposition_rows, lineage = _load_frozen_inputs()
    context = context_row["context_bundle"]
    if context.get("question_id") != QUESTION_ID or baseline.get("predicted") != "None":
        raise RuntimeError("frozen Instagram baseline identity/result differs")
    contract, contract_sha, system_template, user_template = final_reader_contract.load_final_reader_contract()
    if contract_sha != context_row.get("shared_reader_contract_sha256"):
        raise RuntimeError("source context is not bound to the frozen shared-reader contract")

    observations = _make_observations(proposition_rows, lineage["materialization_rows"])
    state = materialize(observations)
    current_rows = state.current(CURRENT_SLOT)
    if len(current_rows) != 1 or current_rows[0].source_memory_ids != (CURRENT_ID,):
        raise RuntimeError("snapshot materializer did not select exactly the 600-follower state")
    historical_time = min(row.observed_at for row in observations)
    as_of_old = state.as_of(CURRENT_SLOT, historical_time)
    as_of_now = state.as_of(
        CURRENT_SLOT,
        max(row.observed_at for row in observations),
    )
    if len(as_of_old) != 1 or as_of_old[0].source_memory_ids != (HISTORICAL_ID,):
        raise RuntimeError("historical as-of projection did not preserve the 500-follower state")
    if len(as_of_now) != 1 or as_of_now[0].source_memory_ids != (CURRENT_ID,):
        raise RuntimeError("current-time as-of projection did not resolve to 600 followers")

    obsolete_ids = {
        memory_id
        for row in state.records
        if row.status == "SUPERSEDED"
        for memory_id in row.source_memory_ids
    }
    bundle_items = context["items"]
    if not {CURRENT_ID, HISTORICAL_ID}.issubset({item["memory_id"] for item in bundle_items}):
        raise RuntimeError("frozen reader bundle does not contain both conflicting Instagram values")
    projected_items = [dict(item) for item in bundle_items if item["memory_id"] not in obsolete_ids]
    if len(projected_items) != len(bundle_items) - 1:
        raise RuntimeError("context projection changed records beyond the single obsolete snapshot")
    if projection_format == "structured-current":
        projected_items = _render_current_state(projected_items, current_rows[0], observations)
    serialized_context = _serialize_context(projected_items)
    original_user = context_row["reader_messages"][1]["content"]
    match = re.search(r"Current Date: (.+?)\nQuestion: (.+)$", original_user, re.DOTALL)
    if not match:
        raise RuntimeError("frozen reader question/date could not be extracted")
    question_date, question = match.group(1), match.group(2)
    messages = final_reader_contract.build_reader_messages(
        question,
        question_date,
        serialized_context,
        system_template=system_template,
        user_template=user_template,
    )
    if (
        messages[0]["content"] != context_row["reader_messages"][0]["content"]
        or messages[1]["content"].split("Current Date: ", 1)[-1]
        != original_user.split("Current Date: ", 1)[-1]
    ):
        raise RuntimeError("controlled prompt changed outside the memory context")
    gold = _load_gold()
    baseline_metrics = _answer_metrics(baseline["predicted"], gold)
    context_token_count_before = context["context_reader_tokens"]
    bundle_projection = {
        "schema_version": 1,
        "question_id": QUESTION_ID,
        "parent_context_bundle_sha256": context_row["context_bundle_sha256"],
        "projection_format": projection_format,
        "projection_policy": (
            "single current-snapshot slot suppression; preserve historical state in materializer"
            if projection_format == "raw"
            else "single current-snapshot slot suppression plus explicit subject/object/attribute/value/time/source rendering"
        ),
        "removed_obsolete_memory_ids": sorted(obsolete_ids),
        "items": projected_items,
        "serialized_context": serialized_context,
    }
    bundle_projection_bytes = _canonical(bundle_projection)
    new_bundle_sha = _sha_bytes(bundle_projection_bytes)

    if urlsplit(ENDPOINT).hostname != "127.0.0.1":
        raise RuntimeError("reader endpoint is not loopback")
    with httpx.Client(timeout=180.0) as client:
        runtime = _runtime_preflight(client)
        reader_context_tokens = _preflight_token_count(client, serialized_context)
        rendered_response = client.post(
            "http://127.0.0.1:8081/apply-template",
            json={
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        rendered_response.raise_for_status()
        rendered_prompt = rendered_response.json().get("prompt")
        if not isinstance(rendered_prompt, str):
            raise RuntimeError("llama.cpp /apply-template returned no prompt")
        full_prompt_tokens = _preflight_token_count(client, rendered_prompt)
        request = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "seed": 42,
            "max_tokens": 256,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    if full_prompt_tokens + 256 >= 131072:
        raise RuntimeError("controlled reader prompt exceeds frozen context budget")
    request_bytes = _canonical(request)
    request_sha = _sha_bytes(request_bytes)
    output_run.mkdir(parents=True, exist_ok=False)
    projection_sha = _freeze_artifact(output_run / "context_projection.json", bundle_projection_bytes)
    materialized_sha = _freeze_json(
        "materialized_state.json",
        {
            "records": [row.__dict__ for row in state.records],
            "input_sha256": state.input_sha256,
            "output_sha256": state.output_sha256,
            "current_projection_memory_ids": [value for record in current_rows for value in record.source_memory_ids],
            "historical_as_of_memory_ids": [value for record in as_of_old for value in record.source_memory_ids],
        },
        output_run=output_run,
    )
    request_file_sha = _freeze_artifact(output_run / "reader_request.json", request_bytes + b"\n")
    reservation = {
        "run_id": output_run.name,
        "status": "ONE_LOCAL_POST_RESERVED",
        "question_id": QUESTION_ID,
        "endpoint": ENDPOINT,
        "reader_model": MODEL,
        "reader_role": "answer model",
        "memory_system": f"FlatProp + deterministic temporal snapshot projection ({projection_format})",
        "embedding_model": "NONE (reused frozen retrieval output)",
        "judge_model": "NONE",
        "hosted_calls": 0,
        "local_posts_authorized": 1,
        "request_sha256": request_sha,
        "request_file_sha256": request_file_sha,
        "context_projection_sha256": projection_sha,
        "materialized_state_sha256": materialized_sha,
        "prediction_sha_freeze_required": True,
    }
    reservation_sha = _freeze_json("run_reservation.json", reservation, output_run=output_run)
    response_bytes: bytes
    with httpx.Client(timeout=180.0) as client:
        response = client.post(
            f"{ENDPOINT}/chat/completions",
            content=request_bytes,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        response_bytes = response.content
    response_sha = _freeze_artifact(output_run / "reader_response.json", response_bytes)
    response_payload = json.loads(response_bytes)
    choice = response_payload["choices"][0]
    predicted = choice.get("message", {}).get("content")
    usage = response_payload.get("usage", {})
    if not isinstance(predicted, str):
        raise RuntimeError("local shared reader returned no answer text")
    predicted = predicted.strip()
    answer_metrics = _answer_metrics(predicted, gold)
    server_prompt_tokens = usage.get("prompt_tokens")
    finish_reason = choice.get("finish_reason")
    call = {
        **runtime,
        "role": "reader_answer",
        "question_id": QUESTION_ID,
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "local_posts": 1,
        "retry_count": 0,
        "prompt_tokens_preflight": full_prompt_tokens,
        "prompt_tokens_server": server_prompt_tokens,
        "prompt_tokens_match": server_prompt_tokens == full_prompt_tokens,
        "context_reader_tokens": reader_context_tokens,
        "completion_tokens_server": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "quality_status": "OK" if finish_reason == "stop" and isinstance(server_prompt_tokens, int) else "INFRA_FAILURE",
    }
    result = {
        "question_id": QUESTION_ID,
        "question": question,
        "gold": gold,
        "baseline_prediction": baseline["predicted"],
        "snapshot_projection_prediction": predicted,
        "baseline_metrics": baseline_metrics,
        "snapshot_projection_metrics": answer_metrics,
        "delta_f1": answer_metrics["f1"] - baseline_metrics["f1"],
        "reader_context_tokens_before": context_token_count_before,
        "reader_context_tokens_after": reader_context_tokens,
        "reader_prompt_tokens_before": context_row["reader_prompt_tokens_preflight"],
        "reader_prompt_tokens_after_preflight": full_prompt_tokens,
        "reader_prompt_tokens_after_server": server_prompt_tokens,
        "context_items_before": len(bundle_items),
        "context_items_after": len(projected_items),
        "context_bundle_sha256_before": context_row["context_bundle_sha256"],
        "context_projection_sha256": projection_sha,
        "materialized_state_sha256": materialized_sha,
        "obsolete_memory_exposures_removed_from_current_context": sorted(obsolete_ids),
        "historical_as_of_memory_ids": [value for record in as_of_old for value in record.source_memory_ids],
        "current_memory_ids": [value for record in current_rows for value in record.source_memory_ids],
        "retrieval_unchanged": True,
        "reader_contract_sha256": contract_sha,
        "finish_reason": finish_reason,
        "quality_status": call["quality_status"],
        "baseline_question_was_frozen_diagnostic_not_public_test": True,
        "projection_format": projection_format,
        "claim_boundary": "one public LongMemEval frozen-10 development case; manual slot/type assignment; no aggregate benchmark claim",
    }
    result_sha = _freeze_json("diagnostic_result.json", result, output_run=output_run)
    call_sha = _freeze_json("reader_call.json", call, output_run=output_run)
    manifest = {
        "run_id": output_run.name,
        "status": "COMPLETE" if call["quality_status"] == "OK" else "INFRA_FAILURE",
        "question_ids": [QUESTION_ID],
        "local_posts": 1,
        "hosted_calls": 0,
        "retries": 0,
        "judge_calls": 0,
        "embedding_calls": 0,
        "source_hashes": lineage["source_hashes"],
        "source_run_manifest_sha256": lineage["run_manifest_sha256"],
        "context_projection_sha256": projection_sha,
        "materialized_state_sha256": materialized_sha,
        "reader_request_sha256": request_sha,
        "reader_request_file_sha256": request_file_sha,
        "reader_response_sha256": response_sha,
        "reader_call_sha256": call_sha,
        "diagnostic_result_sha256": result_sha,
        "reservation_sha256": reservation_sha,
        "prediction_sha_freeze": "PASS" if response_sha and result_sha else "FAIL",
    }
    _freeze_json("run_manifest.json", manifest, output_run=output_run)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-format", choices=("raw", "structured-current"), default="raw")
    args = parser.parse_args()
    print(json.dumps(run_once(projection_format=args.projection_format), ensure_ascii=False, sort_keys=True, indent=2))
