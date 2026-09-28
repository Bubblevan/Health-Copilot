"""Run the frozen MEM-1D3 question-date reader-only counterfactual."""

from __future__ import annotations

import hashlib
import importlib
import json
import mmap
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ROOT / "tools" / "research" / "memory"
MEMEVAL_ROOT = ROOT.parent / "external" / "memory" / "MemEval"
MEMEVAL_SRC = MEMEVAL_ROOT / "src"
D1_RUN = ROOT / "runs" / "memory" / "mem1" / "mem1d1-frozen-10-20260927"
D3_RUN = ROOT / "runs" / "memory" / "mem1" / "mem1d3-reader-v2-20260928"
DATASET_PATH = ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
DATASET_MANIFEST_PATH = ROOT / "docs" / "research" / "memory" / "dataset_manifest.json"
LOCAL_PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem_1_local_only_protocol.json"
MODEL_PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "model_protocol.json"
V2_TEMPLATE_PATH = ROOT / "docs" / "research" / "memory" / "shared_reader_v2_question_date.txt"
QUESTION_DATES_PATH = ROOT / "docs" / "research" / "memory" / "mem_1d3_question_dates.json"
COUNTERFACTUAL_PATH = ROOT / "docs" / "research" / "memory" / "mem_1d3_reader_counterfactual.json"
REPORT_PATH = ROOT / "docs" / "research" / "memory" / "mem_1d3_question_date_contract.md"
D2_OVERLAY_PATH = ROOT / "docs" / "research" / "memory" / "mem_1d2_provenance_overlay.json"
D2_PACKET_PATH = ROOT / "docs" / "research" / "memory" / "mem_1d2_reflection_packet.json"
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
TEMPORAL_IDS = ("gpt4_e061b84g", "gpt4_f420262c")
SYSTEMS = ("fullcontext", "openclaw", "mem0", "simplemem", "propmem")
FROZEN_D1_ARTIFACTS = (
    "context_bundles.jsonl",
    "context_bundles.sha256",
    "predictions.jsonl",
    "predictions.sha256",
    "call_ledger.jsonl",
    "call_ledger.sha256",
    "baseline_warnings.jsonl",
    "baseline_warnings.sha256",
)
QUESTION_ID_PREFIX = re.compile(rb'^\s*\{\s*"question_id"\s*:\s*"([A-Za-z0-9_-]+)"')
ANSWER_MAX_NEW_TOKENS = 256
MAX_MODEL_LENGTH = 131072
MODEL_ALIAS = "health-memory-qwen3-8b"
LOCAL_BASE_URL = "http://127.0.0.1:8081/v1"
OUTCOME_LABELS = (
    "ANSWER_CORRECT",
    "ANSWER_PARTIAL",
    "ANSWER_WRONG",
    "ABSTAINED",
)
FAILURE_LOCUS_LABELS = (
    "WRITE_OR_COMPRESSION_LOSS",
    "RETRIEVAL_SESSION_MISS",
    "SESSION_HIT_EVIDENCE_ITEM_MISS",
    "CONTEXT_HAS_EVIDENCE_READER_FAIL",
    "MULTI_SESSION_COMPOSITION_FAIL",
    "TEMPORAL_ANCHOR_MISSING",
    "TEMPORAL_ORDERING_FAIL",
    "CURRENT_STATE_RESOLUTION_FAIL",
    "CHANGE_REASONING_FAIL",
    "PREFERENCE_SYNTHESIS_FAIL",
    "UNKNOWN",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_selected_records(path: Path, wanted_ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, str], int]:
    """Decode only frozen DEV rows; inspect only leading IDs in other objects."""
    selected: dict[str, dict[str, Any]] = {}
    record_hashes: dict[str, str] = {}
    skipped = 0
    with path.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
            length = len(data)
            index = 0
            while index < length and data[index] in b" \t\r\n":
                index += 1
            if index >= length or data[index] != ord("["):
                raise ValueError("Frozen LongMemEval source must be a top-level JSON array")
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
                    raise ValueError("Frozen LongMemEval contains an unterminated record")

                raw_record = data[start:index]
                match = QUESTION_ID_PREFIX.match(raw_record)
                if not match:
                    raise ValueError("Question ID must be the first top-level LongMemEval field")
                question_id = match.group(1).decode("ascii")
                if question_id not in wanted_ids:
                    skipped += 1
                    continue
                if question_id in selected:
                    raise ValueError(f"Duplicate frozen DEV record: {question_id}")
                record = json.loads(raw_record)
                if record.get("question_id") != question_id:
                    raise ValueError("Question ID prefix disagrees with the decoded selected record")
                selected[question_id] = record
                record_hashes[question_id] = _sha256_bytes(raw_record)

            while index < length and data[index] in b" \t\r\n":
                index += 1
            if index != length:
                raise ValueError("Unexpected trailing content after frozen LongMemEval array")
    if set(selected) != wanted_ids:
        raise ValueError(f"Frozen DEV selection mismatch: missing={sorted(wanted_ids - set(selected))}")
    return selected, record_hashes, skipped


def _verify_hash_sidecar(path: Path, sidecar: Path) -> bool:
    if not path.is_file() or not sidecar.is_file():
        return False
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    return len(fields) == 2 and fields[1] == path.name and fields[0] == _sha256_file(path)


def _freeze_hash_sidecar(path: Path) -> str:
    digest = _sha256_file(path)
    path.with_name(path.name.replace(".jsonl", ".sha256")).write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def _freeze_file_sidecar(path: Path) -> str:
    digest = _sha256_file(path)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def _load_normalizer():
    head = subprocess.check_output(
        ["git", "-C", str(MEMEVAL_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(MEMEVAL_ROOT), "status", "--porcelain"], text=True
    )
    if head != "807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4" or status.strip():
        raise RuntimeError("Pinned MemEval must remain clean for question-date normalization")
    sys.path.insert(0, str(MEMEVAL_SRC))
    module = importlib.import_module("agents_memory.benchmarks.longmemeval")
    return module._normalize


def _load_template() -> tuple[str, str, bytes]:
    raw = V2_TEMPLATE_PATH.read_bytes()
    text = raw.decode("utf-8")
    parts = text.split("---USER---\n")
    if len(parts) != 2:
        raise ValueError("Reader v2 template must contain exactly one ---USER--- separator")
    system_template, user_template = parts
    required = ("<<QUESTION_DATE>>", "<<MEMORY_CONTEXT>>", "<<QUESTION>>")
    if any(text.count(marker) != 1 for marker in required):
        raise ValueError("Reader v2 template placeholders are missing or repeated")
    return system_template.rstrip("\n"), user_template.rstrip("\n"), raw


def build_reader_messages(
    question: str,
    question_date: str,
    context: str,
    system_template: str,
    user_template: str,
) -> list[dict[str, str]]:
    if not isinstance(question_date, str) or not question_date:
        raise ValueError("Reader v2 requires a non-empty official question_date")
    system = system_template.replace("<<QUESTION_DATE>>", question_date)
    user = user_template.replace("<<MEMORY_CONTEXT>>", context).replace("<<QUESTION>>", question)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _argument(command_line: str, name: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(name)}\s+(?:\"([^\"]+)\"|(\S+))", command_line)
    return (match.group(1) or match.group(2)) if match else None


def _live_processes() -> list[dict[str, Any]]:
    script = (
        "$rows=@(netstat -ano -p tcp | Select-String '^\\s*TCP\\s+127\\.0\\.0\\.1:8081\\s+\\S+\\s+LISTENING\\s+(\\d+)\\s*$'); "
        "$p=@(); foreach($row in $rows) { "
        "if($row.Line -match '(\\d+)\\s*$') { $processId=[int]$Matches[1]; "
        "$p+=@(Get-WmiObject Win32_Process -Filter \"ProcessId=$processId\" | "
        "Select-Object ProcessId,ExecutablePath,CommandLine) } }; "
        "ConvertTo-Json -InputObject $p -Compress -Depth 4"
    )
    output = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command", script], text=True
    ).strip()
    if not output:
        return []
    parsed = json.loads(output)
    return parsed if isinstance(parsed, list) else [parsed]


def _runtime_preflight(d1_manifest: dict[str, Any]) -> dict[str, Any]:
    local_protocol = json.loads(LOCAL_PROTOCOL_PATH.read_text(encoding="utf-8"))
    model_protocol = json.loads(MODEL_PROTOCOL_PATH.read_text(encoding="utf-8"))
    reader = d1_manifest["roles"]["reader_answer_model"]
    expected_model_sha = local_protocol["roles"]["reader_answer_model"]["sha256"]
    expected_binary_sha = local_protocol["local_execution_constraints"]["reader_runtime"]["server_binary_sha256"]
    artifact_path = Path(local_protocol["roles"]["reader_answer_model"]["artifact"])
    if model_protocol["main_track"]["sha256"] != expected_model_sha or reader["artifact_sha256"] != expected_model_sha:
        raise RuntimeError("MEM-1D1 and active local protocol disagree on the frozen reader SHA")
    model_sha = _sha256_file(artifact_path)
    if model_sha != expected_model_sha:
        raise RuntimeError("Local Qwen3-8B GGUF hash differs from the frozen reader")

    all_processes = _live_processes()
    matching = [
        process for process in all_processes
        if _argument(process.get("CommandLine", ""), "--host") == "127.0.0.1"
        and _argument(process.get("CommandLine", ""), "--port") == "8081"
    ]
    if len(matching) != 1:
        raise RuntimeError(f"Expected one frozen loopback llama-server process; found {len(matching)}")
    process = matching[0]
    command_line = process.get("CommandLine") or ""
    executable = Path(process.get("ExecutablePath") or "")
    binary_sha = _sha256_file(executable)
    if binary_sha != expected_binary_sha:
        raise RuntimeError("Running llama-server.exe hash differs from the frozen build")
    version = subprocess.check_output([str(executable), "--version"], text=True, stderr=subprocess.STDOUT).strip()
    if "10068" not in version or "571d0d540" not in version:
        raise RuntimeError("Running llama-server version/build differs from the frozen runtime")
    required_args = {
        "--host": "127.0.0.1",
        "--port": "8081",
        "--ctx-size": "131072",
        "--rope-scaling": "yarn",
        "--rope-scale": "4",
        "--cache-type-k": "q4_0",
        "--cache-type-v": "q4_0",
        "--n-gpu-layers": "99",
        "--flash-attn": "on",
        "--parallel": "1",
    }
    for name, expected in required_args.items():
        if _argument(command_line, name) != expected:
            raise RuntimeError(f"Running llama-server argument differs from frozen value: {name}")
    model_arg = _argument(command_line, "-m")
    if not model_arg or model_arg.replace("/", "\\").casefold() != str(artifact_path).replace("/", "\\").casefold():
        raise RuntimeError("Running llama-server does not load the frozen Qwen3-8B artifact")
    if "--no-kv-offload" in command_line:
        raise RuntimeError("Frozen reader requires GPU KV cache; CPU KV override is active")

    with httpx.Client(timeout=15, trust_env=False) as client:
        models_response = client.get(f"{LOCAL_BASE_URL}/models")
        models_response.raise_for_status()
        models = models_response.json().get("data", [])
        model_ids = [str(row.get("id", "")) for row in models]
        if not any(value.replace("/", "\\").casefold() == str(artifact_path).replace("/", "\\").casefold() for value in model_ids):
            raise RuntimeError("Loopback reader endpoint does not expose the frozen GGUF model")
        slots_response = client.get("http://127.0.0.1:8081/slots")
        slots_response.raise_for_status()
        slots = slots_response.json()
    if not isinstance(slots, list) or not slots:
        raise RuntimeError("Loopback llama-server returned no context slot metadata")
    slot_contexts = [slot.get("n_ctx") for slot in slots if isinstance(slot.get("n_ctx"), int)]
    if not slot_contexts or min(slot_contexts) != MAX_MODEL_LENGTH:
        raise RuntimeError("Loopback reader slot is not configured for 131072 tokens")
    if any(slot.get("is_processing") for slot in slots if isinstance(slot, dict)):
        raise RuntimeError("Frozen reader slot is currently processing another request")

    runtime = local_protocol["local_execution_constraints"]["reader_runtime"]
    if runtime["server"] != "llama.cpp llama-server 10068 (571d0d540)":
        raise RuntimeError("Frozen protocol runtime version is inconsistent")
    return {
        "provider": "local_qwen",
        "endpoint": LOCAL_BASE_URL,
        "endpoint_loopback_only": True,
        "proxy_environment_used": False,
        "model": reader["model"],
        "model_artifact": str(artifact_path),
        "model_sha256": model_sha,
        "server_pid": int(process["ProcessId"]),
        "server_executable": str(executable),
        "server_binary_sha256": binary_sha,
        "server_version_output": version,
        "server_arguments": {
            name: _argument(command_line, name) for name in required_args
        },
        "kv_cache_offload": "GPU",
        "flash_attention": True,
        "slot_context_tokens": min(slot_contexts),
        "generation": {
            "temperature": 0,
            "seed": 42,
            "enable_thinking": False,
            "max_new_tokens": ANSWER_MAX_NEW_TOKENS,
        },
        "embedding_model": None,
        "memory_internal_llm_calls": 0,
        "judge_model": None,
        "hosted_api_calls": 0,
        "official_question_dates_required": True,
        "historic_reader_template_sha256": "0ff70b000bd4b43db85ae587731fea35b691febc815cfbb57c2acb1c8b295b2b",
    }


def _render_and_tokenize(client, messages: list[dict[str, str]]) -> tuple[int, str]:
    rendered = client.post(
        "http://127.0.0.1:8081/apply-template",
        json={
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    rendered.raise_for_status()
    prompt = rendered.json().get("prompt")
    if not isinstance(prompt, str):
        raise TypeError("llama.cpp /apply-template returned no rendered prompt")
    tokenized = client.post(
        "http://127.0.0.1:8081/tokenize",
        json={"content": prompt, "add_special": False},
    )
    tokenized.raise_for_status()
    tokens = tokenized.json().get("tokens")
    if not isinstance(tokens, list):
        raise TypeError("llama.cpp /tokenize returned no token IDs")
    return len(tokens), _sha256_bytes(prompt.encode("utf-8"))


def _answer_metrics(run_mem1, prediction: str, answer: str) -> dict[str, float]:
    return run_mem1._answer_metrics(prediction, answer)


def _source_evidence_hashes() -> dict[str, str]:
    hashes = {}
    for name in FROZEN_D1_ARTIFACTS:
        path = D1_RUN / name
        if name.endswith(".sha256"):
            artifact_name = name.removesuffix(".sha256") + ".jsonl"
            artifact = D1_RUN / artifact_name
            if not _verify_hash_sidecar(artifact, path):
                raise RuntimeError(f"Frozen MEM-1D1 evidence sidecar failed verification: {artifact_name}")
        elif not _verify_hash_sidecar(path, D1_RUN / name.replace(".jsonl", ".sha256")):
            raise RuntimeError(f"Frozen MEM-1D1 evidence failed verification: {name}")
        hashes[name] = _sha256_file(path)
    if not D2_OVERLAY_PATH.is_file():
        raise RuntimeError("Required MEM-1D2 provenance overlay is absent")
    hashes["mem_1d2_provenance_overlay.json"] = _sha256_file(D2_OVERLAY_PATH)
    return hashes


def _prepare_question_dates(records, record_hashes, dataset_manifest):
    rows = []
    for question_id in QUESTION_IDS:
        record = records[question_id]
        question_date = record.get("question_date")
        question_type = record.get("question_type")
        if not isinstance(question_date, str) or not question_date:
            raise ValueError(f"Official LongMemEval record {question_id} lacks question_date")
        if not isinstance(question_type, str) or not question_type:
            raise ValueError(f"Official LongMemEval record {question_id} lacks question_type")
        rows.append({
            "question_id": question_id,
            "question_type": question_type,
            "question_date": question_date,
            "source_record_sha256": record_hashes[question_id],
            "source_record_hash_scope": "exact raw JSON object bytes in the pinned LongMemEval-S file",
        })
    return {
        "artifact_version": "mem1d3-question-dates-v1",
        "dataset_id": "longmemeval_s",
        "dataset_revision": dataset_manifest["source_revision"],
        "dataset_sha256": dataset_manifest["expected_sha256"],
        "selection": "exact frozen MEM-1D1 ten DEV question IDs only",
        "test_access": False,
        "non_target_selection_behavior": "stream top-level object boundaries and leading question_id only; do not JSON-decode non-target records",
        "records": rows,
    }


def _build_counterfactual(old_by_key, new_by_key, question_dates):
    qdate_by_id = {row["question_id"]: row["question_date"] for row in question_dates["records"]}
    rows = []
    for system in SYSTEMS:
        for question_id in QUESTION_IDS:
            old = old_by_key[(system, question_id)]
            new = new_by_key[(system, question_id)]
            old_metrics = {
                key: old.get(key)
                for key in ("token_precision", "token_recall", "f1", "normalized_exact_match")
            }
            new_metrics = {
                key: new.get(key)
                for key in ("token_precision", "token_recall", "f1", "normalized_exact_match")
            }
            rows.append({
                "system": system,
                "question_id": question_id,
                "question_type": old["category"],
                "question_date": qdate_by_id[question_id],
                "question": old["question"],
                "gold_answer": old["ground_truth"],
                "old_prediction_v1": old["predicted"],
                "date_anchored_prediction_v2": new["predicted"],
                "old_lexical_metrics": old_metrics,
                "new_lexical_metrics": new_metrics,
                "lexical_metric_delta_new_minus_old": {
                    key: (new_metrics[key] - old_metrics[key])
                    if isinstance(new_metrics[key], (int, float)) and isinstance(old_metrics[key], (int, float))
                    else None
                    for key in ("token_precision", "token_recall", "f1", "normalized_exact_match")
                },
                "semantic_review_status": "PENDING_HUMAN_REVIEW",
                "same_context_bundle_sha256": old["context_bundle_sha256"] == new["context_bundle_sha256"],
                "context_bundle_sha256": new["context_bundle_sha256"],
                "old_prompt_sha256": old["shared_reader_prompt_sha256"],
                "new_prompt_sha256": new["shared_reader_prompt_sha256"],
                "d1d2_evidence_packet_pointer": {
                    "artifact": "docs/research/memory/mem_1d2_reflection_packet.json",
                    "system": system,
                    "question_id": question_id,
                },
                "human_reflection_labels": {
                    "outcome": None,
                    "failure_loci": None,
                    "allowed_outcomes": list(OUTCOME_LABELS),
                    "allowed_failure_loci": list(FAILURE_LOCUS_LABELS),
                },
                "automatic_semantic_or_failure_labels": [],
                "diagnostic_only": True,
            })

    temporal = [row for row in rows if row["question_id"] in TEMPORAL_IDS]
    non_temporal = [row for row in rows if row["question_id"] not in TEMPORAL_IDS]

    def summary(group):
        deltas = [row["lexical_metric_delta_new_minus_old"]["f1"] for row in group]
        numeric_deltas = [value for value in deltas if isinstance(value, (int, float))]
        changed = sum(row["old_prediction_v1"] != row["date_anchored_prediction_v2"] for row in group)
        return {
            "paired_rows": len(group),
            "answer_text_changed_count": changed,
            "mean_raw_f1_delta": sum(numeric_deltas) / len(numeric_deltas) if numeric_deltas else None,
            "interpretation": "descriptive lexical delta only; not an improvement claim",
        }

    return {
        "artifact_version": "mem1d3-reader-counterfactual-v1",
        "run_id": D3_RUN.name,
        "paired_design": "same five systems and ten questions; v1/v2 share each immutable ContextBundle",
        "comparison_basis": "deterministic token metrics only; semantic review pending",
        "temporal_question_ids": list(TEMPORAL_IDS),
        "non_temporal_question_ids": [question_id for question_id in QUESTION_IDS if question_id not in TEMPORAL_IDS],
        "temporal_cases": summary(temporal),
        "non_temporal_cases": summary(non_temporal),
        "no_automatic_improvement_claim": True,
        "rows": rows,
    }


def _report_text(question_dates, run_config, run_manifest, counterfactual, fullcontext_rows, source_hashes_before, source_hashes_after):
    gate = run_manifest["gate"]
    paired = counterfactual

    def fmt(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) else "unavailable"

    lines = [
        "# MEM-1D3 Question-Date Contract Repair",
        "",
        f"- Gate: `MEM1D3_QUESTION_DATE_CONTRACT_REPAIRED={'YES' if gate['passed'] else 'NO'}`",
        "- Scope: exact frozen MEM-1D1 ten DEV IDs × five systems; 50 reader-only calls; no memory ingestion/retrieval, embedding, judge, hosted API, TEST QA payload, 102 DEV, M10-Base, RevMem, or RL.",
        "- Root cause: official LongMemEval records include `question_date`, but the pinned compatibility normalizer previously discarded it; D1 temporal outputs therefore lack the query-time anchor.",
        "- Interpretation: `MEM1D_TEMPORAL_INTERPRETATION_INVALIDATED_BY_MISSING_QUESTION_DATE=YES`; this invalidates only D1 temporal causal interpretation, not the frozen D1 artifacts.",
        "- Question dates are copied exactly from the official source record; no dates are synthesized or inferred.",
        "",
        "## Frozen Inputs And Runtime",
        "",
        f"- DEV records with official question dates and source-record hashes: `{len(question_dates['records'])}/10`.",
        f"- Frozen ContextBundles re-used and hash-verified: `{run_manifest['context_bundles_verified']}/50` (no bundle copy or reconstruction).",
        f"- Reader: `{run_config['reader']['model']}`; model SHA256 `{run_config['reader']['model_sha256']}`.",
        f"- llama.cpp: `{run_config['runtime']['server_version_output']}`; binary SHA256 `{run_config['runtime']['server_binary_sha256']}`.",
        "- Runtime: 131072 context, 99 GPU layers, Flash Attention, Q4_0 GPU KV; generation temperature 0, seed 42, thinking false, max new tokens 256.",
        f"- Reader-v2 raw template SHA256: `{run_config['template']['file_sha256']}`; canonical message-template SHA256: `{run_config['template']['message_template_sha256']}`.",
        f"- Frozen v1 historical template SHA256: `{run_config['template']['v1_template_sha256']}` (not overwritten).",
        "- Hosted API calls: `0`; judge calls: `0`; embedding/model initialization: `0`; memory-system calls: `0`.",
        "",
        "## FullContext Prompt Fit",
        "",
        "Prompt lengths use the running frozen llama.cpp `/apply-template` plus `/tokenize` endpoints with the reader's chat template and `enable_thinking=false`.",
        "",
        "| Question ID | Prompt tokens | Reserve | Context limit | Fits | Truncated |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in fullcontext_rows:
        lines.append(
            f"| {row['question_id']} | {row['prompt_tokens']} | 256 | 131072 | {row['fits']} | {row['truncated']} |"
        )
    lines.extend([
        "",
        "## Paired Counterfactual",
        "",
        f"- Temporal cases (`gpt4_e061b84g`, `gpt4_f420262c`): `{paired['temporal_cases']['paired_rows']}` system×question rows; changed answer strings `{paired['temporal_cases']['answer_text_changed_count']}/{paired['temporal_cases']['paired_rows']}`; mean raw ΔF1 `{fmt(paired['temporal_cases']['mean_raw_f1_delta'])}`.",
        f"- Non-temporal questions (other eight): `{paired['non_temporal_cases']['paired_rows']}` system×question rows; changed answer strings `{paired['non_temporal_cases']['answer_text_changed_count']}/{paired['non_temporal_cases']['paired_rows']}`; mean raw ΔF1 `{fmt(paired['non_temporal_cases']['mean_raw_f1_delta'])}`.",
        "- These are lexical diagnostics only, not improvement claims. All 50 rows remain `PENDING_HUMAN_REVIEW`; no outcome or failure-locus labels were assigned automatically.",
        "- Knowledge-update cases `1cea1afa` and `c4ea545c`, multi-session case `1c549ce4`, and preference cases remain separate axes. D1 temporal answers are historical diagnostics; subsequent temporal interpretation must use reader v2.",
        "",
        "## Evidence Integrity",
        "",
        f"- MEM-1D1 artifact/sidecar and D2 provenance-overlay hashes unchanged: `{source_hashes_before == source_hashes_after}`.",
        f"- Reader-only calls completed: `{run_manifest['reader_calls_successful']}/50`; call-ledger rows: `{run_manifest['call_ledger_rows']}`.",
        f"- FullContext prompt fit and non-truncation: `{run_manifest['fullcontext_fit_all']}`.",
        "- Question-date source audit: `docs/research/memory/mem_1d3_question_dates.json`.",
        "- Paired artifact: `docs/research/memory/mem_1d3_reader_counterfactual.json`.",
        f"- New run directory: `{D3_RUN.relative_to(ROOT).as_posix()}`; ContextBundles are referenced by hash, not copied.",
        "- Research positioning updated in `docs/research/memory/baseline_sources.md`; RevMem is only a harness-native revision-aware memory experiment, not a novelty/SOTA claim.",
        "",
        "STOP: do not proceed to 102 DEV, M10-Base, RevMem, or RL without a separate human review/authorization.",
        "",
    ])
    return "\n".join(lines)


def run() -> dict[str, Any]:
    if D3_RUN.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite existing D3 run directory: {D3_RUN}")
    for path in (QUESTION_DATES_PATH, COUNTERFACTUAL_PATH, REPORT_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing MEM-1D3 artifact: {path}")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(DATASET_PATH)

    sys.path.insert(0, str(TOOLS_DIR))
    run_mem1 = importlib.import_module("run_mem1")
    from context_bundle import verify_context_bundle
    from mem1_artifacts import read_jsonl

    d1_manifest = json.loads((D1_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    if d1_manifest.get("split") != "DEV" or d1_manifest.get("test_access") is not False:
        raise RuntimeError("MEM-1D1 manifest is not a frozen DEV-only run")
    if tuple(d1_manifest.get("question_ids", [])) != QUESTION_IDS:
        raise RuntimeError("MEM-1D1 question IDs differ from the frozen ten-case set")
    dataset_manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    dataset_entry = next(row for row in dataset_manifest["datasets"] if row["dataset_id"] == "longmemeval_s")
    if _sha256_file(DATASET_PATH) != dataset_entry["expected_sha256"]:
        raise RuntimeError("LongMemEval-S file hash differs from the pinned dataset manifest")
    source_hashes_before = _source_evidence_hashes()

    records, raw_record_hashes, non_target_count = load_selected_records(DATASET_PATH, set(QUESTION_IDS))
    if non_target_count != dataset_entry["question_count"] - len(QUESTION_IDS):
        raise RuntimeError("Bounded source selector did not encounter the expected LongMemEval-S record count")
    normalize = _load_normalizer()
    normalized_records = run_mem1.select_records(
        [records[question_id] for question_id in QUESTION_IDS], list(QUESTION_IDS), normalize
    )
    normalized_by_id = {row["qa"][0]["question_id"]: row for row in normalized_records}
    if set(normalized_by_id) != set(QUESTION_IDS):
        raise RuntimeError("Question-date normalization changed the frozen DEV ID set")
    for question_id in QUESTION_IDS:
        original = records[question_id]
        qa = normalized_by_id[question_id]["qa"][0]
        if qa.get("question_date") != original["question_date"]:
            raise RuntimeError(f"Normalizer did not preserve official question_date for {question_id}")
    question_dates = _prepare_question_dates(records, raw_record_hashes, dataset_entry)
    bundles_path = D1_RUN / "context_bundles.jsonl"
    predictions_path = D1_RUN / "predictions.jsonl"
    bundle_rows = read_jsonl(bundles_path)
    old_rows = read_jsonl(predictions_path)
    expected_keys = {(system, question_id) for system in SYSTEMS for question_id in QUESTION_IDS}
    bundles_by_key = {(row["system"], row["question_id"]): row["context_bundle"] for row in bundle_rows}
    old_by_key = {(row["system"], row["question_id"]): row for row in old_rows}
    if set(bundles_by_key) != expected_keys or len(bundle_rows) != 50:
        raise RuntimeError("MEM-1D1 must contain exactly 50 frozen ContextBundles")
    if set(old_by_key) != expected_keys or len(old_rows) != 50:
        raise RuntimeError("MEM-1D1 must contain exactly 50 frozen predictions")
    for key in sorted(expected_keys):
        bundle = bundles_by_key[key]
        old = old_by_key[key]
        if not verify_context_bundle(bundle):
            raise RuntimeError(f"ContextBundle content/canonical SHA invalid: {key}")
        if bundle["context_bundle_sha256"] != old.get("context_bundle_sha256"):
            raise RuntimeError(f"Prediction is not bound to the frozen ContextBundle: {key}")
        if old.get("quality_status") != "OK" or not old.get("shared_reader_prompt_sha256"):
            raise RuntimeError(f"Frozen D1 prediction is incomplete: {key}")
        if old.get("shared_reader_template_sha256") != run_mem1.D1_READER_TEMPLATE_SHA256:
            raise RuntimeError(f"Historical v1 prompt identity differs from locked value: {key}")
        normalized = normalized_by_id[key[1]]["qa"][0]
        if old.get("question") != normalized.get("question"):
            raise RuntimeError(f"Question text differs between D1 and official source: {key[1]}")

    system_template, user_template, template_bytes = _load_template()
    template_file_sha = _sha256_bytes(template_bytes)
    template_messages = build_reader_messages(
        "<<QUESTION>>", "<<QUESTION_DATE>>", "<<MEMORY_CONTEXT>>", system_template, user_template
    )
    template_message_sha = _sha256_bytes(_canonical_json(template_messages))
    prepared: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for question_id in QUESTION_IDS:
            normalized = normalized_by_id[question_id]["qa"][0]
            bundle = bundles_by_key[(system, question_id)]
            messages = build_reader_messages(
                normalized["question"], normalized["question_date"], bundle["serialized_context"],
                system_template, user_template,
            )
            prepared.append({
                "system": system,
                "question_id": question_id,
                "question": normalized["question"],
                "question_date": normalized["question_date"],
                "question_type": normalized["category"],
                "gold_answer": normalized["answer"],
                "context_bundle": bundle,
                "context_bundle_sha256": bundle["context_bundle_sha256"],
                "messages": messages,
                "prompt_sha256": _sha256_bytes(_canonical_json(messages)),
                "source_record_sha256": raw_record_hashes[question_id],
            })

    with httpx.Client(timeout=3600, trust_env=False) as client:
        runtime = _runtime_preflight(d1_manifest)
        preflight_rows = []
        for row in prepared:
            tokens, rendered_sha = _render_and_tokenize(client, row["messages"])
            row["reader_prompt_tokens"] = tokens
            row["rendered_prompt_sha256"] = rendered_sha
            if row["system"] == "fullcontext":
                preflight_rows.append({
                    "question_id": row["question_id"],
                    "prompt_tokens": tokens,
                    "output_reserve": ANSWER_MAX_NEW_TOKENS,
                    "max_model_length": MAX_MODEL_LENGTH,
                    "fits": tokens + ANSWER_MAX_NEW_TOKENS <= MAX_MODEL_LENGTH,
                    "truncated": tokens + ANSWER_MAX_NEW_TOKENS > MAX_MODEL_LENGTH,
                })

    fullcontext_fit_all = len(preflight_rows) == len(QUESTION_IDS) and all(row["fits"] and not row["truncated"] for row in preflight_rows)
    if not fullcontext_fit_all:
        raise RuntimeError("Reader-v2 FullContext prompt plus reserve does not fit; refusing to truncate or generate")

    config = {
        "config_version": "mem1d3-reader-only-v1",
        "run_id": D3_RUN.name,
        "dataset": {
            "id": "longmemeval_s",
            "revision": dataset_entry["source_revision"],
            "sha256": dataset_entry["expected_sha256"],
            "split": "DEV",
            "question_ids": list(QUESTION_IDS),
            "test_access": False,
        },
        "reader": {
            "model": runtime["model"],
            "artifact": runtime["model_artifact"],
            "model_sha256": runtime["model_sha256"],
            "generation": runtime["generation"],
        },
        "runtime": runtime,
        "memory_system": {
            "operation": "reuse frozen ContextBundles only",
            "systems": list(SYSTEMS),
            "ingestion_calls": 0,
            "retrieval_calls": 0,
            "memory_internal_llm_calls": 0,
        },
        "embedding": {"model": None, "loaded": False, "calls": 0},
        "judge": {"model": None, "calls": 0},
        "hosted_api": {"calls": 0, "required_api_key": None},
        "template": {
            "name": "shared_reader_v2_question_date",
            "file": V2_TEMPLATE_PATH.relative_to(ROOT).as_posix(),
            "file_sha256": template_file_sha,
            "message_template_sha256": template_message_sha,
            "v1_template_sha256": run_mem1.D1_READER_TEMPLATE_SHA256,
            "question_date_source": "official LongMemEval record question_date; copied exactly",
            "task_specific_hints": False,
            "question_type_in_prompt": False,
        },
        "context_bundles": {
            "source_run_id": d1_manifest["run_id"],
            "source_jsonl_sha256": source_hashes_before["context_bundles.jsonl"],
            "verified_count": len(bundles_by_key),
            "copied_or_rebuilt": False,
        },
        "fullcontext_prompt_preflight": preflight_rows,
        "expected_reader_calls": 50,
        "tokenizer_preflight_only": True,
        "configuration_sha256": None,
    }
    config["configuration_sha256"] = _sha256_bytes(_canonical_json({key: value for key, value in config.items() if key != "configuration_sha256"}))

    D3_RUN.mkdir(parents=True, exist_ok=False)
    config_path = D3_RUN / "run_config.json"
    _write_json(config_path, config)
    config_sha = _sha256_file(config_path)
    config_path.with_name("run_config.sha256").write_text(
        f"{config_sha}  {config_path.name}\n", encoding="ascii", newline="\n"
    )
    predictions_out = D3_RUN / "predictions.jsonl"
    call_ledger_out = D3_RUN / "call_ledger.jsonl"
    new_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    reader_successes = 0

    with httpx.Client(timeout=3600, trust_env=False) as client:
        for call_index, row in enumerate(prepared, 1):
            call_started = time.perf_counter()
            timestamp = datetime.now(UTC).isoformat()
            request = {
                "model": MODEL_ALIAS,
                "messages": row["messages"],
                "temperature": 0,
                "seed": 42,
                "max_tokens": ANSWER_MAX_NEW_TOKENS,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request_sha = _sha256_bytes(_canonical_json(request))
            answer: str | None = None
            error_type = None
            status_code = None
            usage: dict[str, Any] = {}
            finish_reason = None
            try:
                response = client.post(f"{LOCAL_BASE_URL}/chat/completions", json=request)
                status_code = response.status_code
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                answer_value = choice.get("message", {}).get("content")
                if not isinstance(answer_value, str):
                    raise TypeError("Reader response did not contain text content")
                answer = answer_value.strip()
                usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
                finish_reason = choice.get("finish_reason")
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
                error_type = type(error).__name__
            latency_ms = round((time.perf_counter() - call_started) * 1000, 3)
            success = answer is not None
            if success:
                reader_successes += 1
            call_row = {
                "call_index": call_index,
                "timestamp_utc": timestamp,
                "role": "reader_answer",
                "provider": "local_qwen",
                "endpoint": LOCAL_BASE_URL,
                "model": MODEL_ALIAS,
                "system": row["system"],
                "question_id": row["question_id"],
                "context_bundle_sha256": row["context_bundle_sha256"],
                "prompt_sha256": row["prompt_sha256"],
                "rendered_prompt_sha256": row["rendered_prompt_sha256"],
                "request_sha256": request_sha,
                "temperature": 0,
                "seed": 42,
                "enable_thinking": False,
                "max_new_tokens": ANSWER_MAX_NEW_TOKENS,
                "prompt_tokens_preflight": row["reader_prompt_tokens"],
                "prompt_tokens_server": usage.get("prompt_tokens"),
                "completion_tokens_server": usage.get("completion_tokens"),
                "finish_reason": finish_reason,
                "latency_ms": latency_ms,
                "http_status": status_code,
                "success": success,
                "error_type": error_type,
                "retry_count": 0,
                "hosted_call": False,
            }
            _append_jsonl(call_ledger_out, call_row)

            old = old_by_key[(row["system"], row["question_id"])]
            metrics = _answer_metrics(run_mem1, answer, row["gold_answer"]) if answer is not None else {
                "token_precision": None,
                "token_recall": None,
                "f1": None,
                "normalized_exact_match": None,
            }
            prediction = {
                "system": row["system"],
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "question_date": row["question_date"],
                "question": row["question"],
                "ground_truth": row["gold_answer"],
                "predicted": answer,
                "quality_status": "OK" if success else "INFRA_FAILURE",
                **metrics,
                "reader_prompt_tokens_preflight": row["reader_prompt_tokens"],
                "reader_prompt_tokens_server": usage.get("prompt_tokens"),
                "completion_tokens_server": usage.get("completion_tokens"),
                "reader_latency_ms": latency_ms,
                "finish_reason": finish_reason,
                "output_hit_token_cap": finish_reason == "length",
                "input_truncated": False,
                "context_bundle_sha256": row["context_bundle_sha256"],
                "source_record_sha256": row["source_record_sha256"],
                "shared_reader_template_sha256": template_message_sha,
                "shared_reader_prompt_sha256": row["prompt_sha256"],
                "rendered_prompt_sha256": row["rendered_prompt_sha256"],
                "old_prediction_sha256": _sha256_bytes(_canonical_json({
                    "prediction": old.get("predicted"),
                    "context_bundle_sha256": old.get("context_bundle_sha256"),
                    "prompt_sha256": old.get("shared_reader_prompt_sha256"),
                })),
                "reader_error_type": error_type,
                "call_index": call_index,
            }
            _append_jsonl(predictions_out, prediction)
            new_by_key[(row["system"], row["question_id"])] = prediction

    prediction_sha = _freeze_hash_sidecar(predictions_out)
    call_sha = _freeze_hash_sidecar(call_ledger_out)
    if set(new_by_key) != expected_keys:
        raise RuntimeError("Reader-v2 result set is not the exact frozen 50 system×question pairs")
    _write_json(QUESTION_DATES_PATH, question_dates)
    question_dates_sha = _freeze_file_sidecar(QUESTION_DATES_PATH)
    counterfactual = _build_counterfactual(old_by_key, new_by_key, question_dates)
    _write_json(COUNTERFACTUAL_PATH, counterfactual)
    counterfactual_sha = _freeze_file_sidecar(COUNTERFACTUAL_PATH)
    source_hashes_after = _source_evidence_hashes()
    source_unchanged = source_hashes_before == source_hashes_after
    template_unchanged = _sha256_file(V2_TEMPLATE_PATH) == template_file_sha
    fullcontext_actual = [
        row for row in new_by_key.values() if row["system"] == "fullcontext"
    ]
    fullcontext_fit_after = all(
        row["reader_prompt_tokens_preflight"] + ANSWER_MAX_NEW_TOKENS <= MAX_MODEL_LENGTH
        and row["input_truncated"] is False
        for row in fullcontext_actual
    ) and len(fullcontext_actual) == 10
    ledger_rows = _load_jsonl(call_ledger_out)
    all_reader_only = len(ledger_rows) == 50 and all(
        row["role"] == "reader_answer" and row["provider"] == "local_qwen" and row["hosted_call"] is False
        for row in ledger_rows
    )
    all_success = reader_successes == 50 and all(row["quality_status"] == "OK" for row in new_by_key.values())
    gate_passed = (
        len(question_dates["records"]) == 10
        and len(bundles_by_key) == 50
        and all(row["same_context_bundle_sha256"] for row in counterfactual["rows"])
        and all_success
        and all_reader_only
        and fullcontext_fit_after
        and source_unchanged
        and template_unchanged
        and _verify_hash_sidecar(QUESTION_DATES_PATH, QUESTION_DATES_PATH.with_name(f"{QUESTION_DATES_PATH.name}.sha256"))
        and _verify_hash_sidecar(COUNTERFACTUAL_PATH, COUNTERFACTUAL_PATH.with_name(f"{COUNTERFACTUAL_PATH.name}.sha256"))
        and _verify_hash_sidecar(predictions_out, D3_RUN / "predictions.sha256")
        and _verify_hash_sidecar(call_ledger_out, D3_RUN / "call_ledger.sha256")
        and _verify_hash_sidecar(config_path, D3_RUN / "run_config.sha256")
    )
    run_manifest = {
        "manifest_version": "mem1d3-reader-only-v1",
        "run_id": D3_RUN.name,
        "status": "COMPLETE" if gate_passed else "INCOMPLETE_OR_INVALID",
        "split": "DEV",
        "test_access": False,
        "question_ids": list(QUESTION_IDS),
        "system_count": len(SYSTEMS),
        "question_count": len(QUESTION_IDS),
        "reader_calls_expected": 50,
        "reader_calls_attempted": len(ledger_rows),
        "reader_calls_successful": reader_successes,
        "call_ledger_rows": len(ledger_rows),
        "all_calls_local_reader_only": all_reader_only,
        "embedding_calls": 0,
        "memory_ingestion_or_retrieval_calls": 0,
        "judge_calls": 0,
        "hosted_api_calls": 0,
        "context_bundles_verified": 50,
        "context_bundles_copied_or_rebuilt": False,
        "fullcontext_fit_all": fullcontext_fit_after,
        "fullcontext_prompt_preflight": preflight_rows,
        "source_evidence_hashes_before": source_hashes_before,
        "source_evidence_hashes_after": source_hashes_after,
        "source_evidence_unchanged": source_unchanged,
        "configuration_sha256": config["configuration_sha256"],
        "configuration_file_sha256": config_sha,
        "question_dates_sha256": question_dates_sha,
        "counterfactual_sha256": counterfactual_sha,
        "predictions_sha256": prediction_sha,
        "call_ledger_sha256": call_sha,
        "counterfactual_pairs": len(counterfactual["rows"]),
        "semantic_review_status": "PENDING_HUMAN_REVIEW",
        "gate": {
            "all_question_dates_preserved": len(question_dates["records"]) == 10,
            "frozen_bundles_reused_and_verified": len(bundles_by_key) == 50,
            "exactly_50_reader_only_calls_complete": all_success and len(ledger_rows) == 50 and all_reader_only,
            "no_memory_or_embedding_calls": True,
            "no_test_access": True,
            "fullcontext_non_truncated": fullcontext_fit_after,
            "d1_and_d2_evidence_unchanged": source_unchanged,
            "v2_template_and_config_frozen": template_unchanged and _verify_hash_sidecar(config_path, D3_RUN / "run_config.sha256"),
            "paired_review_artifact_has_50_rows": len(counterfactual["rows"]) == 50 and _verify_hash_sidecar(COUNTERFACTUAL_PATH, COUNTERFACTUAL_PATH.with_name(f"{COUNTERFACTUAL_PATH.name}.sha256")),
            "passed": gate_passed,
        },
    }
    _write_json(D3_RUN / "run_manifest.json", run_manifest)
    report = _report_text(
        question_dates, config, run_manifest, counterfactual, preflight_rows,
        source_hashes_before, source_hashes_after,
    )
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
    return {
        "gate": "MEM1D3_QUESTION_DATE_CONTRACT_REPAIRED=YES" if gate_passed else "MEM1D3_QUESTION_DATE_CONTRACT_REPAIRED=NO",
        "question_dates": len(question_dates["records"]),
        "bundles_verified": len(bundles_by_key),
        "reader_calls": len(ledger_rows),
        "successful_reader_calls": reader_successes,
        "fullcontext_fit_all": fullcontext_fit_after,
        "source_evidence_unchanged": source_unchanged,
        "configuration_sha256": config["configuration_sha256"],
        "predictions_sha256": prediction_sha,
        "call_ledger_sha256": call_sha,
        "temporal_case_summary": counterfactual["temporal_cases"],
        "non_temporal_case_summary": counterfactual["non_temporal_cases"],
    }


def main() -> int:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gate"] == "MEM1D3_QUESTION_DATE_CONTRACT_REPAIRED=YES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
