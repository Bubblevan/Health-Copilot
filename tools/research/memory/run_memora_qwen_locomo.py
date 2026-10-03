"""Run Microsoft's Memora architecture on LoCoMo with the frozen local Qwen stack.

This is a research adapter around the pinned upstream Memora implementation.
It does not modify Memora source files or train a retrieval policy.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import time
import types
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT.parent
EXTERNAL_MEMORY = WORKSPACE / "external" / "memory"
MEMORA_ROOT = EXTERNAL_MEMORY / "Memora"
LOCOMO_ROOT = EXTERNAL_MEMORY / "LoCoMo"
MEMEVAL_ROOT = EXTERNAL_MEMORY / "MemEval"
DATASET_PATH = LOCOMO_ROOT / "data" / "locomo10.json"
MEMENTAL_PATCH = ROOT / "tools" / "research" / "memory" / "patches" / "memeval_qwen_main_v1.patch"
RUN_ROOT = ROOT / "runs" / "memory" / "memora-locomo-qwen-v3"
REPORT_PATH = ROOT / "docs" / "research" / "memory" / "memora_qwen_locomo_results.md"

EXPECTED_MEMORA_COMMIT = "dec3f8f2444eace7004fc084abe1be9f3d88270e"
EXPECTED_LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
EXPECTED_DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
EXPECTED_MEMEVAL_PATCH_SHA256 = "c7e44015527a17d43cf9bbd2e2a6e50d007943ee2efaa244899d3b048fd80eab"
READER_PATH = Path(r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf")
READER_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
LLAMA_SHA256 = "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"
EMBEDDING_PATH = Path(r"E:\Health-Copilot-Models\models\Qwen3-Embedding-0.6B")
EMBEDDING_WEIGHTS_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
ENDPOINT = "http://127.0.0.1:8081/v1"
READER_MODEL = "health-memory-qwen3-8b"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ANSWER_BUDGET = 256
INTERNAL_BUDGET = 8192
BOOTSTRAP_SEED = 42
BOOTSTRAP_SAMPLES = 10_000
SCHEMA_VERSION = 1
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _run_git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(path), *args], text=True, capture_output=True, check=check)


def _verify_repositories() -> dict[str, str]:
    if _git(MEMORA_ROOT, "rev-parse", "HEAD") != EXPECTED_MEMORA_COMMIT:
        raise RuntimeError("Memora checkout is not the audited Microsoft upstream revision")
    remote = _git(MEMORA_ROOT, "remote", "get-url", "origin").lower()
    if not remote.endswith("microsoft/memora.git"):
        raise RuntimeError(f"Memora origin is not microsoft/Memora: {remote}")
    if _run_git(MEMORA_ROOT, "diff", "--quiet", check=False).returncode != 0:
        raise RuntimeError("Tracked upstream Memora files have local modifications")
    if _run_git(MEMORA_ROOT, "diff", "--cached", "--quiet", check=False).returncode != 0:
        raise RuntimeError("Memora has staged source changes; refusing to run against a dirty upstream")
    if _git(LOCOMO_ROOT, "rev-parse", "HEAD") != EXPECTED_LOCOMO_COMMIT:
        raise RuntimeError("LoCoMo checkout differs from the frozen public dataset revision")
    if _sha256_file(DATASET_PATH) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("LoCoMo dataset hash differs from the frozen public dataset")
    return {
        "memora_commit": EXPECTED_MEMORA_COMMIT,
        "memora_origin": remote,
        "locomo_commit": EXPECTED_LOCOMO_COMMIT,
        "locomo_dataset_sha256": EXPECTED_DATASET_SHA256,
    }


def _ensure_memeval_patch() -> bool:
    if _sha256_file(MEMENTAL_PATCH) != EXPECTED_MEMEVAL_PATCH_SHA256:
        raise RuntimeError("Pinned local provider adapter patch SHA mismatch")
    reverse = _run_git(MEMEVAL_ROOT, "apply", "--reverse", "--check", str(MEMENTAL_PATCH), check=False)
    if reverse.returncode == 0:
        return False
    forward = _run_git(MEMEVAL_ROOT, "apply", "--check", str(MEMENTAL_PATCH), check=False)
    if forward.returncode != 0:
        raise RuntimeError("MemEval source is neither clean upstream nor the exact pinned adapter")
    _run_git(MEMEVAL_ROOT, "apply", str(MEMENTAL_PATCH))
    return True


def _remove_owned_memeval_patch(owned: bool) -> None:
    if owned:
        _run_git(MEMEVAL_ROOT, "apply", "--reverse", str(MEMENTAL_PATCH))


def _configure_local_environment() -> None:
    for name in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_AD_TOKEN",
        "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
    ):
        os.environ.pop(name, None)
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "ANONYMIZED_TELEMETRY": "false",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "HC_MEMORY_TRACK": "main_local_only",
        "HC_LOCAL_READER_BASE_URL": ENDPOINT,
        "HC_READER_MODEL": READER_MODEL,
        "HC_LOCAL_EMBEDDING_PATH": str(EMBEDDING_PATH),
        "HC_LOCAL_EMBEDDING_DEVICE": "cuda:0",
        "HC_LOCAL_EMBEDDING_DTYPE": "float16",
        "HC_READER_ANSWER_MAX_NEW_TOKENS": str(ANSWER_BUDGET),
    })


def _install_outbound_network_guard() -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def check(address: Any) -> None:
        host = address[0] if isinstance(address, tuple) and address else address
        try:
            allowed = ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
        except ValueError:
            allowed = str(host).lower() == "localhost"
        if not allowed:
            raise PermissionError(f"Local-only Memora run blocked outbound socket to {host}")

    def guarded_connect(self: socket.socket, address: Any):
        check(address)
        return original_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any):
        try:
            check(address)
        except PermissionError:
            return 10013
        return original_connect_ex(self, address)

    socket.socket.connect = guarded_connect  # type: ignore[assignment]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[assignment]


def _reader_process() -> dict[str, Any]:
    ps = (
        "$p=@(Get-CimInstance Win32_Process -Filter \"name='llama-server.exe'\" | "
        "Select-Object ProcessId,ExecutablePath,CommandLine); $p | ConvertTo-Json -Compress"
    )
    output = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command", ps], text=True).strip()
    processes = json.loads(output) if output else []
    if isinstance(processes, dict):
        processes = [processes]
    matches = [
        row for row in processes
        if "--port 8081" in str(row.get("CommandLine", ""))
        and "127.0.0.1" in str(row.get("CommandLine", ""))
        and str(READER_PATH).lower() in str(row.get("CommandLine", "")).lower()
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one frozen reader on loopback port 8081; this runner never touches 8092")
    process = matches[0]
    executable = Path(process["ExecutablePath"])
    binary_sha = _sha256_file(executable)
    command = str(process["CommandLine"])
    required = (
        "--ctx-size 131072", "--n-gpu-layers 99", "--flash-attn on",
        "--cache-type-k q4_0", "--cache-type-v q4_0", "--parallel 1",
        "--rope-scaling yarn", "--rope-scale 4",
    )
    missing = [flag for flag in required if flag.lower() not in command.lower()]
    if binary_sha != LLAMA_SHA256 or _sha256_file(READER_PATH) != READER_SHA256 or missing:
        raise RuntimeError(f"Frozen reader mismatch: binary_sha={binary_sha}; missing_flags={missing}")
    return {
        "pid": int(process["ProcessId"]),
        "executable": str(executable),
        "binary_sha256": binary_sha,
        "model": str(READER_PATH),
        "model_sha256": READER_SHA256,
        "build": "llama.cpp 10068 / 571d0d540",
        "command_line": command,
    }


def _load_dataset() -> list[dict[str, Any]]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if len(data) != 10 or sum(len(row.get("qa", [])) for row in data) != 1986:
        raise RuntimeError("Frozen LoCoMo dataset must contain ten conversations and 1,986 QA items")
    return data


def _question_rows(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": f"{item['sample_id']}:qa:{index}",
            "sample_id": item["sample_id"],
            "qa_index": index,
            "question": str(qa.get("question", "")),
        }
        for item in dataset
        for index, qa in enumerate(item["qa"])
    ]


def _prepare_manifests(identity: dict[str, Any], dataset: list[dict[str, Any]], run_root: Path) -> str:
    rows = _question_rows(dataset)
    encoded = b"".join(_canonical(row) + b"\n" for row in rows)
    questions_path = run_root / "frozen_questions.jsonl"
    questions_hash_path = run_root / "frozen_questions.sha256"
    run_root.mkdir(parents=True, exist_ok=True)
    if questions_path.exists() and questions_path.read_bytes() != encoded:
        raise FileExistsError("Existing frozen question manifest differs from the pinned public dataset")
    if not questions_path.exists():
        questions_path.write_bytes(encoded)
    question_sha = _sha256_file(questions_path)
    if questions_hash_path.exists() and questions_hash_path.read_text(encoding="ascii").split()[0] != question_sha:
        raise RuntimeError("Frozen question manifest sidecar mismatch")
    if not questions_hash_path.exists():
        questions_hash_path.write_text(f"{question_sha}  {questions_path.name}\n", encoding="ascii")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "track": "controlled architecture transfer to a unified fully-local model stack",
        "identity": identity,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "question_count": len(rows),
        "question_manifest_sha256": question_sha,
        "reader": {
            "model": READER_MODEL,
            "artifact": str(READER_PATH),
            "sha256": READER_SHA256,
            "answer_max_new_tokens": ANSWER_BUDGET,
        },
        "memory_internal_llm": {"model": READER_MODEL, "max_new_tokens": INTERNAL_BUDGET},
        "embedding": {
            "model": EMBEDDING_MODEL,
            "revision": EMBEDDING_REVISION,
            "weights_sha256": EMBEDDING_WEIGHTS_SHA256,
            "device": "cuda:0",
            "dtype": "float16",
            "dimensions": 1024,
            "normalized": True,
            "query_instruction": (
                "Given a conversation-history question, retrieve memory passages that "
                "provide evidence needed to answer the question."
            ),
            "document_instruction": "",
            "max_length": 8192,
            "truncation": True,
        },
        "judge": {
            "model": READER_MODEL,
            "provider": "loopback llama.cpp only",
            "prompt": "official Memora LoCoMo ACCURACY_PROMPT",
            "excluded_category": 5,
        },
        "hosted_api": "NONE",
        "api_key_required": "NONE",
        "retrieval_strategies": ["semantic", "prompt"],
        "method_config": {
            "top_k": 30,
            "enable_cue_index": True,
            "enable_hybrid_search": True,
            "hybrid_search_method": "bm25",
            "hybrid_top_k": 10,
            "enable_segmentation": True,
            "enable_episodic_memory": True,
            "use_segments_as_episodic": True,
            "combined_user": True,
            "prompted_policy_max_steps": 4,
            "workers": 1,
            "grpo": "not trained or evaluated",
        },
        "scoring": {
            "official_memora_token_f1": "set-overlap F1 after official punctuation splitting",
            "official_memora_exact_match": "case-insensitive exact string match",
            "local_qwen_judge": "official Memora ACCURACY_PROMPT; excludes category 5",
            "normalized_em": "lowercase, remove punctuation/articles, normalize whitespace",
            "paired_bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "stratified_by": "LoCoMo category"},
        },
        "warning": "Not an exact reproduction of the paper's proprietary GPT reader/judge stack or published numerical results.",
    }
    manifest_path = run_root / "protocol_manifest.json"
    manifest["protocol_sha256"] = _sha256_bytes(_canonical(manifest))
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise FileExistsError("Existing Memora LoCoMo protocol manifest differs; use a new RUN_ROOT for changes")
    else:
        _write_json(manifest_path, manifest)
    return question_sha


def _install_utils_shim(dataset_path: Path) -> None:
    """Provide only the public LoCoMo loader helpers, preventing upstream URL fetches."""
    module = types.ModuleType("utils")

    def load_data(path: str, subset_idx: int = -1):
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return rows[subset_idx - 1:subset_idx] if subset_idx > 0 else rows

    def generate_debug_data(rows, conversion_idx=0, session_idx=1, num_sessions=3):
        subset = json.loads(json.dumps(rows[: conversion_idx + 1]))
        subset[0]["qa"] = subset[0].get("qa", [])[:10]
        conversation = subset[0]["conversation"]
        selected = {"speaker_a": conversation["speaker_a"], "speaker_b": conversation["speaker_b"]}
        first = int(session_idx) if isinstance(session_idx, int) else int(str(session_idx).split(",")[0])
        for number in range(first, first + num_sessions):
            key = f"session_{number}"
            if key in conversation:
                selected[key] = conversation[key]
                selected[f"{key}_date_time"] = conversation.get(f"{key}_date_time", "")
        subset[0]["conversation"] = selected
        return subset

    def get_session_num(conversation):
        return sum(f"session_{index}" in conversation for index in range(1, 100))

    module.load_data = load_data
    module.generate_debug_data = generate_debug_data
    module.get_session_num = get_session_num
    module.is_valid_image_url = lambda _url, timeout=5: False
    sys.modules["utils"] = module


def _install_local_adapters(provider: Any, config: Any) -> dict[str, Any]:
    sys.path.insert(0, str(MEMORA_ROOT / "src"))
    sys.path.insert(0, str(MEMORA_ROOT / "app" / "locomo"))
    _install_utils_shim(DATASET_PATH)

    from memora.utils import embedding as embedding_module
    from memora.utils import llm as llm_module
    from memora.utils.llm import ChatCompletionModel
    from memora.memora_client import MemoraClient
    from providers.memora.add import MemoraADD
    from providers.memora.search import MemoraSearch

    from agents_memory.healthcopilot_embedding import get_local_embedding_runtime
    from agents_memory.healthcopilot_provider import call_context, current_call_context, reader_client, record_call

    def client_factory(cfg):
        context = current_call_context()
        role = context.role if context.role in {"memory_ingest", "memory_reasoning", "reader_answer", "judge_local"} else "memory_ingest"
        return reader_client(role, "Memora-LoCoMo", context.question_id or "adapter-init", config)

    def forbidden_hosted_client(*_args, **_kwargs):
        raise RuntimeError("Hosted model providers are disabled in Memora local-only reproduction")

    def model_init(self, cfg, token_usage_callback=None):
        self.cfg = cfg
        self.token_usage_callback = token_usage_callback
        self.model_type = "azure"
        self.client = client_factory(cfg)
        self.hf_model = None
        self.hf_tokenizer = None

    def local_invoke(self, messages, response_format, source, **kwargs):
        kwargs.setdefault("max_tokens", INTERNAL_BUDGET)
        kwargs.setdefault("temperature", 0.0)
        kwargs.setdefault("seed", 42)
        extra_body = kwargs.get("extra_body") if isinstance(kwargs.get("extra_body"), dict) else {}
        template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
        template_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = template_kwargs
        kwargs["extra_body"] = extra_body
        if response_format:
            response = self.client.beta.chat.completions.parse(
                messages=messages,
                model=READER_MODEL,
                response_format=response_format,
                **kwargs,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                text = response.choices[0].message.content or ""
                return response_format.model_validate_json(text)
            return parsed
        response = self.client.chat.completions.create(
            messages=messages,
            model=READER_MODEL,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def local_embeddings(self, inputs):
        context = current_call_context()
        role = context.role or "memory_reasoning"
        input_type = "document" if role == "memory_ingest" else "query"
        runtime = get_local_embedding_runtime(str(EMBEDDING_PATH), device="cuda:0", dtype="float16")
        texts = [str(item) for item in inputs]
        vectors, token_count, truncated, latency = runtime.encode(texts, input_type)
        with call_context(context.system, context.question_id, "embedding"):
            record_call(
                role="embedding",
                provider="local_transformers_cuda",
                model=EMBEDDING_MODEL,
                prompt_tokens=token_count,
                completion_tokens=None,
                latency_ms=latency,
                success=True,
                truncated=truncated,
            )
        return vectors

    ChatCompletionModel.__init__ = model_init
    ChatCompletionModel._determine_model_type = lambda self, _model_name: "azure"
    ChatCompletionModel._invoke_azure = local_invoke
    llm_module.get_aoai_chat_completion_client = client_factory
    llm_module.get_openai_chat_completion_client = forbidden_hosted_client
    embedding_module.get_embedding_client = lambda _cfg: object()
    embedding_module.BaseEmbeddingModel.generate_embeddings = local_embeddings

    # The runner instantiates the searcher directly, so replace its imported factory too.
    import providers.memora.search as search_module
    search_module.get_aoai_chat_completion_client = client_factory

    def local_reader_token_count(memories, model=READER_MODEL):
        import httpx
        text = "\n".join(str(memory) for memory in memories)
        url = ENDPOINT.removesuffix("/v1") + "/tokenize"
        with httpx.Client(timeout=120, trust_env=False) as client:
            response = client.post(url, json={"content": text, "add_special": False})
            response.raise_for_status()
            tokens = response.json().get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError("Loopback llama.cpp tokenizer did not return token IDs")
        return {
            "total_tokens": len(tokens),
            "num_memories": len(memories),
            "avg_tokens": round(len(tokens) / len(memories), 2) if memories else 0,
        }
    search_module.count_memories_tokens = local_reader_token_count

    # URL validation in the upstream add path is deliberately disabled for offline LoCoMo.
    import providers.memora.add as add_module
    add_module.is_valid_image_url = lambda _url, timeout=5: False

    # Expose official judge source without importing its unrelated BERTScore/NLTK stack.
    metrics_utils = types.ModuleType("metrics.utils")
    def extract_json(text: str) -> str:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        return match.group(1) if match else text.strip()
    metrics_utils.extract_json = extract_json
    sys.modules["metrics.utils"] = metrics_utils
    from metrics.llm_judge import ACCURACY_PROMPT

    # Strictly fail if a configuration change routes internal client construction to hosted APIs.
    if MemoraClient is None:  # pragma: no cover - import assertion for adapter audit
        raise RuntimeError("Memora public client import failed")
    return {
        "MemoraADD": MemoraADD,
        "MemoraSearch": MemoraSearch,
        "ChatCompletionModel": ChatCompletionModel,
        "call_context": call_context,
        "current_call_context": current_call_context,
        "reader_client": reader_client,
        "ACCURACY_PROMPT": ACCURACY_PROMPT,
        "extract_json": extract_json,
        "count_reader_tokens": local_reader_token_count,
    }


def _config(run_root: Path, strategy: str):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(MEMORA_ROOT / "app" / "locomo" / "conf" / "config.yaml"))
    cfg.general.project_path = str(MEMORA_ROOT / "app" / "locomo")
    cfg.general.data_path = str(LOCOMO_ROOT / "data")
    cfg.general.output_path = str(run_root / "outputs")
    cfg.general.results_path = str(run_root / "results")
    cfg.general.memory_store_path = str(run_root / "memory_store")
    cfg.general.debug = False
    cfg.openai.api_type = "azure"  # all factories are replaced by fail-closed local adapters
    cfg.openai.api_key = "local-only-no-hosted-provider"
    cfg.openai.llm_api_base = ENDPOINT
    cfg.openai.embedding_api_base = ENDPOINT
    cfg.openai.model = READER_MODEL
    cfg.openai.embedding_model = EMBEDDING_MODEL
    cfg.llm.model = READER_MODEL
    cfg.llm.answer_model = READER_MODEL
    cfg.memory.type = "memora"
    cfg.memory.persist_path = str(run_root / "memory_store")
    cfg.memory.force_rebuild = False
    cfg.memory.top_k = 30
    cfg.memory.enable_cue_index = True
    cfg.memory.enable_hybrid_search = True
    cfg.memory.enable_segmentation = True
    cfg.memory.enable_episodic_memory = True
    cfg.memory.use_segments_as_episodic = True
    cfg.memory.multimodal_support = False
    cfg.eval.max_workers = 1
    cfg.eval.use_combined_user = True
    cfg.eval.prompt_template = "mem0"
    cfg.eval.subset_idx = -1
    cfg.retrieval.strategy = strategy
    cfg.retrieval.prompted_policy.max_steps = 4
    cfg.retrieval.enable_llm_filter = False
    return cfg


def _existing_rows(path: Path, expected_ids: set[str], run_identity: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = row.get("question_id")
        if question_id not in expected_ids:
            raise RuntimeError(f"Unexpected question ID in resume artifact: {question_id}")
        if question_id in values:
            raise RuntimeError(f"Duplicate question ID in resume artifact: {question_id}")
        if row.get("run_identity") != run_identity:
            raise RuntimeError(f"Question row identity mismatch for {question_id}")
        values[question_id] = row
    return values


def _official_metrics(prediction: str, reference: str) -> dict[str, float | int]:
    prediction = str(prediction or "").strip()
    reference = str(reference or "").strip()
    if not prediction or not reference:
        return {
            "official_f1": 0.0,
            "official_exact_match": 0,
            "normalized_em": 0,
        }
    official_normalize = lambda text: text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()
    pred_tokens = set(official_normalize(prediction))
    ref_tokens = set(official_normalize(reference))
    overlap = len(pred_tokens & ref_tokens)
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def norm_em(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    return {
        "official_f1": f1,
        "official_exact_match": int(prediction.lower() == reference.lower()),
        "normalized_em": int(norm_em(prediction) == norm_em(reference)),
    }


def _question_call_health(path: Path, question_id: str, strategy: str) -> dict[str, Any]:
    calls = [
        row for row in _read_jsonl(path)
        if row.get("question_id") == question_id and row.get("system") == f"Memora-{strategy}"
    ]
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calls:
        by_role[str(row.get("role", "unknown"))].append(row)
    required = ["reader_answer", "embedding"]
    if strategy == "prompt":
        required.append("memory_reasoning")
    missing = [role for role in required if not any(row.get("success") is True for row in by_role.get(role, []))]
    return {
        "calls": len(calls),
        "successes_by_role": {role: sum(row.get("success") is True for row in rows) for role, rows in by_role.items()},
        "failures_by_role": {role: sum(row.get("success") is False for row in rows) for role, rows in by_role.items()},
        "missing_success_roles": missing,
    }


def _judge(searcher: Any, helpers: dict[str, Any], qa: dict[str, Any], response: str, question_id: str, strategy: str) -> dict[str, Any]:
    prompt = helpers["ACCURACY_PROMPT"].format(
        question=qa["question"], gold_answer=qa["answer"], generated_answer=response
    )
    with helpers["call_context"](f"Memora-{strategy}", question_id, "judge_local"):
        result = searcher.llm_client.chat.completions.create(
            model=READER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            seed=42,
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    payload = json.loads(helpers["extract_json"](result.choices[0].message.content or "{}"))
    label = str(payload.get("label", "")).upper()
    if label not in {"CORRECT", "WRONG"}:
        raise RuntimeError(f"Local Qwen judge returned an invalid label: {label!r}")
    return {"label": label, "correct": int(label == "CORRECT"), "explanation": payload.get("reason", "")}


def _run_ingestion(dataset: list[dict[str, Any]], run_root: Path, identity: str, cfg: Any, helpers: dict[str, Any]) -> dict[str, Any]:
    state_path = run_root / "ingestion_state.json"
    checkpoint_root = run_root / "ingestion" / "checkpoints"
    completed: list[int] = []
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_identity") != identity:
            raise RuntimeError("Ingestion resume state belongs to a different frozen run")
        completed = [int(value) for value in state.get("completed_conversation_indices", [])]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    prior_build_log = []
    for idx in completed:
        checkpoint_path = checkpoint_root / f"{idx:02d}.json"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Completed conversation is missing its durable checkpoint: {idx}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("run_identity") != identity:
            raise RuntimeError(f"Conversation checkpoint identity mismatch: {idx}")
        prior_build_log.extend(checkpoint.get("build_log", []))
    manager = helpers["MemoraADD"](cfg, data_path=str(DATASET_PATH))
    manager.build_log.extend(prior_build_log)
    segmenter_method = manager.segmenter._segment_with_llm
    def tracked_segmenter(messages):
        result = segmenter_method(messages)
        if result is None:
            manager.segmenter._hc_fallback_count = getattr(manager.segmenter, "_hc_fallback_count", 0) + 1
        return result
    manager.segmenter._segment_with_llm = tracked_segmenter
    for idx, item in enumerate(dataset):
        if idx in completed:
            continue
        sample_id = str(item["sample_id"])
        started = time.perf_counter()
        call_id = f"ingest:{sample_id}"
        fallback_before = getattr(manager.segmenter, "_hc_fallback_count", 0)
        build_log_before = len(manager.build_log)
        with helpers["call_context"]("Memora", call_id, "memory_ingest"):
            manager.process_conversation(item, idx)
        elapsed_ms = (time.perf_counter() - started) * 1000
        client_id = f"{item['conversation']['speaker_a']}_{item['conversation']['speaker_b']}_{idx}"
        client = manager.get_memory_client(client_id)
        current = sorted(set(completed + [idx]))
        build_log_delta = manager.build_log[build_log_before:]
        _write_json(checkpoint_root / f"{idx:02d}.json", {
            "run_identity": identity,
            "sample_id": sample_id,
            "conversation_index": idx,
            "wall_time_ms": round(elapsed_ms, 3),
            "memory_count": client.count(),
            "segments": len(build_log_delta),
            "segmenter_fallback_count": getattr(manager.segmenter, "_hc_fallback_count", 0) - fallback_before,
            "build_log": build_log_delta,
        })
        _write_json(state_path, {
            "run_identity": identity,
            "completed_conversation_indices": current,
            "completed_sample_ids": [str(dataset[number]["sample_id"]) for number in current],
            "last_completed_at_utc": datetime.now(UTC).isoformat(),
        })
        completed = current
        print(f"[ingest {len(completed)}/{len(dataset)}] {sample_id}: {elapsed_ms / 1000:.1f}s")
    if sorted(completed) != list(range(len(dataset))):
        raise RuntimeError("Not every LoCoMo conversation has a completed memory build")
    checkpoint_rows = [
        json.loads((checkpoint_root / f"{idx:02d}.json").read_text(encoding="utf-8"))
        for idx in range(len(dataset))
    ]
    conversations_path = run_root / "ingestion" / "conversations.jsonl"
    conversations_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = conversations_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps({key: value for key, value in row.items() if key != "build_log"}, separators=(",", ":")) + "\n" for row in checkpoint_rows), encoding="utf-8")
    os.replace(temporary, conversations_path)
    _write_json(run_root / "ingestion" / "build_log.json", manager.build_log)
    return {"completed_conversations": len(completed), "build_log_entries": len(manager.build_log)}


def _run_strategy(dataset: list[dict[str, Any]], questions: list[dict[str, Any]], run_root: Path, identity: str, cfg: Any, strategy: str, helpers: dict[str, Any]) -> dict[str, Any]:
    predictions_path = run_root / "predictions" / f"{strategy}.jsonl"
    judgments_path = run_root / "judgments" / f"{strategy}.jsonl"
    failures_path = run_root / "failures" / f"{strategy}.jsonl"
    expected = {row["question_id"] for row in questions}
    predictions = _existing_rows(predictions_path, expected, identity)
    judgments = _existing_rows(judgments_path, expected, identity)
    searcher = helpers["MemoraSearch"](
        cfg,
        output_path=str(run_root / "outputs" / f"official-{strategy}-output.json"),
        top_k=30,
        retrieval_strategy=strategy,
    )
    original_search_memory = searcher.search_memory
    active_question_id = ""

    def search_with_memory_role(user_id, query, *args, **kwargs):
        with helpers["call_context"](f"Memora-{strategy}", active_question_id, "memory_reasoning"):
            return original_search_memory(user_id, query, *args, **kwargs)

    searcher.search_memory = search_with_memory_role
    items_by_sample = {str(item["sample_id"]): item for item in dataset}
    indices_by_sample = {str(item["sample_id"]): index for index, item in enumerate(dataset)}
    totals = {"generated": len(predictions), "judged": len(judgments), "infra_failures": 0}
    for position, row in enumerate(questions, 1):
        qid = row["question_id"]
        sample_id = row["sample_id"]
        item = items_by_sample[sample_id]
        qa = item["qa"][row["qa_index"]]
        category = int(qa.get("category", -1))
        if qid not in predictions:
            active_question_id = qid
            speaker_a = item["conversation"]["speaker_a"]
            speaker_b = item["conversation"]["speaker_b"]
            user_id = f"{speaker_a}_{speaker_b}_{indices_by_sample[sample_id]}"
            with helpers["call_context"](f"Memora-{strategy}", qid, "reader_answer"):
                result = searcher.answer_question(user_id, None, str(qa["question"]))
            response = str(result[0] or "")
            provider_health = _question_call_health(run_root / "call_ledger.jsonl", qid, strategy)
            if response.startswith("ERROR:") or not response.strip() or provider_health["missing_success_roles"]:
                _append_jsonl(failures_path, {
                    "question_id": qid,
                    "stage": "reader_answer",
                    "error": response[:500] or "missing successful local call for a required role",
                    "provider_health": provider_health,
                    "run_identity": identity,
                })
                totals["infra_failures"] += 1
                raise RuntimeError(f"Local provider failure at {qid}; no quality score was recorded")
            metrics = _official_metrics(response, str(qa.get("answer", "")))
            answer_row = {
                "question_id": qid,
                "sample_id": sample_id,
                "qa_index": row["qa_index"],
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, "unknown"),
                "question": str(qa["question"]),
                "gold_answer": str(qa.get("answer", "")),
                "response": response,
                **metrics,
                "retrieved_memories": result[1],
                "formatted_memory_context": result[6],
                "retrieval_latency_seconds": round(float(result[3]), 6),
                "answer_latency_seconds": round(float(result[5]), 6),
                "latency_breakdown": result[8],
                "provider_health": provider_health,
                "run_identity": identity,
            }
            _append_jsonl(predictions_path, answer_row)
            predictions[qid] = answer_row
            totals["generated"] += 1

        prediction = predictions[qid]
        if qid not in judgments and category != 5:
            try:
                judge = _judge(searcher, helpers, qa, prediction["response"], qid, strategy)
            except Exception as error:
                _append_jsonl(failures_path, {
                    "question_id": qid,
                    "stage": "judge_local",
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                    "run_identity": identity,
                })
                totals["infra_failures"] += 1
                raise RuntimeError(f"Local Qwen judge failed at {qid}; judge score not recorded") from error
            judge_row = {
                "question_id": qid,
                "category": category,
                **judge,
                "run_identity": identity,
            }
            _append_jsonl(judgments_path, judge_row)
            judgments[qid] = judge_row
            totals["judged"] += 1
        if position % 25 == 0 or position == len(questions):
            print(f"[{strategy} {position}/{len(questions)}] answers={len(predictions)} judges={len(judgments)}")
    return {
        "strategy": strategy,
        "predictions": len(predictions),
        "expected_predictions": len(questions),
        "judgments": len(judgments),
        "expected_judgments": sum(
            int(qa.get("category", -1)) != 5
            for item in dataset
            for qa in item["qa"]
        ),
        "infra_failures_this_invocation": totals["infra_failures"],
        "predictions_path": str(predictions_path),
        "judgments_path": str(judgments_path),
    }


def _bootstrap_delta(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    right_by_id = {row["question_id"]: row for row in right}
    paired = [row for row in left if row["question_id"] in right_by_id]
    if not paired:
        return {"n": 0, "delta": None, "ci95": None}
    deltas = [float(row["official_f1"]) - float(right_by_id[row["question_id"]]["official_f1"]) for row in paired]
    groups: dict[int, list[float]] = defaultdict(list)
    for row, delta in zip(paired, deltas):
        groups[int(row["category"])].append(delta)
    import numpy as np
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.zeros(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for values in groups.values():
        current = np.asarray(values, dtype=np.float64)
        indices = rng.integers(0, len(current), size=(BOOTSTRAP_SAMPLES, len(current)))
        samples += current[indices].mean(axis=1) * (len(current) / len(paired))
    return {
        "n": len(paired),
        "delta": float(mean(deltas)),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "stratified_by": "LoCoMo category",
    }


def _summarize_call_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "calls": len(rows),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows if isinstance(row.get("prompt_tokens"), (int, float))),
        "prompt_token_calls_known": sum(isinstance(row.get("prompt_tokens"), (int, float)) for row in rows),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in rows if isinstance(row.get("completion_tokens"), (int, float))),
        "completion_token_calls_known": sum(isinstance(row.get("completion_tokens"), (int, float)) for row in rows),
        "latency_seconds": round(sum(float(row.get("latency_ms") or 0) for row in rows) / 1000, 3),
    }


def _summarize(
    run_root: Path,
    identity: str,
    questions: list[dict[str, Any]],
    strategies: list[str],
    expected_judgments: int,
) -> dict[str, Any]:
    strategy_results = {}
    predictions_by_strategy = {}
    call_ledger = _read_jsonl(run_root / "call_ledger.jsonl")
    unresolved_failures = []
    shared_ingestion_usage = {}
    ingestion_calls = [row for row in call_ledger if str(row.get("question_id", "")).startswith("ingest:")]
    for role in ("memory_ingest", "embedding"):
        shared_ingestion_usage[role] = _summarize_call_usage([row for row in ingestion_calls if row.get("role") == role])
    for strategy in strategies:
        rows = _read_jsonl(run_root / "predictions" / f"{strategy}.jsonl")
        judgments = _read_jsonl(run_root / "judgments" / f"{strategy}.jsonl")
        failures = _read_jsonl(run_root / "failures" / f"{strategy}.jsonl")
        predictions_by_strategy[strategy] = rows
        judged = {row["question_id"]: row for row in judgments}
        summary: dict[str, Any] = {
            "prediction_count": len(rows),
            "judged_count": len(judgments),
            "overall": {},
            "categories": {},
        }
        for group_name, group_rows in [("overall", rows)] + [
            (CATEGORY_NAMES.get(category, str(category)), [row for row in rows if int(row["category"]) == category])
            for category in range(1, 6)
        ]:
            summary_group = {
                "count": len(group_rows),
                "official_f1": mean([float(row["official_f1"]) for row in group_rows]) if group_rows else None,
                "official_exact_match": mean([int(row["official_exact_match"]) for row in group_rows]) if group_rows else None,
                "normalized_em": mean([int(row["normalized_em"]) for row in group_rows]) if group_rows else None,
                "local_qwen_judge_accuracy": mean([
                    int(judged[row["question_id"]]["correct"])
                    for row in group_rows
                    if row["question_id"] in judged
                ]) if any(row["question_id"] in judged for row in group_rows) else None,
                "local_qwen_judge_count": sum(row["question_id"] in judged for row in group_rows),
                "avg_memory_context_reader_tokens": mean([
                    float((row.get("latency_breakdown") or {}).get("prompt_stats", {}).get("total_tokens"))
                    for row in group_rows
                    if (row.get("latency_breakdown") or {}).get("prompt_stats", {}).get("total_tokens") is not None
                ]) if any((row.get("latency_breakdown") or {}).get("prompt_stats", {}).get("total_tokens") is not None for row in group_rows) else None,
                "avg_retrieval_latency_seconds": mean([float(row["retrieval_latency_seconds"]) for row in group_rows]) if group_rows else None,
                "avg_answer_latency_seconds": mean([float(row["answer_latency_seconds"]) for row in group_rows]) if group_rows else None,
            }
            if group_name == "overall":
                summary["overall"] = summary_group
            else:
                summary["categories"][group_name] = summary_group
        eligible = [row for row in rows if int(row["category"]) != 5 and row["question_id"] in judged]
        summary["local_qwen_judge"] = {
            "count": len(eligible),
            "accuracy": mean([int(judged[row["question_id"]]["correct"]) for row in eligible]) if eligible else None,
            "category_5_excluded": True,
        }
        strategy_calls = [row for row in call_ledger if row.get("system") == f"Memora-{strategy}"]
        usage_by_role = {}
        for role in ("memory_reasoning", "reader_answer", "judge_local", "embedding"):
            role_calls = [row for row in strategy_calls if row.get("role") == role]
            usage_by_role[role] = {
                "calls": len(role_calls),
                "prompt_tokens": sum(int(row["prompt_tokens"]) for row in role_calls if isinstance(row.get("prompt_tokens"), (int, float))),
                "prompt_token_calls_known": sum(isinstance(row.get("prompt_tokens"), (int, float)) for row in role_calls),
                "completion_tokens": sum(int(row["completion_tokens"]) for row in role_calls if isinstance(row.get("completion_tokens"), (int, float))),
                "completion_token_calls_known": sum(isinstance(row.get("completion_tokens"), (int, float)) for row in role_calls),
                "latency_seconds": round(sum(float(row.get("latency_ms") or 0) for row in role_calls) / 1000, 3),
            }
        summary["local_model_usage"] = usage_by_role
        prediction_ids = {row["question_id"] for row in rows}
        judgment_ids = set(judged)
        for failure in failures:
            qid = failure.get("question_id")
            if qid not in prediction_ids or (failure.get("stage") == "judge_local" and qid not in judgment_ids):
                unresolved_failures.append({"strategy": strategy, **failure})
        strategy_results[strategy] = summary
    paired = None
    if "prompt" in predictions_by_strategy and "semantic" in predictions_by_strategy:
        paired = _bootstrap_delta(predictions_by_strategy["prompt"], predictions_by_strategy["semantic"])
    answer_complete = all(len(predictions_by_strategy[strategy]) == len(questions) for strategy in strategies)
    judge_complete = all(
        strategy_results[strategy]["judged_count"] == expected_judgments
        for strategy in strategies
    )
    result = {
        "status": "complete" if answer_complete and judge_complete and not unresolved_failures else "partial",
        "run_identity": identity,
        "question_count": len(questions),
        "expected_local_judgments": expected_judgments,
        "unresolved_infrastructure_failures": unresolved_failures,
        "shared_ingestion_model_usage": shared_ingestion_usage,
        "strategies": strategy_results,
        "prompt_minus_semantic_official_f1": paired,
        "warning": "Local Qwen architecture transfer; not exact numerical reproduction of published proprietary GPT results.",
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(run_root / "summary.json", result)
    _write_markdown_report(result, run_root)
    return result


def _write_markdown_report(result: dict[str, Any], run_root: Path) -> None:
    lines = [
        "# Memora on LoCoMo with Local Qwen",
        "",
        f"Status: `{result['status']}`",
        "",
        "This is a controlled architecture transfer of the pinned Microsoft Memora implementation to a fully local Qwen stack. It is not an exact numerical reproduction of the paper's GPT reader/judge results.",
        "",
        f"Questions answered: {result['question_count']} expected.",
        "",
        "## Main Results",
        "",
        "| Strategy | N | Official token F1 | Official EM | Normalized EM | Local Qwen judge accuracy | Judge N | Mean memory-context tokens | Retrieval sec | Answer sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, values in result["strategies"].items():
        overall = values["overall"]
        judge = values["local_qwen_judge"]
        lines.append(
            f"| {strategy} | {overall['count']} | {_fmt(overall['official_f1'])} | {_fmt(overall['official_exact_match'])} | {_fmt(overall['normalized_em'])} | {_fmt(judge['accuracy'])} | {judge['count']} | {_fmt(overall['avg_memory_context_reader_tokens'])} | {_fmt(overall['avg_retrieval_latency_seconds'])} | {_fmt(overall['avg_answer_latency_seconds'])} |"
        )
    lines.extend(["", "## Category Results", "", "| Strategy | Category | N | Official token F1 | Official EM | Normalized EM | Local Qwen judge accuracy | Mean memory-context tokens | Retrieval sec | Answer sec |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for strategy, values in result["strategies"].items():
        for category, scores in values["categories"].items():
            lines.append(
                f"| {strategy} | {category} | {scores['count']} | {_fmt(scores['official_f1'])} | {_fmt(scores['official_exact_match'])} | {_fmt(scores['normalized_em'])} | {_fmt(scores['local_qwen_judge_accuracy'])} | {_fmt(scores['avg_memory_context_reader_tokens'])} | {_fmt(scores['avg_retrieval_latency_seconds'])} | {_fmt(scores['avg_answer_latency_seconds'])} |"
            )
    paired = result.get("prompt_minus_semantic_official_f1")
    if paired:
        ci = paired.get("ci95")
        lines.extend([
            "",
            "## Paired Strategy Contrast",
            "",
            f"Prompted-policy minus semantic official F1: `{_fmt(paired.get('delta'))}` over {paired.get('n')} paired questions; stratified paired-bootstrap 95% CI `{_fmt(ci[0])} to {_fmt(ci[1])}` (10,000 resamples, seed 42).",
        ])
    lines.extend([
        "",
        "## Shared Memory Ingestion Usage",
        "",
        "| Role | Calls | Prompt tokens known | Completion tokens known | Service sec |",
        "|---|---:|---:|---:|---:|",
    ])
    for role, usage in result.get("shared_ingestion_model_usage", {}).items():
        lines.append(
            f"| {role} | {usage['calls']} | {usage['prompt_tokens']} / {usage['prompt_token_calls_known']} calls | {usage['completion_tokens']} / {usage['completion_token_calls_known']} calls | {_fmt(usage['latency_seconds'])} |"
        )
    lines.extend([
        "",
        "## Local Model Usage",
        "",
        "Role-wise request counts, local-server prompt/completion usage where available, and local service latency:",
        "",
        "| Strategy | Role | Calls | Prompt tokens known | Completion tokens known | Service sec |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for strategy, values in result["strategies"].items():
        for role, usage in values["local_model_usage"].items():
            lines.append(
                f"| {strategy} | {role} | {usage['calls']} | {usage['prompt_tokens']} / {usage['prompt_token_calls_known']} calls | {usage['completion_tokens']} / {usage['completion_token_calls_known']} calls | {_fmt(usage['latency_seconds'])} |"
            )
    lines.extend([
        "",
        "## Protocol Notes",
        "",
        "- Reader, memory-internal LLM, and judge: the same frozen Qwen3-8B Q4_K_M through the loopback llama.cpp service.",
        "- Embedding: frozen Qwen3-Embedding-0.6B, local CUDA FP16, 1024 dimensions, normalized vectors.",
        "- Memory method: Memora's LLM segmentation, primary abstraction plus specific memory values, cue anchors, episodic links, update decisions, local Chroma storage, BM25 hybrid retrieval, and the official semantic/prompted-policy retrieval paths.",
        "- No hosted model or embedding APIs; no API keys; no GRPO training. Category 5 is omitted only from the Memora-style judge accuracy, matching its official LoCoMo evaluation code; deterministic metrics include it.",
        "- Memory-context token counts use the pinned Qwen reader's local llama.cpp tokenizer on formatted memory lines joined by newlines; they are not full reader-prompt token counts. Model prompt/completion token totals are reported from the local server only when it supplies usage fields.",
        "- Predictions, judge rows, call ledger, and persisted memories are resumable local artifacts under the run directory.",
        "- The paper's published GPT-based results are external historical coordinates, not a directly comparable baseline for this Qwen run.",
        "",
        f"Run artifacts: `{run_root}`",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repositories = _verify_repositories()
    if not READER_PATH.is_file() or not EMBEDDING_PATH.is_dir():
        raise RuntimeError("Frozen local Qwen model artifacts are not available")
    _configure_local_environment()
    reader = _reader_process()
    parsed = urlsplit(ENDPOINT)
    if parsed.hostname != "127.0.0.1" or parsed.port != 8081:
        raise RuntimeError("Reader endpoint must be 127.0.0.1:8081")

    dataset = _load_dataset()
    if args.phase == "smoke":
        dataset = json.loads(json.dumps(dataset[:1]))
        dataset[0]["qa"] = dataset[0]["qa"][:1]
        conversation = dataset[0]["conversation"]
        keep = {"speaker_a", "speaker_b", "session_1", "session_1_date_time"}
        dataset[0]["conversation"] = {key: value for key, value in conversation.items() if key in keep}
        run_root = RUN_ROOT / "smoke"
    else:
        run_root = RUN_ROOT
    base_identity = {
        **repositories,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "reader_model_sha256": READER_SHA256,
        "embedding_model_revision": EMBEDDING_REVISION,
        "embedding_weights_sha256": EMBEDDING_WEIGHTS_SHA256,
        "strategies": ["semantic", "prompt"],
        "phase": args.phase,
    }
    base_identity["identity_sha256"] = _sha256_bytes(_canonical(base_identity))
    run_identity = base_identity["identity_sha256"]
    question_hash = _prepare_manifests(base_identity, dataset, run_root)

    owned_patch = _ensure_memeval_patch()
    try:
        sys.path.insert(0, str(MEMEVAL_ROOT / "src"))
        from agents_memory.healthcopilot_provider import ProviderConfig, configure_call_ledger, initialize_local_embedding
        config = ProviderConfig.from_env()
        configure_call_ledger(run_root / "call_ledger.jsonl")
        helpers = _install_local_adapters(sys.modules["agents_memory.healthcopilot_provider"], config)
        _install_outbound_network_guard()
        embedding_runtime = initialize_local_embedding(config)
        if embedding_runtime.artifact.revision != EMBEDDING_REVISION or embedding_runtime.artifact.weights_sha256 != EMBEDDING_WEIGHTS_SHA256:
            raise RuntimeError("Local Qwen embedding artifact did not match the frozen revision")
        runtime_info = {
            "reader": reader,
            "embedding": {
                "repo": embedding_runtime.artifact.repo,
                "revision": embedding_runtime.artifact.revision,
                "model_sha256": embedding_runtime.artifact.model_sha256,
                "weights_sha256": embedding_runtime.artifact.weights_sha256,
                "dimensions": embedding_runtime.artifact.dimensions,
                "normalized": embedding_runtime.artifact.normalized,
                "device": embedding_runtime.artifact.device,
                "dtype": embedding_runtime.artifact.dtype,
                "device_name": embedding_runtime.artifact.device_name,
                "query_instruction": embedding_runtime.artifact.query_instruction,
                "document_instruction": embedding_runtime.artifact.document_instruction,
                "batch_size": embedding_runtime.artifact.batch_size,
                "max_length": embedding_runtime.artifact.max_length,
                "truncation": embedding_runtime.artifact.truncation,
            },
            "question_manifest_sha256": question_hash,
            "hosted_api_calls": 0,
            "api_keys_required": [],
            "software": {
                "python": sys.version.split()[0],
                "torch": __import__("torch").__version__,
                "chromadb": __import__("chromadb").__version__,
                "transformers": __import__("transformers").__version__,
                "openai": __import__("openai").__version__,
            },
        }
        _write_json(run_root / "runtime_manifest.json", runtime_info)
        cfg = _config(run_root, "semantic")
        if args.phase in {"smoke", "full"}:
            ingestion = _run_ingestion(dataset, run_root, run_identity, cfg, helpers)
            questions = _question_rows(dataset)
            strategies = args.strategies
            run_results = []
            for strategy in strategies:
                strategy_cfg = _config(run_root, strategy)
                run_results.append(_run_strategy(dataset, questions, run_root, run_identity, strategy_cfg, strategy, helpers))
            expected_judgments = sum(
                int(qa.get("category", -1)) != 5
                for item in dataset
                for qa in item["qa"]
            )
            summary = _summarize(run_root, run_identity, questions, strategies, expected_judgments)
            summary["ingestion"] = ingestion
            summary["runtime"] = runtime_info
            summary["strategy_runs"] = run_results
            summary["source"] = {
                "paper_arxiv": "2602.03315",
                "local_paper_pdf_sha256": "c6f1b9381bff11a05b24a5dadd6d3a88d200c85e564c364e5813aa1493c1ad760",
                "memora_upstream_commit": EXPECTED_MEMORA_COMMIT,
            }
            _write_json(run_root / "summary.json", summary)
            _write_markdown_report(summary, run_root)
        return {"run_identity": run_identity, "status": args.phase, "run_root": str(run_root)}
    finally:
        _remove_owned_memeval_patch(owned_patch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), default="full")
    parser.add_argument("--strategies", nargs="+", choices=("semantic", "prompt"), default=["semantic", "prompt"])
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
