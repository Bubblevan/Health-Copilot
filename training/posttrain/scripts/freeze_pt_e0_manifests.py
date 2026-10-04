from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
POSTTRAIN_ROOT = REPO_ROOT / "training/posttrain"
DATA_ROOT = Path("/root/gpufree-data/Health-Copilot-PT-E0-data")
MODEL_DIR = Path("/root/gpufree-share/data/Qwen3-8B")
JUDGE_MODEL_DIR = Path("/root/gpufree-share/models/Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M")
JUDGE_FILE = JUDGE_MODEL_DIR / "mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf"
LLAMA_CPP_ROOT = Path("/root/gpufree-data/llama.cpp")
LLAMA_SERVER = LLAMA_CPP_ROOT / "build/bin/llama-server"
MANIFEST_DIR = POSTTRAIN_ROOT / "manifests"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
JUDGE_FILE_SHA256 = "c5743c1bf39db0ae8a5ade5df0374b8e9e492754a199cfdad7ef393c1590f7c0"
LLAMA_CPP_COMMIT = "836d57176dc699a726c55418e4f96b8ca628e1bf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    names = ("torch", "transformers", "trl", "peft", "unsloth", "vllm", "flashinfer-python", "datasets", "accelerate")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def freeze() -> dict[str, Any]:
    nvidia = command_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"])
    driver = command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    gpu = None
    if nvidia:
        fields = [part.strip() for part in nvidia.splitlines()[0].split(",")]
        gpu = {
            "name": fields[0],
            "memory_total": fields[1] if len(fields) > 1 else None,
            "memory_used_at_capture": fields[2] if len(fields) > 2 else None,
            "utilization_at_capture": fields[3] if len(fields) > 3 else None,
        }
    environment = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvidia_driver": driver.splitlines()[0].strip() if driver else None,
        "gpu": gpu,
        "packages": package_versions(),
        "hosted_api_calls": 0,
        "openai_api_key_used": False,
    }
    (MANIFEST_DIR / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    weight_files = [
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    ]
    expected_hashes = {
        "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
        "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
        "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
        "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
        "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
        "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
        "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
    }
    actual_hashes = {}
    for filename, expected in expected_hashes.items():
        path = MODEL_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Local model file hash mismatch: {filename}")
        actual_hashes[filename] = {"sha256": actual, "bytes": path.stat().st_size}
    tokenizer_config = json.loads((MODEL_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))
    chat_template = tokenizer_config.get("chat_template")
    chat_template_hash = hashlib.sha256((chat_template or "").encode("utf-8")).hexdigest()
    if chat_template_hash != "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8":
        raise ValueError("Qwen3 chat template hash mismatch")
    model_manifest = {
        "repo_id": "Qwen/Qwen3-8B",
        "revision": MODEL_REVISION,
        "license": "apache-2.0",
        "local_path": str(MODEL_DIR),
        "adapter": None,
        "config_sha256": actual_hashes["config.json"]["sha256"],
        "safetensors_index_sha256": actual_hashes["model.safetensors.index.json"]["sha256"],
        "weight_files": {name: actual_hashes[name] for name in weight_files},
        "tokenizer_files": {
            name: actual_hashes[name]
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
        },
        "chat_template_sha256": chat_template_hash,
        "all_verified_files": actual_hashes,
    }
    (MANIFEST_DIR / "model/qwen3-8b-base.json").write_text(json.dumps(model_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not JUDGE_FILE.is_file() or not LLAMA_SERVER.is_file():
        raise FileNotFoundError("Local Q4 judge GGUF and llama-server binary must exist")
    judge_hash = sha256_file(JUDGE_FILE)
    if judge_hash != JUDGE_FILE_SHA256:
        raise ValueError("Local Q4 judge file hash mismatch")
    server_version = command_output([str(LLAMA_SERVER), "--version"])
    judge_model_manifest = {
        "repo_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "gguf_repo_id": "bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF",
        "gguf_revision": "f73dfd9e812922fb503a993e3fa5671424f486d3",
        "file": JUDGE_FILE.name,
        "file_sha256": judge_hash,
        "file_bytes": JUDGE_FILE.stat().st_size,
        "quantization": "Q4_K_M",
        "local_path": str(JUDGE_FILE),
        "adapter": None,
        "runtime": "ggml-org/llama.cpp",
        "runtime_commit": LLAMA_CPP_COMMIT,
        "runtime_version": server_version,
        "server_binary_sha256": sha256_file(LLAMA_SERVER),
        "build": {
            "cmake_build_type": "Release",
            "GGML_CUDA": True,
            "GGML_NATIVE": False,
            "CMAKE_CUDA_ARCHITECTURES": "89",
            "LLAMA_BUILD_TESTS": False,
            "LLAMA_BUILD_EXAMPLES": False,
            "LLAMA_BUILD_SERVER": True,
            "cuda_toolkit": "13.0.88",
        },
    }
    judge_model_manifest_path = MANIFEST_DIR / "model/mistral-small-3.1-24b-q4-local-judge.json"
    judge_model_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    judge_model_manifest_path.write_text(json.dumps(judge_model_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset_manifest = DATA_ROOT / "eval/prepared/dataset_manifest.json"
    if not dataset_manifest.is_file():
        raise FileNotFoundError(dataset_manifest)
    shutil.copyfile(dataset_manifest, MANIFEST_DIR / "eval/eval_dataset_manifest.json")
    protocol = {
        "schema_version": "pt-e0-eval-protocol-v1",
        "candidate_system_prompt": "",
        "candidate_system_prompt_sha256": hashlib.sha256(b"").hexdigest(),
        "candidate_chat_template_sha256": chat_template_hash,
        "prompt_builder_sha256": sha256_file(POSTTRAIN_ROOT / "eval/prompts.py"),
        "parser_sha256": sha256_file(POSTTRAIN_ROOT / "eval/parsers.py"),
        "candidate_view_preparer_sha256": sha256_file(POSTTRAIN_ROOT / "scripts/prepare_external_eval.py"),
        "candidate_generation_code_sha256": sha256_file(POSTTRAIN_ROOT / "scripts/run_base_eval.py"),
        "mcq_scorer_code_sha256": sha256_file(POSTTRAIN_ROOT / "scripts/score_base_eval.py"),
        "generation": {
            "temperature": 0,
            "do_sample": False,
            "top_p": 1,
            "max_new_tokens": 2048,
            "enable_thinking": True,
            "RAG": False,
            "memory": False,
            "tools": False,
            "retrieved_evidence": False,
            "answer_autofix": False,
            "system_safety_postprocessing": False,
        },
        "scoring_contract": {
            "DiagnosisArena": "final selected option; exact match",
            "CMB": "exact answer string after canonical letter ordering; multi-answer exact match",
            "HealthBench Professional": "frozen local rubric judge, criterion-level; raw and length-adjusted scores",
            "LiveMedBench": "reserved; no candidate generations and no rubric scoring in PT-E0",
        },
        "candidate_view_manifest_sha256": sha256_file(MANIFEST_DIR / "eval/eval_dataset_manifest.json"),
    }
    (MANIFEST_DIR / "eval/eval_protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from eval.healthbench_judge import JUDGE_PROMPT_TEMPLATE

    judge_protocol = {
        "schema_version": "pt-e0-hbpro-local-rubric-v1",
        "metric_name": "HB-Pro Local-Rubric Score",
        "official_healthbench_leaderboard_score": False,
        "model_manifest": "../model/mistral-small-3.1-24b-q4-local-judge.json",
        "judge_model_manifest_sha256": sha256_file(judge_model_manifest_path),
        "prompt_template_sha256": hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
        "prompt_builder_sha256": sha256_file(POSTTRAIN_ROOT / "eval/healthbench_judge.py"),
        "grader_entrypoint_sha256": sha256_file(POSTTRAIN_ROOT / "scripts/run_healthbench_local_judge.py"),
        "judge_runtime": {
            "launcher_sha256": sha256_file(POSTTRAIN_ROOT / "scripts/start_healthbench_judge_server.sh"),
            "base_url": "http://127.0.0.1:8080",
            "model_alias": "Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M",
            "context_size": 32768,
            "gpu_layers": 99,
            "parallel_slots": 1,
            "flash_attention": "on",
            "threads": 8,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "web_ui": False,
            "host": "127.0.0.1",
            "port": 8080,
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "seed": 20261004,
            "response_format": "json_object",
            "max_attempts_for_invalid_json": 2,
        },
        "candidate_visibility": ["conversation", "Qwen final answer", "one rubric criterion and its points"],
        "hidden_from_judge": ["physician_response", "difficulty", "grader decisions", "other rubric criteria", "checkpoint identity"],
        "score_calculation": {
            "positive_denominator": "sum of positive rubric points",
            "numerator": "sum of signed points for every criterion marked met",
            "case_aggregation": "clipped mean in [0, 1]",
            "length_adjustment_center_characters": 2000,
            "length_penalty_per_500_characters": 0.0147,
            "bootstrap_samples": 10000,
            "bootstrap_seed": 20261004,
            "scoring_code_sha256": sha256_file(POSTTRAIN_ROOT / "eval/healthbench_scoring.py"),
            "scorer_entrypoint_sha256": sha256_file(POSTTRAIN_ROOT / "scripts/score_healthbench_local.py"),
        },
        "reference_scoring_source": "https://github.com/openai/simple-evals/blob/main/healthbench_eval.py",
        "hosted_api_calls": 0,
        "openai_api_key_used": False,
    }
    (MANIFEST_DIR / "eval/healthbench_local_judge.json").write_text(json.dumps(judge_protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"environment": environment, "model_revision": MODEL_REVISION, "chat_template_sha256": chat_template_hash, "judge_gguf_sha256": judge_hash, "llama_cpp_commit": LLAMA_CPP_COMMIT}


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
