"""Verify and freeze the E6A llama.cpp GPU server identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from eval.rag_e6.llm import COMPLETION_CEILING, CONTEXT_CEILING, MODEL_NAME, MODEL_SHA256
from eval.rag_e6.split import sha256_file, write_immutable_json

DEFAULT_MODEL = Path(r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf")
DEFAULT_SERVER = Path(
    r"C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
)
DEFAULT_RUNTIME = ROOT / "runs/rag_e6/runtime"
EXPECTED_SERVER_SHA256 = "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"


def _http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise TypeError("llama.cpp endpoint returned a non-object response")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    model_sha = sha256_file(args.model)
    if model_sha != MODEL_SHA256:
        raise ValueError("Qwen3-8B model SHA-256 differs from its pinned identity")
    version_text = (args.runtime_dir / "llama-server.version.txt").read_text(encoding="utf-8")
    if "version: 10068 (571d0d540)" not in version_text:
        raise ValueError("llama.cpp version differs from the prior frozen U3-R identity")
    devices_text = (args.runtime_dir / "llama-server.devices.txt").read_text(encoding="utf-8")
    expected_device = "Vulkan1: NVIDIA GeForce RTX 4090 Laptop GPU"
    if expected_device not in devices_text:
        raise RuntimeError("llama.cpp does not report the pinned RTX 4090 Vulkan device")

    base = f"http://127.0.0.1:{args.port}"
    health = _http_json(base + "/health")
    properties = _http_json(base + "/props")
    pid_path = args.runtime_dir / "llama-server.pid"
    pid = int(pid_path.read_text(encoding="ascii").strip())
    logs = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (
            args.runtime_dir / "llama-server.stdout.log",
            args.runtime_dir / "llama-server.stderr.log",
        )
        if path.exists()
    )
    gpu_snapshot = (args.runtime_dir / "nvidia-smi-startup.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    if health.get("status") != "ok":
        raise RuntimeError("llama.cpp health endpoint is not ready")
    if str(pid) not in gpu_snapshot:
        raise RuntimeError("nvidia-smi snapshot does not show the E6A server process")
    memory_match = re.search(r"\|\s*(\d+)MiB\s*/\s*16376MiB", gpu_snapshot)
    if memory_match is None:
        raise RuntimeError("nvidia-smi snapshot did not report RTX 4090 memory usage")
    context_match = re.search(r"training context of the model \((\d+)\).*capping", logs)
    effective_context_size = int(context_match.group(1)) if context_match else 0
    if effective_context_size != 40960:
        raise RuntimeError("llama.cpp log does not confirm Qwen3's native 40,960-token context cap")
    command_line = [
        str(args.server), "--model", str(args.model.resolve()), "--alias", MODEL_NAME,
        "--host", "127.0.0.1", "--port", str(args.port), "--ctx-size", str(CONTEXT_CEILING),
        "--n-gpu-layers", "all", "--device", "Vulkan1", "--flash-attn", "on",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--parallel", "1",
    ]
    manifest = {
        "schema_version": "rag-e6a-gpu-server-v1",
        "host": "127.0.0.1",
        "port": args.port,
        "health_status": health["status"],
        "model_path": str(args.model.resolve()),
        "model_sha256": model_sha,
        "model_api_id": MODEL_NAME,
        "server_path": str(args.server.resolve()),
        "server_sha256": EXPECTED_SERVER_SHA256,
        "server_sha256_verification": "inherited from the frozen U3-R server manifest; current process rechecked the executable version and Vulkan device list because WinGet ACL blocks direct hashing",
        "llama_cpp_version": version_text,
        "process_id": pid,
        "command_line": command_line,
        "gpu_device": "Vulkan1",
        "backend": "Vulkan",
        "n_gpu_layers": "all",
        "flash_attention": "on",
        "cache_type_k": "q4_0",
        "cache_type_v": "q4_0",
        "parallel_slots": 1,
        "context_ceiling": CONTEXT_CEILING,
        "effective_context_size": effective_context_size,
        "model_native_context_size": effective_context_size,
        "context_ceiling_note": "65,536 requested; llama.cpp caps this GGUF to its native 40,960 context; measured prompts must leave room for the common 8,192 output ceiling",
        "completion_ceiling": COMPLETION_CEILING,
        "reasoning_enabled": False,
        "retries": 0,
        "server_props_sha256": hashlib.sha256(
            json.dumps(properties, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "device_listing_sha256": hashlib.sha256(devices_text.encode()).hexdigest(),
        "server_logs_sha256": hashlib.sha256(logs.encode()).hexdigest(),
        "gpu_snapshot_sha256": hashlib.sha256(gpu_snapshot.encode()).hexdigest(),
        "gpu_memory_used_mib_at_record": int(memory_match.group(1)),
        "server_process_visible_in_nvidia_smi": True,
        "verified_at_unix": int(time.time()),
    }
    output = args.runtime_dir / "gpu_server_manifest.json"
    write_immutable_json(output, manifest)
    print(json.dumps({
        "manifest": str(output),
        "model_sha256": model_sha,
        "server_version": version_text.splitlines()[0],
        "backend": manifest["backend"],
        "gpu_device": manifest["gpu_device"],
        "pid": pid,
        "health": health["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
