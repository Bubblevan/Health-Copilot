from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from eval.healthbench_judge import JUDGE_PROMPT_TEMPLATE, build_judge_prompt

DATA_ROOT = Path(os.environ.get("PT_E0_DATA_ROOT", "/root/gpufree-data/Health-Copilot-PT-E0-data"))
RUN_DIR = Path(os.environ.get("PT_E0_RUNS", str(DATA_ROOT / "runs/posttrain/pt-e0"))) / "hbpro"
CANDIDATE_VIEW = DATA_ROOT / "eval/prepared/hbpro/candidate_view.jsonl"
SCORER_VIEW = DATA_ROOT / "eval/prepared/hbpro/scorer_view.jsonl"
POSTTRAIN_ROOT = Path(__file__).resolve().parents[1]
JUDGE_PROTOCOL_PATH = POSTTRAIN_ROOT / "manifests/eval/healthbench_local_judge.json"
JUDGE_MODEL_MANIFEST_PATH = POSTTRAIN_ROOT / "manifests/model/mistral-small-3.1-24b-q4-local-judge.json"
JUDGE_MODEL = "Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M"
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 512
SEED = 20261004
MAX_ATTEMPTS = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_grade(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    decoder = json.JSONDecoder()
    for offset, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[offset:])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(parsed, dict)
            and type(parsed.get("criteria_met")) is bool
            and isinstance(parsed.get("explanation"), str)
        ):
            return parsed
    return None


def ensure_local_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("The local judge URL must be plain HTTP on loopback; hosted endpoints are forbidden")
    return base_url.rstrip("/")


def judge_request(base_url: str, prompt: str) -> tuple[str, float]:
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Local judge request failed: {error}") from error
    elapsed = time.perf_counter() - started
    return str(body["choices"][0]["message"]["content"]), elapsed


def frozen_predictions() -> list[dict[str, Any]]:
    path = RUN_DIR / "predictions.jsonl"
    manifest_path = RUN_DIR / "prediction_manifest.json"
    sidecar = RUN_DIR / "predictions.sha256"
    if not path.is_file() or not manifest_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("Frozen HB-Pro candidate predictions and hash manifest are required before judge scoring")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256_file(path)
    sidecar_hash = sidecar.read_text(encoding="ascii").split()[0]
    if actual != manifest.get("predictions_sha256") or actual != sidecar_hash:
        raise ValueError("HB-Pro predictions failed SHA256 verification")
    return read_jsonl(path)


def verify_frozen_judge() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(JUDGE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    model_manifest = json.loads(JUDGE_MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    template_hash = hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    builder_hash = sha256_file(POSTTRAIN_ROOT / "eval/healthbench_judge.py")
    entrypoint_hash = sha256_file(Path(__file__))
    if protocol["prompt_template_sha256"] != template_hash or protocol["prompt_builder_sha256"] != builder_hash:
        raise ValueError("Local judge prompt differs from the frozen protocol")
    if protocol["grader_entrypoint_sha256"] != entrypoint_hash:
        raise ValueError("Local grader entrypoint differs from the frozen protocol")
    if protocol["judge_model_manifest_sha256"] != sha256_file(JUDGE_MODEL_MANIFEST_PATH):
        raise ValueError("Local judge model manifest changed after protocol freeze")
    model_path = Path(model_manifest["local_path"])
    if model_path.stat().st_size != model_manifest["file_bytes"] or sha256_file(model_path) != model_manifest["file_sha256"]:
        raise ValueError("Local Q4 judge GGUF failed frozen size/SHA256 verification")
    runtime_binary = Path("/root/gpufree-data/llama.cpp/build/bin/llama-server")
    if sha256_file(runtime_binary) != model_manifest["server_binary_sha256"]:
        raise ValueError("llama-server binary differs from the frozen runtime manifest")
    if model_manifest["runtime_commit"] != "836d57176dc699a726c55418e4f96b8ca628e1bf":
        raise ValueError("llama.cpp runtime commit differs from the frozen run code")
    if protocol["judge_runtime"]["launcher_sha256"] != sha256_file(POSTTRAIN_ROOT / "scripts/start_healthbench_judge_server.sh"):
        raise ValueError("Local judge launch arguments differ from the frozen protocol")
    expected_runtime = {
        "base_url": "http://127.0.0.1:8080",
        "model_alias": JUDGE_MODEL,
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
    }
    if any(protocol["judge_runtime"].get(key) != value for key, value in expected_runtime.items()):
        raise ValueError("Runtime parameters differ from the frozen local judge configuration")
    expected_generation = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "response_format": "json_object",
        "max_attempts_for_invalid_json": MAX_ATTEMPTS,
    }
    if protocol["generation"] != expected_generation:
        raise ValueError("Judge generation parameters differ from the frozen protocol")
    return protocol, model_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("PT_E0_JUDGE_URL", "http://127.0.0.1:8080"))
    args = parser.parse_args()
    base_url = ensure_local_url(args.base_url)
    frozen_protocol, model_manifest = verify_frozen_judge()
    if base_url != frozen_protocol["judge_runtime"]["base_url"]:
        raise ValueError("Judge endpoint differs from the frozen local endpoint")
    predictions = frozen_predictions()
    candidates = read_jsonl(CANDIDATE_VIEW)
    scorers = read_jsonl(SCORER_VIEW)
    candidate_by_id = {str(row["id"]): row for row in candidates}
    scorer_by_id = {str(row["id"]): row for row in scorers}
    prediction_by_id = {str(row["id"]): row for row in predictions}
    ids = set(candidate_by_id)
    if len(ids) != len(candidates) or ids != set(scorer_by_id) or ids != set(prediction_by_id):
        raise ValueError("Candidate, scorer, and frozen prediction IDs must match one-to-one")

    expected: dict[str, tuple[str, int, dict[str, Any]]] = {}
    for example_id in sorted(ids):
        for index, rubric_item in enumerate(scorer_by_id[example_id]["rubric_items"]):
            task_id = f"{example_id}|{index:04d}"
            prompt = build_judge_prompt(
                candidate_by_id[example_id]["messages"],
                str(prediction_by_id[example_id]["final_answer"]),
                rubric_item,
            )
            expected[task_id] = (prompt, index, rubric_item)

    output = RUN_DIR / "rubric_grades.jsonl"
    partial = RUN_DIR / "rubric_grades.jsonl.partial"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(partial) if partial.exists() else []
    done = {row["task_id"] for row in existing}
    if not done.issubset(expected):
        raise ValueError("Existing partial judge results do not match this frozen run")
    prompt_template_hash = hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    identity = {
        "judge_repo_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "judge_gguf_repo_id": "bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF",
        "judge_gguf_revision": "f73dfd9e812922fb503a993e3fa5671424f486d3",
        "judge_file_sha256": "c5743c1bf39db0ae8a5ade5df0374b8e9e492754a199cfdad7ef393c1590f7c0",
        "judge_quantization": "Q4_K_M",
        "judge_runtime": "llama.cpp OpenAI-compatible local server",
        "llama_cpp_commit": "836d57176dc699a726c55418e4f96b8ca628e1bf",
        "server_version": model_manifest["runtime_version"],
        "server_binary_sha256": model_manifest["server_binary_sha256"],
        "frozen_judge_protocol_sha256": sha256_file(JUDGE_PROTOCOL_PATH),
        "context_size": 32768,
        "gpu_layers": 99,
        "parallel_slots": 1,
        "flash_attention": "on",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "threads": 8,
        "web_ui": False,
        "host": "127.0.0.1",
        "port": 8080,
        "prompt_template_sha256": prompt_template_hash,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "response_format": "json_object",
        "max_attempts_for_invalid_json": MAX_ATTEMPTS,
        "hosted_api_calls": 0,
        "openai_api_key_used": False,
    }
    identity_path = RUN_DIR / "judge_run_manifest.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Judge settings differ from the frozen local judge manifest")
    write_json(identity_path, identity)

    completed_this_run = 0
    with partial.open("a" if partial.exists() else "w", encoding="utf-8", buffering=1) as handle:
        for task_id, (prompt_text, index, rubric_item) in expected.items():
            if task_id in done:
                continue
            example_id = task_id.rsplit("|", 1)[0]
            raw = ""
            elapsed_total = 0.0
            parsed = None
            attempts = 0
            for attempts in range(1, MAX_ATTEMPTS + 1):
                raw, elapsed = judge_request(base_url, prompt_text)
                elapsed_total += elapsed
                parsed = parse_grade(raw)
                if parsed is not None:
                    break
            if parsed is None:
                raise RuntimeError(f"Local judge returned invalid JSON for {task_id} after {attempts} attempts")
            row = {
                "task_id": task_id,
                "id": example_id,
                "criterion_index": index,
                "criterion_text": rubric_item["criterion_text"],
                "points": rubric_item["points"],
                "criteria_met": parsed["criteria_met"],
                "explanation": parsed["explanation"],
                "attempts": attempts,
                "judge_latency_seconds": elapsed_total,
                "raw_judge_output": raw,
                "raw_judge_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "candidate_output_sha256": prediction_by_id[example_id]["raw_output_hash"],
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            done.add(task_id)
            completed_this_run += 1
            if completed_this_run % 25 == 0:
                print(json.dumps({"completed_this_run": completed_this_run, "total_criteria": len(expected), "task_id": task_id}), flush=True)

    rows = read_jsonl(partial)
    if {row["task_id"] for row in rows} != set(expected) or len(rows) != len(expected):
        raise RuntimeError("Local judge criteria output is incomplete or has duplicate task IDs")
    partial.replace(output)
    digest = sha256_file(output)
    (RUN_DIR / "rubric_grades.sha256").write_text(f"{digest}  rubric_grades.jsonl\n", encoding="ascii")
    print(json.dumps({"n_cases": len(ids), "n_criteria": len(expected), "rubric_grades_sha256": digest, "path": str(output)}))


if __name__ == "__main__":
    main()
