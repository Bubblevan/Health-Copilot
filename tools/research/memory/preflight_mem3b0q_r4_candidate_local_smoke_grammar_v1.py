"""Pre-model-load llama.cpp grammar conversion check for the R4C-05 schema."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from tools.research.memory import mem3b0q_r4_candidate_request_v1 as request_builder
from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate


ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs" / "research" / "memory"
PACK_PATH = DOCS / "mem3b0q_r4_candidate_control_pack_v1.json"
OUTPUT_PATH = DOCS / "mem3b0q_r4_candidate_local_smoke_grammar_preflight_v1.json"
OUTPUT_SHA_PATH = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".sha256")
CLI_PATH = Path(frozen_gate.SERVER_PATH).with_name("llama-cli.exe")
EXPECTED_CLI_SHA256 = "48566cf6e2969464b799dbcac7393b3549f9efd6884688074dc803125dbafa85"
EXPECTED_VERSION = "10068 (571d0d540)"
CASE_ID = "R4C-05"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _classify_cli_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    combined = f"{result.stdout}\n{result.stderr}"
    missing_model = bool(
        re.search(r"failed to load model|failed to open model", combined, re.IGNORECASE)
    )
    schema_error = bool(
        re.search(r"json.?schema|grammar", combined, re.IGNORECASE)
        and re.search(r"parse_error|invalid json|failed to parse|grammar.*(error|failed)", combined, re.IGNORECASE)
        and not missing_model
    )
    return {
        "exit_code": result.returncode,
        "missing_model_failure_detected": missing_model,
        "schema_or_grammar_error_detected": schema_error,
        "combined_output_sha256": _sha256(combined.encode("utf-8")),
        "output_excerpt": combined[:600],
    }


def _run_cli(schema_path: Path, missing_model: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(CLI_PATH),
            "-m",
            str(missing_model),
            "--json-schema-file",
            str(schema_path),
            "-n",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
    )
    return _classify_cli_result(result)


def run_preflight() -> dict[str, Any]:
    if not CLI_PATH.is_file():
        raise RuntimeError("pinned_llama_cli_missing")
    cli_sha = _sha256(CLI_PATH.read_bytes())
    if cli_sha != EXPECTED_CLI_SHA256:
        raise RuntimeError("pinned_llama_cli_sha256_mismatch")
    version_result = subprocess.run(
        [str(CLI_PATH), "--version"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    version = f"{version_result.stdout}\n{version_result.stderr}".strip()
    if version_result.returncode != 0 or EXPECTED_VERSION not in version:
        raise RuntimeError("pinned_llama_cli_version_mismatch")

    pack = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    cases = [case for case in pack.get("cases", []) if case.get("case_id") == CASE_ID]
    if (
        pack.get("status") != "FROZEN_PROTOCOL_NO_INFERENCE_AUTHORIZATION"
        or len(cases) != 1
    ):
        raise RuntimeError("frozen_candidate_case_mismatch")
    request = request_builder.build_candidate_request(
        cases[0], model=frozen_gate.MODEL_PATH
    )
    schema = request["response_format"]["json_schema"]["schema"]
    Draft202012Validator.check_schema(schema)
    schema_bytes = (
        json.dumps(schema, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    schema_sha = _sha256(
        json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    missing_model = Path(tempfile.gettempdir()) / "__mem3b0q_r4_candidate_no_model__.gguf"
    if missing_model.exists():
        raise RuntimeError("intentional_missing_model_path_exists")
    with tempfile.TemporaryDirectory(prefix="mem3b0q-r4-candidate-schema-") as temp_dir:
        schema_path = Path(temp_dir) / f"{CASE_ID}.schema.json"
        schema_path.write_bytes(schema_bytes)
        valid_schema_result = _run_cli(schema_path, missing_model)
        malformed_schema_path = Path(temp_dir) / "invalid.schema.json"
        malformed_schema_path.write_text("{not-json", encoding="utf-8")
        malformed_schema_result = _run_cli(malformed_schema_path, missing_model)

    valid_pass = (
        valid_schema_result["exit_code"] != 0
        and valid_schema_result["missing_model_failure_detected"]
        and not valid_schema_result["schema_or_grammar_error_detected"]
    )
    malformed_pass = (
        malformed_schema_result["exit_code"] != 0
        and malformed_schema_result["schema_or_grammar_error_detected"]
        and not malformed_schema_result["missing_model_failure_detected"]
    )
    passed = valid_pass and malformed_pass
    return {
        "preflight_id": "mem3b0q-r4-candidate-local-smoke-grammar-preflight-v1",
        "status": "PASS_GRAMMAR_CONVERSION_PRE_MODEL_LOAD" if passed else "FAIL",
        "case_id": CASE_ID,
        "schema_sha256": schema_sha,
        "llama_cli": {
            "path": str(CLI_PATH),
            "sha256": cli_sha,
            "version": EXPECTED_VERSION,
            "version_output_sha256": _sha256(version.encode("utf-8")),
        },
        "valid_schema_control": {
            **valid_schema_result,
            "status": "PASS" if valid_pass else "FAIL",
        },
        "malformed_schema_negative_control": {
            **malformed_schema_result,
            "status": "PASS" if malformed_pass else "FAIL",
        },
        "grammar_conversion": "VERIFIED_PRE_MODEL_LOAD" if passed else "FAIL",
        "sampler_initialization": "NOT_VERIFIED",
        "inference_performed": False,
        "model_loaded": False,
        "local_endpoint_called": False,
        "hosted_api_called": False,
        "interpretation": (
            "The pinned CLI converted the exact R4C-05 JSON Schema through its "
            "pre-model-load schema/grammar path before failing on an intentionally "
            "absent model; malformed schema was rejected at schema parsing. This "
            "does not verify server-side sampler initialization or model output."
        ),
    }


def main() -> int:
    result = run_preflight()
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    OUTPUT_PATH.write_bytes(encoded.encode("utf-8"))
    digest = _sha256(OUTPUT_PATH.read_bytes())
    OUTPUT_SHA_PATH.write_text(f"{digest}  {OUTPUT_PATH.name}\n", encoding="ascii")
    print(f"{result['status']} {OUTPUT_PATH}")
    return 0 if result["status"] == "PASS_GRAMMAR_CONVERSION_PRE_MODEL_LOAD" else 1


if __name__ == "__main__":
    raise SystemExit(main())
