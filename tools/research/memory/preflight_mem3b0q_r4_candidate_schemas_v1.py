"""Offline CLI schema-file parsing preflight; never loads a model or contacts a service."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from tools.research.memory.mem3b0q_r4_candidate_builder_v1 import build_candidate_schema
from tools.research.memory.run_mem3b0q_r4_gate import _load_frozen_inputs


ROOT = Path(__file__).resolve().parents[3]
SERVER_PATH = Path(
    r"C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages"
    r"\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
)
CLI_PATH = SERVER_PATH.with_name("llama-cli.exe")
EXPECTED_CLI_SHA256 = "48566cf6e2969464b799dbcac7393b3549f9efd6884688074dc803125dbafa85"
EXPECTED_VERSION = "10068 (571d0d540)"
OUTPUT_PATH = ROOT / "docs/research/memory/mem3b0q_r4_candidate_schema_preflight_v1_1.json"
OUTPUT_SHA_PATH = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".sha256")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_cli(schema_path: Path, model_path: Path) -> dict[str, Any]:
    command = [
        str(CLI_PATH),
        "-m",
        str(model_path),
        "--json-schema-file",
        str(schema_path),
        "-n",
        "0",
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    missing_model = bool(
        re.search(r"failed to load model|failed to open model", combined, re.IGNORECASE)
    )
    schema_parse_error = any(
        re.search(r"--json-schema-file|json.?schema", line, re.IGNORECASE)
        and re.search(
            r"parse_error|invalid json|failed to parse|error while handling argument",
            line,
            re.IGNORECASE,
        )
        and not re.search(
            r"failed to load model|failed to open model", line, re.IGNORECASE
        )
        for line in combined.splitlines()
    )
    return {
        "exit_code": result.returncode,
        "schema_file_parse_error_detected": schema_parse_error,
        "missing_model_failure_detected": missing_model,
        "combined_output_sha256": _sha256(combined.encode("utf-8")),
        "output_excerpt": combined[:600],
    }


def run_preflight() -> dict[str, Any]:
    if not CLI_PATH.is_file():
        raise RuntimeError("pinned_llama_cli_missing")
    cli_sha256 = _sha256(CLI_PATH.read_bytes())
    if cli_sha256 != EXPECTED_CLI_SHA256:
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
    version_output = f"{version_result.stdout}\n{version_result.stderr}".strip()
    if version_result.returncode != 0 or EXPECTED_VERSION not in version_output:
        raise RuntimeError("pinned_llama_cli_version_mismatch")

    _, pack, _, _ = _load_frozen_inputs()
    propositions = pack.get("propositions")
    if not isinstance(propositions, list) or len(propositions) != 20:
        raise RuntimeError("frozen_r4_control_pack_shape_mismatch")

    model_path = Path(tempfile.gettempdir()) / "__mem3b0q_r4_absent_model__.gguf"
    if model_path.exists():
        raise RuntimeError("intentional_missing_model_path_exists")

    case_results = []
    with tempfile.TemporaryDirectory(prefix="mem3b0q-r4-schema-preflight-") as temp_dir:
        temp_path = Path(temp_dir)
        for proposition in propositions:
            source_id = proposition.get("source_id")
            source_text = proposition.get("proposition_text")
            if not isinstance(source_id, str) or not isinstance(source_text, str):
                raise RuntimeError("frozen_r4_proposition_shape_mismatch")
            schema = build_candidate_schema(source_id, source_text)
            Draft202012Validator.check_schema(schema)
            schema_bytes = (
                json.dumps(schema, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            ).encode("utf-8")
            schema_path = temp_path / f"{source_id}.schema.json"
            schema_path.write_bytes(schema_bytes)
            cli_result = _run_cli(schema_path, model_path)
            passed = (
                cli_result["exit_code"] != 0
                and cli_result["missing_model_failure_detected"]
                and not cli_result["schema_file_parse_error_detected"]
            )
            case_results.append(
                {
                    "source_id": source_id,
                    "schema_sha256": _sha256(schema_bytes),
                    "schema_well_formed": True,
                    "atoms_max_items": schema["properties"]["atoms"]["maxItems"],
                    **cli_result,
                    "status": "PASS" if passed else "FAIL",
                }
            )

        bad_schema_path = temp_path / "invalid.schema.json"
        bad_schema_path.write_text("{not-json", encoding="utf-8")
        negative_control = _run_cli(bad_schema_path, model_path)

    negative_control_passed = (
        negative_control["exit_code"] != 0
        and negative_control["schema_file_parse_error_detected"]
        and not negative_control["missing_model_failure_detected"]
    )
    all_cases_passed = all(row["status"] == "PASS" for row in case_results)
    overall_passed = all_cases_passed and negative_control_passed
    return {
        "preflight_id": "mem3b0q-r4-candidate-schema-preflight-v1.1",
        "status": "PASS_SCHEMA_FILE_PARSE_ONLY" if overall_passed else "FAIL",
        "inference_performed": False,
        "model_loaded": False,
        "local_endpoint_called": False,
        "hosted_api_called": False,
        "schema_file_parsing": "PASS" if overall_passed else "FAIL",
        "grammar_conversion": "NOT_VERIFIED",
        "sampler_initialization": "NOT_VERIFIED",
        "frozen_r4_manifest_sha256": _sha256(
            (ROOT / "docs/research/memory/mem3b0q_r4_freeze_manifest.json").read_bytes()
        ),
        "frozen_pack_sha256": _sha256(
            (ROOT / "docs/research/memory/mem3b0q_r4_control_pack_v1.json").read_bytes()
        ),
        "llama_cli": {
            "path": str(CLI_PATH),
            "sha256": cli_sha256,
            "version": EXPECTED_VERSION,
            "observed_version_output_sha256": _sha256(
                version_output.encode("utf-8")
            ),
        },
        "preflight_runner_sha256": _sha256(Path(__file__).resolve().read_bytes()),
        "candidate_builder_sha256": _sha256(
            (ROOT / "tools/research/memory/mem3b0q_r4_candidate_builder_v1.py").read_bytes()
        ),
        "intentional_missing_model_path": str(model_path),
        "schemas_checked": len(case_results),
        "schema_parse_negative_control": {
            **negative_control,
            "status": "PASS" if negative_control_passed else "FAIL",
        },
        "cases": case_results,
        "interpretation": (
            "Verifies Draft 2020-12 validity and that valid JSON schema files pass the "
            "pinned CLI schema-file parsing stage before the intentionally absent model "
            "fails to load. The malformed-schema negative control verifies parse rejection. "
            "Does not prove grammar conversion or sampler initialization, serve a request, "
            "or establish model quality."
        ),
    }


def main() -> int:
    result = run_preflight()
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    OUTPUT_PATH.write_bytes(encoded.encode("utf-8"))
    digest = _sha256(OUTPUT_PATH.read_bytes())
    OUTPUT_SHA_PATH.write_text(f"{digest}  {OUTPUT_PATH.name}\n", encoding="ascii")
    print(f"{result['status']} {OUTPUT_PATH}")
    return 0 if result["status"] == "PASS_SCHEMA_FILE_PARSE_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
