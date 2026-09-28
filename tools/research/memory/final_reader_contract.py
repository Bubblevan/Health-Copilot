"""Load and validate the permanently frozen shared-reader contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "docs" / "research" / "memory" / "final_reader_contract.json"
CONTRACT_SIDECAR = CONTRACT_PATH.with_name(f"{CONTRACT_PATH.name}.sha256")
TEMPLATE_PATH = ROOT / "docs" / "research" / "memory" / "shared_reader_v3_final.txt"
PINNED_FINAL_READER_CONTRACT_SHA256 = "57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3"
SYSTEM_V1 = (
    "Answer questions using only the supplied conversation memory context. "
    "Answer concisely but completely, using exact wording when possible. "
    "If the requested information is not present, answer None. "
    "Do not guess or add unsupported facts."
)
V1_USER_TEMPLATE = "Memory context:\n<<MEMORY_CONTEXT>>\n\nQuestion: <<QUESTION>>"
V3_USER_TEMPLATE = (
    "Memory context:\n<<MEMORY_CONTEXT>>\n\n"
    "Current Date: <<QUESTION_DATE>>\nQuestion: <<QUESTION>>"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_template(raw: bytes) -> tuple[str, str]:
    text = raw.decode("utf-8")
    parts = text.split("---USER---\n")
    if len(parts) != 2:
        raise RuntimeError("Final reader template must contain exactly one ---USER--- separator")
    system_template, user_template = (part.rstrip("\n") for part in parts)
    if system_template != SYSTEM_V1:
        raise RuntimeError("Final reader system prompt differs from the frozen v1 answering policy")
    if user_template != V3_USER_TEMPLATE:
        raise RuntimeError("Final reader user prompt differs from the protocol-frozen v3 template")
    if user_template.replace("Current Date: <<QUESTION_DATE>>\n", "") != V1_USER_TEMPLATE:
        raise RuntimeError("Final reader v3 is not a date-only extension of the v1 user template")
    if text.count("<<MEMORY_CONTEXT>>") != 1 or text.count("<<QUESTION_DATE>>") != 1 or text.count("<<QUESTION>>") != 1:
        raise RuntimeError("Final reader template placeholders must occur exactly once")
    return system_template, user_template


def _message_template_sha256(system_template: str, user_template: str) -> str:
    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user_template},
    ]
    return _sha256_bytes(_canonical_json(messages))


def contract_binding(contract: dict[str, Any], contract_sha256: str) -> dict[str, str]:
    return {
        "contract_version": contract["contract_version"],
        "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "sha256": contract_sha256,
        "template_sha256": contract["template"]["canonical_message_template_sha256"],
    }


def verify_contract_binding(
    binding: dict[str, Any] | None,
    contract: dict[str, Any],
    contract_sha256: str,
) -> None:
    if binding != contract_binding(contract, contract_sha256):
        raise RuntimeError("Run manifest is not bound to the exact frozen final reader contract")


def load_final_reader_contract() -> tuple[dict[str, Any], str, str, str]:
    """Fail closed unless contract, sidecar, and raw template all match."""
    if not CONTRACT_PATH.is_file() or not CONTRACT_SIDECAR.is_file() or not TEMPLATE_PATH.is_file():
        raise RuntimeError("Frozen final reader contract, sidecar, or template is missing")
    contract_sha = _sha256_file(CONTRACT_PATH)
    if contract_sha != PINNED_FINAL_READER_CONTRACT_SHA256:
        raise RuntimeError("Final reader contract differs from the permanently pinned MEM-1D4 contract")
    sidecar_fields = CONTRACT_SIDECAR.read_text(encoding="ascii").strip().split()
    if sidecar_fields != [contract_sha, CONTRACT_PATH.name]:
        raise RuntimeError("Final reader contract SHA256 sidecar mismatch")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    raw_template = TEMPLATE_PATH.read_bytes()
    raw_sha = _sha256_bytes(raw_template)
    system_template, user_template = _parse_template(raw_template)
    message_sha = _message_template_sha256(system_template, user_template)
    template_contract = contract.get("template", {})
    v1_template_sha = _message_template_sha256(SYSTEM_V1, V1_USER_TEMPLATE.replace(
        "<<MEMORY_CONTEXT>>", "<CONTEXT>"
    ).replace("<<QUESTION>>", "<QUESTION>"))
    if (
        contract.get("locked_for_mem2_plus") is not True
        or template_contract.get("raw_sha256") != raw_sha
        or template_contract.get("canonical_message_template_sha256") != message_sha
        or template_contract.get("path") != TEMPLATE_PATH.relative_to(ROOT).as_posix()
        or template_contract.get("system_message") != system_template
        or template_contract.get("user_message_template") != user_template
        or template_contract.get("v1_canonical_message_template_sha256") != v1_template_sha
    ):
        raise RuntimeError("Final reader contract does not match its frozen template hashes")
    if contract.get("contract_version") != "shared_reader_v3_minimal_current_date":
        raise RuntimeError("Unsupported final reader contract version")
    return contract, contract_sha, system_template, user_template


def build_reader_messages(
    question: str,
    question_date: str,
    serialized_context: str,
    *,
    system_template: str,
    user_template: str,
) -> list[dict[str, str]]:
    if not isinstance(question_date, str) or not question_date:
        raise ValueError("Final shared-reader contract requires the official question_date")
    system = system_template
    user = (
        user_template
        .replace("<<MEMORY_CONTEXT>>", serialized_context)
        .replace("<<QUESTION_DATE>>", question_date)
        .replace("<<QUESTION>>", question)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
