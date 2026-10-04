from __future__ import annotations

from typing import Any

# No custom system message is injected into candidate context.
SYSTEM_PROMPT: str | None = None
SYSTEM_PROMPT_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _options_text(options: Any) -> str:
    if isinstance(options, dict):
        return "\n".join(f"{str(key).strip()}. {value}" for key, value in options.items() if value is not None and str(value).strip())
    if isinstance(options, list):
        rows = []
        for index, item in enumerate(options):
            if isinstance(item, dict):
                label = item.get("label", item.get("key", chr(65 + index)))
                value = item.get("text", item.get("value", ""))
            else:
                label, value = chr(65 + index), item
            rows.append(f"{label}. {value}")
        return "\n".join(rows)
    return str(options or "")


def diagnosisarena_prompt(row: dict[str, Any]) -> str:
    pieces = [
        str(row.get("Case Information", "")).strip(),
        str(row.get("Physical Examination", "")).strip(),
        str(row.get("Diagnostic Tests", "")).strip(),
        _options_text(row.get("Options")),
        "Select the single best option. Give the final option label.",
    ]
    return "\n\n".join(piece for piece in pieces if piece)


def cmb_prompt(row: dict[str, Any]) -> str:
    pieces = [str(row.get("question", "")).strip(), _options_text(row.get("option"))]
    instruction = "请选择所有正确选项，并在最后给出选项字母。" if row.get("question_type") == "多项选择题" else "请选择一个最佳选项，并在最后给出选项字母。"
    pieces.append(instruction)
    return "\n\n".join(piece for piece in pieces if piece)


def healthbench_messages(conversation: Any) -> list[dict[str, str]]:
    """Represent only the HealthBench conversation as a user-visible transcript."""
    if isinstance(conversation, dict) and isinstance(conversation.get("messages"), list):
        conversation = conversation["messages"]
    if isinstance(conversation, str):
        return [{"role": "user", "content": conversation}]
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("HealthBench conversation must be a non-empty string or message list")
    turns = []
    for item in conversation:
        if not isinstance(item, dict):
            raise ValueError("HealthBench conversation entries must be objects")
        role = str(item.get("role", item.get("speaker", "user"))).lower()
        label = "Assistant" if role in {"assistant", "doctor", "bot"} else "User"
        content = item.get("content", item.get("message", item.get("text", "")))
        if not isinstance(content, str):
            raise ValueError("HealthBench conversation content must be text")
        turns.append(f"{label}: {content}")
    return [{"role": "user", "content": "\n\n".join(turns)}]


def livemedbench_prompt(row: dict[str, Any]) -> str:
    """Candidate-view only: narrative and core request; excludes advice and rubric."""
    return "\n\n".join(part for part in (str(row.get("narrative", "")).strip(), str(row.get("core_request", "")).strip()) if part)
