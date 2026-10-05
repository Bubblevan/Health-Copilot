"""Answer-schema parsing and response-side identity checks."""

from __future__ import annotations

import json
import re

from .contracts import AnswerSchema

_ABSTENTION_MARKERS = (
    "insufficient evidence", "cannot determine", "can't determine", "unable to determine",
    "cannot reliably answer", "not enough information", "无法判断", "证据不足", "无法可靠",
)
ANSWER_PARSER_REVISION = "deterministic-mcq-parser-v7"


def parse_answer(text: str, schema: AnswerSchema) -> str | tuple[str, ...] | None:
    answer = _clean_answer(text)
    if not answer:
        return None
    if schema in {AnswerSchema.FREE_TEXT, AnswerSchema.ABSTAINABLE}:
        if schema is AnswerSchema.ABSTAINABLE and any(
            marker in answer.casefold() for marker in _ABSTENTION_MARKERS
        ):
            return None
        return answer
    if schema is AnswerSchema.EXACT_TOKEN:
        match = re.search(r"[A-Za-z0-9_.:-]+", answer)
        return match.group(0) if match else None
    if schema is AnswerSchema.SINGLE_CHOICE:
        json_value = _json_choice_value(answer)
        if json_value is not None:
            return _single_choice_label(json_value)
        candidate = _final_choice_line(answer)
        candidate = _unwrap_markdown(candidate)
        candidate = re.sub(r"[*`]", "", candidate)
        candidate = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", candidate)
        patterns = (
            r"^(?:单选题答案|单选答案|(?:正确|最佳)?(?:答案|选项|选项字母)(?:是)?|多选题答案)\s*[:：=-]?\s*([A-Z])(?:\s*[.)、,:：]|\s|$)",
            r"^(?:single[_ ]choice|final\s+answer)\s*[:：=-]?\s*([A-Z])(?:\s*[.)\]:,]|\s|$)",
            r"^(?:final\s+)?(?:correct\s+)?(?:answer|options?|choices?)\s*[:=-]?\s*([A-Z])(?:\s*[.)\]:,]|\s|$)",
            r"^(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-Z])(?:\b|$)",
            r"^(?:the\s+)?correct\s+option\s+is\s+([A-Z])(?:\b|$)",
            r"^([A-Z])(?:\s*[.)]\s*.*|\s*)$",
        )
        for pattern in patterns:
            match = re.match(pattern, candidate, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        for line in reversed([item for item in answer.splitlines() if item.strip()]):
            trailing = _unwrap_markdown(line)
            match = re.fullmatch(r"([A-Z])\s*[.)]?", trailing, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None
    if schema is AnswerSchema.MULTI_SELECT:
        json_value = _json_choice_value(answer)
        if json_value is not None:
            return _choice_labels(json_value)
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        explicit = next((line for line in reversed(lines) if re.match(
            r"^(?:final\s+)?(?:correct\s+)?(?:answer|options?|choices?|答案|正确答案|多选题答案|选项字母|正确选项|最佳选项)",
            _unwrap_markdown(line), re.IGNORECASE,
        )), None)
        compact = next((line for line in reversed(lines) if re.fullmatch(
            r"\s*(?:[A-Z](?:[A-Z]|\s*[,;/&+]\s*[A-Z])*)(?:\s*[.)])?\s*",
            _unwrap_markdown(line), re.IGNORECASE,
        )), None)
        candidate = _unwrap_markdown(explicit or compact or (lines[0] if lines else ""))
        candidate = re.sub(r"[*`]", "", candidate)
        candidate = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", candidate)
        if explicit is not None:
            enumerated = _enumerated_multi_choice_labels(candidate)
            if enumerated:
                return enumerated
        candidate = re.sub(r"\bAND\b", ",", candidate, flags=re.IGNORECASE)
        candidate = re.sub(
            r"^(?:多选题答案|正确答案|答案|选项字母|正确选项|最佳选项|选项)(?:是)?\s*[:：=]?\s*",
            "", candidate,
        )
        candidate = re.sub(r"[、，；]", ",", candidate)
        match = re.match(
            r"^(?:(?:final\s+)?(?:correct\s+)?(?:answer|options?|choices?)\s*[:=-]?\s*)?"
            r"([A-Z](?:[A-Z]|\s*[,;/&+]\s*[A-Z])*)"
            r"(?:\s*[.)]\s*.*|\s+(?:because|since|as|because of|因为|由于|理由是|解释[:：]?).*)?"
            r"\s*[.;:]?\s*$",
            candidate,
            flags=re.IGNORECASE,
        )
        return tuple(sorted(set(re.findall(r"[A-Z]", match.group(1).upper())))) if match else None
    raise ValueError(f"unsupported answer schema: {schema}")


def _enumerated_multi_choice_labels(text: str) -> tuple[str, ...] | None:
    """Read option labels from an explicit answer line that includes option text.

    For example, ``正确选项：B. 心、C. 肾、D. 脾`` is an answer set, not
    the single label B followed by rationale. Restrict subsequent labels to
    item separators so ordinary capital letters in the option text are ignored.
    """
    value = re.sub(
        r"^(?:final\s+)?(?:correct\s+)?"
        r"(?:answer|options?|choices?|答案|正确答案|多选题答案|选项字母|正确选项|最佳选项)"
        r"\s*[:：=-]?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    labels = [
        match.group(1).upper()
        for match in re.finditer(
            r"(?:^|[,，、;；])\s*([A-E])\s*[.)](?=\s*\S)",
            value,
        )
    ]
    if len(labels) < 2:
        return None
    return tuple(sorted(set(labels)))


def is_explicit_abstention(text: str) -> bool:
    normalized = _clean_answer(text).casefold()
    return any(marker in normalized for marker in _ABSTENTION_MARKERS)


def _clean_answer(text: str) -> str:
    answer = text.strip()
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", answer.strip(), flags=re.IGNORECASE)
    answer = re.sub(r"</?answer>", "", answer, flags=re.IGNORECASE)
    return answer.strip()


def _unwrap_markdown(text: str) -> str:
    value = text.strip()
    for marker in ("**", "__", "*", "_", "`"):
        if value.startswith(marker) and value.endswith(marker) and len(value) >= 2 * len(marker):
            return value[len(marker):-len(marker)].strip()
    return value


def _final_choice_line(answer: str) -> str:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        cleaned = _unwrap_markdown(line)
        if re.match(
            r"^(?:final\s+)?(?:answer|option|choice|correct\s+answer)\b"
            r"|^(?:最终答案|正确答案|答案|选项字母|正确选项|最佳选项|多选题答案|选项)\s*[:：=]",
            cleaned,
            re.IGNORECASE,
        ):
            return line
    return lines[0]


def _json_choice_value(answer: str) -> object | None:
    candidates = [answer]
    match = re.search(r"\[[^\[\]]*\]|\{[^{}]*\}", answer, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, (list, tuple)):
            return value
        if isinstance(value, str) and re.fullmatch(
            r"[A-Z](?:[A-Z]|\s*[,;/&+]\s*[A-Z])*", value.strip(), re.IGNORECASE,
        ):
            return value
        if isinstance(value, dict):
            for key in (
                "answer", "option", "options", "choice", "choices",
                "single_choice", "multi_select", "final_answer",
            ):
                if key in value:
                    return value[key]
    # Recover an unambiguous leading answer field when malformed rationale
    # text makes an otherwise clear JSON object invalid (for example, an
    # unescaped quote later in the rationale string).
    leading_answer = re.match(
        r"^\s*\{\s*[\"'](?:single_choice|answer|option|choice|final_answer)[\"']"
        r"\s*:\s*[\"']([A-Z])[\"']\s*[,}]",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if leading_answer:
        return leading_answer.group(1).upper()
    return None


def _choice_labels(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        normalized = re.sub(r"\bAND\b", ",", value.strip(), flags=re.IGNORECASE)
        normalized = normalized.strip(" [](){}'\"`")
        if not re.fullmatch(r"[A-Z](?:[A-Z]|\s*[,/&+]\s*[A-Z])*", normalized, flags=re.IGNORECASE):
            return ()
        labels = re.findall(r"[A-Z]", normalized.upper())
    elif isinstance(value, (list, tuple)):
        labels = []
        for item in value:
            label = str(item).strip().upper()
            if not re.fullmatch(r"[A-Z]", label):
                return ()
            labels.append(label)
    else:
        return ()
    return tuple(sorted(set(labels)))


def _single_choice_label(value: object) -> str | None:
    labels = _choice_labels(value)
    if len(labels) == 1:
        return labels[0]
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    patterns = (
        r"^(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s+(?:is\s+)?([A-Z])\b",
        r"^([A-Z])\s*[.)\]:：、](?:\s*.*)?$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, candidate, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None
