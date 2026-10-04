from __future__ import annotations

import re

_FINAL_CUE = re.compile(
    r"(?:final\s+(?:answer|choice|option)|answer\s*(?:is|:|=)|choice\s*(?:is|:|=)|option\s*(?:is|:|=)|"
    r"最终答案|最终选择|答案\s*(?:是|为|[:：=])|应选|故选|选择\s*(?:是|为|[:：=]))",
    re.IGNORECASE,
)
_OPTION_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-F])(?![A-Za-z0-9])", re.IGNORECASE)
_COMPACT_OPTIONS = re.compile(r"(?<![A-Za-z0-9])([A-F]{2,6})(?![A-Za-z0-9])", re.IGNORECASE)


def final_response(raw_generation: str) -> str:
    """Remove Qwen thinking blocks while preserving raw generation elsewhere."""
    text = str(raw_generation).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return re.sub(r"<\|(?:im_end|endoftext)\|>\s*$", "", text).strip()


def extract_mcq_answer(raw_generation: str, valid_options: set[str] | list[str] | tuple[str, ...], *, allow_multiple: bool = False) -> str | None:
    """Extract the final selected label(s), favoring the last explicit answer cue."""
    valid = {str(option).strip().upper() for option in valid_options}
    if not valid:
        return None
    text = final_response(raw_generation)
    cues = list(_FINAL_CUE.finditer(text))
    segment = text[cues[-1].end():] if cues else text
    segment = segment[-240:]
    if allow_multiple:
        compact_groups = list(_COMPACT_OPTIONS.finditer(segment))
        if compact_groups:
            group = compact_groups[-1].group(1).upper()
            if len(group) > 1 and set(group).issubset(valid):
                return "".join(sorted(set(group)))
        tokens = [m.group(1).upper() for m in _OPTION_TOKEN.finditer(segment) if m.group(1).upper() in valid]
        if not tokens:
            return None
        boundary = max(segment.rfind("。"), segment.rfind("."), segment.rfind("\n"), segment.rfind(";"), segment.rfind("；"))
        tail = segment[boundary + 1:]
        tail_tokens = [m.group(1).upper() for m in _OPTION_TOKEN.finditer(tail) if m.group(1).upper() in valid]
        return "".join(sorted(set(tail_tokens or tokens[-1:])))
    tokens = [m.group(1).upper() for m in _OPTION_TOKEN.finditer(segment) if m.group(1).upper() in valid]
    return tokens[-1] if tokens else None
