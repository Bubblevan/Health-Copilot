from __future__ import annotations

import hashlib
import re
import unicodedata

CANONICALIZATION_VERSION = "pt-e0-nfkc-ws-punctuation-latin-lower-v1"
_PUNCTUATION = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-", "−": "-"})
_OPTION_PREFIX = re.compile(r"^\s*([A-Fa-f])\s*[.、:：)）]\s*")


def canonicalize(text: str) -> str:
    """Normalize text for exact fingerprints without semantic rewriting."""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n").translate(_PUNCTUATION)
    value = "\n".join(" ".join(line.split()) for line in value.split("\n"))
    value = re.sub(r"\n+", "\n", value).strip()
    return value.lower()


def canonicalize_options(text: str) -> str:
    """Canonicalize an MCQ prompt, including option labels and spacing."""
    lines = []
    for line in unicodedata.normalize("NFKC", str(text)).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        compact = " ".join(line.translate(_PUNCTUATION).split())
        match = _OPTION_PREFIX.match(compact)
        if match:
            compact = f"{match.group(1).upper()}. {compact[match.end():].strip()}"
        lines.append(compact)
    return canonicalize("\n".join(lines))


def fingerprint(text: str, *, options: bool = False) -> str:
    canonical = canonicalize_options(text) if options else canonicalize(text)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
