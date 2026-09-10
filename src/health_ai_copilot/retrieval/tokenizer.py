"""Small Chinese-first tokenizer used by the M0 BM25 baseline."""

import re
import unicodedata

import jieba

_MEANINGFUL_TOKEN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def tokenize(text: str) -> list[str]:
    """Segment Chinese text while retaining meaningful ASCII/alphanumeric terms."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return [
        token.strip()
        for token in jieba.lcut(normalized, cut_all=False)
        if token.strip() and _MEANINGFUL_TOKEN.search(token)
    ]
