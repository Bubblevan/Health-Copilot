from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz

CHAR_NGRAM_SIZE = 5
MINHASH_PERMUTATIONS = 64
MINHASH_LSH_THRESHOLD = 0.25
TOKEN_SET_RATIO_THRESHOLD = 95
LONG_SUBSTRING_CHARS = 64
LONG_INDEX_SHINGLE_CHARS = 32
LONG_INDEX_STRIDE = 32
DECONTAMINATION_VERSION = "pt-e0-char5-minhash64-lsh025-token-set95-substring64-v1"


@dataclass(frozen=True)
class EvalPrompt:
    key: str
    benchmark: str
    example_id: str
    text: str


def char_minhash(text: str) -> MinHash:
    signature = MinHash(num_perm=MINHASH_PERMUTATIONS)
    if len(text) < CHAR_NGRAM_SIZE:
        if text:
            signature.update(text.encode("utf-8"))
        return signature
    shingles = {text[index:index + CHAR_NGRAM_SIZE].encode("utf-8") for index in range(len(text) - CHAR_NGRAM_SIZE + 1)}
    signature.update_batch(list(shingles))
    return signature


def build_near_duplicate_index(prompts: Iterable[EvalPrompt]) -> tuple[MinHashLSH, dict[str, EvalPrompt]]:
    index = MinHashLSH(threshold=MINHASH_LSH_THRESHOLD, num_perm=MINHASH_PERMUTATIONS)
    records: dict[str, EvalPrompt] = {}
    for prompt in prompts:
        if not prompt.text:
            continue
        if prompt.key in records:
            continue
        records[prompt.key] = prompt
        index.insert(prompt.key, char_minhash(prompt.text))
    return index, records


def find_near_duplicate_matches(
    text: str,
    index: MinHashLSH,
    prompts_by_key: dict[str, EvalPrompt],
    *,
    exact_text: str | None = None,
) -> list[EvalPrompt]:
    if not text or len(text) < CHAR_NGRAM_SIZE:
        return []
    candidates = index.query(char_minhash(text))
    matches = []
    for key in candidates:
        prompt = prompts_by_key[key]
        if exact_text is not None and prompt.text == exact_text:
            continue
        if fuzz.token_set_ratio(text, prompt.text) >= TOKEN_SET_RATIO_THRESHOLD:
            matches.append(prompt)
    return sorted(matches, key=lambda row: row.key)


def build_long_substring_index(prompts: Iterable[EvalPrompt]) -> dict[str, list[tuple[str, int]]]:
    index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    size = LONG_INDEX_SHINGLE_CHARS
    for prompt in prompts:
        text = prompt.text
        if len(text) < LONG_SUBSTRING_CHARS:
            continue
        positions = list(range(0, len(text) - size + 1, LONG_INDEX_STRIDE))
        final_position = len(text) - size
        if positions[-1] != final_position:
            positions.append(final_position)
        for position in positions:
            index[text[position:position + size]].append((prompt.key, position))
    return dict(index)


def _common_run_at(text_a: str, position_a: int, text_b: str, position_b: int, seed_size: int) -> int:
    left = 0
    while position_a - left - 1 >= 0 and position_b - left - 1 >= 0 and text_a[position_a - left - 1] == text_b[position_b - left - 1]:
        left += 1
    right = seed_size
    while position_a + right < len(text_a) and position_b + right < len(text_b) and text_a[position_a + right] == text_b[position_b + right]:
        right += 1
    return left + right


def find_long_substring_matches(
    text: str,
    index: dict[str, list[tuple[str, int]]],
    prompts_by_key: dict[str, EvalPrompt],
) -> list[EvalPrompt]:
    size = LONG_INDEX_SHINGLE_CHARS
    if len(text) < LONG_SUBSTRING_CHARS:
        return []
    found: dict[str, EvalPrompt] = {}
    seen_seed_pairs: set[tuple[int, str, int]] = set()
    for train_pos in range(0, len(text) - size + 1):
        seed = text[train_pos:train_pos + size]
        for prompt_key, eval_pos in index.get(seed, ()):
            pair = (train_pos, prompt_key, eval_pos)
            if pair in seen_seed_pairs:
                continue
            seen_seed_pairs.add(pair)
            prompt = prompts_by_key[prompt_key]
            if _common_run_at(text, train_pos, prompt.text, eval_pos, size) >= LONG_SUBSTRING_CHARS:
                found[prompt_key] = prompt
    return [found[key] for key in sorted(found)]
