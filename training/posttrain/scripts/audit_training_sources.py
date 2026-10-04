from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data.canonicalize import CANONICALIZATION_VERSION, canonicalize
from data.decontam import (
    DECONTAMINATION_VERSION,
    EvalPrompt,
    build_long_substring_index,
    build_near_duplicate_index,
    find_long_substring_matches,
    find_near_duplicate_matches,
)

DATA_ROOT = Path(os.environ.get("PT_E0_DATA_ROOT", "/root/gpufree-data/Health-Copilot-PT-E0-data"))
SOURCE_DATA_ROOT = Path(os.environ.get("PT_E0_SOURCE_DATA_ROOT", "/root/gpufree-share/data/posttrain"))
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = DATA_ROOT / "runs/posttrain/pt-e0"
OUT = RUN_DIR / "contamination"
SPLITS = RUN_DIR / "stage_splits"
SFT_DEV_TARGET = 2_000
SPLIT_SEED = 20261004
LINK_BUCKET_RULE = {"sft": [0, 49], "rl_train": [50, 96], "rl_dev": [97, 99]}

SOURCE_FILES = {
    "o1_en": SOURCE_DATA_ROOT / "sft/medical-o1-reasoning-SFT/medical_o1_sft.json",
    "o1_zh": SOURCE_DATA_ROOT / "sft/medical-o1-reasoning-SFT/medical_o1_sft_Chinese.json",
    "medreason": SOURCE_DATA_ROOT / "sft/MedReason/ours_quality_33000.jsonl",
    "huatuo": SOURCE_DATA_ROOT / "sft/Huatuo26M-Lite/format_data.jsonl",
    "rl": SOURCE_DATA_ROOT / "rl/medical-o1-verifiable-problem/medical_o1_verifiable_problem.json",
}
SOURCE_REPOS = {
    "o1_en": ("FreedomIntelligence/medical-o1-reasoning-SFT", "medical_o1_sft.json"),
    "o1_zh": ("FreedomIntelligence/medical-o1-reasoning-SFT", "medical_o1_sft_Chinese.json"),
    "medreason": ("UCSC-VLAA/MedReason", "ours_quality_33000.jsonl"),
    "huatuo": ("FreedomIntelligence/Huatuo26M-Lite", "format_data.jsonl"),
    "rl": ("FreedomIntelligence/medical-o1-verifiable-problem", "medical_o1_verifiable_problem.json"),
}
EVAL_BENCHMARKS = ("diagnosisarena", "cmb", "hbpro", "livemedbench")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return sha256_file(path)


def load_source(source: str, path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array for {source}")
    return value


def row_question(source: str, row: dict[str, Any]) -> str:
    if source.startswith("o1_"):
        return str(row.get("Question", ""))
    if source == "medreason":
        return str(row.get("question", ""))
    if source == "huatuo":
        return str(row.get("question", ""))
    if source == "rl":
        return str(row.get("Open-ended Verifiable Question", ""))
    raise KeyError(source)


def stable_row_id(source: str, row: dict[str, Any], index: int) -> str:
    if source.startswith("o1_"):
        return f"{source}:{index:06d}"
    if source == "medreason":
        return f"medreason:{row.get('id_in_dataset', 'missing')}:{index:06d}"
    if source == "huatuo":
        return f"huatuo:{row.get('id', 'missing')}:{index:06d}"
    return f"rl:{index:06d}"


def read_repo_metadata(repo_id: str) -> dict[str, Any]:
    for path in SOURCE_DATA_ROOT.glob("**/.hfd/repo_metadata.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("id") == repo_id:
            return {
                "revision": data.get("sha"),
                "license": (data.get("cardData") or {}).get("license"),
            }
    raise FileNotFoundError(f"Missing .hfd repo metadata for {repo_id}")


def load_training_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw_counts: dict[str, Any] = {}
    for source, path in SOURCE_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        source_rows = load_source(source, path)
        provenance = Counter()
        seen_ids = set()
        for index, row in enumerate(source_rows):
            question = row_question(source, row)
            canonical = canonicalize(question)
            prompt_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            row_id = stable_row_id(source, row, index)
            if row_id in seen_ids:
                raise ValueError(f"Stable source row ID collision: {row_id}")
            seen_ids.add(row_id)
            dataset_name = str(row.get("dataset_name", "")) if source == "medreason" else None
            if source == "medreason":
                provenance[dataset_name or "<missing>"] += 1
            records.append({
                "source": source,
                "row_id": row_id,
                "original_row_id": row.get("id_in_dataset", row.get("id", index)),
                "language": "en" if source in {"o1_en", "medreason", "rl"} else "zh",
                "raw_text": question,
                "canonical_text": canonical,
                "prompt_sha256": prompt_sha,
                "dataset_name": dataset_name,
                "provenance_medxpert_excluded": source == "medreason" and (dataset_name or "").strip().casefold() == "medxpertqa",
            })
        source_records = [row for row in records if row["source"] == source]
        unique_families = len({row["prompt_sha256"] for row in source_records})
        repo_id, repo_file = SOURCE_REPOS[source]
        raw_counts[source] = {
            "raw_rows": len(source_rows),
            "unique_question_families": unique_families,
            "within_source_duplicate_rows": len(source_rows) - unique_families,
            "within_source_duplicate_rate": (len(source_rows) - unique_families) / len(source_rows) if source_rows else 0.0,
            "raw_file": str(path),
            "raw_file_sha256": sha256_file(path),
            "repo_id": repo_id,
            "repo_file": repo_file,
            "repo_metadata": read_repo_metadata(repo_id),
            "medreason_dataset_name_counts": dict(provenance) if source == "medreason" else None,
        }
    return records, raw_counts


def load_eval_prompts() -> list[EvalPrompt]:
    prompts: list[EvalPrompt] = []
    for benchmark in EVAL_BENCHMARKS:
        candidate_path = DATA_ROOT / "eval/prepared" / benchmark / "candidate_view.jsonl"
        if benchmark == "livemedbench":
            candidate_path = DATA_ROOT / "eval/prepared/livemedbench/reserved1024_candidate_view.jsonl"
        for row in read_jsonl(candidate_path):
            text = row.get("fingerprint_text")
            if not isinstance(text, str):
                if isinstance(row.get("messages"), list):
                    text = "\n".join(str(message.get("content", "")) for message in row["messages"])
                else:
                    text = str(row.get("prompt", ""))
            prompts.append(EvalPrompt(
                key=f"{benchmark}|{row['id']}",
                benchmark=benchmark,
                example_id=str(row["id"]),
                text=canonicalize(text),
            ))
    return prompts


def overlap_matrix(records: list[dict[str, Any]], eval_prompts: list[EvalPrompt]) -> list[dict[str, Any]]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        counters[row["source"]][row["prompt_sha256"]] += 1
    for prompt in eval_prompts:
        prompt_hash = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        counters[prompt.benchmark][prompt_hash] += 1
    sources = list(SOURCE_FILES) + list(EVAL_BENCHMARKS)
    rows = []
    for i, left in enumerate(sources):
        for right in sources[i:]:
            common = set(counters[left]) & set(counters[right])
            if left == right:
                families = len(counters[left])
                raw_pairs = sum(counters[left].values())
            else:
                families = len(common)
                raw_pairs = sum(counters[left][fp] * counters[right][fp] for fp in common)
            rows.append({
                "source_a": left,
                "source_b": right,
                "raw_overlap_count": raw_pairs,
                "unique_prompt_family_count": families,
            })
    return rows


def stage_bucket(prompt_hash: str) -> int:
    return int(prompt_hash[:16], 16) % 100


def linked_family_stage(prompt_hash: str) -> str:
    bucket = stage_bucket(prompt_hash)
    if LINK_BUCKET_RULE["sft"][0] <= bucket <= LINK_BUCKET_RULE["sft"][1]:
        return "sft"
    if LINK_BUCKET_RULE["rl_train"][0] <= bucket <= LINK_BUCKET_RULE["rl_train"][1]:
        return "rl_train"
    if LINK_BUCKET_RULE["rl_dev"][0] <= bucket <= LINK_BUCKET_RULE["rl_dev"][1]:
        return "rl_dev"
    raise ValueError(f"Prompt hash bucket is not assigned to a stage: {bucket}")


def include_rl_stage(stage: str) -> bool:
    return stage in {"rl_train", "rl_dev"}


def select_stratified(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0:
        return []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["language"])].append(row)
    exact = {key: n * len(group) / len(rows) for key, group in groups.items()}
    quotas = {key: min(len(groups[key]), int(exact[key])) for key in groups}
    remaining = n - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - int(exact[key])), key))
    while remaining:
        changed = False
        for key in order:
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
        if not changed:
            raise ValueError("Unable to allocate internal SFT DEV quotas")
    selected = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: hashlib.sha256(f"{seed}|{row['row_id']}".encode()).hexdigest())
        selected.extend(group[:quotas[key]])
    return selected


def main() -> None:
    started = time.time()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)
    records, source_counts = load_training_rows()
    fingerprint_rows = [
        {key: row[key] for key in ("source", "row_id", "original_row_id", "language", "raw_text", "canonical_text", "prompt_sha256", "dataset_name")}
        for row in records
    ]
    fingerprint_sha = write_jsonl(RUN_DIR / "fingerprinted_training_records.jsonl", fingerprint_rows)

    eval_prompts = load_eval_prompts()
    exact_index: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for prompt in eval_prompts:
        prompt_hash = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        exact_index[prompt_hash][prompt.benchmark].append(prompt.example_id)

    near_index, prompts_by_key = build_near_duplicate_index(eval_prompts)
    long_index = build_long_substring_index(eval_prompts)
    unique_text_to_hash = {}
    for row in records:
        unique_text_to_hash.setdefault(row["prompt_sha256"], row["canonical_text"])
    contamination_by_hash: dict[str, dict[str, set[str]]] = {}
    for position, (prompt_hash, text) in enumerate(unique_text_to_hash.items(), start=1):
        exact = {benchmark: set(ids) for benchmark, ids in exact_index.get(prompt_hash, {}).items()}
        near_rows = find_near_duplicate_matches(text, near_index, prompts_by_key, exact_text=text)
        long_rows = find_long_substring_matches(text, long_index, prompts_by_key)
        result: dict[str, set[str]] = defaultdict(set)
        for benchmark, ids in exact.items():
            result[benchmark].update(f"exact:{value}" for value in ids)
        for prompt in near_rows:
            result[prompt.benchmark].add(f"near:{prompt.example_id}")
        for prompt in long_rows:
            result[prompt.benchmark].add(f"substring64:{prompt.example_id}")
        contamination_by_hash[prompt_hash] = result
        if position % 25_000 == 0:
            print(json.dumps({"phase": "decontamination", "unique_training_prompts_checked": position, "total_unique": len(unique_text_to_hash)}), flush=True)

    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions = []
    contamination_counts = {
        source: {benchmark: Counter() for benchmark in EVAL_BENCHMARKS}
        for source in SOURCE_FILES
    }
    for row in records:
        matches = contamination_by_hash[row["prompt_sha256"]]
        provenance_excluded = row["provenance_medxpert_excluded"]
        external_excluded = bool(matches)
        reasons = []
        if not row["canonical_text"]:
            reasons.append("empty_question")
        if provenance_excluded:
            reasons.append("medreason_provenance_medxpertqa")
        if external_excluded:
            reasons.append("external_eval_contamination")
        if reasons:
            exclusions.append({
                "source": row["source"],
                "row_id": row["row_id"],
                "prompt_sha256": row["prompt_sha256"],
                "reasons": reasons,
                "benchmark_match_counts": {
                    benchmark: {
                        "exact": sum(value.startswith("exact:") for value in matches.get(benchmark, set())),
                        "near": sum(value.startswith("near:") for value in matches.get(benchmark, set())),
                        "substring64": sum(value.startswith("substring64:") for value in matches.get(benchmark, set())),
                    }
                    for benchmark in EVAL_BENCHMARKS
                },
            })
        for benchmark in EVAL_BENCHMARKS:
            for key in ("exact", "near", "substring64"):
                contamination_counts[row["source"]][benchmark][key] += int(
                    any(value.startswith(f"{key}:") for value in matches.get(benchmark, set()))
                )
        row["external_excluded"] = external_excluded
        row["provenance_excluded"] = provenance_excluded
        row["empty_question"] = not bool(row["canonical_text"])
        rows_by_source[row["source"]].append(row)

    matrix = overlap_matrix(records, eval_prompts)
    sft_sources_for_dedup = {"o1_en", "o1_zh", "medreason", "huatuo"}
    raw_sft_family_sources: dict[str, set[str]] = defaultdict(set)
    for row in records:
        if row["source"] in sft_sources_for_dedup:
            raw_sft_family_sources[row["prompt_sha256"]].add(row["source"])
    raw_sft_union_families = set(raw_sft_family_sources)
    cross_sft_duplicate_families = {family for family, sources in raw_sft_family_sources.items() if len(sources) > 1}
    cross_sft_raw_row_pairs = sum(
        int(row["raw_overlap_count"])
        for row in matrix
        if row["source_a"] in sft_sources_for_dedup
        and row["source_b"] in sft_sources_for_dedup
        and row["source_a"] != row["source_b"]
    )
    cross_sft_duplicate_rate = len(cross_sft_duplicate_families) / len(raw_sft_union_families) if raw_sft_union_families else 0.0
    exact_dedup_summary = {
        "cross_sft_duplicate_unique_prompt_families": len(cross_sft_duplicate_families),
        "cross_sft_unique_prompt_family_denominator": len(raw_sft_union_families),
        "cross_sft_duplicate_family_rate": cross_sft_duplicate_rate,
        "cross_sft_raw_matching_row_pairs": cross_sft_raw_row_pairs,
    }
    with (OUT / "overlap_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_a", "source_b", "raw_overlap_count", "unique_prompt_family_count"])
        writer.writeheader()
        writer.writerows(matrix)

    unique_by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for source, source_rows in rows_by_source.items():
        unique_by_source[source] = {}
        for row in source_rows:
            if row["external_excluded"] or row["provenance_excluded"] or row["empty_question"]:
                continue
            unique_by_source[source].setdefault(row["prompt_sha256"], row)

    o1_sources = {"o1_en", "o1_zh"}
    sft_sources = ("o1_en", "o1_zh", "medreason", "huatuo")
    all_o1 = set(unique_by_source["o1_en"]) | set(unique_by_source["o1_zh"])
    all_sft = set().union(*(set(unique_by_source[source]) for source in sft_sources))
    all_rl = set(unique_by_source["rl"])
    linked_clean = all_sft & all_rl
    linked_o1_clean = all_o1 & all_rl
    raw_o1 = {row["prompt_sha256"] for row in records if row["source"] in o1_sources}
    raw_sft = {row["prompt_sha256"] for row in records if row["source"] in sft_sources}
    raw_rl = {row["prompt_sha256"] for row in records if row["source"] == "rl"}
    linked_raw_o1 = raw_o1 & raw_rl
    linked_raw_sft = raw_sft & raw_rl
    unmatched_o1_raw = raw_o1 - raw_rl
    unmatched_sft_raw = raw_sft - raw_rl

    sft_stage: dict[str, str] = {}
    rl_stage: dict[str, str] = {}
    for prompt_hash in linked_clean:
        assigned_stage = linked_family_stage(prompt_hash)
        if assigned_stage == "sft":
            sft_stage[prompt_hash] = "sft"
            rl_stage[prompt_hash] = "withheld_sft_family"
        else:
            sft_stage[prompt_hash] = "withheld_rl_family"
            rl_stage[prompt_hash] = assigned_stage

    sft_candidates = []
    for source in sft_sources:
        for prompt_hash, row in unique_by_source[source].items():
            if prompt_hash not in linked_clean or sft_stage.get(prompt_hash) == "sft":
                sft_candidates.append(row)

    source_rank = {"o1_en": 0, "o1_zh": 1, "medreason": 2, "huatuo": 3}
    sft_candidates.sort(key=lambda row: (source_rank[row["source"]], row["row_id"]))
    sft_by_hash: dict[str, dict[str, Any]] = {}
    for row in sft_candidates:
        sft_by_hash.setdefault(row["prompt_sha256"], row)
    sft_pool = list(sft_by_hash.values())

    rl_train = []
    rl_dev = []
    for prompt_hash, row in unique_by_source["rl"].items():
        stage = rl_stage.get(prompt_hash)
        if stage is None:
            stage = "rl_dev" if stage_bucket(prompt_hash) >= LINK_BUCKET_RULE["rl_dev"][0] else "rl_train"
        if not include_rl_stage(stage):
            continue
        (rl_dev if stage == "rl_dev" else rl_train).append(row)

    sft_families = {row["prompt_sha256"] for row in sft_pool}
    rl_train_families = {row["prompt_sha256"] for row in rl_train}
    rl_dev_families = {row["prompt_sha256"] for row in rl_dev}
    if sft_families & (rl_train_families | rl_dev_families):
        raise RuntimeError("SFT and RL candidate prompt families are not disjoint")

    dev_n = min(SFT_DEV_TARGET, max(0, len(sft_pool) - 1))
    sft_dev = select_stratified(sft_pool, dev_n, SPLIT_SEED)
    sft_dev_ids = {row["row_id"] for row in sft_dev}
    sft_train = [row for row in sft_pool if row["row_id"] not in sft_dev_ids]

    split_values = {
        "sft_train_ids.json": [row["row_id"] for row in sorted(sft_train, key=lambda row: row["row_id"])],
        "sft_dev_ids.json": [row["row_id"] for row in sorted(sft_dev, key=lambda row: row["row_id"])],
        "rl_train_ids.json": [row["row_id"] for row in sorted(rl_train, key=lambda row: row["row_id"])],
        "rl_dev_ids.json": [row["row_id"] for row in sorted(rl_dev, key=lambda row: row["row_id"])],
    }
    split_hashes = {name: write_json(SPLITS / name, values) for name, values in split_values.items()}
    split_manifest = {
        "schema_version": "pt-e0-stage-splits-v1",
        "seed": SPLIT_SEED,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "canonicalizer_sha256": sha256_file(Path(__file__).resolve().parents[1] / "data/canonicalize.py"),
        "link_rule": {
            "hash": "uint64(first 16 hex chars of SHA256(question-only canonical prompt)) modulo 100",
            "buckets": LINK_BUCKET_RULE,
            "reason_for_adjustment": "3% RL dev hash space targets about 1.2k clean prompts after all-source SFT/RL family separation; frozen before any model scoring or training.",
        },
        "sft_internal_dev": {"target": SFT_DEV_TARGET, "actual": len(sft_dev), "stratification": ["source", "language"]},
        "counts": {
            "medical_o1_linked_families_raw": len(linked_raw_o1),
            "medical_o1_linked_families_clean": len(linked_o1_clean),
            "all_sft_rl_linked_families_raw": len(linked_raw_sft),
            "all_sft_rl_linked_families_clean": len(linked_clean),
            "medical_o1_unmatched_sft_prompt_families_raw": len(unmatched_o1_raw),
            "all_sft_unmatched_prompt_families_raw": len(unmatched_sft_raw),
            "sft_rl_candidate_family_overlap": len(sft_families & (rl_train_families | rl_dev_families)),
            "sft_candidate_families_before_dev": len(sft_pool),
            "sft_train": len(sft_train),
            "sft_dev": len(sft_dev),
            "rl_train": len(rl_train),
            "rl_dev": len(rl_dev),
            "rl_clean_unique_before_stage": len(unique_by_source["rl"]),
        },
        "file_sha256": split_hashes,
    }
    write_json(SPLITS / "split_manifest.json", split_manifest)

    provenance_exclusions = {
        "MedXpertQA": sum(row["source"] == "medreason" and row["provenance_medxpert_excluded"] for row in records),
        "MedQA": sum(row["source"] == "medreason" and (row.get("dataset_name") or "").strip().casefold() == "medqa" for row in records),
        "MedMCQA": sum(row["source"] == "medreason" and (row.get("dataset_name") or "").strip().casefold() == "medmcqa" for row in records),
        "PubMedQA": sum(row["source"] == "medreason" and (row.get("dataset_name") or "").strip().casefold().startswith("pubmedqa") for row in records),
        "MMLU": sum(row["source"] == "medreason" and (row.get("dataset_name") or "").strip().casefold() == "mmlu" for row in records),
    }
    count_payload = {
        "schema_version": "pt-e0-source-audit-v1",
        "canonicalization_version": CANONICALIZATION_VERSION,
        "canonicalizer_sha256": sha256_file(Path(__file__).resolve().parents[1] / "data/canonicalize.py"),
        "decontamination_version": DECONTAMINATION_VERSION,
        "decontamination_code_sha256": sha256_file(Path(__file__).resolve().parents[1] / "data/decontam.py"),
        "thresholds": {"char_ngram": 5, "minhash_permutations": 64, "lsh_threshold": 0.25, "rapidfuzz_token_set_ratio": 95, "long_substring_chars": 64},
        "raw_source_counts": source_counts,
        "exact_deduplication": exact_dedup_summary,
        "provenance_exclusions": provenance_exclusions,
        "external_contamination_by_source_benchmark": {
            source: {benchmark: dict(values) for benchmark, values in by_benchmark.items()}
            for source, by_benchmark in contamination_counts.items()
        },
        "overlap_matrix_sha256": sha256_file(OUT / "overlap_matrix.csv"),
        "fingerprinted_training_records_sha256": fingerprint_sha,
        "split_manifest": split_manifest,
        "external_eval_prompt_counts": dict(Counter(row.benchmark for row in eval_prompts)),
        "wall_time_seconds": round(time.time() - started, 2),
    }
    write_json(OUT / "source_counts.json", count_payload)
    write_jsonl(OUT / "exclusions.jsonl", exclusions)

    clean_counts = {
        source: {
            "clean_unique_families": len(unique_by_source[source]),
            "raw_rows_removed_by_external_eval": sum(row["source"] == source and row["external_excluded"] for row in records),
            "raw_rows_removed_by_medxpert_provenance": sum(row["source"] == source and row["provenance_excluded"] for row in records),
            "raw_rows_removed_empty_question": sum(row["source"] == source and row["empty_question"] for row in records),
        }
        for source in SOURCE_FILES
    }
    write_json(OUT / "clean_source_counts.json", clean_counts)

    report_lines = [
        "# PT-E0 Training Source Contamination Audit",
        "",
        "This report contains counts and stable IDs only. It does not print HealthBench Professional or LiveMedBench prompts or rubric text.",
        "",
        "## Frozen raw source counts and exact dedup",
        "",
        "| Source | Raw rows | Unique question families | Within-source duplicate rows | Within-source duplicate rate | Clean unique families |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in SOURCE_FILES:
        raw = source_counts[source]
        clean = clean_counts[source]
        report_lines.append(f"| {source} | {raw['raw_rows']} | {raw['unique_question_families']} | {raw['within_source_duplicate_rows']} | {raw['within_source_duplicate_rate']:.4%} | {clean['clean_unique_families']} |")
    report_lines.extend([
        "",
        "## Exact cross-source overlap",
        "",
        "| Source A | Source B | Raw matching row pairs | Unique prompt families |",
        "|---|---|---:|---:|",
    ])
    for row in matrix:
        if row["source_a"] != row["source_b"] and row["raw_overlap_count"]:
            report_lines.append(f"| {row['source_a']} | {row['source_b']} | {row['raw_overlap_count']} | {row['unique_prompt_family_count']} |")
    report_lines.extend([
        "",
        f"Cross-SFT exact duplicate prompt families: **{exact_dedup_summary['cross_sft_duplicate_unique_prompt_families']} / {exact_dedup_summary['cross_sft_unique_prompt_family_denominator']} union-unique SFT families ({exact_dedup_summary['cross_sft_duplicate_family_rate']:.4%})**; raw matching cross-SFT row pairs: **{exact_dedup_summary['cross_sft_raw_matching_row_pairs']}**.",
        "The rate denominator is the union of exact-unique raw prompt families across the four SFT sources; a prompt duplicated across more than two sources is counted once in the numerator.",
        "",
        "## External evaluation contamination removed",
        "",
        "| Training source | Benchmark | Exact rows | Near-duplicate rows | 64-character overlap rows |",
        "|---|---|---:|---:|---:|",
    ])
    for source in SOURCE_FILES:
        for benchmark in EVAL_BENCHMARKS:
            values = contamination_counts[source][benchmark]
            report_lines.append(f"| {source} | {benchmark} | {values['exact']} | {values['near']} | {values['substring64']} |")
    report_lines.extend([
        "",
        "## MedReason provenance",
        "",
        "| dataset_name | Rows |",
        "|---|---:|",
    ])
    for name, count in sorted(source_counts["medreason"]["medreason_dataset_name_counts"].items()):
        report_lines.append(f"| {name} | {count} |")
    report_lines.extend([
        f"Rows excluded unconditionally because dataset_name == MedXpertQA: **{provenance_exclusions['MedXpertQA']}**.",
        f"Recorded provenance counts for MedQA / MedMCQA / PubMedQA / MMLU: **{provenance_exclusions['MedQA']} / {provenance_exclusions['MedMCQA']} / {provenance_exclusions['PubMedQA']} / {provenance_exclusions['MMLU']}**.",
        "",
        "## SFT/RL prompt-family separation",
        "",
        f"- Raw shared families across all SFT sources and RL: **{len(linked_raw_sft)}**.",
        f"- Clean shared families assigned once across stages: **{len(linked_clean)}**.",
        f"- Raw shared medical-o1 families: **{len(linked_raw_o1)}**; clean: **{len(linked_o1_clean)}**.",
        f"- Raw unmatched medical-o1 SFT-only families: **{len(unmatched_o1_raw)}**; all-SFT unmatched families: **{len(unmatched_sft_raw)}**.",
        f"- Clean medical-o1 shared families assigned to SFT: **{sum(sft_stage.get(value) == 'sft' for value in linked_o1_clean)}**; RL train: **{sum(rl_stage.get(value) == 'rl_train' for value in linked_o1_clean)}**; RL dev: **{sum(rl_stage.get(value) == 'rl_dev' for value in linked_o1_clean)}**.",
        "- Frozen buckets: SFT 0–49, RL train 50–96, RL dev 97–99 from first 16 SHA256 hex digits modulo 100. This is applied to every exact SFT/RL shared prompt family, including MedReason and Huatuo rows.",
        f"- Candidate SFT/RL prompt-family overlap after splitting: **{len(sft_families & (rl_train_families | rl_dev_families))}**.",
        "",
        "## Final clean candidates",
        "",
        f"- SFT candidate families before internal DEV: **{len(sft_pool)}**.",
        f"- SFT internal DEV: **{len(sft_dev)}**; SFT train candidates: **{len(sft_train)}**.",
        f"- RL train candidates: **{len(rl_train)}**; RL internal DEV: **{len(rl_dev)}**.",
        "",
        "Exact source row IDs and split lists are stored in the ignored PT-E0 run directory. No SFT or RL weights were updated in PT-E0.",
    ])
    report_path = REPO_ROOT / "docs/research/posttrain/pt_e0_contamination_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    training_manifest = {
        "schema_version": "pt-e0-training-source-manifest-v1",
        "sources": source_counts,
        "source_counts_sha256": sha256_file(OUT / "source_counts.json"),
        "clean_source_counts_sha256": sha256_file(OUT / "clean_source_counts.json"),
        "exclusions_sha256": sha256_file(OUT / "exclusions.jsonl"),
        "overlap_matrix_sha256": sha256_file(OUT / "overlap_matrix.csv"),
        "stage_split_manifest_sha256": sha256_file(SPLITS / "split_manifest.json"),
        "report_sha256": sha256_file(report_path),
    }
    write_json(REPO_ROOT / "training/posttrain/manifests/training_sources/source_manifest.json", training_manifest)
    print(json.dumps({
        "raw_source_counts": {source: value["raw_rows"] for source, value in source_counts.items()},
        "medical_o1_linked_raw_families": len(linked_raw_o1),
        "medical_o1_linked_clean_families": len(linked_o1_clean),
        "all_sft_rl_linked_raw_families": len(linked_raw_sft),
        "all_sft_rl_linked_clean_families": len(linked_clean),
        "sft_rl_candidate_family_overlap": len(sft_families & (rl_train_families | rl_dev_families)),
        "sft_train": len(sft_train),
        "sft_dev": len(sft_dev),
        "rl_train": len(rl_train),
        "rl_dev": len(rl_dev),
        "contamination_report": str(report_path),
        "elapsed_seconds": round(time.time() - started, 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
