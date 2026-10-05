"""Gold-blind exact and lexical-overlap audit for Common KB vs frozen eval prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIEWS = {
    "cmb_common1024": Path(
        "/root/gpufree-data/Health-Copilot-PT-E0-data/eval/prepared/"
        "cmb_common1024/candidate_view.jsonl"
    ),
    "diagnosisarena915": Path(
        "/root/gpufree-data/Health-Copilot-PT-E0-data/eval/prepared/"
        "diagnosisarena/candidate_view.jsonl"
    ),
}
DEFAULT_ID_MANIFESTS = {
    "cmb_common1024": ROOT / "configs/eval/cmb_common1024_ids.json",
    "diagnosisarena915": ROOT / "configs/eval/diagnosisarena915_ids.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in normalized if char.isalnum() or "\u4e00" <= char <= "\u9fff")


def _grams(text: str, width: int = 5) -> set[str]:
    normalized = _normalize(text)
    if len(normalized) < width:
        return set()
    return {normalized[index : index + width] for index in range(len(normalized) - width + 1)}


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit() -> dict:
    kb = json.loads((ROOT / "configs/eval/common_medical_kb_v1.json").read_text(encoding="utf-8"))
    source_dir = ROOT / "data/knowledge_cards"
    corpus = []
    for item in kb["source_inventory"]:
        row = json.loads((source_dir / item["path"]).read_text(encoding="utf-8"))
        corpus.append((item["id"], row["title"] + "\n" + row["content"]))
    normalized_docs = {
        source_id: _normalize(text) for source_id, text in corpus
    }
    doc_grams = {source_id: _grams(text) for source_id, text in corpus}

    datasets = {}
    exact_pairs = []
    lexical_candidates = []
    for dataset_id, candidate_path in DEFAULT_VIEWS.items():
        rows = _rows(candidate_path)
        ids_manifest = json.loads(DEFAULT_ID_MANIFESTS[dataset_id].read_text(encoding="utf-8"))
        if isinstance(ids_manifest, list):
            expected_ids = ids_manifest
        else:
            expected_ids = ids_manifest.get("case_ids", ids_manifest.get("ids"))
        actual_ids = [row["id"] for row in rows]
        if (
            expected_ids is None
            or len(actual_ids) != len(set(actual_ids))
            or set(actual_ids) != set(expected_ids)
        ):
            raise ValueError(f"candidate IDs do not match the frozen manifest for {dataset_id}")
        query_fields = 0
        for case in rows:
            strings = [value for value in (case.get("prompt"), case.get("fingerprint_text")) if isinstance(value, str)]
            if not strings:
                continue
            query_fields += len(strings)
            normalized_queries = [_normalize(value) for value in strings]
            max_ngram_query = max(strings, key=len)
            query_grams = _grams(max_ngram_query)
            for source_id, doc_text in normalized_docs.items():
                if any(
                    query and (query in doc_text or doc_text in query)
                    for query in normalized_queries
                ):
                    exact_pairs.append({"dataset_id": dataset_id, "case_id": case["id"], "source_id": source_id})
                grams = doc_grams[source_id]
                if len(query_grams) >= 10 and len(grams) >= 10:
                    containment = len(query_grams & grams) / min(len(query_grams), len(grams))
                    if containment >= 0.95:
                        lexical_candidates.append(
                            {
                                "dataset_id": dataset_id,
                                "case_id": case["id"],
                                "source_id": source_id,
                                "fivegram_containment_of_shorter": round(containment, 6),
                            }
                        )
        datasets[dataset_id] = {
            "candidate_view_path": str(candidate_path),
            "candidate_view_sha256": _sha256(candidate_path),
            "candidate_row_count": len(rows),
            "id_manifest_path": str(DEFAULT_ID_MANIFESTS[dataset_id]),
            "id_manifest_sha256": _sha256(DEFAULT_ID_MANIFESTS[dataset_id]),
            "candidate_ids_match_manifest": True,
            "runtime_candidate_text_fields_checked": query_fields,
        }
    return {
        "schema_version": "common-kb-eval-overlap-audit-v1",
        "gold_opened": False,
        "scorer_view_opened": False,
        "source_corpus_id": kb["corpus_id"],
        "source_corpus_sha256": kb["corpus_sha256"],
        "source_document_count": len(corpus),
        "normalization": "Unicode NFKC + casefold + alphanumeric/CJK only",
        "exact_test": "normalized full prompt/fingerprint containment in normalized source-card title+content",
        "near_duplicate_candidate_test": {
            "feature": "set of five-character shingles",
            "threshold": 0.95,
            "metric": "intersection divided by smaller shingle-set size",
            "interpretation": "candidate discovery only; requires gold-blind text review",
        },
        "datasets": datasets,
        "exact_overlap_pair_count": len(exact_pairs),
        "exact_overlap_pairs": exact_pairs,
        "near_duplicate_candidate_count": len(lexical_candidates),
        "near_duplicate_candidates": lexical_candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/common_eval/harness-v1-base-20261005/common-kb-v1-qualification/overlap_audit.json",
    )
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "datasets": {
                    key: value["candidate_row_count"] for key, value in result["datasets"].items()
                },
                "exact_overlap_pair_count": result["exact_overlap_pair_count"],
                "near_duplicate_candidate_count": result["near_duplicate_candidate_count"],
                "gold_opened": result["gold_opened"],
                "scorer_view_opened": result["scorer_view_opened"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
