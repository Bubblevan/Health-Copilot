"""Print corpus counts and stable samples for quick human inspection."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter

from _common import DATA_ROOT, REPO_ROOT, load_jsonl

sys.path.insert(0, str(REPO_ROOT / "src"))

from health_ai_copilot.huiyi.validation import validate_corpus


def main() -> int:
    stats = validate_corpus(DATA_ROOT)
    documents = load_jsonl(DATA_ROOT / "normalized" / "documents.jsonl")
    chunks = load_jsonl(DATA_ROOT / "chunks" / "chunks.jsonl")
    corpus_identity = json.loads((DATA_ROOT / "index" / "corpus_manifest.json").read_text(encoding="utf-8"))
    index_manifest_path = DATA_ROOT / "index" / "index_manifest.json"
    index_identity = json.loads(index_manifest_path.read_text(encoding="utf-8")) if index_manifest_path.is_file() else {}
    sample = sorted(chunks, key=lambda row: hashlib.sha256(row["chunk_id"].encode()).hexdigest())[:8]
    output = {
        **stats,
        "document_types": dict(Counter(row["document_type"] for row in documents)),
        "review_statuses": dict(Counter(row["review_status"] for row in documents)),
        "freshness_classes": dict(Counter(row["freshness_class"] for row in documents)),
        "corpus_identity_sha256": corpus_identity.get("corpus_identity_sha256"),
        "index_identity_sha256": index_identity.get("index_identity_sha256"),
        "deterministic_chunk_samples": [
            {key: row[key] for key in ("chunk_id", "source_id", "document_type", "primary_topic", "topics", "title", "section_path", "text", "source_url")}
            for row in sample
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
