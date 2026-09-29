"""Build the Lucene-analyzed LuceneBM25Model indexes without loading PyTorch."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_ROOT = Path("D:/MyLab/Jianli/external/rag_e5")
DEFAULT_JAVA_HOME = Path("D:/jdk-21.0.4")
DEFAULT_PYSERINI_ROOT = Path("E:/Health-Copilot-Models/cache/python-packages/pyserini")
BM25_K1 = 0.9
BM25_B = 0.4
BM25_DEPTH = 100
GUIDELINE_SMOKE_QUERIES = (
    ("Guideline for the pharmacological treatment of hypertension in adults", "who-hypertension-pharmacological-2021"),
    ("blood pressure threshold for initiation of pharmacological treatment", "who-hypertension-pharmacological-2021"),
    ("laboratory testing before and during pharmacological treatment", "who-hypertension-pharmacological-2021"),
    ("first-line drug classes for adults with hypertension", "who-hypertension-pharmacological-2021"),
    ("WHO guidelines on physical activity and sedentary behaviour", "who-physical-activity-sedentary-2020"),
    ("physical activity recommendation for adults aged 18 to 64", "who-physical-activity-sedentary-2020"),
    ("sedentary behaviour recommendation for older adults aged 65 and older", "who-physical-activity-sedentary-2020"),
    ("physical activity recommendations for pregnant and postpartum women", "who-physical-activity-sedentary-2020"),
    ("physical activity recommendation for adults living with disability", "who-physical-activity-sedentary-2020"),
    ("Total fat intake for the prevention of unhealthy weight gain in adults and children", "who-total-fat-weight-gain-2023"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any) -> str:
    data = _canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256(data)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _tokenize_lucene(
    rows: list[tuple[str, str]], *, java_home: Path, pyserini_root: Path, java_source: Path, build_root: Path
) -> tuple[list[list[str]], dict[str, str]]:
    jar = pyserini_root / "resources/jars/anserini-1.3.0-fatjar.jar"
    java = java_home / "bin/java.exe"
    javac = java_home / "bin/javac.exe"
    if not all(path.is_file() for path in (jar, java, javac)):
        raise FileNotFoundError("pinned Anserini fatjar and JDK 21 are required")
    classes = build_root / "java-classes"
    classes.mkdir(parents=True, exist_ok=True)
    compiled = subprocess.run(
        [str(javac), "-encoding", "UTF-8", "-classpath", str(jar), "-d", str(classes), str(java_source)],
        capture_output=True,
        text=True,
        check=True,
    )
    input_rows = [
        f"{row_id}\t{base64.b64encode(text.encode('utf-8')).decode('ascii')}"
        for row_id, text in rows
    ]
    result = subprocess.run(
        [
            str(java),
            "-Dfile.encoding=UTF-8",
            "--add-modules=jdk.incubator.vector",
            "-classpath",
            os.pathsep.join((str(classes), str(jar))),
            "E5LuceneAnalyzerCli",
        ],
        input=("\n".join(input_rows) + "\n").encode("ascii"),
        capture_output=True,
        check=True,
    )
    token_rows: list[list[str]] = []
    observed_ids: list[str] = []
    for line in result.stdout.decode("ascii").splitlines():
        row_id, token_text = base64.b64decode(line, validate=True).decode("utf-8").split("\t", maxsplit=1)
        observed_ids.append(row_id)
        token_rows.append(token_text.split())
    if observed_ids != [row_id for row_id, _ in rows]:
        raise ValueError("Lucene analyzer changed the requested ID order")
    return token_rows, {
        "analyzer": "io.anserini.analysis.DefaultEnglishAnalyzer.newStemmingInstance(porter)",
        "analyzer_jar_sha256": _file_sha256(jar),
        "analyzer_jar_bytes": jar.stat().st_size,
        "analyzer_jar_name": jar.name,
        "javac_stderr": compiled.stderr.strip(),
        "java_stderr": result.stderr.decode("utf-8", errors="replace").strip(),
    }


def _smoke_queries(view_id: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    if view_id == "GUIDELINE_ONLY":
        pairs = GUIDELINE_SMOKE_QUERIES
        return [
            {"query_id": f"guideline-{i:02d}", "query": query, "expected_source_id": source}
            for i, (query, source) in enumerate(pairs, start=1)
        ]
    public = sorted(
        (row for row in chunks if row["source_family"] == "public_health"),
        key=lambda row: row["source_id"],
    )
    if view_id == "PUBLIC_HEALTH_ONLY":
        return [
            {"query_id": f"public-health-{i:02d}", "query": " ".join(row["section_path"]), "expected_source_id": row["source_id"]}
            for i, row in enumerate(public[:10], start=1)
        ]
    mixed = [
        {"query_id": f"combined-guideline-{i:02d}", "query": query, "expected_source_id": source}
        for i, (query, source) in enumerate(GUIDELINE_SMOKE_QUERIES[:5], start=1)
    ]
    mixed.extend(
        {"query_id": f"combined-public-{i:02d}", "query": " ".join(row["section_path"]), "expected_source_id": row["source_id"]}
        for i, row in enumerate(public[:5], start=1)
    )
    return mixed


def build(*, repo_root: Path, external_root: Path, java_home: Path, pyserini_root: Path) -> dict[str, Any]:
    if not hasattr(scipy.linalg, "triu"):
        scipy.linalg.triu = np.triu
    import gensim
    from gensim import matutils
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    corpus_report_path = external_root / "e5a3/corpus_build_report.json"
    corpus_report = json.loads(corpus_report_path.read_text(encoding="utf-8"))
    build_root = external_root / "e5a3/indexes"
    java_source = repo_root / "eval/rag_e5/E5LuceneAnalyzerCli.java"
    views: list[dict[str, Any]] = []

    for view in corpus_report["views"]:
        view_id = view["corpus_view_id"]
        chunks_path = Path(view["chunks_jsonl_path"])
        documents_path = Path(view["search_documents_jsonl_path"])
        if _file_sha256(chunks_path) != view["chunks_jsonl_sha256"]:
            raise ValueError(f"chunk SHA mismatch for {view_id}")
        if _file_sha256(documents_path) != view["search_documents_jsonl_sha256"]:
            raise ValueError(f"search-document SHA mismatch for {view_id}")
        chunks = _read_jsonl(chunks_path)
        documents = _read_jsonl(documents_path)
        doc_ids = [row["id"] for row in documents]
        if doc_ids != [row["chunk_id"] for row in chunks]:
            raise ValueError(f"document and chunk order mismatch for {view_id}")
        queries = _smoke_queries(view_id, chunks)
        if len(queries) < 10:
            raise ValueError(f"{view_id} needs at least 10 deterministic title/heading queries")
        analyzer_rows = [
            *[(row["id"], row["contents"]) for row in documents],
            *[(f"query:{row['query_id']}", row["query"]) for row in queries],
        ]
        tokenized, analyzer_info = _tokenize_lucene(
            analyzer_rows,
            java_home=java_home,
            pyserini_root=pyserini_root,
            java_source=java_source,
            build_root=build_root / view_id.lower(),
        )
        document_tokens = tokenized[: len(doc_ids)]
        query_tokens = tokenized[len(doc_ids) :]
        if any(not terms for terms in document_tokens):
            raise ValueError(f"Lucene analyzer produced an empty document in {view_id}")

        directory = build_root / view_id.lower()
        directory.mkdir(parents=True, exist_ok=True)
        token_rows = [
            {"chunk_id": doc_id, "tokens": terms}
            for doc_id, terms in zip(doc_ids, document_tokens, strict=True)
        ]
        token_bytes = b"".join(_canonical_json(row) + b"\n" for row in token_rows)
        token_path = directory / "bm25_lucene_tokens.jsonl"
        token_path.write_bytes(token_bytes)

        dictionary = Dictionary(document_tokens)
        bows = [dictionary.doc2bow(row) for row in document_tokens]
        model = LuceneBM25Model(dictionary=dictionary, k1=BM25_K1, b=BM25_B)
        weighted_docs = matutils.corpus2csc(
            list(model[bows]), num_terms=len(dictionary), num_docs=len(doc_ids)
        ).T.tocsr()
        index = SparseMatrixSimilarity(
            weighted_docs,
            num_docs=len(doc_ids),
            num_terms=len(dictionary),
            normalize_queries=False,
            normalize_documents=False,
        )
        chunks_by_id = {row["chunk_id"]: row for row in chunks}
        ranking_rows: list[dict[str, Any]] = []
        hit_count = 0
        for query, terms in zip(queries, query_tokens, strict=True):
            weighted_query = model[dictionary.doc2bow(terms)]
            scores = np.asarray(index[weighted_query]).reshape(-1)
            ranking = sorted(
                ((doc_id, float(score)) for doc_id, score in zip(doc_ids, scores, strict=True)),
                key=lambda pair: (-pair[1], pair[0]),
            )
            top100 = [{"chunk_id": doc_id, "score": score} for doc_id, score in ranking[:BM25_DEPTH]]
            top10_sources = [
                chunks_by_id[row["chunk_id"]]["source_id"] for row in top100[:10] if row["score"] > 0
            ]
            passed = query["expected_source_id"] in top10_sources
            hit_count += int(passed)
            ranking_rows.append(
                {
                    **query,
                    "lucene_tokens": terms,
                    "expected_source_hit_at_10": passed,
                    "top10_source_ids": top10_sources,
                    "top100": top100,
                }
            )

        matrix_path = directory / "bm25_document_term_weights.npz"
        sparse.save_npz(matrix_path, weighted_docs, compressed=True)
        model_payload = {
            "terms": [dictionary[index] for index in range(len(dictionary))],
            "idfs": [float(model.idfs[index]) for index in range(len(dictionary))],
            "num_documents": len(doc_ids),
            "avg_document_length": float(model.avgdl),
            "k1": BM25_K1,
            "b": BM25_B,
        }
        model_path = directory / "bm25_dictionary_model.json"
        model_sha = _write_json(model_path, model_payload)
        token_sha = _sha256(token_bytes)
        matrix_sha = _file_sha256(matrix_path)
        doc_order_sha = _sha256(_canonical_json(doc_ids))
        ranking_sha = _sha256(_canonical_json(ranking_rows))
        identity = _sha256(
            _canonical_json(
                {
                    "corpus_sha256": view["corpus_sha256"],
                    "analyzer": analyzer_info["analyzer"],
                    "analyzer_jar_sha256": analyzer_info["analyzer_jar_sha256"],
                    "implementation": "gensim.models.LuceneBM25Model",
                    "gensim_version": gensim.__version__,
                    "k1": BM25_K1,
                    "b": BM25_B,
                    "document_order_sha256": doc_order_sha,
                    "token_artifact_sha256": token_sha,
                    "dictionary_model_sha256": model_sha,
                    "document_matrix_sha256": matrix_sha,
                }
            )
        )
        manifest = {
            "schema_version": "rag-e5-bm25-index-v1",
            "corpus_view_id": view_id,
            "corpus_sha256": view["corpus_sha256"],
            "source_families": view["source_families"],
            "analyzer": analyzer_info,
            "implementation": "gensim.models.LuceneBM25Model + SparseMatrixSimilarity",
            "gensim_version": gensim.__version__,
            "scipy_triu_compatibility_shim": "process-local scipy.linalg.triu = numpy.triu when missing",
            "k1": BM25_K1,
            "b": BM25_B,
            "top_k": BM25_DEPTH,
            "document_count": len(doc_ids),
            "term_count": len(dictionary),
            "document_order_sha256": doc_order_sha,
            "token_artifact_path": str(token_path),
            "token_artifact_sha256": token_sha,
            "dictionary_model_path": str(model_path),
            "dictionary_model_sha256": model_sha,
            "document_term_matrix_path": str(matrix_path),
            "document_term_matrix_sha256": matrix_sha,
            "smoke_query_count": len(ranking_rows),
            "smoke_source_hit_at_10": hit_count,
            "smoke_rankings_sha256": ranking_sha,
            "bm25_index_identity": identity,
        }
        manifest_sha = _write_json(directory / "bm25_index_manifest.json", manifest)
        smoke = {
            "schema_version": "rag-e5-bm25-source-smoke-v1",
            "corpus_view_id": view_id,
            "corpus_sha256": view["corpus_sha256"],
            "query_count": len(queries),
            "hits_at_10": hit_count,
            "pass": hit_count == len(queries),
            "rankings": ranking_rows,
        }
        smoke_path = directory / "bm25_source_title_smoke.json"
        smoke_sha = _write_json(smoke_path, smoke)
        views.append(
            {
                "corpus_view_id": view_id,
                "source_families": view["source_families"],
                "source_count": view["source_count"],
                "chunk_count": view["chunk_count"],
                "corpus_sha256": view["corpus_sha256"],
                "bm25_index_path": str(matrix_path),
                "bm25_index_manifest_path": str(directory / "bm25_index_manifest.json"),
                "bm25_index_manifest_sha256": manifest_sha,
                "bm25_index_identity": identity,
                "bm25_smoke_hits_at_10": hit_count,
                "bm25_smoke_query_count": len(queries),
                "bm25_smoke_path": str(smoke_path),
                "bm25_smoke_sha256": smoke_sha,
                "qualification_pass": hit_count == len(queries),
            }
        )

    report = {
        "schema_version": "rag-e5-e5a3-bm25-qualification-v1",
        "stage": "E5-A3-BM25-index-qualification",
        "external_corpus_identity": corpus_report["external_corpus_identity"],
        "source_manifest_sha256": corpus_report["source_manifest_sha256"],
        "owner_decision_record_sha256": corpus_report["owner_decision_record_sha256"],
        "bm25_profile": {
            "analyzer": "Pyserini/Anserini DefaultEnglishAnalyzer (Porter stemming, English stopwords)",
            "index_model": "gensim.models.LuceneBM25Model",
            "gensim_version": gensim.__version__,
            "k1": BM25_K1,
            "b": BM25_B,
            "top_k": BM25_DEPTH,
        },
        "views": views,
        "bm25_indexes_ready": all(view["bm25_smoke_hits_at_10"] == 10 for view in views),
        "index_output_root": str(build_root),
    }
    report_path = external_root / "e5a3/bm25_qualification_report.json"
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--java-home", type=Path, default=DEFAULT_JAVA_HOME)
    parser.add_argument("--pyserini-root", type=Path, default=DEFAULT_PYSERINI_ROOT)
    args = parser.parse_args()
    result = build(
        repo_root=args.repo_root,
        external_root=args.external_root,
        java_home=args.java_home,
        pyserini_root=args.pyserini_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
