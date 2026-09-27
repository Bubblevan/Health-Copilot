#!/usr/bin/env python3
"""Data-free retrieval-path audit for the pinned PyPI and official SimpleMem releases."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MEMORY_ROOT = ROOT.parent / "external" / "memory"
MEMEVAL_ROOT = MEMORY_ROOT / "MemEval"
OFFICIAL_ROOT = MEMORY_ROOT / "SimpleMem"
QUESTION = "What medication did Alice mention?"
TARGET_TEXT = "Alice's medication was metformin from prescription RX-17."
DISTRACTOR_TEXT = "Alice listens to classical music and enjoys the violin."
FIXTURE = {
    "question": QUESTION,
    "entries": [
        {
            "entry_id": "target",
            "lossless_restatement": TARGET_TEXT,
            "keywords": ["metformin", "medication", "RX-17"],
            "persons": ["Alice"],
            "entities": ["RX-17"],
            "location": "Boston",
            "timestamp": "2025-01-02T00:00:00",
            "synthetic_vector": [0.0, 1.0],
        },
        {
            "entry_id": "distractor",
            "lossless_restatement": DISTRACTOR_TEXT,
            "keywords": ["classical music", "violin"],
            "persons": ["Alice"],
            "entities": ["MUSIC-3"],
            "location": "Boston",
            "timestamp": "2025-01-02T00:00:00",
            "synthetic_vector": [1.0, 0.0],
        },
    ],
    "query_vector": [1.0, 0.0],
    "purpose": "Make semantic top-1 disagree with the lexical target.",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_source_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def prompt_template_hashes(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}

    def template(node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                elif isinstance(value, ast.FormattedValue):
                    parts.append("{dynamic_value}")
            return "".join(parts)
        return ast.dump(node, include_attributes=False)

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Assign(self, node: ast.Assign) -> None:
            if any(isinstance(target, ast.Name) and target.id == "prompt" for target in node.targets):
                key = ".".join(self.functions) or "module"
                result[key] = sha256_bytes(template(node.value).encode("utf-8"))
            self.generic_visit(node)

    Visitor().visit(tree)
    return dict(sorted(result.items()))


def planning_path_checks(path: Path) -> dict[str, bool]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_retrieve_with_planning":
            method = node
            break
    if method is None:
        return {"method_present": False}

    calls: list[str] = []
    has_reflection_condition = False

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            self.generic_visit(node)

        def visit_If(self, node: ast.If) -> None:
            nonlocal has_reflection_condition
            if isinstance(node.test, ast.Name) and node.test.id == "should_use_reflection":
                has_reflection_condition = True
            self.generic_visit(node)

    Visitor().visit(method)
    positions = {name: calls.index(name) if name in calls else None for name in (
        "_semantic_search",
        "_analyze_query",
        "_keyword_search",
        "_structured_search",
        "_merge_and_deduplicate_entries",
        "_retrieve_with_intelligent_reflection",
    )}
    ordered = [value for value in positions.values() if value is not None]
    return {
        "method_present": True,
        "semantic_call_present": positions["_semantic_search"] is not None,
        "query_analysis_present": positions["_analyze_query"] is not None,
        "lexical_call_present": positions["_keyword_search"] is not None,
        "structured_call_present": positions["_structured_search"] is not None,
        "merge_dedup_call_present": positions["_merge_and_deduplicate_entries"] is not None,
        "optional_reflection_call_present": positions["_retrieve_with_intelligent_reflection"] is not None,
        "optional_reflection_condition_present": has_reflection_condition,
        "required_stage_calls_in_order": ordered == sorted(ordered) and len(ordered) == 6,
        "call_order": calls,
    }


def uv_lock_artifact_hashes() -> dict[str, str | None]:
    lock_path = MEMEVAL_ROOT / "uv.lock"
    if not lock_path.exists():
        return {"wheel_sha256": None, "sdist_sha256": None}
    import tomllib

    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    package = next(
        (item for item in lock.get("package", []) if item.get("name") == "simplemem"),
        None,
    )
    if package is None:
        return {"wheel_sha256": None, "sdist_sha256": None}
    wheel_hash = next(
        (item.get("hash", "").removeprefix("sha256:") for item in package.get("wheels", [])),
        None,
    )
    sdist = package.get("sdist", {})
    sdist_hash = sdist.get("hash", "").removeprefix("sha256:") or None
    return {"wheel_sha256": wheel_hash, "sdist_sha256": sdist_hash}


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def load_official_config() -> types.ModuleType:
    source = OFFICIAL_ROOT / "config.py.example"
    config = types.ModuleType("simplemem_fidelity_config")
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), config.__dict__)
    config.OPENAI_API_KEY = None
    config.OPENAI_BASE_URL = None
    config.LLM_MODEL = "synthetic-fake-llm"
    config.USE_STREAMING = False
    config.ENABLE_PARALLEL_RETRIEVAL = False
    config.MAX_RETRIEVAL_WORKERS = 1
    config.ENABLE_REFLECTION = False
    config.MAX_REFLECTION_ROUNDS = 0
    config.SEMANTIC_TOP_K = 2
    config.KEYWORD_TOP_K = 2
    config.STRUCTURED_TOP_K = 2
    sys.modules["config"] = config
    return config


class FixtureEmbedding:
    dimension = 2

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = [
            [0.0, 1.0] if "metformin" in text.lower() else [1.0, 0.0]
            for text in texts
        ]
        return np.asarray(vectors, dtype=np.float32)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FixtureLLM:
    """Static in-process planner responses; this never contacts a model provider."""

    def chat_completion(self, messages: list[dict[str, str]], **_: Any) -> str:
        system = messages[0]["content"]
        if "information requirement analyst" in system:
            return json.dumps(
                {
                    "question_type": "factual",
                    "key_entities": ["Alice"],
                    "required_info": [
                        {"info_type": "medication", "description": "medication Alice mentioned", "priority": "high"}
                    ],
                    "relationships": [],
                    "minimal_queries_needed": 1,
                }
            )
        if "query generation specialist" in system:
            return json.dumps({"reasoning": "synthetic fixture", "queries": [QUESTION]})
        if "query analysis assistant" in system:
            return json.dumps(
                {
                    "keywords": ["metformin"],
                    "persons": ["Alice"],
                    "time_expression": None,
                    "location": None,
                    "entities": ["RX-17"],
                }
            )
        raise AssertionError(f"Unexpected planner role in synthetic audit: {system}")

    @staticmethod
    def extract_json(response: str) -> dict[str, Any]:
        return json.loads(response)


def call_git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def official_provenance() -> dict[str, Any]:
    if not OFFICIAL_ROOT.exists():
        return {"path": str(OFFICIAL_ROOT), "present": False}
    head = call_git("rev-parse", "HEAD", cwd=OFFICIAL_ROOT)
    tag = call_git("describe", "--tags", "--exact-match", cwd=OFFICIAL_ROOT)
    status = call_git("status", "--porcelain", cwd=OFFICIAL_ROOT)
    return {
        "repository": "https://github.com/aiming-lab/SimpleMem.git",
        "path": str(OFFICIAL_ROOT),
        "tag": tag,
        "commit": head,
        "expected_commit": "7da777f56a15db81bb261d296c89cad5915e8d67",
        "worktree_clean": not bool(status),
        "source_tree_sha256": sha256_source_tree(OFFICIAL_ROOT),
    }


def load_target(target: str):
    if target == "official":
        config = load_official_config()
        sys.path.insert(0, str(OFFICIAL_ROOT))
        from core.hybrid_retriever import HybridRetriever
        from database.vector_store import VectorStore
        from models.memory_entry import MemoryEntry

        return HybridRetriever, VectorStore, MemoryEntry, config

    sys.path.insert(0, str(ROOT))
    from simplemem.core.hybrid_retriever import HybridRetriever
    from simplemem.database.vector_store import VectorStore
    from simplemem.models.memory_entry import MemoryEntry
    from simplemem.config import get_config

    return HybridRetriever, VectorStore, MemoryEntry, get_config()


def make_entry(memory_entry: type, item: dict[str, Any]):
    return memory_entry(
        entry_id=item["entry_id"],
        lossless_restatement=item["lossless_restatement"],
        keywords=item["keywords"],
        persons=item["persons"],
        entities=item["entities"],
        location=item["location"],
        timestamp=item["timestamp"],
    )


def trace_retrieval(
    target: str,
    hybrid_retriever: type,
    vector_store: Any,
) -> dict[str, Any]:
    calls = {"semantic": 0, "keyword": 0, "structured": 0, "merge": 0}
    for label, method_name in (
        ("semantic", "semantic_search"),
        ("keyword", "keyword_search"),
        ("structured", "structured_search"),
    ):
        original = getattr(vector_store, method_name)

        def wrapped(*args: Any, __name: str = label, __original=original, **kwargs: Any):
            calls[__name] += 1
            return __original(*args, **kwargs)

        setattr(vector_store, method_name, wrapped)

    if target == "official":
        retriever = hybrid_retriever(
            llm_client=FixtureLLM(),
            vector_store=vector_store,
            semantic_top_k=2,
            keyword_top_k=2,
            structured_top_k=2,
            enable_planning=True,
            enable_reflection=False,
            enable_parallel_retrieval=False,
            max_retrieval_workers=1,
        )
        merge_name = "_merge_and_deduplicate_entries"
    else:
        retriever = hybrid_retriever(
            llm_client=None,
            vector_store=vector_store,
            enable_planning=True,
            enable_reflection=False,
            enable_parallel_retrieval=False,
            max_retrieval_workers=1,
        )
        retriever._analyze_information_requirements = lambda _query: {
            "required_info": [{"info_type": "synthetic"}]
        }
        retriever._generate_targeted_queries = lambda query, _plan: [query]
        merge_name = "_merge_and_deduplicate_entries"

    if hasattr(retriever, merge_name):
        original_merge = getattr(retriever, merge_name)

        def counted_merge(*args: Any, **kwargs: Any):
            calls["merge"] += 1
            return original_merge(*args, **kwargs)

        setattr(retriever, merge_name, counted_merge)

    result = retriever.retrieve(QUESTION, enable_reflection=False)
    return {"call_counts": calls, "result_ids": [entry.entry_id for entry in result]}


def audit(target: str) -> dict[str, Any]:
    if target == "official":
        load_official_config()
        package_version = "v0.1.0"
        package_source = "official tagged source checkout"
        source_paths = [
            "core/hybrid_retriever.py",
            "database/vector_store.py",
            "models/memory_entry.py",
            "utils/llm_client.py",
            "utils/embedding.py",
            "config.py.example",
        ]
    else:
        package_version = metadata.version("simplemem")
        package_source = "PyPI wheel pinned by MemEval uv.lock"
        package_root = Path(metadata.distribution("simplemem").locate_file("simplemem"))
        config_root = package_root
        source_paths = [
            "core/hybrid_retriever.py",
            "database/vector_store.py",
            "config/__init__.py",
            "system.py",
        ]

    hybrid_retriever, vector_store_class, memory_entry, _loaded_config = load_target(target)

    if target == "official":
        source_tree_root = OFFICIAL_ROOT
        source_files = {name: sha256_file(OFFICIAL_ROOT / name) for name in source_paths}
        prompt_file = OFFICIAL_ROOT / "core" / "hybrid_retriever.py"
        config_file_hash = sha256_file(OFFICIAL_ROOT / "config.py.example")
    else:
        source_tree_root = package_root
        source_files = {name: sha256_file(package_root / name) for name in source_paths}
        prompt_file = package_root / "core" / "hybrid_retriever.py"
        config_file_hash = sha256_file(package_root / "config" / "__init__.py")

    with tempfile.TemporaryDirectory(prefix=f"simplemem-{target}-fidelity-") as temp:
        store = vector_store_class(
            db_path=str(Path(temp) / "lancedb"),
            embedding_model=FixtureEmbedding(),
            table_name="synthetic_fidelity",
        )
        store.add_entries([make_entry(memory_entry, item) for item in FIXTURE["entries"]])
        semantic_ids = [entry.entry_id for entry in store.semantic_search(QUESTION, top_k=1)]
        keyword_ids = [entry.entry_id for entry in store.keyword_search(["metformin"], top_k=1)]
        fts_initialized = bool(getattr(store, "_fts_initialized", False))

        false_counts = {"semantic": 0, "keyword": 0, "structured": 0, "merge": 0}
        originals = {
            name: getattr(store, method)
            for name, method in (
                ("semantic", "semantic_search"),
                ("keyword", "keyword_search"),
                ("structured", "structured_search"),
            )
        }
        for label, method in (("semantic", "semantic_search"), ("keyword", "keyword_search"), ("structured", "structured_search")):
            original = originals[label]

            def counted(*args: Any, __label: str = label, __original=original, **kwargs: Any):
                false_counts[__label] += 1
                return __original(*args, **kwargs)

            setattr(store, method, counted)

        false_retriever = hybrid_retriever(
            llm_client=FixtureLLM() if target == "official" else None,
            vector_store=store,
            semantic_top_k=2,
            keyword_top_k=2,
            structured_top_k=2,
            enable_planning=False,
            enable_reflection=False,
            enable_parallel_retrieval=False,
            max_retrieval_workers=1,
        )
        false_result = false_retriever.retrieve(QUESTION, enable_reflection=False)
        false_ids = [entry.entry_id for entry in false_result]
        false_counts = dict(false_counts)
        for label, method in (
            ("semantic", "semantic_search"),
            ("keyword", "keyword_search"),
            ("structured", "structured_search"),
        ):
            setattr(store, method, originals[label])

        true_trace = trace_retrieval(target, hybrid_retriever, store)

    package_record = uv_lock_artifact_hashes()
    official = official_provenance()
    official_retriever_hash = (
        sha256_file(OFFICIAL_ROOT / "core" / "hybrid_retriever.py")
        if OFFICIAL_ROOT.exists()
        else None
    )
    package_dir_hash = sha256_source_tree(source_tree_root)
    versions = {
        name: installed_version(name)
        for name in ("simplemem", "lancedb", "tantivy", "pylance", "lance-namespace", "lance-namespace-urllib3-client")
    }
    fixture_hash = sha256_bytes(json.dumps(FIXTURE, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    configured = {
        "planning_false": {"enable_planning": False, "enable_reflection": False},
        "planning_true": {
            "enable_planning": True,
            "enable_reflection": False,
            "enable_parallel_retrieval": False,
            "semantic_top_k": 2,
            "keyword_top_k": 2,
            "structured_top_k": 2,
        },
        "llm": "static in-process fixture fake; outbound calls=0",
        "embedding": "fixed 2D vectors; outbound calls=0",
        "answer_generator_invoked": False,
    }
    config_digest = sha256_bytes(json.dumps(configured, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    if target == "official":
        planning_counts = true_trace["call_counts"]
        source_checks = planning_path_checks(prompt_file)
        fidelity_pass = (
            fts_initialized
            and semantic_ids == ["distractor"]
            and keyword_ids == ["target"]
            and false_counts["semantic"] >= 1
            and false_counts["keyword"] == 0
            and false_counts["structured"] == 0
            and planning_counts["semantic"] >= 1
            and planning_counts["keyword"] >= 1
            and planning_counts["structured"] >= 1
            and planning_counts["merge"] >= 1
            and "target" in true_trace["result_ids"]
            and all(
                source_checks.get(key, False)
                for key in (
                    "semantic_call_present",
                    "query_analysis_present",
                    "lexical_call_present",
                    "structured_call_present",
                    "merge_dedup_call_present",
                    "optional_reflection_call_present",
                    "optional_reflection_condition_present",
                    "required_stage_calls_in_order",
                )
            )
        )
        gate_name = "SIMPLEMEM_OFFICIAL_TAG_HYBRID_FIDELITY"
    else:
        planning_counts = true_trace["call_counts"]
        fidelity_pass = (
            fts_initialized
            and semantic_ids == ["distractor"]
            and keyword_ids == ["target"]
            and false_counts["semantic"] >= 1
            and false_counts["keyword"] == 0
            and false_counts["structured"] == 0
            and planning_counts["semantic"] >= 1
            and planning_counts["keyword"] >= 1
            and planning_counts["structured"] >= 1
            and planning_counts["merge"] >= 1
        )
        source_checks = planning_path_checks(prompt_file)
        gate_name = "SIMPLEMEM_HYBRID_FIDELITY"

    return {
        "audit_version": "simplemem-fidelity-v1",
        "target": target,
        "test_type": "synthetic_no_benchmark_data",
        "benchmark_data_accessed": False,
        "gate": gate_name,
        "decision": "PASS" if fidelity_pass else "FAIL",
        "provenance_mismatch_with_official_v0_1_0_tag": target == "pypi",
        "baseline_interpretation": (
            "SimpleMem-PyPI-0.1.0-MemEval: direct FTS is operational, but the planning retrieval path is semantic-only; not eligible to represent official SimpleMem architecture."
            if target == "pypi"
            else "Official aiming-lab/SimpleMem v0.1.0 source checkout."
        ),
        "provenance_comparison": {
            "pypi_hybrid_retriever_sha256": source_files.get("core/hybrid_retriever.py")
            if target == "pypi"
            else sha256_file(Path(metadata.distribution("simplemem").locate_file("simplemem")) / "core" / "hybrid_retriever.py"),
            "official_v0_1_0_hybrid_retriever_sha256": official_retriever_hash,
            "source_hashes_match": (
                source_files.get("core/hybrid_retriever.py") == official_retriever_hash
                if target == "pypi"
                else sha256_file(Path(metadata.distribution("simplemem").locate_file("simplemem")) / "core" / "hybrid_retriever.py") == official_retriever_hash
            ),
        },
        "tested_module_file": str(Path(sys.modules[hybrid_retriever.__module__].__file__).resolve()),
        "planning_source_verification": source_checks,
        "package": {
            "name": "simplemem",
            "version": package_version,
            "source": package_source,
            "wheel_sha256": package_record["wheel_sha256"] if target == "pypi" else None,
            "sdist_sha256": package_record["sdist_sha256"] if target == "pypi" else None,
            "source_tree_sha256": package_dir_hash,
            "installed_source_tree_sha256": package_dir_hash if target == "pypi" else None,
        },
        "official_release": official,
        "runtime_versions": versions,
        "fts": {
            "index_creation_succeeded": fts_initialized,
            "implementation": "LanceDB local Tantivy FTS, tokenizer en_stem",
            "direct_keyword_search_result_ids": keyword_ids,
            "semantic_result_ids": semantic_ids,
        },
        "fixture": {"definition": FIXTURE, "sha256": fixture_hash},
        "call_counts": {
            "planning_false": false_counts,
            "planning_true": planning_counts,
        },
        "planning_false_result_ids": false_ids,
        "planning_true_result_ids": true_trace["result_ids"],
        "source_hashes": source_files,
        "prompt_template_sha256": prompt_template_hashes(prompt_file),
        "config": {
            "source_sha256": config_file_hash,
            "effective_test_config": configured,
            "effective_test_config_sha256": config_digest,
        },
        "providers": {
            "llm": "in-process static fake",
            "embedding": "in-process deterministic synthetic vectors",
            "hosted_calls": 0,
            "native_answer_head_invoked": False,
        },
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("pypi", "official"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit(args.target)
    default_name = (
        "simplemem_pypi_010_fidelity.json"
        if args.target == "pypi"
        else "simplemem_official_v010_fidelity.json"
    )
    output = args.output or (ROOT / "docs" / "research" / "memory" / default_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
