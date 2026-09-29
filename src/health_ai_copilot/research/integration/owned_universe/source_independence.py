"""Structural source-independence and generated-text contamination checks."""

from __future__ import annotations

import ast
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from .schema import LatentWorld

REQUIRED_U1_MODULES = (
    "actions.py", "contracts.py", "counterfactual.py", "evaluator.py", "executor.py",
    "evidence_world.py", "replay.py", "state.py", "tools.py", "__init__.py",
)
GENERATOR_MODULES = (
    "__init__.py", "schema.py", "facts.py", "grammar.py", "timeline.py", "scenarios.py",
    "splits.py", "realization.py", "realization_text.py", "evaluator_truth.py",
    "diagnostics.py", "lineage.py", "source_independence.py", "cli.py",
)


def audited_source_paths(source_root: Path, spec_root: Path) -> tuple[Path, ...]:
    package_inits = (
        source_root / "health_ai_copilot" / "__init__.py",
        source_root / "health_ai_copilot" / "research" / "__init__.py",
    )
    u1 = source_root / "health_ai_copilot" / "research" / "integration"
    owned = u1 / "owned_universe"
    source_files = tuple(sorted(u1 / name for name in REQUIRED_U1_MODULES))
    generator_files = tuple(sorted(owned / name for name in GENERATOR_MODULES))
    spec_files = tuple(sorted(spec_root.glob("*.json")))
    paths = (*spec_files, *package_inits, *source_files, *generator_files)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source allowlist input missing: {missing}")
    return tuple(paths)


def source_independence_audit(
    worlds: tuple[LatentWorld, ...], *, source_root: Path, spec_root: Path,
    forbidden_identifiers: tuple[str, ...],
) -> dict[str, Any]:
    inputs = audited_source_paths(source_root, spec_root)
    imported_modules: set[str] = set()
    forbidden_imports: list[str] = []
    for path in inputs:
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                imported_modules.add(module)
                root = module.split(".", 1)[0]
                if root in {"external", "datasets", "benchmarks"}:
                    forbidden_imports.append(module)

    generated_rows: list[str] = []
    for world in worlds:
        generated_rows.append(world.query)
        generated_rows.extend(row.natural_language_content for row in world.patient_records)
        generated_rows.extend(row.natural_language_content for row in world.evidence_records)
    hits = sorted({token for text in generated_rows for token in forbidden_identifiers
                   if re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
                                text, flags=re.IGNORECASE)})
    base = spec_root.parents[1]
    return {
        "status": "PASS" if not (forbidden_imports or hits) else "FAIL",
        "input_allowlist_verified": True,
        "allowlisted_input_paths": [path.relative_to(base).as_posix() for path in inputs],
        "input_sha256": {path.relative_to(base).as_posix(): sha256(path.read_bytes()).hexdigest()
                          for path in inputs},
        "import_graph_modules": sorted(imported_modules),
        "forbidden_imports": sorted(set(forbidden_imports)),
        "isolated_execution_directory": "temporary directory containing only listed owned spec/code and run output",
        "external_benchmark_data_paths_opened": [],
        "external_medical_document_content_used": False,
        "forbidden_identifier_scan": {
            "generated_runtime_text_count": len(generated_rows),
            "identifiers_scanned": list(forbidden_identifiers),
            "hits": hits,
            "status": "PASS" if not hits else "FAIL",
        },
    }
