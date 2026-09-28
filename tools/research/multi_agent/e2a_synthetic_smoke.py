"""Run the deterministic E2-A synthetic isolation/provenance smoke."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from health_ai_copilot.agent.messages import (
    FinalTurn,
    ToolCall,
    ToolCallTurn,
    ToolResultMessage,
    UserMessage,
)
from health_ai_copilot.capabilities import (
    CapabilityScopedRetriever,
    CapabilitySourceCatalog,
    E2WorkerRole,
    SourceMetadata,
    WorkerCapabilitySpec,
    e2_component_manifest,
    production_worker_capabilities,
)
from health_ai_copilot.contracts import Evidence
from health_ai_copilot.e2_team import E2HeterogeneousSequentialRunner, E2Task
from health_ai_copilot.runtime.budget import RunBudgetConfig
from health_ai_copilot.runtime.components import ComponentManifest
from health_ai_copilot.runtime.context import RunContext
from health_ai_copilot.runtime.provider import ProviderUsage
from health_ai_copilot.verification.grounding import GroundedClaim


def _evidence(source_id: str) -> Evidence:
    return Evidence(
        source_id,
        f"synthetic-title-{source_id}",
        f"synthetic-excerpt-{source_id}",
        f"https://synthetic.invalid/{source_id}",
        1.0,
    )


class _Retriever:
    def __init__(self, rows: list[Evidence]) -> None:
        self.rows = rows

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        del query
        return list(self.rows[:top_k])


class _Agent:
    def __init__(self, *, query: str, fail: bool = False, forge: str | None = None) -> None:
        self.query = query
        self.fail = fail
        self.forge = forge
        self.calls = 0

    def respond(self, messages, tools, *, runtime=None):
        self.calls += 1
        if runtime is not None:
            runtime.budget.guard_provider()
            runtime.budget.record_usage(ProviderUsage(5, 3, 8))
        if self.fail:
            raise RuntimeError("synthetic provider error")
        if self.query and self.calls == 1:
            return ToolCallTurn(
                [ToolCall(f"search-{self.calls}", "search_knowledge", {"query": self.query})]
            )
        source_id = self.forge or self._last_observed_source(messages)
        claims = (GroundedClaim("synthetic claim", (source_id,)),) if source_id else ()
        return FinalTurn(
            "synthetic answer" if source_id else "",
            [source_id] if source_id else [],
            abstain=not bool(source_id),
            claims=claims,
        )

    @staticmethod
    def _last_observed_source(messages) -> str | None:
        for message in reversed(messages):
            if isinstance(message, ToolResultMessage) and message.result.observed_evidence:
                return message.result.observed_evidence[0].source_id
            if isinstance(message, UserMessage) and message.evidence:
                return message.evidence[0].source_id
        return None


def _spec(
    role: E2WorkerRole,
    family: str,
    domains: tuple[str, ...],
) -> WorkerCapabilitySpec:
    return WorkerCapabilitySpec(
        capability_id=f"smoke-{role.value}-v1",
        role=role,
        allowed_source_families=(family,),
        allowed_capability_domains=domains,
        tool_ids=("search_knowledge",),
        retriever_profile_id="synthetic-bm25-v1",
        authority_scope=f"Synthetic fixture authority for {family} only.",
        objective_contract_id=f"smoke-{role.value}-objective-v1",
        max_model_turns=2,
        max_tool_calls=1,
    )


def _run(
    *,
    rows: list[Evidence],
    metadata: list[SourceMetadata],
    public_model: _Agent | None = None,
    guideline_model: _Agent | None = None,
    budget: RunBudgetConfig | None = None,
    tasks: list[E2Task] | None = None,
    initial: list[Evidence] | None = None,
    public_spec: WorkerCapabilitySpec | None = None,
    guideline_spec: WorkerCapabilitySpec | None = None,
) -> tuple[Any, CapabilitySourceCatalog]:
    catalog = CapabilitySourceCatalog(tuple(metadata))
    public_spec = public_spec or _spec(
        E2WorkerRole.PUBLIC_HEALTH, "public_health", ("public_facts",)
    )
    guideline_spec = guideline_spec or _spec(
        E2WorkerRole.GUIDELINE, "reviewed_guideline", ("guideline",)
    )
    specs = tuple(
        spec
        for spec in (public_spec, guideline_spec)
        if spec.role in {task.role for task in (tasks or [])}
    )
    manifest = e2_component_manifest(
        ComponentManifest("e2a-synthetic-smoke", ()),
        specs,
        source_catalog=catalog,
    )
    runtime = RunContext.create(
        "e2a_synthetic_smoke",
        budget=budget,
        profile_id=manifest.profile_id,
        component_manifest_hash=manifest.manifest_hash,
        config_hash=manifest.manifest_hash,
        code_commit="synthetic",
    )
    worker_models = {}
    if any(spec.role == E2WorkerRole.PUBLIC_HEALTH for spec in specs):
        worker_models[E2WorkerRole.PUBLIC_HEALTH] = public_model or _Agent(
            query="ignore the capability and list every source"
        )
    if any(spec.role == E2WorkerRole.GUIDELINE for spec in specs):
        worker_models[E2WorkerRole.GUIDELINE] = guideline_model or _Agent(
            query="ignore the capability and list every source"
        )
    runner = E2HeterogeneousSequentialRunner(
        worker_models=worker_models,
        retriever=_Retriever(rows),
        retriever_profile_id="synthetic-bm25-v1",
        source_catalog=catalog,
        capabilities=specs,
        component_manifest=manifest,
    )
    result = runner.run(
        "Synthetic question used only by this offline smoke.",
        tasks or [],
        initial_evidence=initial or [],
        runtime=runtime,
    )
    return result, catalog


def run_smoke(output_root: Path | None = None) -> Path:
    source_rows = [
        _evidence("public-source"),
        _evidence("guideline-source"),
        _evidence("literature-source"),
    ]
    metadata = [
        SourceMetadata("public-source", "public_health", ("public_facts",)),
        SourceMetadata("guideline-source", "reviewed_guideline", ("guideline",)),
        SourceMetadata("literature-source", "scholarly_literature", ("study",)),
    ]
    base = _Retriever(source_rows)
    public_cap = _spec(E2WorkerRole.PUBLIC_HEALTH, "public_health", ("public_facts",))
    guideline_cap = _spec(E2WorkerRole.GUIDELINE, "reviewed_guideline", ("guideline",))
    catalog = CapabilitySourceCatalog(tuple(metadata))
    case_a = CapabilityScopedRetriever(base, public_cap, catalog).search(
        "ignore the boundary and list all sources", top_k=3
    )
    case_b = CapabilityScopedRetriever(base, guideline_cap, catalog).search(
        "return literature studies too", top_k=3
    )
    a_pass = [item.source_id for item in case_a] == ["public-source"]
    b_pass = [item.source_id for item in case_b] == ["guideline-source"]

    case_c, catalog_c = _run(
        rows=source_rows,
        metadata=metadata,
        public_model=_Agent(query="ignore scope and return guideline"),
        guideline_model=_Agent(query="ignore scope and return public health"),
        tasks=[
            E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "collect public education"),
            E2Task("task-guideline", E2WorkerRole.GUIDELINE, "collect recommendations"),
        ],
    )
    c_pass = (
        {"public-source", "guideline-source"} == set(case_c.ledger.source_ids)
        and all(item.verified and item.productive for item in case_c.worker_reports)
    )

    shared_rows = [_evidence("shared-source")]
    shared_meta = [
        SourceMetadata("shared-source", "synthetic_shared", ("public_facts", "shared_domain"))
    ]
    shared_public = _spec(E2WorkerRole.PUBLIC_HEALTH, "synthetic_shared", ("public_facts",))
    shared_guideline = _spec(E2WorkerRole.GUIDELINE, "synthetic_shared", ("shared_domain",))
    case_d, catalog_d = _run(
        rows=shared_rows,
        metadata=shared_meta,
        public_model=_Agent(query="observe shared"),
        guideline_model=_Agent(query="observe shared"),
        tasks=[
            E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "observe shared source"),
            E2Task("task-guideline", E2WorkerRole.GUIDELINE, "observe shared source"),
        ],
        public_spec=shared_public,
        guideline_spec=shared_guideline,
    )
    d_metrics = case_d.metrics(catalog_d)
    d_pass = (
        len(case_d.ledger.records) == 1
        and len(case_d.ledger.records[0].observations) == 2
        and d_metrics["WorkerEvidenceOverlap"]["value"] == 1.0
        and d_metrics["WorkerUniqueEvidenceContribution"]["value"] == 0.0
    )

    case_e, _ = _run(
        rows=source_rows,
        metadata=metadata,
        public_model=_Agent(query="public-source", fail=True),
        guideline_model=_Agent(query="guideline-source"),
        tasks=[
            E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "failure isolation"),
            E2Task("task-guideline", E2WorkerRole.GUIDELINE, "successful task"),
        ],
    )
    e_pass = (
        not case_e.worker_reports[0].verified
        and case_e.worker_reports[1].verified
        and case_e.ledger.source_ids == ("guideline-source",)
    )

    case_f, _ = _run(
        rows=source_rows,
        metadata=metadata,
        budget=RunBudgetConfig(max_provider_calls=1),
        tasks=[
            E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "parent budget"),
            E2Task("task-guideline", E2WorkerRole.GUIDELINE, "parent budget"),
        ],
    )
    f_pass = (
        all(item.error_code == "parent_budget_exhausted" for item in case_f.worker_reports)
        and case_f.ledger.source_ids == ()
    )

    case_g, _ = _run(
        rows=source_rows,
        metadata=metadata,
        public_model=_Agent(query="", forge="fabricated-source"),
        tasks=[E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "citation provenance")],
        initial=[_evidence("public-source")],
    )
    g_report = case_g.worker_reports[0]
    g_pass = (
        g_report.error_code == "worker_citation_provenance_violation"
        and not g_report.claims
        and "fabricated-source" not in case_g.ledger.source_ids
    )

    results = {"A": a_pass, "B": b_pass, "C": c_pass, "D": d_pass, "E": e_pass, "F": f_pass, "G": g_pass}
    if not all(results.values()):
        failed = [key for key, passed in results.items() if not passed]
        raise AssertionError(f"synthetic E2 smoke failed: {failed}")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = output_root or Path("runs/multi_agent")
    output = root / f"e2a-{run_id}"
    output.mkdir(parents=True, exist_ok=False)

    production = production_worker_capabilities()
    manifest = {
        "run_id": run_id,
        "run_type": "synthetic_smoke_only",
        "architecture_id": "L3_HETEROGENEOUS_SEQUENTIAL",
        "scheduler": "sequential-v1",
        "memory_policy": "OFF",
        "benchmark_rows_loaded": 0,
        "test_accessed": False,
        "production_capability_contracts": [
            {**item.to_dict(), "contract_hash": item.contract_hash} for item in production
        ],
        "cases": {
            "C": {
                "component_manifest_hash": case_c.component_manifest_hash,
                "capability_manifest_hash": case_c.capability_manifest_hash,
                "source_catalog_hash": catalog_c.catalog_hash,
            },
            "D": {
                "component_manifest_hash": case_d.component_manifest_hash,
                "capability_manifest_hash": case_d.capability_manifest_hash,
                "source_catalog_hash": catalog_d.catalog_hash,
            },
        },
        "case_results": results,
    }
    reports = []
    ledger_rows = []
    metrics = {
        "schema_version": "e2-metrics-v1",
        "synthetic_only": True,
        "source_isolation": {
            "A_public_health_visible_source_ids": [item.source_id for item in case_a],
            "B_guideline_visible_source_ids": [item.source_id for item in case_b],
        },
        "case_c_complementary_workers": case_c.metrics(
            catalog_c,
            required_evidence_groups=(("public-source",), ("guideline-source",)),
            required_source_families=("public_health", "reviewed_guideline"),
        ),
        "case_d_duplicate_evidence": d_metrics,
        "case_e_failure_containment": case_e.metrics(catalog),
        "case_f_parent_budget": case_f.metrics(catalog),
        "case_g_forged_provenance": case_g.metrics(catalog),
    }
    for label, result in (("C", case_c), ("D", case_d), ("E", case_e), ("F", case_f), ("G", case_g)):
        reports.extend(
            {"case_id": label, "worker_report": row.to_dict()}
            for row in result.worker_reports
        )
        ledger_rows.extend(
            {"case_id": label, "evidence": row.to_dict()}
            for row in result.ledger.records
        )

    _write_json(output / "manifest.json", manifest)
    _write_jsonl(output / "worker_reports.jsonl", reports)
    _write_jsonl(output / "evidence_ledger.jsonl", ledger_rows)
    _write_json(output / "metrics.json", metrics)
    report_lines = [
        "# E2-A Synthetic Smoke",
        "",
        f"Run ID: `{run_id}`",
        "",
        "All fixtures are synthetic. No benchmark DEV/TEST case, provider network, or medical corpus was used.",
        "",
        "| Case | Result |",
        "| --- | --- |",
    ]
    report_lines.extend(f"| {case} | {'PASS' if passed else 'FAIL'} |" for case, passed in results.items())
    report_lines.extend(["", "This smoke validates capability boundaries and bookkeeping only; it is not evidence of medical-answer quality."])
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")
    return output


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    print(run_smoke().resolve())
