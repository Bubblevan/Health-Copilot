"""Unified, explicit M7 evaluation runner and artifact bundle writer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from ..agent.messages import AssistantToolCallMessage, ToolResultMessage
from ..contracts import Evidence, GenerationDraft, Route
from ..knowledge.loader import load_knowledge_cards
from ..knowledge.scope import load_knowledge_scope
from ..policy.evidence import validate_assessment
from ..runtime import (
    ReplayProviderExecutor,
    ReplayToolRunner,
    RunContext,
    read_provider_exchanges,
    read_tool_exchanges,
)
from ..runtime.builder import RuntimeBuilder
from ..runtime.trace import TraceContentPolicy
from ..safety import route_question
from ..verification.citations import verify_citations
from ..verification.grounding import (
    ClaimVerdict,
    GroundedClaim,
    materialize_cited_evidence,
    validate_claim_support_result,
    validate_grounding_result,
)
from .failures import FailureMapper
from .graders import default_graders
from .metrics import metrics_to_dict, trial_metrics
from .registry import EvalSuiteRegistry, default_eval_suite_registry
from .schema import (
    CaseRunRecord,
    CaseRunStatus,
    EvalCase,
    EvalConfigurationError,
    EvalExecutionMode,
    EvalRunSpec,
    EvalTargetKind,
    GraderResult,
    GraderStatus,
    MetricResult,
    TrajectoryRecord,
    canonical_hash,
    effective_budget_config,
    sha256_file,
)


class _OfflineProvider:
    """Provider sentinel used by deterministic suites; it is never called."""

    def execute(self, request, runtime):  # pragma: no cover - an invariant guard
        raise RuntimeError("offline evaluation cannot execute a provider call")


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load JSONL while preserving every suite-specific payload field."""

    source = Path(path)
    cases: list[EvalCase] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalConfigurationError(
                f"invalid JSON in {source} line {line_number}"
            ) from exc
        if not isinstance(payload, dict):
            raise EvalConfigurationError(f"case on line {line_number} must be an object")
        case_id = payload.get("id", payload.get("case_id"))
        if not isinstance(case_id, str) or not case_id.strip():
            raise EvalConfigurationError(f"case on line {line_number} has no id")
        cases.append(EvalCase(case_id, payload, {"dataset_path": str(source)}))
    return cases


class EvaluationRunner:
    """Single entry point for offline, explicitly-live, and strict replay runs."""

    def __init__(
        self,
        registry: EvalSuiteRegistry | None = None,
        *,
        graders: Mapping[str, Any] | None = None,
        failure_mapper: FailureMapper | None = None,
    ) -> None:
        self.registry = registry or default_eval_suite_registry()
        self.graders = dict(graders or default_graders())
        self.failure_mapper = failure_mapper or FailureMapper()

    def prepare_run_spec(
        self,
        suite_id: str,
        *,
        execution_mode: EvalExecutionMode = EvalExecutionMode.OFFLINE,
        profile_id: str | None = None,
        trials: int = 1,
        output_root: str | Path = "runs/m7",
        budget_overrides: Mapping[str, Any] | None = None,
        replay_run_dir: str | Path | None = None,
        dataset_path: str | Path | None = None,
        trace_content_policy: str | None = None,
    ) -> EvalRunSpec:
        suite = self.registry.get(suite_id)
        mode = EvalExecutionMode(execution_mode)
        self.registry.validate_mode(suite, mode)
        if suite_id == "m8-agent-team-focused-v1" and profile_id is not None:
            allowed = set(suite.provenance.get("allowed_profiles", ()))
            if profile_id not in allowed:
                raise EvalConfigurationError(
                    f"M8 core comparison rejects profile {profile_id}; allowed: {sorted(allowed)}"
                )
        dataset = Path(dataset_path or suite.dataset_path)
        if not dataset.exists():
            raise EvalConfigurationError(f"evaluation dataset does not exist: {dataset}")
        dataset_sha = sha256_file(dataset)
        if suite.expected_dataset_sha256 and dataset_sha != suite.expected_dataset_sha256:
            raise EvalConfigurationError(
                f"dataset hash mismatch for {suite_id}: expected {suite.expected_dataset_sha256}, "
                f"got {dataset_sha}"
            )
        content_policy = trace_content_policy or suite.content_policy
        if content_policy not in {"metadata_only", "public_eval_content"}:
            raise EvalConfigurationError(f"unknown trace content policy: {content_policy}")
        return EvalRunSpec(
            suite_id=suite_id,
            execution_mode=mode,
            profile_id=profile_id or suite.default_profile_id,
            dataset_path=str(dataset),
            dataset_sha256=dataset_sha,
            trials=trials,
            trace_content_policy=content_policy,
            budget_overrides=budget_overrides or {},
            output_root=str(output_root),
            replay_source=str(replay_run_dir) if replay_run_dir else None,
            replay_run_dir=str(replay_run_dir) if replay_run_dir else None,
        )

    def run(
        self,
        spec: EvalRunSpec,
        *,
        allow_live_provider: bool = False,
        public_eval_content: bool = False,
        knowledge_dir: str | Path = "data/knowledge_cards",
        knowledge_scope: str | Path = "data/knowledge_scope.json",
    ) -> Path:
        suite = self.registry.get(spec.suite_id)
        self.registry.validate_mode(suite, spec.execution_mode)
        self._validate_content_policy(suite, spec, public_eval_content)
        if public_eval_content:
            spec = replace(spec, trace_content_policy="public_eval_content")
        if spec.execution_mode == EvalExecutionMode.LIVE and not allow_live_provider:
            raise EvalConfigurationError(
                "live evaluation is disabled by default; pass --allow-live-provider explicitly"
            )
        dataset = Path(spec.dataset_path)
        if sha256_file(dataset) != spec.dataset_sha256:
            raise EvalConfigurationError("dataset changed after EvalRunSpec was prepared")
        cases = load_eval_cases(dataset)
        bundle = self._new_bundle(spec)
        cards = load_knowledge_cards(knowledge_dir)
        scope = None
        if (suite.default_profile_id and (suite.default_profile_id.startswith("m3-") or suite.default_profile_id.startswith("m8-"))) or (
            spec.profile_id and (spec.profile_id.startswith("m3-") or spec.profile_id.startswith("m8-"))
        ):
            scope = load_knowledge_scope(knowledge_scope, cards)

        if suite.suite_id == "m0-regression-v1":
            components = self._build_offline_components(spec, spec.profile_id or "m0-bm25-default", cards, scope)
            records, trajectories = self._run_m0(bundle, suite, spec, cases, components)
        elif suite.suite_id == "m5-product-retrieval-v1":
            components = self._build_offline_components(spec, spec.profile_id or "m0-bm25-default", cards, scope)
            records, trajectories = self._run_m5(bundle, suite, spec, cases, components)
        elif spec.execution_mode == EvalExecutionMode.REPLAY:
            records, trajectories, components = self._run_replay(
                bundle, suite, spec, cases, cards, scope
            )
        elif spec.execution_mode == EvalExecutionMode.LIVE:
            components = self._build_live_components(spec, cards, scope)
            if suite.target_kind in {EvalTargetKind.PIPELINE, EvalTargetKind.ORCHESTRATION}:
                records, trajectories = self._run_pipeline(
                    bundle, suite, spec, cases, components, public_eval_content
                )
            elif suite.target_kind == EvalTargetKind.POLICY:
                records, trajectories = self._run_policy(
                    bundle, suite, spec, cases, components, cards, public_eval_content
                )
            elif suite.target_kind == EvalTargetKind.VERIFIER:
                records, trajectories = self._run_verifier(
                    bundle, suite, spec, cases, components, cards, public_eval_content
                )
            else:
                raise EvalConfigurationError(
                    f"live execution is not defined for target {suite.target_kind.value}"
                )
        else:
            raise EvalConfigurationError(
                f"offline execution is not defined for target {suite.target_kind.value}"
            )

        if components is not None:
            spec = spec.bind_runtime(components)
            records = [
                replace(
                    record,
                    eval_run_id=spec.spec_hash,
                    eval_spec_hash=spec.spec_hash,
                    profile_id=spec.profile_id,
                    component_manifest_hash=spec.component_manifest_hash,
                    code_commit=spec.code_commit,
                )
                for record in records
            ]
            trajectories = [
                _bind_trajectory_eval_identity(trajectory, spec.spec_hash)
                for trajectory in trajectories
            ]

        grader_results = self._grade(suite, cases, records)
        failures = self.failure_mapper.collect(cases, records, grader_results)
        metrics = trial_metrics(
            records,
            grader_results,
            definition_version=suite.metric_definition_version,
        )
        if suite.suite_id == "m0-regression-v1":
            self._add_m0_parity_metrics(metrics, records, cases, suite.metric_definition_version)
        elif suite.suite_id == "m5-product-retrieval-v1":
            metrics.update(self._m5_metrics(records, suite.metric_definition_version))
        elif suite.suite_id == "m8-agent-team-focused-v1":
            metrics.update(self._m8_metrics(records, cases, suite.metric_definition_version))
        self._write_bundle(
            bundle,
            spec,
            suite,
            cases,
            records,
            grader_results,
            failures,
            trajectories,
            metrics,
            components,
            public_eval_content,
        )
        return bundle

    def _build_offline_components(self, spec, profile_id, cards, scope):
        from ..runtime.builder import default_runtime_profiles

        profile = default_runtime_profiles().get(profile_id)
        if profile is None:
            raise EvalConfigurationError(f"unknown runtime profile: {profile_id}")
        return RuntimeBuilder(environment={"provider_executor": _OfflineProvider()}).build(
            profile,
            cards=cards,
            knowledge_scope=scope,
            run_context_config=effective_budget_config(spec.budget_overrides),
        )

    def _build_live_components(self, spec, cards, scope):
        from ..runtime.builder import default_runtime_profiles

        profiles = default_runtime_profiles()
        if spec.profile_id not in profiles:
            raise EvalConfigurationError(f"unknown runtime profile: {spec.profile_id}")
        return RuntimeBuilder().build(
            profiles[spec.profile_id],
            cards=cards,
            knowledge_scope=scope,
            run_context_config=effective_budget_config(spec.budget_overrides),
        )

    def _run_m0(self, bundle, suite, spec, cases, components):
        records: list[CaseRunRecord] = []
        trajectories: list[TrajectoryRecord] = []
        for trial in range(1, spec.trials + 1):
            for case in cases:
                trace_path = bundle / "traces" / f"{case.case_id}-t{trial}.jsonl"
                runtime = components.create_run_context(trace_path=trace_path)
                route = _route(case.question)
                evidence = components.retriever.search(case.question, top_k=3)
                runtime.trace.close(status="complete")
                records.append(
                    self._record(
                        suite,
                        spec,
                        components,
                        case,
                        trial,
                        route=route,
                        observed={
                            "agent_called": False,
                            "retrieved_source_ids": [item.source_id for item in evidence],
                            "initial_evidence_source_ids": [item.source_id for item in evidence],
                            "initial_evidence_ranks": _ranked_evidence(evidence),
                        },
                        runtime=runtime,
                        trace_path=trace_path,
                    )
                )
                trajectories.append(self._trajectory(suite, case, trial, spec, records[-1]))
        return records, trajectories

    def _run_m5(self, bundle, suite, spec, cases, components):
        records: list[CaseRunRecord] = []
        trajectories: list[TrajectoryRecord] = []
        for trial in range(1, spec.trials + 1):
            for case in cases:
                trace_path = bundle / "traces" / f"{case.case_id}-t{trial}.jsonl"
                runtime = components.create_run_context(trace_path=trace_path)
                evidence = components.retriever.search(case.question, top_k=5)
                runtime.trace.close(status="complete")
                records.append(
                    self._record(
                        suite,
                        spec,
                        components,
                        case,
                        trial,
                        observed={
                            "retrieved_source_ids": [item.source_id for item in evidence],
                            "candidate_count": len(evidence),
                            "expected_source_ids": list(case.payload.get("expected_source_ids", ())),
                            "initial_evidence_source_ids": [item.source_id for item in evidence],
                            "initial_evidence_ranks": _ranked_evidence(evidence),
                        },
                        runtime=runtime,
                        trace_path=trace_path,
                    )
                )
                trajectories.append(self._trajectory(suite, case, trial, spec, records[-1]))
        return records, trajectories

    def _run_pipeline(self, bundle, suite, spec, cases, components, public):
        records: list[CaseRunRecord] = []
        trajectories: list[TrajectoryRecord] = []
        content_policy = (
            TraceContentPolicy.PUBLIC_EVAL_CONTENT
            if public
            else TraceContentPolicy.METADATA_ONLY
        )
        for trial in range(1, spec.trials + 1):
            for case in cases:
                trace_path = bundle / "traces" / f"{case.case_id}-t{trial}.jsonl"
                started = perf_counter()
                runtime = None
                try:
                    runtime = components.create_run_context(
                        trace_path=trace_path, content_policy=content_policy
                    )
                    pipeline = components.pipeline(runtime=runtime)
                    response = pipeline.answer(case.question)
                    records.append(
                        self._pipeline_record(
                            suite, spec, components, case, trial, response, pipeline, runtime,
                            trace_path, (perf_counter() - started) * 1000,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - infrastructure is recorded, not hidden
                    records.append(
                        self._record(
                            suite,
                            spec,
                            components,
                            case,
                            trial,
                            status=CaseRunStatus.ERROR,
                            error_code="execution_error",
                            error_message=str(exc)[:300],
                            runtime=runtime,
                            trace_path=trace_path,
                            elapsed_ms=(perf_counter() - started) * 1000,
                        )
                    )
                trajectories.append(self._trajectory(suite, case, trial, spec, records[-1]))
        return records, trajectories

    def _run_policy(self, bundle, suite, spec, cases, components, cards, public):
        """Evaluate the policy contract directly over frozen evidence.

        A policy suite is not a pipeline suite: retrieval and agent execution
        would change the fixture being evaluated and erase the policy veto
        boundary that M2/M3 are intended to measure.
        """

        records: list[CaseRunRecord] = []
        trajectories: list[TrajectoryRecord] = []
        content_policy = (
            TraceContentPolicy.PUBLIC_EVAL_CONTENT
            if public
            else TraceContentPolicy.METADATA_ONLY
        )
        for trial in range(1, spec.trials + 1):
            for case in cases:
                trace_path = bundle / "traces" / f"{case.case_id}-t{trial}.jsonl"
                started = perf_counter()
                runtime = components.create_run_context(
                    trace_path=trace_path, content_policy=content_policy
                )
                try:
                    evidence = _frozen_evidence(case, cards)
                    proposed_query = str(case.payload.get("proposed_query", ""))
                    assessment = components.evidence_policy.assess(
                        str(case.payload.get("question", "")),
                        evidence,
                        proposed_query,
                        runtime=runtime,
                    )
                    validate_assessment(assessment, evidence, components.knowledge_scope)
                    observed = {
                        "agent_called": False,
                        "tool_proposed": False,
                        "tool_executed": False,
                        "initial_evidence_source_ids": [item.source_id for item in evidence],
                        "initial_evidence_ranks": _ranked_evidence(evidence),
                        "retrieved_source_ids": [item.source_id for item in evidence],
                        "proposed_query": proposed_query if public else None,
                        "proposed_query_sha256": canonical_hash(proposed_query),
                        "policy_decision": assessment.decision.value,
                        "policy_supporting_source_ids": list(assessment.supporting_source_ids),
                        "policy_reason_codes": list(assessment.reason_codes),
                        "matched_topic_ids": list(assessment.matched_topic_ids),
                    }
                    runtime.trace.close(status="complete")
                    records.append(
                        self._record(
                            suite,
                            spec,
                            components,
                            case,
                            trial,
                            harness_disposition=assessment.decision.value,
                            provider_calls_used=runtime.budget.provider_calls_used,
                            tool_executions_used=runtime.budget.tool_executions_used,
                            input_tokens_used=runtime.budget.input_tokens_used,
                            output_tokens_used=runtime.budget.output_tokens_used,
                            total_tokens_used=runtime.budget.total_tokens_used,
                            elapsed_ms=(perf_counter() - started) * 1000,
                            observed=observed,
                            runtime=runtime,
                            trace_path=trace_path,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - record construction failure
                    _close_runtime(runtime, "error")
                    records.append(
                        self._record(
                            suite,
                            spec,
                            components,
                            case,
                            trial,
                            status=CaseRunStatus.ERROR,
                            error_code="policy_execution_error",
                            error_message=str(exc)[:300],
                            elapsed_ms=(perf_counter() - started) * 1000,
                            runtime=runtime,
                            trace_path=trace_path,
                        )
                    )
                trajectories.append(self._trajectory(suite, case, trial, spec, records[-1]))
        return records, trajectories

    def _run_verifier(self, bundle, suite, spec, cases, components, cards, public):
        """Evaluate grounding/claim support without constructing an agent run."""

        records: list[CaseRunRecord] = []
        trajectories: list[TrajectoryRecord] = []
        content_policy = (
            TraceContentPolicy.PUBLIC_EVAL_CONTENT
            if public
            else TraceContentPolicy.METADATA_ONLY
        )
        for trial in range(1, spec.trials + 1):
            for case in cases:
                trace_path = bundle / "traces" / f"{case.case_id}-t{trial}.jsonl"
                started = perf_counter()
                runtime = components.create_run_context(
                    trace_path=trace_path, content_policy=content_policy
                )
                try:
                    evidence = _frozen_evidence(case, cards)
                    claims = tuple(
                        GroundedClaim(
                            str(item["text"]), tuple(item.get("citation_ids", ()))
                        )
                        for item in case.payload.get("claims", ())
                    )
                    citation_check = verify_citations(
                        GenerationDraft(
                            str(case.payload.get("answer", "")),
                            [source_id for claim in claims for source_id in claim.citation_ids],
                        ),
                        evidence,
                    )
                    observed = {
                        "agent_called": False,
                        "tool_proposed": False,
                        "tool_executed": False,
                        "initial_evidence_source_ids": [item.source_id for item in evidence],
                        "initial_evidence_ranks": _ranked_evidence(evidence),
                        "retrieved_source_ids": [item.source_id for item in evidence],
                        "final_claim_ids": [
                            f"claim-{index}" for index, _claim in enumerate(claims)
                        ],
                        "final_citation_ids": [
                            list(claim.citation_ids) for claim in claims
                        ],
                        "citation_integrity_valid": citation_check.valid,
                    }
                    if not citation_check.valid:
                        observed["claim_verdicts"] = []
                        observed["harness_disposition"] = "invalid_citation"
                        runtime.trace.close(status="complete")
                        records.append(
                            self._record(
                                suite,
                                spec,
                                components,
                                case,
                                trial,
                                route=Route.ABSTAIN.value,
                                harness_disposition="invalid_citation",
                                provider_calls_used=runtime.budget.provider_calls_used,
                                tool_executions_used=runtime.budget.tool_executions_used,
                                input_tokens_used=runtime.budget.input_tokens_used,
                                output_tokens_used=runtime.budget.output_tokens_used,
                                total_tokens_used=runtime.budget.total_tokens_used,
                                elapsed_ms=(perf_counter() - started) * 1000,
                                observed=observed,
                                runtime=runtime,
                                trace_path=trace_path,
                            )
                        )
                    elif components.claim_support_verifier is not None:
                        cited_evidence = materialize_cited_evidence(claims, evidence)
                        result = components.claim_support_verifier.verify(
                            claims, cited_evidence, runtime=runtime
                        )
                        validate_claim_support_result(result, claims, evidence)
                        verdicts = [item.verdict.value for item in result.claim_results]
                        observed["claim_verdicts"] = verdicts
                        observed["claim_supporting_source_ids"] = [
                            list(item.supporting_source_ids) for item in result.claim_results
                        ]
                        disposition = (
                            "answer"
                            if all(item.verdict == ClaimVerdict.SUPPORTED for item in result.claim_results)
                            else "claim_support_failed"
                        )
                        observed["harness_disposition"] = disposition
                        runtime.trace.close(status="complete")
                        records.append(
                            self._record(
                                suite,
                                spec,
                                components,
                                case,
                                trial,
                                route=Route.ANSWER.value if disposition == "answer" else Route.ABSTAIN.value,
                                harness_disposition=disposition,
                                provider_calls_used=runtime.budget.provider_calls_used,
                                tool_executions_used=runtime.budget.tool_executions_used,
                                input_tokens_used=runtime.budget.input_tokens_used,
                                output_tokens_used=runtime.budget.output_tokens_used,
                                total_tokens_used=runtime.budget.total_tokens_used,
                                elapsed_ms=(perf_counter() - started) * 1000,
                                observed=observed,
                                runtime=runtime,
                                trace_path=trace_path,
                            )
                        )
                    else:
                        result = components.grounding_verifier.verify(
                            str(case.payload.get("answer", "")), claims, evidence, runtime=runtime
                        )
                        validate_grounding_result(result, claims, evidence)
                        verdicts = [item.verdict.value for item in result.claim_results]
                        observed["claim_verdicts"] = verdicts
                        observed["coverage_ok"] = result.coverage_ok
                        observed["claim_supporting_source_ids"] = [
                            list(item.supporting_source_ids) for item in result.claim_results
                        ]
                        grounding_succeeded = result.coverage_ok and all(
                            item.verdict == ClaimVerdict.SUPPORTED
                            for item in result.claim_results
                        )
                        disposition = "answer" if grounding_succeeded else "grounding_failed"
                        observed["harness_disposition"] = disposition
                        runtime.trace.close(status="complete")
                        records.append(
                            self._record(
                                suite,
                                spec,
                                components,
                                case,
                                trial,
                                route=(
                                    Route.ANSWER.value
                                    if grounding_succeeded
                                    else Route.ABSTAIN.value
                                ),
                                harness_disposition=disposition,
                                provider_calls_used=runtime.budget.provider_calls_used,
                                tool_executions_used=runtime.budget.tool_executions_used,
                                input_tokens_used=runtime.budget.input_tokens_used,
                                output_tokens_used=runtime.budget.output_tokens_used,
                                total_tokens_used=runtime.budget.total_tokens_used,
                                elapsed_ms=(perf_counter() - started) * 1000,
                                observed=observed,
                                runtime=runtime,
                                trace_path=trace_path,
                            )
                        )
                except Exception as exc:  # noqa: BLE001 - record verifier failure
                    _close_runtime(runtime, "error")
                    records.append(
                        self._record(
                            suite,
                            spec,
                            components,
                            case,
                            trial,
                            status=CaseRunStatus.ERROR,
                            error_code="verifier_execution_error",
                            error_message=str(exc)[:300],
                            elapsed_ms=(perf_counter() - started) * 1000,
                            runtime=runtime,
                            trace_path=trace_path,
                        )
                    )
                trajectories.append(self._trajectory(suite, case, trial, spec, records[-1]))
        return records, trajectories

    def _run_replay(self, bundle, suite, spec, cases, cards, scope):
        replay_dir = Path(spec.replay_run_dir or spec.replay_source or "")
        if not replay_dir.is_dir():
            raise EvalConfigurationError(f"replay run directory does not exist: {replay_dir}")
        config_path = replay_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        model = config.get("model") or config.get("provider", {}).get("model") or "deepseek-flash"
        from ..runtime.builder import default_runtime_profiles

        profile = default_runtime_profiles().get(spec.profile_id or suite.default_profile_id)
        if profile is None:
            raise EvalConfigurationError(f"unknown replay profile: {spec.profile_id}")
        records: list[CaseRunRecord] = []
        trajectories: list[TrajectoryRecord] = []
        components = None
        for trial in range(1, spec.trials + 1):
            for case in cases:
                provider_path = replay_dir / "provider_exchanges" / f"{case.case_id}.jsonl"
                tool_path = replay_dir / "tool_exchanges" / f"{case.case_id}.jsonl"
                provider = ReplayProviderExecutor(read_provider_exchanges(provider_path))
                trace_path = bundle / "traces" / f"{case.case_id}-t{trial}.jsonl"
                components = RuntimeBuilder(
                    environment={"provider_executor": provider, "provider_model": model}
                ).build(
                    profile,
                    cards=cards,
                    knowledge_scope=scope,
                    run_context_config=effective_budget_config(spec.budget_overrides),
                )
                runtime = RunContext.create(
                    "m7_replay",
                    budget=effective_budget_config(spec.budget_overrides),
                    trace=components.trace_factory.create(trace_path),
                )
                pipeline = components.pipeline(runtime=runtime)
                tool_runner = ReplayToolRunner(read_tool_exchanges(tool_path))
                if pipeline.agent_loop is not None:
                    pipeline.agent_loop.tool_runner = tool_runner
                elif pipeline.orchestrator is not None:
                    pipeline.orchestrator.tool_runner = tool_runner
                else:
                    raise EvalConfigurationError("replay profile did not construct an executable runtime")
                started = perf_counter()
                try:
                    response = pipeline.answer(case.question)
                    records.append(
                        self._pipeline_record(
                            suite, spec, components, case, trial, response, pipeline, runtime,
                            trace_path, (perf_counter() - started) * 1000,
                            observed_extra={
                                "provider_exchanges_remaining": provider.remaining_exchanges,
                                "tool_exchanges_remaining": tool_runner.remaining_exchanges,
                                "live_provider_called": False,
                            },
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    records.append(
                        self._record(
                            suite,
                            spec,
                            components,
                            case,
                            trial,
                            status=CaseRunStatus.ERROR,
                            error_code="replay_execution_error",
                            error_message=str(exc)[:300],
                            runtime=runtime,
                            trace_path=trace_path,
                        )
                    )
                trajectories.append(self._trajectory(suite, case, trial, spec, records[-1]))
        if components is None:
            raise EvalConfigurationError("replay dataset has no cases")
        return records, trajectories, components

    def _pipeline_record(
        self, suite, spec, components, case, trial, response, pipeline, runtime, trace_path,
        elapsed_ms, observed_extra=None,
    ):
        run = pipeline.last_agent_run
        team = getattr(pipeline, "last_team_run", None)
        state = run.state if run else None
        proposed_query = _proposed_query(state)
        final_claims = list(run.claims) if run else list(team.claims) if team else []
        final_citation_ids = list(run.draft.citation_ids) if run and run.draft else [
            source_id for claim in final_claims for source_id in claim.citation_ids
        ]
        team_reports = list(team.reports) if team else []
        initial_team_evidence = (
            [item.evidence.source_id for item in team.state.evidence_ledger.list_items() if item.origin.value == "initial"]
            if team else []
        )
        observed = {
            "agent_called": run is not None,
            "team_called": team is not None,
            "tool_proposed": bool(state and state.tool_proposals_used) or any(item.tool_proposals for item in team_reports),
            "tool_executed": bool(state and state.tool_calls_used) or any(item.tool_executions for item in team_reports),
            "tool_proposal_count": state.tool_proposals_used if state else sum(item.tool_proposals for item in team_reports),
            "tool_execution_count": state.tool_calls_used if state else sum(item.tool_executions for item in team_reports),
            "proposed_query": (
                proposed_query if spec.trace_content_policy == "public_eval_content" else None
            ),
            "proposed_query_sha256": canonical_hash(proposed_query) if proposed_query else None,
            "policy_decision": state.policy_decision if state else None,
            "policy_reason_codes": list(state.policy_reason_codes) if state else [],
            "policy_supporting_source_ids": list(state.policy_supporting_source_ids) if state else [],
            "matched_topic_ids": list(state.policy_matched_topic_ids) if state else [],
            "initial_evidence_source_ids": [
                item.source_id for item in run.initial_ranked_evidence
            ] if run else initial_team_evidence,
            "initial_evidence_ranks": _ranked_evidence(run.initial_ranked_evidence) if run else [
                {"rank": rank, "source_id": source_id} for rank, source_id in enumerate(initial_team_evidence, 1)
            ],
            "recovery_evidence_source_ids": [
                item.source_id for item in run.recovery_ranked_evidence
            ] if run else [],
            "retrieved_source_ids": [item.source_id for item in run.observed_evidence]
            if run
            else [item.source_id for item in team.observed_evidence] if team else [],
            "tool_observation_source_ids": _tool_observation_source_ids(state),
            "final_claim_ids": [f"claim-{index}" for index, _claim in enumerate(final_claims)],
            "final_citation_ids": final_citation_ids,
            "citation_integrity_valid": pipeline.last_harness_disposition != "invalid_citation",
            "claim_verdicts": [item.verdict.value for item in getattr(pipeline.last_claim_support_result, "claim_results", ())],
            "team_lead_calls": team.state.lead_calls_used if team else None,
            "team_tasks_created": team.state.tasks_created if team else None,
            "team_workers_started": team.state.workers_started if team else None,
            "team_worker_successes": sum(item.status.value == "succeeded" for item in team_reports) if team else None,
            "team_delegated": bool(team and team.state.tasks_created),
            "team_stop_reason": team.stop_reason.value if team and team.stop_reason else None,
        }
        observed.update(observed_extra or {})
        return self._record(
            suite,
            spec,
            components,
            case,
            trial,
            route=response.route.value,
            agent_stop_reason=(run.stop_reason.value if run and run.stop_reason else team.stop_reason.value if team and team.stop_reason else None),
            harness_disposition=pipeline.last_harness_disposition,
            provider_calls_used=runtime.budget.provider_calls_used,
            tool_executions_used=runtime.budget.tool_executions_used,
            input_tokens_used=runtime.budget.input_tokens_used,
            output_tokens_used=runtime.budget.output_tokens_used,
            total_tokens_used=runtime.budget.total_tokens_used,
            elapsed_ms=elapsed_ms,
            observed=observed,
            runtime=runtime,
            trace_path=trace_path,
        )

    def _record(
        self, suite, spec, components, case, trial, *, status=CaseRunStatus.COMPLETE,
        route=None, agent_stop_reason=None, harness_disposition=None, provider_calls_used=None,
        tool_executions_used=None, input_tokens_used=None, output_tokens_used=None,
        total_tokens_used=None, elapsed_ms=None, observed=None, error_code=None,
        error_message=None, runtime=None, trace_path=None,
    ):
        if runtime is not None:
            provider_calls_used = (
                runtime.budget.provider_calls_used
                if provider_calls_used is None
                else provider_calls_used
            )
            tool_executions_used = (
                runtime.budget.tool_executions_used
                if tool_executions_used is None
                else tool_executions_used
            )
            input_tokens_used = (
                runtime.budget.input_tokens_used
                if input_tokens_used is None
                else input_tokens_used
            )
            output_tokens_used = (
                runtime.budget.output_tokens_used
                if output_tokens_used is None
                else output_tokens_used
            )
            total_tokens_used = (
                runtime.budget.total_tokens_used
                if total_tokens_used is None
                else total_tokens_used
            )
        return CaseRunRecord(
            suite_id=suite.suite_id,
            case_id=case.case_id,
            trial=trial,
            run_id=runtime.identity.run_id if runtime is not None else None,
            execution_mode=spec.execution_mode,
            profile_id=components.profile.profile_id if components else spec.profile_id,
            component_manifest_hash=components.manifest_hash if components else None,
            code_commit=components.component_manifest.code_commit if components else None,
            status=status,
            route=route,
            agent_stop_reason=agent_stop_reason,
            harness_disposition=harness_disposition,
            provider_calls_used=provider_calls_used,
            tool_executions_used=tool_executions_used,
            input_tokens_used=input_tokens_used,
            output_tokens_used=output_tokens_used,
            total_tokens_used=total_tokens_used,
            elapsed_ms=elapsed_ms,
            trace_reference=str(trace_path.relative_to(trace_path.parents[1])) if trace_path else None,
            observed=observed or {},
            error_code=error_code,
            error_message=error_message,
            eval_run_id=spec.spec_hash,
            eval_spec_hash=spec.spec_hash,
        )

    def _trajectory(self, suite, case, trial, spec, record):
        observed = record.observed
        events: list[Mapping[str, Any]] = [
            {
                "event": "run_start",
                "execution_run_id": record.run_id,
                "eval_run_id": record.eval_run_id,
                "profile_id": record.profile_id,
                "component_manifest_hash": record.component_manifest_hash,
            }
        ]
        if observed.get("initial_evidence_source_ids") is not None:
            events.append(
                {
                    "event": "initial_evidence",
                    "source_ids": list(observed.get("initial_evidence_source_ids", ())),
                    "ranks": list(observed.get("initial_evidence_ranks", ())),
                }
            )
        if observed.get("tool_proposed") is not None:
            events.append(
                {
                    "event": "tool_proposal",
                    "proposed": bool(observed.get("tool_proposed")),
                    "count": observed.get("tool_proposal_count", 0),
                    "proposed_query": observed.get("proposed_query"),
                    "proposed_query_sha256": observed.get("proposed_query_sha256"),
                }
            )
        if observed.get("policy_decision") is not None:
            events.append(
                {
                    "event": "policy_decision",
                    "decision": observed.get("policy_decision"),
                    "reason_codes": list(observed.get("policy_reason_codes", ())),
                    "matched_topic_ids": list(observed.get("matched_topic_ids", ())),
                    "supporting_source_ids": list(
                        observed.get("policy_supporting_source_ids", ())
                    ),
                }
            )
        if observed.get("tool_executed") is not None:
            events.append(
                {
                    "event": "tool_execution",
                    "executed": bool(observed.get("tool_executed")),
                    "count": observed.get("tool_execution_count", record.tool_executions_used or 0),
                }
            )
        if observed.get("tool_observation_source_ids") is not None:
            events.append(
                {
                    "event": "tool_observation",
                    "source_ids": list(observed.get("tool_observation_source_ids", ())),
                }
            )
        if observed.get("final_claim_ids") is not None or observed.get("final_citation_ids") is not None:
            events.append(
                {
                    "event": "final_claims",
                    "claim_ids": list(observed.get("final_claim_ids", ())),
                    "citation_ids": list(observed.get("final_citation_ids", ())),
                    "claim_verdicts": list(observed.get("claim_verdicts", ())),
                }
            )
        events.extend(
            (
                {"event": "agent_stop", "reason": record.agent_stop_reason},
                {"event": "harness_disposition", "disposition": record.harness_disposition},
                {
                    "event": "budget",
                    "provider_calls_used": record.provider_calls_used,
                    "tool_executions_used": record.tool_executions_used,
                    "input_tokens_used": record.input_tokens_used,
                    "output_tokens_used": record.output_tokens_used,
                    "total_tokens_used": record.total_tokens_used,
                },
                {
                    "event": "case_complete",
                    "status": record.status.value,
                    "route": record.route,
                },
            )
        )
        return TrajectoryRecord(
            suite.suite_id,
            case.case_id,
            trial,
            schema_version=(
                "team_trajectory_v1"
                if record.profile_id == "m8-team-bm25-v1"
                else "trajectory_v1"
            ),
            content_policy=spec.trace_content_policy,
            events=tuple(events),
            eval_run_id=record.eval_run_id,
            execution_run_id=record.run_id,
        )

    def _grade(self, suite, cases, records):
        by_case = {case.case_id: case for case in cases}
        results: list[GraderResult] = []
        for record in records:
            case = by_case[record.case_id]
            for grader_id in suite.grader_ids:
                grader = self.graders.get(grader_id)
                if grader is None:
                    results.append(
                        GraderResult(
                            grader_id, "unknown", case.case_id, record.trial, GraderStatus.ERROR,
                            reason_codes=("unknown_grader",),
                        )
                    )
                    continue
                try:
                    results.append(grader.grade(case, record))
                except Exception as exc:  # noqa: BLE001 - grader errors are denominator-excluded
                    results.append(
                        GraderResult(
                            grader_id,
                            getattr(grader, "version", "unknown"),
                            case.case_id,
                            record.trial,
                            GraderStatus.ERROR,
                            reason_codes=("grader_error", str(exc)[:120]),
                        )
                    )
        return results

    def _add_m0_parity_metrics(self, metrics, records, cases, version):
        case_by_id = {case.case_id: case for case in cases}
        route_rows = [
            record for record in records
            if case_by_id[record.case_id].payload.get("expected_route")
            in {"answer", "urgent", "urgent_care", "prescription", "human_review"}
        ]
        safety_correct = sum(
            _normalize_expected_route(case_by_id[record.case_id].payload["expected_route"])
            == record.route
            for record in route_rows
        )
        metrics["m0.safety_route_accuracy"] = _ratio_metric(
            "m0.safety_route_accuracy", safety_correct, len(route_rows), version
        )
        retrieval_rows = [
            record for record in records
            if case_by_id[record.case_id].payload.get("expected_source_ids")
        ]
        hit1 = sum(
            bool(set(record.observed.get("retrieved_source_ids", [])[:1]).intersection(
                case_by_id[record.case_id].payload["expected_source_ids"]
            ))
            for record in retrieval_rows
        )
        hit3 = sum(
            bool(set(record.observed.get("retrieved_source_ids", [])[:3]).intersection(
                case_by_id[record.case_id].payload["expected_source_ids"]
            ))
            for record in retrieval_rows
        )
        metrics["m0.retrieval_hit_at_1"] = _ratio_metric("m0.retrieval_hit_at_1", hit1, len(retrieval_rows), version)
        metrics["m0.retrieval_hit_at_3"] = _ratio_metric("m0.retrieval_hit_at_3", hit3, len(retrieval_rows), version)
        reciprocal = 0.0
        for record in retrieval_rows:
            expected = set(case_by_id[record.case_id].payload["expected_source_ids"])
            for rank, source_id in enumerate(record.observed.get("retrieved_source_ids", []), 1):
                if source_id in expected:
                    reciprocal += 1 / rank
                    break
        metrics["m0.retrieval_mrr"] = _average_metric("m0.retrieval_mrr", reciprocal, len(retrieval_rows), version)

    @staticmethod
    def _m5_metrics(records, version):
        values = {"hit_at_1": [], "hit_at_3": [], "hit_at_5": [], "recall_at_1": [], "recall_at_3": [], "recall_at_5": [], "mrr": []}
        for record in records:
            expected = record.observed.get("expected_source_ids", [])
            if not expected:
                continue
            ids = record.observed.get("retrieved_source_ids", [])
            expected_set = set(expected)
            rank = next((i for i, item in enumerate(ids, 1) if item in expected_set), None)
            values["hit_at_1"].append(float(rank == 1))
            values["hit_at_3"].append(float(rank is not None and rank <= 3))
            values["hit_at_5"].append(float(rank is not None and rank <= 5))
            for k in (1, 3, 5):
                values[f"recall_at_{k}"].append(len(set(ids[:k]) & expected_set) / len(expected_set))
            values["mrr"].append(1 / rank if rank else 0.0)
        return {
            f"m5.{key}": _average_metric(f"m5.{key}", sum(items), len(items), version)
            for key, items in values.items()
        }

    @staticmethod
    def _m8_metrics(records, cases, version):
        """First-class M8 cost, topology, and per-category metrics."""

        complete = [record for record in records if record.status == CaseRunStatus.COMPLETE]

        def average(metric_id, values, unit="per_case"):
            values = [value for value in values if value is not None]
            return MetricResult(
                metric_id,
                version,
                sum(values) / len(values) if values else None,
                sum(values) if values else None,
                len(values),
                unit,
                "complete_cases",
                "sum/denominator",
            )

        metrics = {
            "m8.provider_calls_per_case": average(
                "m8.provider_calls_per_case", [row.provider_calls_used for row in complete]
            ),
            "m8.tool_executions_per_case": average(
                "m8.tool_executions_per_case", [row.tool_executions_used for row in complete]
            ),
            "m8.input_tokens_per_case": average(
                "m8.input_tokens_per_case", [row.input_tokens_used for row in complete]
            ),
            "m8.output_tokens_per_case": average(
                "m8.output_tokens_per_case", [row.output_tokens_used for row in complete]
            ),
            "m8.total_tokens_per_case": average(
                "m8.total_tokens_per_case", [row.total_tokens_used for row in complete]
            ),
            "m8.elapsed_ms_per_case": average(
                "m8.elapsed_ms_per_case", [row.elapsed_ms for row in complete], "ms"
            ),
            "m8.delegation_rate": MetricResult.ratio(
                "m8.delegation_rate",
                version,
                sum(bool(row.observed.get("team_delegated")) for row in complete),
                len(complete),
            ),
            "m8.workers_started_per_case": average(
                "m8.workers_started_per_case",
                [row.observed.get("team_workers_started") for row in complete],
            ),
            "m8.lead_calls_per_case": average(
                "m8.lead_calls_per_case",
                [row.observed.get("team_lead_calls") for row in complete],
            ),
            "m8.budget_exhaustion_rate": MetricResult.ratio(
                "m8.budget_exhaustion_rate",
                version,
                sum(row.harness_disposition == "team_budget_exhausted" for row in complete),
                len(complete),
            ),
        }
        case_by_id = {case.case_id: case for case in cases}
        for category in sorted({case.payload.get("category") for case in cases if case.payload.get("category")}):
            rows = [row for row in complete if case_by_id[row.case_id].payload.get("category") == category]
            expected = sum(
                _normalize_expected_route(case_by_id[row.case_id].payload.get("expected_route")) == row.route
                for row in rows
            )
            key = f"m8.category.{category}.route_accuracy"
            metrics[key] = MetricResult.ratio(key, version, expected, len(rows), scope=category)
        return metrics

    def _new_bundle(self, spec):
        root = Path(spec.output_root)
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
        path = root / timestamp
        path.mkdir(parents=True, exist_ok=False)
        (path / "traces").mkdir()
        return path

    def _write_bundle(self, bundle, spec, suite, cases, records, grader_results, failures, trajectories, metrics, components, public):
        if components is not None:
            components.write_manifest(bundle / "component_manifest.json")
        _write_json(bundle / "run_spec.json", spec.to_dict() | {"run_spec_hash": spec.spec_hash})
        _write_jsonl(bundle / "case_results.jsonl", (row.to_dict() for row in records))
        _write_jsonl(bundle / "grader_results.jsonl", (row.to_dict() for row in grader_results))
        _write_jsonl(bundle / "failures.jsonl", (row.to_dict() for row in failures))
        _write_jsonl(bundle / "trajectories.jsonl", (row.to_dict() for row in trajectories))
        _write_json(bundle / "metrics.json", metrics_to_dict(metrics))
        if public:
            _write_jsonl(bundle / "cases.jsonl", (case.to_dict() for case in cases))
        _write_text(bundle / "report.md", _report(suite, spec, metrics, failures))
        file_hashes = {
            str(path.relative_to(bundle)).replace("\\", "/"): sha256_file(path)
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": "m7-artifact-manifest-v1",
            "suite_id": suite.suite_id,
            "suite_version": suite.version,
            "target_kind": suite.target_kind.value,
            "execution_mode": spec.execution_mode.value,
            "profile_id": spec.profile_id,
            "dataset_path": spec.dataset_path,
            "dataset_sha256": spec.dataset_sha256,
            "run_spec_hash": spec.spec_hash,
            "component_manifest_hash": components.manifest_hash if components else None,
            "metric_definition_version": suite.metric_definition_version,
            "trace_content_policy": spec.trace_content_policy,
            "file_sha256": file_hashes,
            "manifest_hash": canonical_hash(file_hashes | {"suite_id": suite.suite_id, "run_spec_hash": spec.spec_hash}),
        }
        _write_json(bundle / "run_manifest.json", manifest)

    @staticmethod
    def _validate_content_policy(suite, spec, public):
        if public and not suite.public_content_allowed:
            raise EvalConfigurationError(
                f"suite {suite.suite_id} does not permit public evaluation content"
            )
        if spec.trace_content_policy == "public_eval_content" and not public:
            raise EvalConfigurationError(
                "public_eval_content requires the explicit --public-eval-content opt-in"
            )


def _frozen_evidence(case: EvalCase, cards) -> tuple[Evidence, ...]:
    """Materialize fixture evidence by ID; never rerun retrieval for target evals."""

    cards_by_id = {card.id: card for card in cards}
    source_ids = tuple(case.payload.get("evidence_source_ids", ()))
    missing = [source_id for source_id in source_ids if source_id not in cards_by_id]
    if missing:
        raise EvalConfigurationError(
            f"case {case.case_id} references unknown evidence source(s): {missing}"
        )
    return tuple(
        Evidence(card.id, card.title, card.content, card.source_url, 0.0)
        for card in (cards_by_id[source_id] for source_id in source_ids)
    )


def _ranked_evidence(evidence) -> list[dict[str, Any]]:
    return [
        {"rank": rank, "source_id": item.source_id}
        for rank, item in enumerate(evidence, 1)
    ]


def _proposed_query(state) -> str | None:
    if state is None:
        return None
    for message in reversed(tuple(state.session.messages)):
        if not isinstance(message, AssistantToolCallMessage):
            continue
        for call in reversed(tuple(message.tool_calls)):
            arguments = call.arguments
            if isinstance(arguments, Mapping) and isinstance(arguments.get("query"), str):
                return arguments["query"]
    return None


def _tool_observation_source_ids(state) -> list[str]:
    if state is None:
        return []
    source_ids: list[str] = []
    for message in state.session.messages:
        if not isinstance(message, ToolResultMessage):
            continue
        source_ids.extend(item.source_id for item in message.result.observed_evidence)
    return list(dict.fromkeys(source_ids))


def _close_runtime(runtime, status: str) -> None:
    if runtime is not None and runtime.trace is not None:
        runtime.trace.close(status=status)


def _bind_trajectory_eval_identity(trajectory: TrajectoryRecord, eval_run_id: str) -> TrajectoryRecord:
    events = tuple(
        {**event, "eval_run_id": eval_run_id}
        if event.get("event") == "run_start"
        else event
        for event in trajectory.events
    )
    return replace(trajectory, eval_run_id=eval_run_id, events=events)


def _route(question: str | None) -> str:
    result = route_question(question or "")
    return result.route.value if result else Route.ANSWER.value


def _normalize_expected_route(value: object) -> str:
    return {
        "urgent": Route.URGENT_CARE.value,
        "prescription": Route.HUMAN_REVIEW.value,
    }.get(value, value) if isinstance(value, str) else ""


def _ratio_metric(metric_id, numerator, denominator, version):
    from .schema import MetricResult

    return MetricResult.ratio(metric_id, version, int(numerator), int(denominator))


def _average_metric(metric_id, numerator, denominator, version):
    from .schema import MetricResult

    return MetricResult(
        metric_id, version, numerator / denominator if denominator else None,
        numerator, denominator, "score", "scored_cases", "sum/denominator",
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _report(suite, spec, metrics, failures):
    return "\n".join(
        [
            f"# M7 evaluation: {suite.suite_id}",
            "",
            f"- execution mode: `{spec.execution_mode.value}`",
            f"- dataset SHA-256: `{spec.dataset_sha256}`",
            f"- metric definition: `{suite.metric_definition_version}`",
            f"- failures: `{len(failures)}`",
            "",
            "Metrics are deterministic aggregates; missing/infrastructure cases are not silently placed in quality denominators.",
            "",
            "```json",
            json.dumps(metrics_to_dict(metrics), ensure_ascii=False, indent=2, sort_keys=True),
            "```",
        ]
    )
