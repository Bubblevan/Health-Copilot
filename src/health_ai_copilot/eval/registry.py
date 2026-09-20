"""Explicit source-defined M7 suite registry."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import EvalExecutionMode, EvalSuite, EvalTargetKind


class EvalSuiteRegistryError(RuntimeError):
    """Base registry error."""


class DuplicateEvalSuiteError(EvalSuiteRegistryError):
    pass


class UnknownEvalSuiteError(EvalSuiteRegistryError):
    pass


class UnsupportedEvalExecutionMode(EvalSuiteRegistryError):
    pass


@dataclass(frozen=True)
class EvalSuiteRegistration:
    suite: EvalSuite


class EvalSuiteRegistry:
    """Maps explicit suite IDs to versioned, trusted suite definitions."""

    def __init__(self) -> None:
        self._registrations: dict[str, EvalSuiteRegistration] = {}

    def register(self, suite: EvalSuite) -> None:
        if suite.suite_id in self._registrations:
            raise DuplicateEvalSuiteError(f"duplicate eval suite: {suite.suite_id}")
        self._registrations[suite.suite_id] = EvalSuiteRegistration(suite)

    def get(self, suite_id: str) -> EvalSuite:
        try:
            return self._registrations[suite_id].suite
        except KeyError as exc:
            raise UnknownEvalSuiteError(f"unknown eval suite: {suite_id}") from exc

    def list(self) -> tuple[EvalSuite, ...]:
        return tuple(self._registrations[key].suite for key in sorted(self._registrations))

    def validate_mode(self, suite: EvalSuite, mode: EvalExecutionMode) -> None:
        mode = EvalExecutionMode(mode)
        if mode not in suite.supported_execution_modes:
            supported = ", ".join(item.value for item in suite.supported_execution_modes)
            raise UnsupportedEvalExecutionMode(
                f"suite {suite.suite_id} does not support {mode.value}; supported: {supported}"
            )


def default_eval_suite_registry() -> EvalSuiteRegistry:
    """Return all historical suites without constructing a runtime/provider."""

    registry = EvalSuiteRegistry()
    registrations = (
        EvalSuite(
            "m0-regression-v1",
            "1",
            EvalTargetKind.DETERMINISTIC_GATE,
            "evals/m0.jsonl",
            (EvalExecutionMode.OFFLINE,),
            "m0-bm25-default",
            ("route", "safety_short_circuit", "retrieval_source"),
            "m0-parity-v1",
            provenance={
                "source_dataset": "M0.3 reviewed eval pack",
                "review_status": "reviewed field and source mapping; not clinical validation",
                "gold_semantics": "deterministic safety route and expected source IDs",
            },
            expected_dataset_sha256="b40ec6f88ae5900a9708dd75fe78eb35436799443cf6e45c818cc19573c53a46",
        ),
        EvalSuite(
            "m1-focused-v1",
            "1",
            EvalTargetKind.PIPELINE,
            "evals/m1_recovery.jsonl",
            (EvalExecutionMode.LIVE, EvalExecutionMode.REPLAY),
            "m1-bm25-default",
            ("route", "recovery_tool_behavior", "budget_termination"),
            "m1-focused-v1",
            public_content_allowed=True,
            provenance={"gold_semantics": "focused recovery diagnostic; not generalization"},
            expected_dataset_sha256="b76ac092f9d203563ad9f9d997599a071e44fae6c60e056416b22a96d72ff0c3",
        ),
        EvalSuite(
            "m2-policy-v1",
            "1",
            EvalTargetKind.POLICY,
            "evals/m2_policy.jsonl",
            (EvalExecutionMode.LIVE, EvalExecutionMode.REPLAY),
            "m2-bm25-default",
            ("policy_decision", "capability_topic"),
            "m2-policy-v1",
            provenance={"gold_semantics": "standalone evidence-policy fixture decisions"},
            expected_dataset_sha256="6b95736e91c42134ffcb8f89adac3f4f87c62b4159f93d0f854c67321df6f805",
        ),
        EvalSuite(
            "m2-grounding-v1",
            "1",
            EvalTargetKind.VERIFIER,
            "evals/m2_grounding.jsonl",
            (EvalExecutionMode.LIVE,),
            "m2-bm25-default",
            ("claim_verdict", "citation_integrity"),
            "m2-grounding-v1",
            provenance={"gold_semantics": "standalone grounding relation fixtures"},
            expected_dataset_sha256="15d7b08544e362b647ea1273b5d9eee4af1a8137608639467f7cc3dcad86b458",
        ),
        EvalSuite(
            "m2-focused-v1",
            "1",
            EvalTargetKind.PIPELINE,
            "evals/m1_recovery.jsonl",
            (EvalExecutionMode.LIVE, EvalExecutionMode.REPLAY),
            "m2-bm25-default",
            ("route", "recovery_tool_behavior", "citation_integrity", "budget_termination"),
            "m2-focused-v1",
            public_content_allowed=True,
            provenance={"gold_semantics": "focused M2 diagnostic over the historical recovery pack"},
            expected_dataset_sha256="b76ac092f9d203563ad9f9d997599a071e44fae6c60e056416b22a96d72ff0c3",
        ),
        EvalSuite(
            "m3-capability-v1",
            "1",
            EvalTargetKind.POLICY,
            "evals/m3_capability.jsonl",
            (EvalExecutionMode.LIVE,),
            "m3-bm25-default",
            ("policy_decision", "capability_topic"),
            "m3-capability-v1",
            provenance={
                "gold_semantics": "frozen evidence/query/scope state; BM25 is never rerun to make gold"
            },
            expected_dataset_sha256="e56f20ba986e6eaa3600e10719f4b3af01b919c22940c5f39bc1364cfbaf1f20",
        ),
        EvalSuite(
            "m3-claim-support-v1",
            "1",
            EvalTargetKind.VERIFIER,
            "evals/m3_claim_support.jsonl",
            (EvalExecutionMode.LIVE,),
            "m3-bm25-default",
            ("claim_verdict", "citation_integrity"),
            "m3-claim-support-v1",
            provenance={
                "gold_semantics": "disposition-level rejection and fine-grained claim verdict are separate"
            },
            expected_dataset_sha256="4c04ac90089ab3cdad77406c8bf943938bdc2a48c823269fd9626e7231851e92",
        ),
        EvalSuite(
            "m3-focused-v1",
            "1",
            EvalTargetKind.PIPELINE,
            "evals/m1_recovery.jsonl",
            (EvalExecutionMode.LIVE, EvalExecutionMode.REPLAY),
            "m3-bm25-default",
            (
                "route",
                "recovery_tool_behavior",
                "citation_integrity",
                "claim_verdict",
                "budget_termination",
            ),
            "m3-focused-v1",
            public_content_allowed=True,
            provenance={"gold_semantics": "focused M2-vs-M3 compatibility diagnostic"},
            expected_dataset_sha256="b76ac092f9d203563ad9f9d997599a071e44fae6c60e056416b22a96d72ff0c3",
        ),
        EvalSuite(
            "m4-replay-v1",
            "1",
            EvalTargetKind.REPLAY,
            "evals/m4_replay.jsonl",
            (EvalExecutionMode.REPLAY,),
            "m3-bm25-default",
            ("route", "replay_consistency", "budget_termination"),
            "m4-replay-v1",
            public_content_allowed=True,
            provenance={
                "source_dataset": "fixed six-case subset of evals/m1_recovery.jsonl",
                "review_status": "public reviewed fixture replay diagnostic; not generalization",
            },
            expected_dataset_sha256="08edbdfb0b84d9555ac3c01e14e20d5c9191a8d4a56c56f49e0a3035d6aaac65",
        ),
        EvalSuite(
            "m5-product-retrieval-v1",
            "1",
            EvalTargetKind.RETRIEVER,
            "evals/retrieval/m5_product_retrieval_v1.jsonl",
            (EvalExecutionMode.OFFLINE,),
            "m0-bm25-default",
            ("retrieval_source",),
            "m5-retrieval-v1",
            public_content_allowed=True,
            provenance={
                "source_dataset": "mechanically reused frozen/reviewed component rows",
                "review_status": "component-derived; not independent external generalization",
                "gold_semantics": "expected source IDs and corpus-uncovered controls",
            },
            expected_dataset_sha256="f6d580971aefbf3764da9ad1edb56c922b13da1e78093f535f79505cf6a06d33",
        ),
        EvalSuite(
            "m5-focused-e2e-v1",
            "1",
            EvalTargetKind.PIPELINE,
            "evals/m4_replay.jsonl",
            (EvalExecutionMode.LIVE, EvalExecutionMode.REPLAY),
            "m3-bm25-default",
            ("route", "recovery_tool_behavior", "budget_termination"),
            "m5-focused-e2e-v1",
            public_content_allowed=True,
            provenance={"gold_semantics": "six-case compatibility diagnostic; not causal retrieval evidence"},
            expected_dataset_sha256="08edbdfb0b84d9555ac3c01e14e20d5c9191a8d4a56c56f49e0a3035d6aaac65",
        ),
        EvalSuite(
            "m8-agent-team-focused-v1",
            "1",
            EvalTargetKind.ORCHESTRATION,
            "evals/m8_agent_team_focused_v1.jsonl",
            (EvalExecutionMode.LIVE, EvalExecutionMode.REPLAY),
            "m8-team-bm25-v1",
            (
                "route",
                "safety_short_circuit",
                "citation_integrity",
                "claim_verdict",
                "evidence_group_coverage",
                "budget_termination",
                "tool_execution",
                "team_metrics",
                "ood_answer",
            ),
            "m8-metrics-v1",
            public_content_allowed=True,
            provenance={
                "gold_semantics": "focused candidate diagnostic for L0 workflow, L1 single agent, and L2 bounded Agent Team",
                "annotation_manifest": "evals/m8_agent_team_focused_v1.annotation_manifest.json",
                "review_status": "pending_human_review",
                "allowed_profiles": [
                    "m8-workflow-bm25-v1",
                    "m3-bm25-default",
                    "m8-team-bm25-v1",
                ],
            },
            expected_dataset_sha256="78a417bef691892fb0b248911044f0325849ae4a4ae1065ec7ad009968180549",
        ),
    )
    for suite in registrations:
        registry.register(suite)
    return registry
