# M7 evaluation: m8-agent-team-focused-v1

- execution mode: `live`
- dataset SHA-256: `78a417bef691892fb0b248911044f0325849ae4a4ae1065ec7ad009968180549`
- metric definition: `m8-metrics-v2`
- failures: `43`

Metrics are deterministic aggregates; missing/infrastructure cases are not silently placed in quality denominators.

```json
{
  "case.all_trials_pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 12,
    "metric_id": "case.all_trials_pass_rate",
    "numerator": 4,
    "scope": "case",
    "unit": "rate",
    "value": 0.3333333333333333
  },
  "case.harness_disposition_consistency": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 12,
    "metric_id": "case.harness_disposition_consistency",
    "numerator": 7,
    "scope": "case",
    "unit": "rate",
    "value": 0.5833333333333334
  },
  "case.route_consistency": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 12,
    "metric_id": "case.route_consistency",
    "numerator": 8,
    "scope": "case",
    "unit": "rate",
    "value": 0.6666666666666666
  },
  "grader.budget_termination.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.budget_termination.pass_rate",
    "numerator": 36,
    "scope": "budget_termination",
    "unit": "rate",
    "value": 1.0
  },
  "grader.budget_termination.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.budget_termination.scored_count",
    "numerator": 36,
    "scope": "budget_termination",
    "unit": "count",
    "value": 36
  },
  "grader.citation_integrity.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.citation_integrity.pass_rate",
    "numerator": 36,
    "scope": "citation_integrity",
    "unit": "rate",
    "value": 1.0
  },
  "grader.citation_integrity.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.citation_integrity.scored_count",
    "numerator": 36,
    "scope": "citation_integrity",
    "unit": "count",
    "value": 36
  },
  "grader.claim_verdict.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 0,
    "metric_id": "grader.claim_verdict.pass_rate",
    "numerator": 0,
    "scope": "claim_verdict",
    "unit": "rate",
    "value": null
  },
  "grader.claim_verdict.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.claim_verdict.scored_count",
    "numerator": 0,
    "scope": "claim_verdict",
    "unit": "count",
    "value": 0
  },
  "grader.evidence_group_coverage.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.evidence_group_coverage.pass_rate",
    "numerator": 12,
    "scope": "evidence_group_coverage",
    "unit": "rate",
    "value": 0.3333333333333333
  },
  "grader.evidence_group_coverage.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.evidence_group_coverage.scored_count",
    "numerator": 36,
    "scope": "evidence_group_coverage",
    "unit": "count",
    "value": 36
  },
  "grader.ood_answer.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 6,
    "metric_id": "grader.ood_answer.pass_rate",
    "numerator": 6,
    "scope": "ood_answer",
    "unit": "rate",
    "value": 1.0
  },
  "grader.ood_answer.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.ood_answer.scored_count",
    "numerator": 6,
    "scope": "ood_answer",
    "unit": "count",
    "value": 6
  },
  "grader.route.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.route.pass_rate",
    "numerator": 27,
    "scope": "route",
    "unit": "rate",
    "value": 0.75
  },
  "grader.route.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.route.scored_count",
    "numerator": 36,
    "scope": "route",
    "unit": "count",
    "value": 36
  },
  "grader.safety_short_circuit.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 3,
    "metric_id": "grader.safety_short_circuit.pass_rate",
    "numerator": 3,
    "scope": "safety_short_circuit",
    "unit": "rate",
    "value": 1.0
  },
  "grader.safety_short_circuit.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.safety_short_circuit.scored_count",
    "numerator": 3,
    "scope": "safety_short_circuit",
    "unit": "count",
    "value": 3
  },
  "grader.team_metrics.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 33,
    "metric_id": "grader.team_metrics.pass_rate",
    "numerator": 33,
    "scope": "team_metrics",
    "unit": "rate",
    "value": 1.0
  },
  "grader.team_metrics.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.team_metrics.scored_count",
    "numerator": 33,
    "scope": "team_metrics",
    "unit": "count",
    "value": 33
  },
  "grader.tool_execution.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 0,
    "metric_id": "grader.tool_execution.pass_rate",
    "numerator": 0,
    "scope": "tool_execution",
    "unit": "rate",
    "value": null
  },
  "grader.tool_execution.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "grader.tool_execution.scored_count",
    "numerator": 0,
    "scope": "tool_execution",
    "unit": "count",
    "value": 0
  },
  "m8.budget_exhaustion_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.budget_exhaustion_rate",
    "numerator": 0,
    "scope": "all",
    "unit": "rate",
    "value": 0.0
  },
  "m8.category.cross_source_comparison.route_accuracy": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 6,
    "metric_id": "m8.category.cross_source_comparison.route_accuracy",
    "numerator": 3,
    "scope": "cross_source_comparison",
    "unit": "rate",
    "value": 0.5
  },
  "m8.category.decomposable.route_accuracy": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 12,
    "metric_id": "m8.category.decomposable.route_accuracy",
    "numerator": 9,
    "scope": "decomposable",
    "unit": "rate",
    "value": 0.75
  },
  "m8.category.ood_uncovered.route_accuracy": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 6,
    "metric_id": "m8.category.ood_uncovered.route_accuracy",
    "numerator": 6,
    "scope": "ood_uncovered",
    "unit": "rate",
    "value": 1.0
  },
  "m8.category.simple_direct.route_accuracy": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 12,
    "metric_id": "m8.category.simple_direct.route_accuracy",
    "numerator": 9,
    "scope": "simple_direct",
    "unit": "rate",
    "value": 0.75
  },
  "m8.delegation_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.delegation_rate",
    "numerator": 7,
    "scope": "all",
    "unit": "rate",
    "value": 0.19444444444444445
  },
  "m8.elapsed_ms_per_case": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.elapsed_ms_per_case",
    "numerator": 563312.6559999655,
    "scope": "complete_cases",
    "unit": "ms",
    "value": 15647.573777776819
  },
  "m8.input_tokens_per_case": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.input_tokens_per_case",
    "numerator": 63293,
    "scope": "complete_cases",
    "unit": "per_case",
    "value": 1758.138888888889
  },
  "m8.lead_calls_per_case": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 33,
    "metric_id": "m8.lead_calls_per_case",
    "numerator": 40,
    "scope": "complete_cases",
    "unit": "per_case",
    "value": 1.2121212121212122
  },
  "m8.lead_failure_rate.contract_validation": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.lead_failure_rate.contract_validation",
    "numerator": 0,
    "scope": "all",
    "unit": "rate",
    "value": 0.0
  },
  "m8.lead_failure_rate.empty_response": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.lead_failure_rate.empty_response",
    "numerator": 0,
    "scope": "all",
    "unit": "rate",
    "value": 0.0
  },
  "m8.lead_failure_rate.internal": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.lead_failure_rate.internal",
    "numerator": 0,
    "scope": "all",
    "unit": "rate",
    "value": 0.0
  },
  "m8.lead_failure_rate.json_decode": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.lead_failure_rate.json_decode",
    "numerator": 0,
    "scope": "all",
    "unit": "rate",
    "value": 0.0
  },
  "m8.lead_failure_rate.provider": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.lead_failure_rate.provider",
    "numerator": 0,
    "scope": "all",
    "unit": "rate",
    "value": 0.0
  },
  "m8.output_tokens_per_case": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.output_tokens_per_case",
    "numerator": 115495,
    "scope": "complete_cases",
    "unit": "per_case",
    "value": 3208.1944444444443
  },
  "m8.provider_calls_per_case": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.provider_calls_per_case",
    "numerator": 78,
    "scope": "complete_cases",
    "unit": "per_case",
    "value": 2.1666666666666665
  },
  "m8.tool_executions_per_case": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.tool_executions_per_case",
    "numerator": 0,
    "scope": "complete_cases",
    "unit": "per_case",
    "value": 0.0
  },
  "m8.total_tokens_per_case": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "m8.total_tokens_per_case",
    "numerator": 178788,
    "scope": "complete_cases",
    "unit": "per_case",
    "value": 4966.333333333333
  },
  "m8.worker_completion_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 9,
    "metric_id": "m8.worker_completion_rate",
    "numerator": 0,
    "scope": "worker_reports",
    "unit": "rate",
    "value": 0.0
  },
  "m8.worker_evidence_overlap": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 2,
    "metric_id": "m8.worker_evidence_overlap",
    "numerator": 2.0,
    "scope": "complete_cases",
    "unit": "per_case",
    "value": 1.0
  },
  "m8.worker_productive_report_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 0,
    "metric_id": "m8.worker_productive_report_rate",
    "numerator": 0,
    "scope": "worker_reports",
    "unit": "rate",
    "value": null
  },
  "m8.worker_recovery_evidence_overlap": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 2,
    "metric_id": "m8.worker_recovery_evidence_overlap",
    "numerator": 0.0,
    "scope": "complete_cases",
    "unit": "per_case",
    "value": 0.0
  },
  "m8.worker_unique_evidence_contribution": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 2,
    "metric_id": "m8.worker_unique_evidence_contribution",
    "numerator": 0.0,
    "scope": "complete_cases",
    "unit": "per_case",
    "value": 0.0
  },
  "m8.worker_unique_recovery_contribution": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 2,
    "metric_id": "m8.worker_unique_recovery_contribution",
    "numerator": 0.0,
    "scope": "complete_cases",
    "unit": "per_case",
    "value": 0.0
  },
  "m8.workers_started_per_case": {
    "aggregation": "sum/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 33,
    "metric_id": "m8.workers_started_per_case",
    "numerator": 9,
    "scope": "complete_cases",
    "unit": "per_case",
    "value": 0.2727272727272727
  },
  "run.case_count": {
    "aggregation": "count",
    "definition_version": "m8-metrics-v2",
    "denominator": 12,
    "metric_id": "run.case_count",
    "numerator": 12,
    "scope": "run",
    "unit": "count",
    "value": 12
  },
  "run.trial_count": {
    "aggregation": "count",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "run.trial_count",
    "numerator": 36,
    "scope": "run",
    "unit": "count",
    "value": 36
  },
  "trajectory.completeness": {
    "aggregation": "numerator/denominator",
    "definition_version": "m8-metrics-v2",
    "denominator": 36,
    "metric_id": "trajectory.completeness",
    "numerator": 36,
    "scope": "all",
    "unit": "rate",
    "value": 1.0
  }
}
```
