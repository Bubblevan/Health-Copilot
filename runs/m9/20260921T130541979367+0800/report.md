# M7 evaluation: m9-mcp-security-v1

- execution mode: `offline`
- dataset SHA-256: `610c3d7d9ca0da4de8619352cd86f3829c370d98108512ee8fdd9b4821b37176`
- metric definition: `m9-security-metrics-v1`
- failures: `0`

Metrics are deterministic aggregates; missing/infrastructure cases are not silently placed in quality denominators.

```json
{
  "case.all_trials_pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 12,
    "metric_id": "case.all_trials_pass_rate",
    "numerator": 12,
    "scope": "case",
    "unit": "rate",
    "value": 1.0
  },
  "case.harness_disposition_consistency": {
    "aggregation": "numerator/denominator",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 12,
    "metric_id": "case.harness_disposition_consistency",
    "numerator": 12,
    "scope": "case",
    "unit": "rate",
    "value": 1.0
  },
  "case.route_consistency": {
    "aggregation": "numerator/denominator",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 12,
    "metric_id": "case.route_consistency",
    "numerator": 12,
    "scope": "case",
    "unit": "rate",
    "value": 1.0
  },
  "grader.security_control.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 12,
    "metric_id": "grader.security_control.pass_rate",
    "numerator": 12,
    "scope": "security_control",
    "unit": "rate",
    "value": 1.0
  },
  "grader.security_control.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 12,
    "metric_id": "grader.security_control.scored_count",
    "numerator": 12,
    "scope": "security_control",
    "unit": "count",
    "value": 12
  },
  "m9.permission_cases_passed": {
    "aggregation": "numerator/denominator",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 4,
    "metric_id": "m9.permission_cases_passed",
    "numerator": 4,
    "scope": "security_controls",
    "unit": "count",
    "value": 4
  },
  "m9.protocol_cases_passed": {
    "aggregation": "numerator/denominator",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 3,
    "metric_id": "m9.protocol_cases_passed",
    "numerator": 3,
    "scope": "security_controls",
    "unit": "count",
    "value": 3
  },
  "m9.sandbox_contract_cases_passed": {
    "aggregation": "numerator/denominator",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 5,
    "metric_id": "m9.sandbox_contract_cases_passed",
    "numerator": 5,
    "scope": "security_controls",
    "unit": "count",
    "value": 5
  },
  "m9.security_control_pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 12,
    "metric_id": "m9.security_control_pass_rate",
    "numerator": 12,
    "scope": "security_controls",
    "unit": "rate",
    "value": 1.0
  },
  "run.case_count": {
    "aggregation": "count",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 12,
    "metric_id": "run.case_count",
    "numerator": 12,
    "scope": "run",
    "unit": "count",
    "value": 12
  },
  "run.trial_count": {
    "aggregation": "count",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 12,
    "metric_id": "run.trial_count",
    "numerator": 12,
    "scope": "run",
    "unit": "count",
    "value": 12
  },
  "trajectory.completeness": {
    "aggregation": "numerator/denominator",
    "definition_version": "m9-security-metrics-v1",
    "denominator": 12,
    "metric_id": "trajectory.completeness",
    "numerator": 12,
    "scope": "all",
    "unit": "rate",
    "value": 1.0
  }
}
```
