# M7 evaluation: m0-regression-v1

- execution mode: `offline`
- dataset SHA-256: `b40ec6f88ae5900a9708dd75fe78eb35436799443cf6e45c818cc19573c53a46`
- metric definition: `m0-parity-v1`
- failures: `7`

Metrics are deterministic aggregates; missing/infrastructure cases are not silently placed in quality denominators.

```json
{
  "case.all_trials_pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m0-parity-v1",
    "denominator": 80,
    "metric_id": "case.all_trials_pass_rate",
    "numerator": 73,
    "scope": "case",
    "unit": "rate",
    "value": 0.9125
  },
  "case.route_consistency": {
    "aggregation": "numerator/denominator",
    "definition_version": "m0-parity-v1",
    "denominator": 80,
    "metric_id": "case.route_consistency",
    "numerator": 80,
    "scope": "case",
    "unit": "rate",
    "value": 1.0
  },
  "grader.retrieval_source.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m0-parity-v1",
    "denominator": 62,
    "metric_id": "grader.retrieval_source.pass_rate",
    "numerator": 59,
    "scope": "retrieval_source",
    "unit": "rate",
    "value": 0.9516129032258065
  },
  "grader.retrieval_source.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m0-parity-v1",
    "denominator": 80,
    "metric_id": "grader.retrieval_source.scored_count",
    "numerator": 62,
    "scope": "retrieval_source",
    "unit": "count",
    "value": 62
  },
  "grader.route.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m0-parity-v1",
    "denominator": 80,
    "metric_id": "grader.route.pass_rate",
    "numerator": 76,
    "scope": "route",
    "unit": "rate",
    "value": 0.95
  },
  "grader.route.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m0-parity-v1",
    "denominator": 80,
    "metric_id": "grader.route.scored_count",
    "numerator": 80,
    "scope": "route",
    "unit": "count",
    "value": 80
  },
  "grader.safety_short_circuit.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m0-parity-v1",
    "denominator": 14,
    "metric_id": "grader.safety_short_circuit.pass_rate",
    "numerator": 14,
    "scope": "safety_short_circuit",
    "unit": "rate",
    "value": 1.0
  },
  "grader.safety_short_circuit.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m0-parity-v1",
    "denominator": 80,
    "metric_id": "grader.safety_short_circuit.scored_count",
    "numerator": 14,
    "scope": "safety_short_circuit",
    "unit": "count",
    "value": 14
  },
  "m0.retrieval_hit_at_1": {
    "aggregation": "numerator/denominator",
    "definition_version": "m0-parity-v1",
    "denominator": 62,
    "metric_id": "m0.retrieval_hit_at_1",
    "numerator": 56,
    "scope": "all",
    "unit": "rate",
    "value": 0.9032258064516129
  },
  "m0.retrieval_hit_at_3": {
    "aggregation": "numerator/denominator",
    "definition_version": "m0-parity-v1",
    "denominator": 62,
    "metric_id": "m0.retrieval_hit_at_3",
    "numerator": 59,
    "scope": "all",
    "unit": "rate",
    "value": 0.9516129032258065
  },
  "m0.retrieval_mrr": {
    "aggregation": "sum/denominator",
    "definition_version": "m0-parity-v1",
    "denominator": 62,
    "metric_id": "m0.retrieval_mrr",
    "numerator": 57.5,
    "scope": "scored_cases",
    "unit": "score",
    "value": 0.9274193548387096
  },
  "m0.safety_route_accuracy": {
    "aggregation": "numerator/denominator",
    "definition_version": "m0-parity-v1",
    "denominator": 76,
    "metric_id": "m0.safety_route_accuracy",
    "numerator": 76,
    "scope": "all",
    "unit": "rate",
    "value": 1.0
  },
  "run.case_count": {
    "aggregation": "count",
    "definition_version": "m0-parity-v1",
    "denominator": 80,
    "metric_id": "run.case_count",
    "numerator": 80,
    "scope": "run",
    "unit": "count",
    "value": 80
  },
  "run.trial_count": {
    "aggregation": "count",
    "definition_version": "m0-parity-v1",
    "denominator": 80,
    "metric_id": "run.trial_count",
    "numerator": 80,
    "scope": "run",
    "unit": "count",
    "value": 80
  },
  "trajectory.completeness": {
    "aggregation": "numerator/denominator",
    "definition_version": "m0-parity-v1",
    "denominator": 80,
    "metric_id": "trajectory.completeness",
    "numerator": 80,
    "scope": "all",
    "unit": "rate",
    "value": 1.0
  }
}
```
