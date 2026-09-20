# M7 evaluation: m4-replay-v1

- execution mode: `replay`
- dataset SHA-256: `08edbdfb0b84d9555ac3c01e14e20d5c9191a8d4a56c56f49e0a3035d6aaac65`
- metric definition: `m4-replay-v1`
- failures: `0`

Metrics are deterministic aggregates; missing/infrastructure cases are not silently placed in quality denominators.

```json
{
  "case.all_trials_pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m4-replay-v1",
    "denominator": 6,
    "metric_id": "case.all_trials_pass_rate",
    "numerator": 6,
    "scope": "case",
    "unit": "rate",
    "value": 1.0
  },
  "case.route_consistency": {
    "aggregation": "numerator/denominator",
    "definition_version": "m4-replay-v1",
    "denominator": 6,
    "metric_id": "case.route_consistency",
    "numerator": 6,
    "scope": "case",
    "unit": "rate",
    "value": 1.0
  },
  "grader.budget_termination.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m4-replay-v1",
    "denominator": 6,
    "metric_id": "grader.budget_termination.pass_rate",
    "numerator": 6,
    "scope": "budget_termination",
    "unit": "rate",
    "value": 1.0
  },
  "grader.budget_termination.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m4-replay-v1",
    "denominator": 6,
    "metric_id": "grader.budget_termination.scored_count",
    "numerator": 6,
    "scope": "budget_termination",
    "unit": "count",
    "value": 6
  },
  "grader.replay_consistency.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m4-replay-v1",
    "denominator": 6,
    "metric_id": "grader.replay_consistency.pass_rate",
    "numerator": 6,
    "scope": "replay_consistency",
    "unit": "rate",
    "value": 1.0
  },
  "grader.replay_consistency.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m4-replay-v1",
    "denominator": 6,
    "metric_id": "grader.replay_consistency.scored_count",
    "numerator": 6,
    "scope": "replay_consistency",
    "unit": "count",
    "value": 6
  },
  "grader.route.pass_rate": {
    "aggregation": "numerator/denominator",
    "definition_version": "m4-replay-v1",
    "denominator": 6,
    "metric_id": "grader.route.pass_rate",
    "numerator": 6,
    "scope": "route",
    "unit": "rate",
    "value": 1.0
  },
  "grader.route.scored_count": {
    "aggregation": "scored/all",
    "definition_version": "m4-replay-v1",
    "denominator": 6,
    "metric_id": "grader.route.scored_count",
    "numerator": 6,
    "scope": "route",
    "unit": "count",
    "value": 6
  },
  "run.case_count": {
    "aggregation": "count",
    "definition_version": "m4-replay-v1",
    "denominator": 6,
    "metric_id": "run.case_count",
    "numerator": 6,
    "scope": "run",
    "unit": "count",
    "value": 6
  },
  "run.trial_count": {
    "aggregation": "count",
    "definition_version": "m4-replay-v1",
    "denominator": 6,
    "metric_id": "run.trial_count",
    "numerator": 6,
    "scope": "run",
    "unit": "count",
    "value": 6
  },
  "trajectory.completeness": {
    "aggregation": "numerator/denominator",
    "definition_version": "m4-replay-v1",
    "denominator": 6,
    "metric_id": "trajectory.completeness",
    "numerator": 6,
    "scope": "all",
    "unit": "rate",
    "value": 1.0
  }
}
```
