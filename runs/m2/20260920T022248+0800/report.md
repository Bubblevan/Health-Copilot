# M2 focused M1-vs-M2 run: `20260920T022248+0800`

This is a focused diagnostic, not a generalization or medical-accuracy claim.

## Configuration

- `commit_sha`: `9043d316772d331df49420f3c5889d91d256cc05`
- `knowledge_pack_version`: `m0.2-2026-09-15`
- `eval_pack_sha256`: `b76ac092f9d203563ad9f9d997599a071e44fae6c60e056416b22a96d72ff0c3`
- `model`: `deepseek-flash`
- `policy_model`: `deepseek-flash`
- `verifier_model`: `deepseek-flash`
- `same_model_verification`: `True`
- `temperature`: `0.1`
- `policy_temperature`: `0`
- `verifier_temperature`: `0`
- `base_url`: `https://api.deepseek.com`
- `trials`: `3`
- `max_model_turns`: `2`
- `max_tool_calls`: `1`
- `initial_top_k`: `5`
- `recovery_top_k`: `3`

## Metrics

```json
{
  "m1": {
    "run_cases": 30,
    "initial_hit@3": 0.5,
    "initial_hit@3_cases": 18,
    "recovery_attempt_rate": 0.8888888888888888,
    "recovery_expected_cases": 9,
    "recovery_success@3": 0.8888888888888888,
    "post_recovery_hit@3": 0.8888888888888888,
    "unnecessary_recovery_rate": 0.0,
    "ood_tool_activation_rate": 1.0,
    "ood_answer_rate": 0.0,
    "ood_abstain_rate": 1.0,
    "expected_answer_cases": 18,
    "expected_answer_rate": 0.9444444444444444,
    "unexpected_abstain_rate": 0.05555555555555555,
    "safety_short_circuit_accuracy": 1.0,
    "mean_model_turns": 1.6666666666666667,
    "mean_tool_calls": 0.6666666666666666,
    "budget_exhaustion_rate": 0.0,
    "citation_integrity_pass_rate": 1.0
  },
  "m2": {
    "run_cases": 30,
    "tool_proposal_rate": 0.6666666666666666,
    "tool_execution_rate": 0.5666666666666667,
    "policy_veto_rate": 0.1,
    "ood_tool_proposal_rate": 0.9166666666666666,
    "ood_tool_execution_rate": 0.6666666666666666,
    "ood_answer_rate": 0.0,
    "ood_abstain_rate": 1.0,
    "expected_answer_rate": 0.9444444444444444,
    "unexpected_abstain_rate": 0.05555555555555555,
    "direct_hit_policy_false_veto_rate": 0.0,
    "mean_model_turns": 1.5666666666666667,
    "mean_tool_proposals": 0.6666666666666666,
    "mean_tool_executions": 0.5666666666666667,
    "mean_policy_calls": 0.6666666666666666,
    "mean_verifier_calls": 0.6,
    "budget_exhaustion_rate": 0.0,
    "grounding_rejection_rate": 0.0
  },
  "trial_count": 3,
  "pack_case_count": 12,
  "trajectory_count": 36
}
```

- OOD answer failure records: `0`

Same-model verification is not an independent judge. M3 was not started.
