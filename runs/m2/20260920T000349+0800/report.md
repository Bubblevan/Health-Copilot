# M2 focused M1-vs-M2 run: `20260920T000349+0800`

This is a focused diagnostic, not a generalization or medical-accuracy claim.

## Configuration

- `commit_sha`: `d7cceecfaeef733ffa39f072e3e2c34631081630`
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
    "recovery_attempt_rate": 1.0,
    "recovery_expected_cases": 9,
    "recovery_success@3": 1.0,
    "post_recovery_hit@3": 1.0,
    "unnecessary_recovery_rate": 0.0,
    "ood_tool_activation_rate": 1.0,
    "ood_answer_rate": 0.0,
    "ood_abstain_rate": 1.0,
    "expected_answer_cases": 18,
    "expected_answer_rate": 1.0,
    "unexpected_abstain_rate": 0.0,
    "safety_short_circuit_accuracy": 1.0,
    "mean_model_turns": 1.7,
    "mean_tool_calls": 0.7,
    "budget_exhaustion_rate": 0.0,
    "citation_integrity_pass_rate": 1.0
  },
  "m2": {
    "run_cases": 30,
    "tool_proposal_rate": 0.7,
    "tool_execution_rate": 0.0,
    "policy_veto_rate": 0.7,
    "ood_tool_proposal_rate": 1.0,
    "ood_tool_execution_rate": 0.0,
    "ood_answer_rate": 0.0,
    "ood_abstain_rate": 1.0,
    "expected_answer_rate": 0.5,
    "unexpected_abstain_rate": 0.5,
    "direct_hit_policy_false_veto_rate": 0.0,
    "mean_model_turns": 1.0,
    "mean_tool_proposals": 0.7,
    "mean_tool_executions": 0.0,
    "mean_policy_calls": 0.7,
    "mean_verifier_calls": 0.3,
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
