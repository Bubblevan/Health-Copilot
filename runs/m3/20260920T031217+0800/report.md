# M3 focused M2-vs-M3 run: `20260920T031217+0800`

M3 materializes user-visible output from verified claims and does not use semantic coverage_ok.

```json
{
  "m2": {
    "run_cases": 30,
    "tool_proposal_rate": 0.7,
    "tool_execution_rate": 0.6666666666666666,
    "policy_veto_rate": 0.03333333333333333,
    "ood_tool_proposal_rate": 1.0,
    "ood_tool_execution_rate": 0.9166666666666666,
    "ood_answer_rate": 0.0,
    "ood_abstain_rate": 1.0,
    "expected_answer_rate": 0.8888888888888888,
    "unexpected_abstain_rate": 0.1111111111111111,
    "direct_hit_policy_false_veto_rate": 0.0,
    "mean_model_turns": 1.6666666666666667,
    "mean_tool_proposals": 0.7,
    "mean_tool_executions": 0.6666666666666666,
    "mean_policy_calls": 0.7,
    "mean_verifier_calls": 0.6,
    "budget_exhaustion_rate": 0.0,
    "grounding_rejection_rate": 0.03333333333333333
  },
  "m3": {
    "run_cases": 30,
    "tool_proposal_rate": 0.6333333333333333,
    "tool_execution_rate": 0.26666666666666666,
    "policy_veto_rate": 0.36666666666666664,
    "ood_tool_proposal_rate": 0.9166666666666666,
    "ood_tool_execution_rate": 0.0,
    "ood_answer_rate": 0.0,
    "ood_abstain_rate": 1.0,
    "expected_answer_rate": 0.8333333333333334,
    "unexpected_abstain_rate": 0.16666666666666666,
    "direct_hit_policy_false_veto_rate": 0.0,
    "mean_model_turns": 1.2666666666666666,
    "mean_tool_proposals": 0.6333333333333333,
    "mean_tool_executions": 0.26666666666666666,
    "mean_policy_calls": 0.6333333333333333,
    "mean_verifier_calls": 0.5666666666666667,
    "budget_exhaustion_rate": 0.0,
    "grounding_rejection_rate": 0.0,
    "claim_support_rejection_rate": 0.06666666666666667
  },
  "frozen_m2_baseline": {
    "ood_tool_execution_rate": 0.6666666666666666,
    "expected_answer_rate": 0.9444444444444444
  },
  "trial_count": 3,
  "pack_case_count": 12,
  "trajectory_count": 72
}
```

- failure records: `16`
- Same-model verification is not an independent judge.
