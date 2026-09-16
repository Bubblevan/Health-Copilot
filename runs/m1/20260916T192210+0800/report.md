# M1 focused regression run: `20260916T192210+0800`

这是同一模型配置下的 focused regression / diagnostic result，不是泛化性能声明。

## Configuration

- commit: `285b4af4e26009aca03883a38607744761d84bc7`
- model: `deepseek-flash`
- temperature: `0.1`
- trials: `3`
- initial_top_k: `5`
- recovery_top_k: `3`

## Metrics

- `run_cases`: `30`
- `initial_hit@3`: `0.5`
- `initial_hit@3_cases`: `18`
- `recovery_attempt_rate`: `1.0`
- `recovery_expected_cases`: `9`
- `recovery_success@3`: `1.0`
- `post_recovery_hit@3`: `1.0`
- `unnecessary_recovery_rate`: `0.0`
- `ood_tool_activation_rate`: `0.9166666666666666`
- `ood_answer_rate`: `0.0`
- `ood_abstain_rate`: `1.0`
- `expected_answer_cases`: `18`
- `expected_answer_rate`: `1.0`
- `unexpected_abstain_rate`: `0.0`
- `safety_short_circuit_accuracy`: `1.0`
- `mean_model_turns`: `1.6666666666666667`
- `mean_tool_calls`: `0.6666666666666666`
- `budget_exhaustion_rate`: `0.0`
- `citation_integrity_pass_rate`: `1.0`
- `trial_count`: `3`
- `pack_case_count`: `12`
- `trajectory_count`: `36`

## Hard acceptance failures

- hard failure records: `0`

Evidence stages are preserved separately in `trajectories.jsonl`; final citation verification uses the observed union.

Diagnostic warning: OOD tool activation is intentionally reported separately; it is not counted as a hard failure.
