# E5-B4 policy interpretation erratum

This note clarifies the interpretation of the frozen B4 CPU report. It does not alter the raw outputs, arm artifacts, scores, bootstrap, or report JSON.

## Quality-oracle headroom is not policy value

The B4 per-case quality oracle has zero headroom over the best fixed action, **STRONG**, whose measured E2E quality is `0.8611`. This means no action-selection policy can improve the measured quality above Always-STRONG on these 60 controlled cases. It does **not** mean that execution policy has no value: cheaper actions tie STRONG's maximal measured outcome on some cases.

The exact-quality, minimum-cost action labels in the frozen report are:

| Minimal action | Cases |
|---|---:|
| OFF | 20 |
| STANDARD | 10 |
| STRONG | 30 |

Always-STRONG activates retrieval on 60 cases and the bridge on 60 cases. The cost-aware minimum-cost oracle activates retrieval on the 10 STANDARD plus 30 STRONG cases (40 total), and the bridge on the 30 STRONG cases. Relative to Always-STRONG, this is a structural reduction of **33.3% retrieval activations** (`60 → 40`) and **50.0% bridge activations** (`60 → 30`) while retaining the same measured quality (`0.8611`).

This is a counterfactual mechanism result on a small, deliberately controlled 60-case B4 set. The oracle has privileged access to sibling outcomes and is not a deployable policy; B4 does not establish that the minimum-cost action is predictable from runtime-observable features. U3-R must remeasure the outcome labels and cost frontier on the owned DEV universe before any policy-training decision.

## Legacy action-space field

The frozen JSON field `recommended_action_space = OFF|STANDARD` is **`LEGACY_DERIVED_FIELD_NOT_COST_AWARE`** and must not be used as an experimental conclusion or as evidence that STRONG should be removed. The legacy fallback selects OFF|STANDARD when the STANDARD-vs-OFF fixed-action value is positive after quality-oracle headroom gates fail. That fallback does not encode the cost-aware tie structure above: it omits that STRONG is globally strongest while cheaper actions match its outcome on some cases.

The report JSON remains byte-for-byte unchanged. This interpretation erratum records the meaning of its fields without editing the frozen result.
