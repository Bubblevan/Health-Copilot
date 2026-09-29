# U1.1 Training-View Semantics

## Status

The training views remain schema probes only. No policy was sampled, no SFT
training was run, no GRPO optimization was run, and no OPD optimization was
run.

## Student action provenance

`build_training_views` requires an explicit valid evaluated action and a
`StudentActionSource`. U1.1 synthetic runs use `SCRIPTED_PROBE`; they do not
claim that the action was sampled from a student policy. Synthetic fixtures
cannot be labeled `POLICY_SAMPLE`. A future policy sample must include an
explicit policy identity and sample ID.

The StudentPacket is formed from the selected arm's execution-visible outcome,
not evaluator success, gold answers, sibling-arm outcomes, or teacher
attribution. It serializes `student_action_source` directly.

## OPD gate

`PrivilegedTeacherPacket` carries `opd_data_status`:

```text
SCRIPTED_PROBE -> SCHEMA_PROBE
POLICY_SAMPLE  -> POLICY_VISITED
```

All U1.1 artifacts use `SCHEMA_PROBE`. `POLICY_VISITED` is only available when
the builder receives an explicitly identified policy sample. The presence of a
teacher-packet schema does not authorize optimization.

## GRPO provenance

The current `PolicyRolloutGroup` enumerates deterministic counterfactual arms.
Its serialized `rollout_source` is `COUNTERFACTUAL_ENUMERATION`, and the JSON
uses `actions`, not `sampled_actions`. This establishes reward-schema
compatibility only; it is not a sampled rollout group from policy πθ.

## SFT label provenance

`SFTCandidate.acceptable_actions` is derived from
`MinimalSuccessfulActionSet`. Its provenance records:

- `label_source = DETERMINISTIC_COUNTERFACTUAL_ORACLE`
- execution backend and evaluator version
- cost model and epsilon
- every evaluated arm

Real data cannot be promoted to a training label from this synthetic result.
It will also need its actual execution backend, evaluator, cost model, epsilon,
and all evaluated arms recorded before a minimum action set can serve as a
label.

## Plane separation

Runtime episode serialization contains no evaluation gold. StudentPacket
contains neither outcome success flags nor counterfactual/privileged values.
PrivilegedTeacherPacket is serialized separately and explicitly reports the
OPD status. This preserves a typed separation between the observed student
trajectory and evaluator/teacher information.
