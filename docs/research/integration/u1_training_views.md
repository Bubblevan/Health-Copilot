# U1 Post-Training Compatibility Views

This stage proves only that typed artifacts can be formed. It does not train a
model or invoke a provider or training framework.

## SFTCandidate

```json
{
  "observable_state": {"history_exists": true},
  "acceptable_actions": ["MEMORY"],
  "provenance": {"episode_hash": "<sha256>", "schema": "u1-training-views-v1"}
}
```

It contains the pre-decision state and acceptable action set only. It excludes
gold answer, future patient state, raw sibling outcomes, and privileged failure
analysis.

## PolicyRolloutGroup

The deterministic GRPO-compatible view groups multiple action samples from the
same episode with each sample's outcome, abstract cost, and a versioned relative
reward. No `verl`, `rllm`, Agent Lightning, or optimizer is connected.

```json
{
  "episode_id": "U1-ALL",
  "sampled_actions": ["NONE", "MEMORY", "RAG", "TEAM", "MEMORY+RAG",
                      "MEMORY+TEAM", "RAG+TEAM", "ALL"],
  "outcomes": [{"task_success": false}, {"task_success": false},
               {"task_success": false}, {"task_success": false},
               {"task_success": false}, {"task_success": false},
               {"task_success": false}, {"task_success": true}],
  "group_rewards": [0.0, -0.01, -0.01, -0.02, -0.02, -0.03, -0.03, 0.96],
  "reward_version": "u1-success-safety-cost-v1"
}
```

## OPD student and teacher

`StudentPacket` has visited observable state, the student's action, its own
runtime-visible answer/usage, and provenance. `PrivilegedTeacherPacket` is a
separate type and output stream; it can add sibling outcomes, the minimal
successful action set, and failure attribution. `StudentPacket` has no field or
reference for a teacher packet. Tests serialize both independently and assert
that privileged keys/labels do not appear in the student view.

Example boundary (values abbreviated):

```json
{"student_packet":{"student_action":"NONE","student_outcome":{"answer":"..."}}}
{"privileged_plane":"PRIVILEGED_TRAINING",
 "minimal_successful_action_set":["MEMORY+RAG"],
 "failure_attribution":["MISSING_MEMORY_READ"]}
```
