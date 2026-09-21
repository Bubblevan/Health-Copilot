# M10.1 / H7 Closeout — Executable Context Projection

M10.1 closes the gap between deterministic context selection and the actual
provider request. `ContextProjector` is the boundary: each model turn builds a
fresh `ContextPlan`, and only the projector's messages are passed to the
provider. Persistent history and selected memory remain data-bearing,
untrusted context; current user input, Evidence, system pins, and complete
tool exchanges are protected.

## Frozen implementation contract

- `ContextManager` selects; `ContextProjector` projects; neither boundary
  executes tools, writes memory, or grants authority.
- Retrieval completes before the first plan, and the second model turn gets a
  new plan containing the actual tool observation.
- Safety routing runs before session resume, memory query, retrieval, or any
  provider call. Projection and budget failures fail closed.
- Session turns commit atomically as one revisioned batch containing user
  input, tool call/result observations, and final output. Hidden reasoning and
  automatic memory writes are never persisted.
- Replay identity optionally binds session revision, memory snapshot hash, and
  one or more context-plan hashes. A recorded M10.1 artifact with state
  identity cannot replay against a changed state.
- Context-manager component identity is `context-manager-v2`, implementation
  version `m10-context-v2`; SQLite state schema remains `v1`.

## Offline integration suite

`m10-context-integration-v1` contains ten deterministic synthetic cases and
uses provider-capture fakes only. It covers history reachability, dropped
context leakage, active/superseded memory, protected Evidence, turn-two tool
atomicity, safety precedence, budget fail-closed behavior, atomic session
commit, and replay identity. The suite reports separate numerator/denominator
metrics:

- `m10.context_projection_accuracy`
- `m10.dropped_context_leakage_rate`
- `m10.protected_context_retention_rate`
- `m10.tool_atomicity_pass_rate`
- `m10.safety_precedes_memory_rate`
- `m10.atomic_turn_commit_rate`
- `m10.memory_replay_identity_pass_rate`

The frozen 24-case `m10-memory-v1` suite remains unchanged. No live-provider
smoke or clinical conclusion is implied by these fixtures.

M11 remains reserved for learned memory policy, post-training, and learned
context editing; it is not part of this closeout.
