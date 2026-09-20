# M6 — Composable Runtime Profiles & Component Registry

M6 formalizes H3（Extensible / Plugin Runtime）without turning trusted Python
components into a security boundary or a plugin marketplace.

## Boundary

`RuntimeProfile` is declarative. It contains component IDs, a profile ID, an
execution mode, and JSON-compatible configuration only. It never contains a
Python object, import path supplied by a user, or dynamically installed package.

`ComponentRegistry` maps the source-defined pair `(ComponentKind, component_id)`
to a factory. Duplicate IDs, unknown IDs, invalid configuration, unavailable
optional dependencies, and factory failures are startup errors. There is no
fallback to BM25, hashing, token overlap, another provider, or another model.

`ComponentRegistry` constructs subsystems. The existing `ToolRegistry` remains
the separate run-time dispatcher for the single `search_knowledge` Agent tool.

## Runtime graph

```text
RuntimeProfile
    -> RuntimeBuilder
        -> provider / retrieval / policy / verification / tool / trace
            -> RuntimeComponents + ComponentManifest
                -> HealthCopilotPipeline / AgentLoop
```

`RuntimeComponents` is long-lived and reusable. `RunContext`, budget state,
trace, AgentState, and AgentSession are created per run. A future Session/Memory
lifetime is intentionally not implemented here.

The component lifetime is deliberately different from the run lifetime: a
provider executor, retriever/index, policy adapter, verifier adapter and tool
registry are constructed once by `RuntimeBuilder` and can serve multiple
`RunContext`s. No adapter stores a mutable per-run `_runtime`; Pipeline and
AgentLoop pass the active context explicitly to each provider/policy/verifier
operation. This keeps interleaved runs' budgets, traces and sessions separate.

`RuntimeComponents.answer(question)` is the profile-aware primary execution
entry point. It creates a fresh `RunContext`, trace and budget for every call,
then uses the same `HealthCopilotPipeline`/`AgentLoop` classes regardless of
the selected retrieval profile. The CLI uses this entry point rather than
constructing a pipeline that bypasses the profile-owned context.

## Identity and learned artifacts

Every built component has:

```text
ComponentIdentity(
  kind, component_id, implementation, version,
  artifact_revision, config_hash, learned_artifacts
)
```

Learned artifacts additionally record provider/family, model ID, revision,
`local_files_only`, and embedding dimension when applicable. No network lookup
is performed to invent a revision. An explicit revision is preferred; absent a
safely available local revision, the identity records `revision: null` and the
run must not be described as bit-exact frozen.

The deterministic manifest includes profile ID, sorted component identities,
knowledge-pack version, and knowledge-scope version. Its SHA-256 is persisted
with the manifest and propagated to `RunIdentity.config_hash` and trace
`RUN_START` metadata. The manifest also includes the source `code_commit` when
it can be discovered locally (or an explicit `HEALTH_COPILOT_BUILD_COMMIT` /
builder value); unresolved provenance is represented as `null`. Metadata-only
traces never contain medical content.

Provider model routing is role-aware without collapsing domain contracts:
`agent`/`generator`, `policy`, and `verifier` may each select a model in the
profile. They all share the same `ProviderExecutor` transport boundary, while
`AgentModel`, `EvidencePolicy`, and claim/grounding verifiers remain separate
adapters. If a role is omitted, it uses the explicit provider default; an
injected test executor receives the deterministic `fixture-provider-model`
default. Resolution is strictly per role: `agent` reads only
`profile.config.agent.model`, `generator` reads only
`profile.config.generator.model`, `policy` reads only
`profile.config.policy.model`, and `verifier` reads only
`profile.config.verifier.model`. A missing role-specific override falls back
directly to the provider default; no role inherits another role's override.
It never silently changes to another model or retrieval component.

## Registered retrieval IDs

The source registry currently registers:

- `bm25-v1`
- `dense-hashing-demo-v1`
- `hybrid-hashing-demo-v1`
- `hybrid-token-rerank-demo-v1`
- `dense-st-multilingual-minilm-v1`
- `hybrid-rrf-st-v1`
- `hybrid-rerank-st-mmarco-v1`

The first four are deterministic demo components. The last three are learned
local components and require the `sentence-transformers` dependency plus local
artifacts. CI never downloads model weights and no weight files are committed.

The primary CLI form is:

```text
python -m health_ai_copilot.cli --mode m3 \
  --profile m3-bm25-default --question "..."
```

`m3-hybrid-rerank-local` selects the M5 learned Hybrid+CrossEncoder component.
The legacy `--retriever hybrid_rerank` form is deprecated and explicitly maps
to `hybrid-token-rerank-demo-v1`; it does not claim to select the learned M5 arm.

## Replay

M4 exchange replay remains valid when no M6 identity is expected. New replay
callers can pass `ReplayMetadata(profile_id, component_manifest_hash,
code_commit)`; if the expected runtime identity differs, replay fails closed.
The replay-only initial-evidence component hashes question keys and the full
evidence content (`source_id`, title, excerpt, URL and score), not just the
question list. A replay using recorded initial evidence and tool exchanges may
build a replay-only evidence source and does not instantiate a live learned
retriever. When recorded tool exchanges supply the needed observations, replay
does not require constructing the live learned retriever.

Learned SentenceTransformer/CrossEncoder profiles record model family, model
ID, revision, local-files-only policy and embedding dimension where applicable.
An explicit revision is preferred; a safely discoverable local revision may be
used, otherwise `revision: null` means unresolved/non-frozen provenance. M6
does not perform network lookup to invent a revision and does not claim
bit-exact reproducibility in that state. Missing optional dependencies,
artifacts, revision/index mismatches and invalid configuration fail during
construction; there is no hashing/BM25/token-overlap fallback.

M6 does not add Agent tools, MCP, Sandbox, Memory, dynamic plugin installation,
hot reload, web search, or post-training.
