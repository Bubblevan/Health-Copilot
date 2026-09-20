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
`RUN_START` metadata. Metadata-only traces never contain medical content.

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
callers can pass `ReplayMetadata(profile_id, component_manifest_hash)`; if the
expected runtime identity differs, replay fails closed. A replay using recorded
initial evidence and tool exchanges may build a replay-only evidence source and
does not instantiate a live learned retriever.

M6 does not add Agent tools, MCP, Sandbox, Memory, dynamic plugin installation,
hot reload, web search, or post-training.
