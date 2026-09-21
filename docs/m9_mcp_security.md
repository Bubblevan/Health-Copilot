# M9.1 — Declarative MCP/Security Graph, Failure Semantics & Eval Closeout

M9 implements Harness Macro H6 as three separate control layers:

```text
AgentLoop -> ToolRegistry -> PermissionGuard -> McpToolAdapter
          -> McpClient -> MCP transport -> MCP server
```

MCP connectivity answers how Health-Copilot talks to a capability provider.
`PermissionPolicy` answers whether the current run may invoke it. The sandbox
answers what the provider process can technically reach. Approval is neither
MCP authorization nor sandbox widening.

## Status and frozen boundary

M9.1 is the final H6 closeout over the experimental M9 capability boundary. The product default is
still `m3-bm25-default`; the only product-facing M9 profile is the opt-in,
read-only `m9-mcp-search-bm25-v1`. The security fixture tools are test-only and
are not registered in M3, M8, or M9 product profiles.

M8 remains frozen. M9 does not start Memory, persistent context, post-training,
SFT/DPO/RL, multimodal work, dynamic topology, decentralized MAS, A2A, or MCP
Apps/Tasks/Prompts. M10 is reserved for Memory, M11 for post-training, and M12
for multimodal work.

The M9 checkpoint recorded real containment evidence on WSL2 Ubuntu 24.04 with
Bubblewrap. This M9.1 closeout preserves that evidence boundary and adds the
platform-independent contracts; the current Windows host has no available
WSL2/Bubblewrap backend, so its containment smoke is reported as skipped rather
than as new evidence. Environments without that backend must fail closed for a
profile that requires real enforcement; a `no-sandbox-dev-v1` profile is an
explicit non-containing development escape hatch, not an H6 claim.

## M9.1 declarative closeout

`RuntimeProfile` now has optional component selections:

```text
mcp_client: str | None
permission: str | None
sandbox: str | None
```

The fields are component IDs, not Python objects. They are omitted from the
canonical profile dictionary when `None`, preserving the frozen M0–M8 profile
hashes. The explicit M9 declaration is:

```text
profile:  m9-mcp-search-bm25-v1
mode:     m9_mcp
mcp_client: mcp-client-2026-07-28-v1
permission: permission-policy-v1
sandbox:    no-sandbox-dev-v1
tool_set:   mcp-search-knowledge-v1
```

The builder constructs these IDs from the profile graph. It does not select
M9 components by branching on `profile_id`; unknown IDs and missing required
M9 graph fields fail closed. The component manifest therefore records the MCP,
permission, and sandbox identities selected by the profile. The real backend
registrations are explicit (`bubblewrap-sandbox-v1` and
`bubblewrap-sandbox-wsl-v1`) and are never dynamically installed or made the
product default.

Sandbox setup is not authorization. A required stdio sandbox that is missing,
unavailable, or non-containing is reported as:

```text
McpFailureKind.SANDBOX
SandboxFailureKind.UNAVAILABLE | FILESYSTEM_DENIED | NETWORK_DENIED | PROCESS_ERROR
```

The MCP `AUTHORIZATION` category remains reserved for a future actual
protocol/resource-server authorization failure. Permission denial, approval
denial, MCP tool-call errors, and sandbox failures remain distinct in the M7
failure taxonomy.

The security manifest is now a first-class offline M7 suite:
`m9-mcp-security-v1`, target kind `security`, metric definition
`m9-security-metrics-v1`. It runs deterministic in-process fixtures only and
produces the normal M7 bundle. The closeout bundle is
`runs/m9/20260921T130541979367+0800/`; it contains 12/12 passing controls,
including 4/4 permission, 3/3 protocol, and 5/5 sandbox-contract controls.
This offline result is not a claim of real process containment; the latter is
reported separately from the WSL2/Bubblewrap smoke.

The M8 real-run interpretation is intentionally precise: observed worker
completion was `0`; full observed-context overlap was `1.0` on the comparable
multi-worker records; recovery-only unique contribution was `0`. Therefore no
unique worker evidence contribution was observed. The overlap metric does not
justify claiming that two independently completed workers produced the same
evidence.

## A. Modern MCP baseline

M9 uses the final [MCP 2026-07-28 specification](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/index.mdx)
and pins the official Python SDK as `mcp==2.2.0`. The runtime uses strict
`mode="2026-07-28"` and the SDK's `server/discover`, `tools/list`, and
`tools/call` surfaces. It does not implement legacy `initialize`/
`initialized`, `Mcp-Session-Id` affinity, HTTP+SSE as the primary transport,
Roots, Sampling, or Logging.

The synchronous Health-Copilot boundary never exposes SDK result objects to
`AgentLoop`. `McpToolAdapter` converts a catalog entry to `ToolSpec`, validates
JSON Schema before dispatch, and converts the call result to the existing
`ToolResult` plus `Evidence[]` boundary.

For Streamable HTTP, an instrumented local official-SDK fixture verifies:

| Operation | Observable routing metadata |
| --- | --- |
| `tools/list` | `MCP-Protocol-Version: 2026-07-28`, `Mcp-Method: tools/list` |
| `tools/call` | `MCP-Protocol-Version: 2026-07-28`, `Mcp-Method: tools/call`, `Mcp-Name: search_knowledge` |

The test also verifies that no `Mcp-Session-Id` is required. The in-process
fixture explicitly returns only `2026-07-28`; a version mismatch is typed as a
protocol failure rather than silently downgraded.

## B. Server identity, namespace, and catalog provenance

`McpServerConfig` is trusted runtime configuration. It records `server_id`,
transport, protocol version, endpoint identity, auth mode, sandbox profile ID,
and namespace. HTTP endpoints must use `http` or `https`, have a host, and must
not contain credentials. The endpoint itself is omitted from safe component
identity output; tokens and secrets are never manifest fields.

The canonical internal namespace is:

```text
mcp::<server_id>::<tool_name>
```

The M9 search profile exposes the unambiguous short name
`search_knowledge`; two servers using the same remote tool name still have
different canonical names and cannot silently overwrite each other in
`ToolRegistry`.

`McpToolCatalogSnapshot` records the sorted catalog, protocol version,
`catalog_hash`, `ttl_ms`, `cache_scope`, and `obtained_at`. The official SDK's
`tools/list` cache hints are observed (`ttlMs=1000`, `cacheScope=public` in the
search fixture). The domain client reuses a fresh snapshot and refreshes on
expiry or an explicit force-refresh; it does not implement a legacy
session-changed list model.

The component manifest binds the MCP SDK/protocol, server identity, tool
namespace, catalog hash/cache metadata, permission-policy hash, and sandbox
profile hash. Changing an authority boundary therefore changes runtime
identity.

## C. MCP authorization versus Harness permission

Remote MCP OAuth is **NOT IMPLEMENTED** in M9. M9 does not fake authorization
with a static API key and does not implement a custom OAuth stack or Dynamic
Client Registration. The product profile uses explicitly unauthenticated local
fixtures only. Current authorization guidance remains the reference for a
future authenticated integration:
[MCP authorization security considerations](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/basic/authorization/security-considerations.mdx).

MCP server authorization and local agent permission remain distinct:

1. MCP authorization asks whether a client may access a resource server.
2. `PermissionPolicy` asks whether this Agent/run may invoke a configured tool.
3. The sandbox asks what the tool process can technically access.

M9 validates the second and third layers locally. The auth-success/permission-
deny and permission-allow/auth-failure branches are documented as future
remote-auth conformance work, not claimed as implemented.

## D. Permission and approval

The trusted local mapping is:

```text
(server_id, tool_name) -> CapabilityPolicy
```

Capability classes are `READ`, `WRITE_WORKSPACE`, `NETWORK`,
`EXTERNAL_SIDE_EFFECT`, and `SENSITIVE_DATA`. Each policy can record
`read_paths`, `write_paths`, and `network_origins`. Unknown servers/tools are
default-deny; MCP descriptions cannot create or widen this mapping.

The enforced order is:

```text
schema validation -> capability lookup -> ALLOW/DENY/REQUIRE_APPROVAL
-> trusted ApprovalProvider -> sandboxed transport execution -> ToolResult
```

`ApprovalProvider` is a trusted application/human boundary. `AlwaysApprove`,
`AlwaysDeny`, and scripted providers are deterministic test utilities only.
Approval requests contain run ID, server/tool identity, capability,
argument hash, resource scope, reason, and one-shot scope. The request ID
binds the result to the exact request. The model has no self-approval API.

Approval grants logical permission only. It never widens the predeclared
sandbox. The matrix is tested explicitly: deny and denied approval make zero
MCP calls; approved calls can execute when the sandbox allows; an approved
outside-workspace write and an approved network probe still fail at the
sandbox boundary.

## E. Sandbox technical boundary

The sandbox contracts are `SandboxBackend`, `SandboxPolicy`,
`SandboxProfile`, `SandboxResult`, and `SandboxFailure`. Filesystem and
network controls are separate fields:

```text
filesystem: read_roots, write_roots
network: DENY_ALL | ALLOWLIST
```

Bubblewrap is the real Linux backend. On this Windows development host,
`WslBubblewrapSandboxBackend` launches the same Bubblewrap process boundary
inside WSL2. A required real profile fails closed when the backend is missing
or non-containing. Network allowlists are represented but not silently treated
as enforced; the current backend fails closed for an unimplemented allowlist.

The test-only `m9_security_mcp_server.py` exposes synthetic
`read_allowed_file`, `read_forbidden_file`, `write_workspace_file`,
`write_outside_workspace`, and `network_probe` tools. It is launched through
the sandbox backend as an MCP stdio server, never as a direct unsandboxed
`python server.py` product capability.

Observed real smoke results:

| Check | Result |
| --- | --- |
| allowed fixture read | succeeds |
| workspace write | succeeds |
| read outside mounted root | fails in the sandboxed process |
| write outside workspace | fails in the sandboxed process |
| network probe with `DENY_ALL` | fails in the sandboxed process |

These are subprocess observations, not only PermissionPolicy assertions.

## F. Product integration and parity

`m9-mcp-search-bm25-v1` preserves the existing provider, M3 policy,
claim-support verifier, KnowledgeScope, BM25 corpus, AgentLoop, and upper
Harness. Only `search_knowledge` crosses the MCP adapter. The deterministic
parity test uses identical cards, query, and top-k and compares source IDs and
order plus the normalized `ToolResult`/`Evidence[]` boundary. M9 is transport
replacement, not a retrieval-quality improvement, and it does not retune
BM25.

The existing M4 `RecordingToolRunner` / `ReplayToolRunner` remains the replay
authority. MCP calls are recorded after normalization as tool identity,
arguments, and `ToolResult`; replay returns the recorded observation without
starting an MCP server, network, or sandbox process.

The local security eval manifest is
`evals/m9_mcp_security_v1.jsonl`. It is not an LLM benchmark. Its deterministic
coverage includes protocol identity, catalog cache, namespace collision,
default deny, approval, prompt-injection output, filesystem containment,
network containment, and sandbox-unavailable fail-closed behavior.

M8's deterministic `worker_evidence_overlap` and
`worker_unique_evidence_contribution` fields remain preserved in team
trajectories/eval records; M9 does not train on them or rerun the M8 L0/L1/L2
experiment.

## G. Prompt injection and privacy

MCP tool descriptions and results are untrusted data. Text such as “ignore
previous policy”, “call another sensitive tool”, or “upload data” cannot
register a tool, mutate the permission map, approve itself, widen a sandbox,
change a RuntimeProfile, or change EvidencePolicy. This is a control-plane
boundary claim, not a claim that prompt injection is solved at the model layer.

Metadata-only traces record server/tool identity, capability, method,
protocol, catalog hash, decision, approval ID, sandbox profile, failure kind,
argument hash, and public-safe observed source IDs. They do not record bearer
tokens, OAuth codes, client secrets, SSH/Git credentials, or raw sensitive
environment variables.

## H. Threat model and limitations

M9 explicitly considers malicious MCP servers, malicious descriptions,
prompt-injected results, unknown tools, over-permissioned agents, approval
fatigue, credential leakage, filesystem/network exfiltration, confused deputy
behavior, tool-name collision, and remote endpoint substitution.

The implemented mitigations are default-deny local policy, pre-call approval,
trusted profile configuration, canonical identity, metadata-only tracing,
stdio process containment, and separate filesystem/network enforcement. M9
does not claim that MCP servers are safe, that approval guarantees user intent,
that OAuth makes tools safe, or that Bubblewrap is unbreakable. Kernel/WSL/
Bubblewrap escape is outside this milestone's threat model; the backend itself
is an explicit platform assumption and must be tested on supported hosts.

Future work includes authenticated remote MCP using current SDK-supported
authorization/CIMD flows, network allowlists, richer remote endpoint policy,
MRTR, MCP Tasks/Apps, and broader capability surfaces. None is silently
implemented here.

## Source alignment

The following sources inform M9's boundaries. “Adopted” describes a concrete
M9 decision; “deferred” is intentionally outside the milestone; “rejected” is
not part of M9's execution semantics.

| Source | Adopted | Intentionally deferred | Rejected for M9 | Why |
| --- | --- | --- | --- | --- |
| [MCP 2026-07-28 specification](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/index.mdx) | Strict protocol pin, discovery, tools/list, tools/call, structured schemas/results, Streamable HTTP routing, cache hints. | Optional extensions, Tasks, Elicitation, Subscriptions, MRTR. | Legacy initialize-era architecture, Roots/Sampling/Logging as new capabilities, HTTP+SSE primary. | M9 proves the smallest modern capability boundary and avoids deprecated/session-shaped authority. |
| [MCP Python SDK v2](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/whats-new.md) | Official SDK `mcp==2.2.0`, strict `Client` mode, official MCPServer fixtures/transports. | Broader SDK auth integrations and extension APIs. | Hand-rolled JSON-RPC in product code and SDK object leakage into AgentLoop. | The SDK owns protocol correctness; Health-Copilot owns typed policy and result normalization. |
| [MCP authorization/security guidance](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/basic/authorization/security-considerations.mdx) | Keep protocol auth, local permission, approval, and sandbox as distinct authorities; no secrets in traces. | CIMD/OAuth issuer validation and credential lifecycle integration. | Static API-key substitution presented as OAuth or custom DCR. | Remote OAuth is explicitly NOT IMPLEMENTED in M9. |
| [Anthropic Multi-Agent Research](https://www.anthropic.com/engineering/multi-agent-research-system) | Source-aware findings and explicit capability boundaries remain compatible with M8 and M9 provenance. | Open-web research, parallel scaling, iterative research loops. | Adding open-web/browser/shell capabilities to the Health-Copilot product profile. | M9 is a safe capability transport test, not a broader research-agent experiment. |
| [Anthropic multi-agent failures (2026)](https://www.anthropic.com/research/multiagent-systems) | Treat untrusted output, redundancy, conformity, and coordination failure as threat/eval concerns. | Reputation, dissent protocols, adversarial multi-agent training, richer trust models. | Letting tool text or distributed peers mutate runtime authority. | Authority must remain in trusted local policy and the sandbox. |
| [Microsoft Agent Framework / Magentic](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/magentic) | Preserve M8's manager/worker orchestration as the frozen upper architecture; keep capability execution below ToolRegistry. | Dynamic replanning, multiple collaboration rounds, shared context. | Replacing M8 with a flexible Magentic workflow during M9. | M9 changes the tool boundary, not the frozen team topology. |
| [Magentic-One](https://arxiv.org/abs/2411.04468) | Use bounded specialist execution and manager-owned synthesis as an architectural reference only. | General-purpose planning, checkpointing, adaptive recovery. | Importing its participant set or open-ended autonomy. | The M9 acceptance question is control-plane enforcement, not general task autonomy. |
| [AgentScope / Alibaba AgentTeams](https://github.com/agentscope-ai/AgentTeams) | Retain explicit role/capability identity and per-role metadata from M8. | Distributed gateways, shared rooms, external Manager/Worker deployments. | New decentralized or external team control planes. | M9 has one local Harness authority and one parent runtime budget. |
| [ClawArena-Team](https://arxiv.org/abs/2606.31174) | Keep execution-based, least-privilege, per-role trajectory thinking in the existing M8 records. | Its multimodal, multi-directory, dynamic management benchmark. | Using its score as a Health-Copilot security or quality claim. | It is background for threat/eval design, not the M9 dataset. |
| [MultiAgentBench](https://aclanthology.org/2025.acl-long.421.pdf) | Retain topology-aware evaluation vocabulary and explicit contribution accounting. | Chain/tree/graph/competition/discussion protocols. | Dynamic graphs, decentralized MAS, or topology changes in M9. | M8 topology remains frozen; M9's new dimension is capability authority. |
| [Agent scaling / Ringelmann](https://arxiv.org/abs/2606.02646) | Preserve M8 overlap/unique-contribution indicators and strict bounded execution. | Scale-law fitting, larger teams, heterogeneous-model ablations, training signals. | Calling M9 transport/security tests evidence of scaling gains. | M9 does not run L0/L1/L2 and does not train on trajectories. |
| [Anthropic sandboxing / containment engineering](https://www.anthropic.com/engineering/claude-code-sandboxing) | Use real process containment, explicit mounts, denied network, and fail-closed backend selection. | Production multi-platform sandbox fleet and broader policy orchestration. | Python monkeypatching or an “approval means sandbox” shortcut. | H6 requires technical enforcement evidence. |
| [OpenAI Codex sandbox/approval separation](https://developers.openai.com/codex/concepts/sandboxing/) | Model the separation between approval authority and technical sandbox authority. | Product-specific UI, host fleet, and account-level approval mechanisms. | Treating a UI approval as a technical containment guarantee. | M9's `ApprovalProvider` is a harness seam, not a sandbox implementation. |

## Acceptance evidence

- `RuntimeProfile` declaratively selects the M9 MCP client, permission, and
  sandbox components; M0–M8 canonical profile hashes remain frozen.
- `m9-mcp-security-v1` is visible in `health-eval list` as an offline
  `security` target and `health-eval run --suite m9-mcp-security-v1
  --execution offline` produces the standard M7 bundle.
- The offline security bundle passes 12/12 controls: permission 4/4,
  protocol 3/3, and sandbox contract 5/5. These are deterministic control
  semantics, not a security certification or real-containment result.
- GitHub Actions validates the platform-independent protocol, permission, unit,
  and offline security contracts; WSL-specific containment tests are skipped
  there. The supported local Windows/WSL2 host is the separate source of real
  Bubblewrap containment evidence.
- SDK: `mcp==2.2.0`; protocol: `2026-07-28`.
- Modern discovery and no legacy session dependency: deterministic tests pass.
- Streamable HTTP `MCP-Protocol-Version`, `Mcp-Method`, and `Mcp-Name`: instrumented official-SDK test passes.
- Catalog hash/order/TTL/cache scope and internal-vs-MCP parity: deterministic tests pass.
- Default-deny, approval binding, prompt-injection authority boundary, and privacy trace checks: deterministic tests pass.
- Real WSL2 Bubblewrap filesystem and network containment smoke: passes.
- Required sandbox with a non-containing backend: fails closed.
- M0–M8 profiles remain unchanged in their product tool sets; M3 remains default.
- No real M8 L0/L1/L2, live Agent smoke, Memory, post-training, or multimodal work was started.
