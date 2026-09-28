# AgentTeams reference audit for E2-A

Repository inspected: `D:\MyLab\Jianli\external\multi-agent\AgentTeams`

Pinned local checkout: `89562fb342564f83ed7c88c1d9d2f5f981897d41`.

## Relevant files and mechanisms

| Upstream path | Relevant mechanism | E2-A interpretation |
| --- | --- | --- |
| `agentteams-controller/api/v1beta1/types.go` (`WorkerSpec`, `WorkerStatus`) | Worker declaration binds model/provider, runtime, identity/role material, skills, permissions, and desired lifecycle. Status is separate from desired spec. | Keep declared capability and execution state separate. E2 stores a typed contract and an execution report; it does not need a CRD. |
| `agentteams-controller/internal/controller/worker_controller.go` | Reconcile loop moves a declared worker toward its requested lifecycle state. | Adopt fixed worker identities and explicit eligibility. Reject a general worker lifecycle/controller. |
| `docs/design/teamharness/project-task-runtime-design.md` | Project and task metadata are distinct; assignments carry context and acceptance criteria; task state survives worker turns. | Use a typed, immutable assignment and a bounded status/report. No cross-run project store is needed for one sequential research run. |
| `docs/design/teamharness/task-transition-engine.md` | Central transition table, transition history, progress action, and structured completion/result. | Adopt one harness-owned state transition path and explicit completion/error fields. Do not adopt durable task history for this bounded runtime. |
| `plugins/teamharness/skills/team/task-delegation/SKILL.md` and `skills/team/task-execution/SKILL.md` | Assignment contains context, expected result, acceptance criteria, and a completion report. | Keep objective separate from role capability; do not accept worker-invented IDs or completion state. |
| `manager/agent/worker-agent/skills/task-progress/SKILL.md` | Progress and final completion are distinct reports. | Report completion separately from contribution/productivity. E2 productivity is computed from verified evidence, not a status label. |
| `docs/design/capability-foundation.md` | Defines closed human/cloud privileges such as `external_sources` and `approval_policy`, with grant/revoke audit. | **Explicitly rejected as an evidence-capability model.** These are identity/security privileges, not corpus/domain capabilities. E2 uses source IDs/families, tool allowlists, authority scope, and budget caps. |

## Ideas adopted

- A worker declaration has stable identity and an explicit runtime binding.
- Assignment state and completion are harness-owned and structured.
- The worker reports progress/completion; the manager independently validates acceptance.
- Capability declarations are closed and auditable. E2 rejects unknown source IDs and unsupported tools.

## Ideas rejected

- Kubernetes, CRDs/controllers, containers, Matrix, MinIO, Higress, dashboards, long-lived worker processes, worker creation, remote skill registries, and persistent task/project stores. They solve deployment and collaboration problems outside this local bounded research harness.
- The AgentTeams `Human.spec.capabilities` contract as the meaning of evidence specialization. It governs privileged operations and must not be confused with evidence authority.
- Runtime/model changes between E2 workers for the first controlled comparison. The worker model is held fixed; capability boundaries change.

## Health-Copilot mapping

- Worker declaration → `WorkerCapabilitySpec` in `src/health_ai_copilot/capabilities.py`.
- Assignment → fixed-role `E2Task` in `src/health_ai_copilot/e2_team.py`.
- State and acceptance → `E2WorkerReport`; `productive` requires completion and either new ledger-backed evidence or a verified ledger-backed citation.
- Lifecycle trace → `E2_WORKER_STARTED`, `E2_WORKER_FINISHED`, and `E2_WORKER_REPORT`, sharing the parent run identity and trace sink.
- Runtime binding → fixed `AgentModel` mapping, `CapabilityScopedSearchTool`, and the parent/worker budget wrapper. There is no worker spawning or runtime switching.
