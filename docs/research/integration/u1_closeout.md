# U1 Closeout

## Result

U1 implements a research-only immutable episode contract, separate runtime,
evaluation, and privileged-training serializers, subject-scoped time-safe
patient snapshots, a distinct synthetic external evidence world, action masks,
Single/Team capability-union enforcement, deterministic counterfactual replay,
and schema-only SFT/GRPO/OPD artifacts.

It demonstrates reproducible execution contracts on synthetic fixtures only.
It does not complete a unified benchmark, evaluate clinical quality, establish
production Memory/RAG eligibility, or start E2-B/L4/post-training.

## Repository isolation

U1 branch: `integration-u1-longitudinal-env-20260929`, created directly from
`9e023dbddc1f8025e3608bfae4c5390c1a7957ef` (E2-A). The starting worktree was
clean. U0 was committed separately as `df8cac1` with only the 8
`docs/research/multi_agent/u0_*` files and 5 `runs/multi_agent/u0-*` files. The
live checkout at task start was `codex/rag-e5-foundation-20260929` at `5ca4951`, rather than
the Memory branch named in the U0 note; `df8cac1` does not include the
`5f666b2` Memory audit commit. U1 branches from E2-A and contains neither that
U0 commit nor the RAG foundation commit.

## Adapter and boundaries

The ESL-like adapter proposal maps profile to PROFILE, timeline events to EVENT,
exam rows to EXAM, measurements to MEASUREMENT, and conversation history to
CONVERSATION. It reads no ESL evaluation query or answer. Patient-state rows are
not thereby declared training data. U1 does not modify Memory or RAG modules or
the frozen E2-A contract.

## Gates

Machine-generated gates and synthetic outcomes are recorded in
`runs/integration/u1-synthetic-20260929-01/`. The run reports zero provider
calls, no training, no ESL evaluation content access, and no Memory/RAG track
changes. Verification used the existing repository `.venv`:

- U1-focused tests: 15 passed.
- Full suite: 607 passed, 2 skipped, 1 existing `jieba` deprecation warning;
  pytest used an isolated basetemp because the default pytest temp root has
  stale ACL-restricted directories.
- `ruff check .`: all checks passed.
- `python -m compileall -q src`: passed.
- `git diff --check`: passed before commit.

## Next stage

Recommended next stage: **U2-D — resolve dataset/split/license blockers first**.
U1 intentionally stayed synthetic; no real longitudinal rows, external
evidence snapshot, or benchmark split was qualified. Do not begin model
training, E2-B, or L4 from this prototype alone.
