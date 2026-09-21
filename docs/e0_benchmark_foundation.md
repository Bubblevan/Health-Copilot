# E0 — Benchmark Foundation

状态：`IMPLEMENTED / REVIEW PENDING`

E0 只建立可复现的 benchmark substrate，不执行外部评分，不调用 judge，不启动
E2 heterogeneous Team、E3 swarm、E4 routing、E5 self-evolving Harness、M11 或
M12。M0–M10.1 的代码、M8 negative-result artifacts、M8 frozen gold、M7 metric
definitions、RuntimeProfile hashes 和 M10.1 ContextProjector semantics 保持不变。

## 研究边界

Health-Copilot 的安全边界仍是 patient education / public medical information、
reviewed evidence、citation verification、safety gate、permission 和 replay。外部
medical QA 分数不等于 clinical validation；HealthBench rubric judge 不等于 medical
truth；MIRAGE exam/literature QA 不等于 patient-care performance；NFCorpus retrieval
不等于 medical answer correctness。

基础原则：

```text
SOURCE DEFINES DATA
MANIFEST DEFINES INTERPRETATION
RUNNER DOES NOT SILENTLY CHANGE EITHER
```

## 三层评测分类

| 层 | 内容 | E0 行为 |
| --- | --- | --- |
| Layer A | M0–M10.1 internal diagnostics | 保持原 suite、gold 和 metric 不变 |
| Layer B | NFCorpus external retrieval | 只准备 manifest、adapter、graded qrels contract |
| Layer C | Medical MIRAGE / HealthBench quality | 只准备来源、schema 和 protocol，不跑分 |

这些层不合并成一个 `Health-Copilot Score`。

## Source alignment

| Source / benchmark | Canonical identity | Used in E0 | Deferred to E1 | License status | Metric family | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| NFCorpus | upstream NFCorpus + BEIR `v2.2.0` at `6ef8c9097ebfb203ad360bd64e0cfb93e64f4a44` | manifest and offline adapter | retrieval nDCG/Recall/MRR | `REVIEW_REQUIRED` | `RETRIEVAL` | BEIR packaging terms are not upstream license approval |
| Medical MIRAGE | `gzxiong/MIRAGE` at `392943af99cd94cafd50a0de2e7fca24bbf65494` | manifest and subdataset-preserving adapter | RAG protocol metrics | `REVIEW_REQUIRED` | `EXTERNAL_QUALITY` | canonical repo is specifically `gzxiong/MIRAGE` |
| HealthBench | OpenAI `simple-evals` at `652c89d0ca9df547706735883097e9537d40dc47` | artifact and judge manifest | rubric judge execution | `REVIEW_REQUIRED` | `EXTERNAL_QUALITY` | judge protocol is pinned but not invoked |
| M8 frozen internal diagnostic | `evals/m8_agent_team_focused_v1.jsonl` | preserved, not migrated | none | existing frozen policy | `HARNESS_DETERMINISTIC` | E0 does not repair or relabel M8 |
| research-architecture-v1 | source-controlled candidate pack | 48 cases, profiles, split/audit contract | E2–E4 comparisons | human review pending | harness/team contribution | no expected-winner field |

## Source-controlled layout

```text
benchmarks/
  registry.json
  nfcorpus/{manifest.json,normalization.json,license_review.md}
  medical_mirage/{manifest.json,normalization.json,license_review.md}
  healthbench/{manifest.json,normalization.json,judge_protocol.json,license_review.md}
  research_architecture_v1/
    {manifest.json,annotation_manifest.json,cases.jsonl,task_profiles.jsonl}
    {fairness_contract.json,cost_model.json}
```

Raw external data is never committed. The default ignored local cache is
`.health-bench-data/`, or the path from `HEALTH_COPILOT_BENCH_DATA`. Its layout is:

```text
<data-root>/raw/<benchmark-id>/
<data-root>/normalized/<benchmark-id>/
<data-root>/cache/<benchmark-id>/
```

Each raw artifact has a `RawArtifactIdentity` containing URL, size, SHA-256, download
time and upstream HTTP metadata. URL alone is never a data identity. Each normalized
artifact has a `NormalizedDatasetIdentity` bound to raw SHA(s), adapter version, schema
version, case count, normalized SHA and split-manifest SHA.

## Contracts and adapters

`BenchmarkManifest` records canonical source identity, revision, license/admissibility,
privacy, medical-content classification, split/gold/metric protocol and optional judge
protocol. `BenchmarkRegistry` is an explicit source-controlled mapping; it does not scan
plugins, import arbitrary paths or perform network I/O.

`BenchmarkCase` keeps benchmark-specific payload and gold separate. NFCorpus preserves
graded qrels and canonical splits. MIRAGE preserves subdataset, options, answer and
question-only retrieval metadata. HealthBench preserves conversation and rubric
structure, but never runs the grader in E0.

Adapters have the local-only contract:

```text
inspect(raw_root)
normalize(raw_root, output_root)
validate(normalized_root)
```

Normalization contains no hidden download, provider call, model call or scoring step.
Synthetic fixtures are used for CI so tests do not copy externally restricted medical
questions into the repository.

## research-architecture-v1

The candidate pack has 48 cases, six per category:

| Category | Count |
| --- | ---: |
| `SIMPLE_DIRECT` | 6 |
| `SINGLE_SOURCE` | 6 |
| `BREADTH_MULTI_SOURCE` | 6 |
| `CROSS_SOURCE` | 6 |
| `CONFLICTING_EVIDENCE` | 6 |
| `TEMPORAL_EVIDENCE` | 6 |
| `SERIAL_DEPENDENCY` | 6 |
| `OOD_INSUFFICIENT` | 6 |

DEV and TEST each contain 24 cases. The split is checked for duplicate IDs, source-group
and question-family leakage, and exact/normalized text duplicates. A `TaskProfile` contains
task family, answerability, required evidence groups, source families, independent
subtasks, dependency edges, conflict/temporal metadata and derived features. It contains
no `best_architecture`, `team_should_win` or `single_should_win` field.

Task profiles and gold are eval-only. The case payload sent to a future Agent contains no
task-family, evidence-group, serial-depth, expected-architecture or OOD gold hints. A
required evidence group means at least one acceptable final verified citation; retrieved
does not mean cited.

The pack is currently `candidate` with `human_review.status = pending`. The deterministic
review export is produced by:

```powershell
health-bench review-export research-architecture-v1
```

Only after human review may the deliberate action below succeed:

```powershell
health-bench freeze research-architecture-v1
```

The freeze action is fail-closed and does not silently change annotation status.

## Fairness contracts

Three future comparison regimes are defined in
`benchmarks/research_architecture_v1/fairness_contract.json`:

- `NATIVE_BUDGET`: each architecture uses its declared bounded native budget;
- `COST_MATCHED`: provider calls, tool calls and observed token ceiling are shared;
- `DEADLINE_MATCHED`: wall-clock deadline is shared and measured by a monotonic clock
  from execution start through the final verified response.

Wall-clock latency, sum of worker/provider execution time, provider calls, tool calls and
tokens remain separate dimensions. Volatile API prices are not embedded in gold.

## CLI contract

```text
health-bench list
health-bench inspect <benchmark>
health-bench prepare <benchmark>
health-bench verify <benchmark>
health-bench normalize <benchmark>
health-bench audit [<benchmark>]
health-bench profile [research-architecture-v1]
health-bench review-export research-architecture-v1
health-bench freeze research-architecture-v1
```

`list`, `inspect` and `profile` are network-free. `prepare` only creates the explicit
local cache directories and tells the operator where a reviewed raw artifact must be
placed; it does not download. E0 intentionally has no `health-bench run` command.

## Stage gate

The closeout artifact is `runs/e0/benchmark_foundation_closeout.json`. Its current
status is `READY_FOR_E1 = no` and `READY_FOR_E2 = no` because external raw/normalized
hashes and admissibility reviews are not complete, and the internal research pack has
not received human review. This is the intended `IMPLEMENTED / REVIEW PENDING` state.

E0 is not `COMPLETE / FROZEN` merely because the registry or adapter tests pass. E0 can
freeze only after source/version audit, license review, raw/normalized hashes, reviewed
gold/task profiles, leakage audit and fairness contracts are frozen. Until then E1,
E2 and E3 remain blocked.

## Explicit non-claims

At this stage:

- HealthBench has no score claim;
- Medical MIRAGE has no performance claim;
- NFCorpus has no new E1 result claim;
- E1 External Evaluation has not started;
- E2 Heterogeneous Parallel Team has not started;
- E3 Bounded Swarm has not started;
- M11 Post-training has not started;
- M12 Multimodal has not started.
