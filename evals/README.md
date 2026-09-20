# M0 Evaluation

`m0.jsonl` 是 M0.3 Eval Pack，当前包含 80 条人工构造案例。它们不含个人信息，也不
提供自动生成的“专家 gold answer”。患者教育案例的 `expected_source_ids` 是基于当前
M0.2 Knowledge Pack 的人工来源映射；urgent、prescription 和 unanswerable 案例用于
policy regression 或 OOD 检索诊断。

新增案例前应记录问题设计依据、来源卡片版本和数据许可情况。`status=reviewed` 表示
已检查字段、路由意图和 expected source mapping，不表示临床验证。

M0 的离线评测只覆盖可确定验证的能力：

- `safety_route_accuracy`；
- 当案例提供 `expected_source_ids` 时的 retrieval `Hit@1`、`Hit@3` 和 `MRR`。

`safety_route_accuracy` 只评估 deterministic safety gate：安全检查拦截的问题，
应得到对应的 `URGENT_CARE` 或 `HUMAN_REVIEW`；安全检查放行的问题在该指标中记为
`ANSWER`。它不是包含检索、生成和拒答的 end-to-end route accuracy，因此
`unanswerable` 案例不进入这个指标。

M0.3 Eval Pack 的检索指标只统计带有人工核对 `expected_source_ids` 的
`patient_education` 案例：`Hit@K` 判断前 K 名是否包含任一预期来源，`MRR` 取首个预期
来源的倒数排名。没有 `expected_source_ids` 的案例不会被伪造为“可评测答案”。

案例中的 `challenge_type` 用于 failure analysis，而不是直接作为失败标签：

- `segmentation_failure`
- `synonym_paraphrase`
- `overly_generic_term`
- `source_overlap`
- `source_conflict`
- `ood_false_retrieval`

只有在实际运行 BM25 并检查 ranked evidence 后，才把案例归入最终 failure table。

M0 不使用 LLM-as-Judge，也不声称测量诊断正确率、临床安全性或语义蕴含。M2 的 grounding
评测已建立在可复现的 standalone trace 和人工审核证据关系标准上；它仍不是医疗正确性评测。

## M1 recovery pack

`m1_recovery.jsonl` 是一个聚焦回归包，不是新的大规模质量声明。它包含 M0 已观察到的
3 条 synonym/paraphrase miss、3 条 direct-hit control、4 条 OOD false-retrieval control，
以及 urgent/prescription safety control。`recovery_expected` 只用于评估设计意图，不会被
生产运行时读取，也不包含固定 rewrite 字符串。

`health_ai_copilot.eval.m1.summarize_m1_runs` 只对调用方实际提供的 `AgentRunResult` 聚合
阶段分离的指标：`initial_hit@3` 看初始 ranked evidence，`recovery_success@3` 同时要求
初始未命中、实际调用 `search_knowledge` 且 recovery ranked top-3 命中，`post_recovery_hit@3`
才看整个 observed trajectory。另有 recovery attempt、非必要恢复、OOD tool/answer/abstain、
safety short-circuit accuracy、turn/tool budget 和 citation integrity 指标；不存在名为
`recovery_hit_at_3` 的混合指标。没有 live model run 时，不会伪造 M1 数字。

配置好 `HEALTH_COPILOT_API_KEY`、`HEALTH_COPILOT_BASE_URL`（可选）和
`HEALTH_COPILOT_MODEL` 后，可用 `python tools/run_m1_focused_eval.py --trials 3` 运行同一
模型配置下的 focused diagnostic；每次运行会在 `runs/m1/<timestamp>/` 保存配置、原始案例、
分阶段 trajectory、metrics、failures 和报告。该结果不代表泛化性能。

## 外部 benchmark

`tools/fetch_m0_data.py benchmarks` 会把 HealthBench 和 MIRAGE 下载到被 Git 忽略的
`artifacts/benchmarks/`。它们不是 M0 自建 `expected_source_ids` 的替代品：HealthBench
更适合后续回答质量、安全与沟通评测；MIRAGE 更适合比较医学 RAG 的检索/语料组合。
在使用或再分发前，先阅读各自上游的数据和代码许可。

## M2 focused packs

`m2_policy.jsonl` contains 24 reviewed, current-KnowledgeCard policy fixtures: direct sufficient,
recoverable paraphrase, and OOD insufficient. No conflict fixture is included because this knowledge
pack has no defensible reviewed source conflict. `m2_grounding.jsonl` is a small evidence-relation pack
for supported, unsupported, contradicted, coverage-missing, and fabricated-citation outcomes. It does
not represent clinical truth or medical accuracy.

BEIR NFCorpus 是单独的标准检索轨道。运行
`python -m health_ai_copilot.eval.nfcorpus --data-dir artifacts/benchmarks/nfcorpus`
可在本地 corpus/queries/qrels 上测当前 BM25 的 Recall@K、MRR 和 nDCG@K。它与
HealthBench 的 rubric 以及产品 policy regression set 保持不同数据模型。

## M3 capability and claim-support packs

`m3_capability.jsonl` freezes the exact Policy state: question, observed evidence IDs, proposed query,
reviewed scope ID, expected decision, and manually reviewed expected topic IDs. Its evaluator materializes
those IDs directly and never reruns BM25, so future retriever changes cannot mutate gold inputs.
`m3_claim_support.jsonl` evaluates only claim support (including multi-claim, fabricated citation, and wrong
citation binding). It intentionally has no `coverage_missing` category because M3 removes free-answer coverage
classification from the output path by constructing visible text from verified claims.

Claim-support reports separate two layers of evidence: disposition-level accept/reject correctness (whether the
pipeline safely accepts or rejects a fixture) and `fine_grained_claim_verdict_accuracy` (per semantic
SUPPORTED/UNSUPPORTED/CONTRADICTED verdict). A fabricated-citation fixture has no semantic-verdict denominator,
because deterministic citation integrity rejects it before the verifier.

Approved M3 expansion releases live in `evals/expansion/` with review manifests that bind the exact dataset SHA256.
Their results are separate from frozen M3 packs. See `evals/expansion/m3_expansion_annotation_audit.md` for the
two capability cases under annotation review and the expansion fine-grained verdict analysis.

`evals/retrieval/m5_product_retrieval_v1.jsonl` is an 80-case M5 retrieval suite mechanically assembled from frozen M0
rows and the already review-manifest-approved M0 expansion. Its manifest binds every component hash and explicitly does
not assert a new annotation or new user approval; retrieval-only metrics for it are separate from M0 frozen baselines.

## M4 public record/replay diagnostic

`m4_replay.jsonl` is a fixed six-case subset of previously reviewed `m1_recovery.jsonl`, with source-dataset
provenance per row. It exercises public recording and semantic replay only; it does not change M1/M2/M3 gold and
is not a new answer-quality or generalization benchmark. `runs/m4/` public artifacts contain reviewed fixture
content by explicit design; production `metadata_only` traces must not.

## M7 unified evaluation entry point

M7 adds an explicit source-registered suite layer without replacing the historical
evaluators above. List suites with:

```powershell
python -m health_ai_copilot.eval.cli list
```

Run deterministic M0/M5 parity or the fixed M4 replay with `health-eval run`; live
execution requires `--allow-live-provider`, and public content additionally requires
`--public-eval-content`. Dataset SHA-256 values are checked before runtime construction.
Unified artifacts live under `runs/m7/<timestamp>/` and include run spec, component
manifest, case records, `trajectory_v1`, deterministic grader results, inspectable
metrics, failure records, traces, and a hashed run manifest. M7 keeps M3 capability
payloads intact, separates claim-level verdicts from final disposition, excludes
UNGRADED/infrastructure errors from quality denominators, and adds no LLM judge. See
[`docs/m7_eval.md`](../docs/m7_eval.md).

## M8 Agent Team candidate diagnostic

`m8_agent_team_focused_v1.jsonl` is a 12-case focused candidate diagnostic:
four direct education cases, four decomposable multi-evidence cases, two
cross-source comparison cases, and two OOD/safety cases. The composed-question
annotations and required evidence groups are recorded in
`m8_agent_team_focused_v1.annotation_manifest.json` with
`review_status=pending_human_review`; they are not a validated benchmark.

The unified suite `m8-agent-team-focused-v1` permits only these compatible
arms: `m8-workflow-bm25-v1` (L0 deterministic workflow), `m3-bm25-default`
(L1 frozen single Agent), and `m8-team-bm25-v1` (L2 bounded Team). All arms
share the same dataset SHA, BM25 retriever, KnowledgeScope, provider model,
and claim-support verifier. Team-only rates are null/not applicable for L0
and L1. Cost dimensions remain first-class; no weighted overall score is
reported.

The frozen Team identity is `topology=star-supervisor-v1` and
`scheduler=sequential-v1`. Team trajectories retain one metadata record per
role, including provider calls, tool proposals/executions, observed source IDs,
and final cited source contribution. The M8 aggregate metrics also expose
`m8.worker_evidence_overlap` (pairwise observed-source Jaccard overlap) and
`m8.worker_unique_evidence_contribution` (the fraction of union sources seen by
exactly one role). These are deterministic diagnostics for redundancy and
coverage. Recovery-only counterparts use only worker recovery evidence and are
reported as `m8.worker_recovery_evidence_overlap` and
`m8.worker_unique_recovery_contribution`. Completion and productivity remain
separate: `m8.worker_completion_rate` counts normal worker task completion,
while `m8.worker_productive_report_rate` requires claims or recovery evidence.
These metrics are part of metric definition `m8-metrics-v2`; they are not
training signals and do not imply a validated benchmark. The
suite's `allowed_profiles` is enforced during both run-spec preparation and
runtime construction to prevent retrieval-confounded M8 artifacts.
