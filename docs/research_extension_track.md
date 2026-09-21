# Research Extension Track

本文档定义 M10.1 冻结之后的研究路线。它把“核心 Harness 是否完成”和“研究问题是否还有价值”分开，避免为了做出更高的 Team 分数而修改已经冻结的实现或 benchmark 语义。

当前基线为 `main@59f817d`；M0–M10.1 Core Harness 已完成并冻结。以下 E0–E6 均为 planned，除非对应阶段门通过，否则不应把结果写成已完成能力或正向结论。

## 研究问题

核心问题不是“Multi-Agent 是否总能赢”，而是：

> 在什么任务结构、证据分布、成本和 deadline 约束下，Harness 应该选择 Single、Workflow、Homogeneous Team 或 Heterogeneous Parallel Team？

研究要保持 Health-Copilot 的产品边界：patient education / public medical information、reviewed evidence、citation verification、safety gate、permission 和 replay。不得为了 Team 实验重新引入诊断、处方或真实 EHR action。

## 阶段与交付物

### E0 — Benchmark Foundation

目标是把现有数据资产和外部数据变成可审计的 benchmark foundation。

交付物：

- external dataset manifest：来源、版本、license、下载 hash、字段映射、PII/medical-content handling；
- 明确区分 internal deterministic metrics 与 external judge-based quality metrics；
- 固定 query/evidence/corpus split，防止在结果出来后修改 gold；
- 为每个 case 增加 task profile：required evidence groups、source families、independent subtasks、serial depth、OOD/insufficient 标记；
- 保留 M0–M10.1 的 frozen suites，不把外部数据偷偷并入旧 suite。

阶段门：manifest 可复现、数据版本可追溯、gold 语义和隐私策略经过人工 review。

### E1 — External Evaluation

按成本和复用价值分层推进：

1. NFCorpus：BM25、learned Dense、Hybrid、Hybrid+Rerank 的完整 external retrieval ablation；
2. MIRAGE：medical RAG external track，报告 corpus/retriever/LLM 组合，不与 M0 source-ID Hit@K 直接横比；
3. HealthBench：patient-facing answer quality、communication、uncertainty、context seeking 和 safety 的独立 external quality track。

外部 benchmark 的 judge 分数不能替代 M7 deterministic harness metrics，也不能被包装成临床验证。每个 adapter 都要报告 dataset version、judge/protocol、失败样例、成本和可复现限制。

阶段门：至少一个 external track 完成可复现 adapter、manifest、baseline 和 failure analysis；未完成的 benchmark debt 要明确列出，不用 headline 数字掩盖。

### E2 — Heterogeneous Parallel Agent Team

这是新的 Team 假设，不修改冻结的 M8 `m8-agent-team-focused-v1`。新 Team 的互补性来自 capability，而不是复制多个相同 worker：

```text
Research Lead
 ├─ Public Health Worker   → WHO / CDC / NHC source family
 ├─ Guideline Worker       → reviewed guideline corpus
 └─ Literature Worker      → PubMed / NFCorpus-style literature corpus
```

每个 worker 必须有不同的 corpus/tool/authority/objective，并保持 worker session 隔离。第一版 contract 建议为：worker turn 1 最多一次 bounded retrieval，worker turn 2 只能 FINAL 或 ABSTAIN，提交带来源的 structured report；禁止 worker 无限搜索或自动写 memory。

交付物：异构 capability manifest、EvidenceLedger、parallel fan-out runtime、父级 deadline/token/provider-call budget、worker completion telemetry，以及与旧 M8 的兼容性隔离证明。

### E3 — Bounded Swarm

在 E2 的有限 fan-out 上研究 bounded swarm，而不是直接实现无界自治。重点是：

- 明确 topology：star / chain / tree / graph；
- 限制 worker 数、层数、工具调用和总预算；
- 记录 delegation precision、worker completion、unique evidence contribution、redundancy、contradiction handling；
- 对 worker failure、partial completion、deadline expiry 和 evidence conflict 保持 fail closed；
- 保留 L0 Workflow、L1 Single、L2 Frozen Old Team 作为控制组。

E3 的成功标准不是“开更多 agent”，而是能说明何种 bounded topology 在何种 task profile 下有可解释收益。

### E4 — Architecture Selection

实现 deterministic ArchitecturePolicy，先不引入 learned classifier：

```text
TaskProfile
├─ required evidence domains
├─ independent subtasks
├─ source families
├─ estimated serial depth
└─ corpus capability
          ↓
ArchitecturePolicy
├─ simple / single-source       → Single or Workflow
├─ multi-source independent     → Parallel Team
├─ high-dependency sequential   → Single
└─ OOD / insufficient           → Abstain
```

必须做三种公平性比较：

1. Native Budget：各 architecture 使用自身合理预算；
2. Cost-Matched：统一 provider calls、token ceiling 或可解释 cost envelope；
3. Deadline-Matched：统一 wall-clock deadline，比较并行是否真正带来 speedup。

primary quality metric 应优先使用 `EvidenceGroupCoverage`，并同时报告 source-family coverage、final citation coverage、unsupported claim、contradiction resolution、latency、tokens、provider calls、redundancy 和 unique worker contribution。不要只报告 route accuracy 或一个总冠军分数。

### E5 — Self-Evolving Harness

只有在 E0–E4 有稳定失败 taxonomy 和可审计 trace 后，才研究 harness 自我改进。范围应先限于：

- 从失败轨迹提出候选 routing、budget、delegation 或 context policy 变更；
- 离线 replay、counterfactual control 和 held-out regression 验证；
- 人工批准后才能更新配置；
- 禁止运行时自改 permission、sandbox、safety gate、citation authority 或 memory policy。

E5 不是直接启动 RL，也不是把模型输出当成新 policy。任何 learned policy 都必须有版本、训练数据 hash、held-out split、rollback 和 replay identity。

### E6 — Final Research / Resume Freeze

最终产物应是一个可面试、可复现、可诚实解释的 research freeze：

- Core Harness M0–M10.1 frozen baseline；
- external benchmark manifest、adapter、baseline 和 limitation；
- L0/L1/L2/L3 architecture comparison；
- task-profile → architecture routing law；
- quality/cost/latency Pareto analysis；
- positive、negative 和 inconclusive findings 分开；
- final commit、dataset hash、component manifest、metric definition 和 replay artifact 全部绑定。

推荐的结论形式是：

```text
Simple Direct       → Single 更便宜/更稳定
Breadth-First       → Team 可能有 coverage 优势
Cross-Source        → Team 假设重点验证区
OOD / Insufficient  → 两者都应 fail closed
```

这只是实验设计目标，不是预先写死的结果。任何 Team 优势都必须同时通过 contribution、cost/deadline control、failure analysis 和 held-out evaluation 支撑。

## 预注册与禁止事项

- 不改写 M8 负结果；它是 homogeneous/sequential Team 的第一轮 negative experiment。
- 不针对少量 case 调 prompt 直到 Team 获胜；新 benchmark 至少应包含 Single-friendly controls、breadth、cross-source、conflict/temporal evidence 和 OOD/insufficient cases，目标规模为 48–60 条，最终以 E0 review 为准。
- 不把 HealthBench/MIRAGE/NFCorpus 的不同指标直接合成一个总分。
- 不把 external judge、retrieval hit rate、route accuracy、EvidenceGroupCoverage 和临床质量混为同一 ground truth。
- 不为了研究 Team 绕过 safety、permission、sandbox、citation 或 replay boundary。
- 在 E0 的数据、gold、成本和公平性方案通过前，不实现 E2/E3 的实验代码。
