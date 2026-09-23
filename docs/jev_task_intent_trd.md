# TRD 草案：Jev 证据任务意图识别与自适应路由

状态：已按本 TRD 加入 research-only 实验臂。未调用 live Jev API；测试尚未新增或运行。

## 1. 目标

在复杂 RAG 检索前，识别问题要求的**证据任务结构**，再由确定性 Harness policy 选择 Single 或 Heterogeneous Parallel Team，并选择可能需要的来源 Worker。

这里的“意图”指检索与证据工作方式，例如多主题覆盖、跨机构比较、时效更新、冲突梳理、串行依赖。它不替用户做临床分诊，也不判断医学陈述或来源哪个正确。

## 2. 当前代码边界

- `src/health_ai_copilot/routing/jev.py` 已封装 TypeSafe typed API 调用。
- `src/health_ai_copilot/routing/architecture.py` 的 `JevArchitectureRouter` 当前一次请求判断 Single/Team，并给 WHO、CDC、NHC、Literature Worker 输出概率。
- `tools/run_resume_multiagent.py` 提供研究用 `jev-routed` runner，路由失败时走全量 Team fallback。
- `runtime/context_manager.py` 中的 Jev priority hints 是另一项独立研究功能，本方案不改它。
- 产品 pipeline 的确定性 `route_question()`、EvidencePolicy、verifier 和默认产品路径均不纳入本次改动。

## 3. 方案概要

新增一条版本化研究实验路径 `jev-intent-routed`。它与现有 `jev-routed` 并列：

| 实验臂 | Jev 输出 | 执行策略 |
|---|---|---|
| `jev-routed` | 当前 architecture + Worker 概率 | 现有策略，作为对照保留 |
| `jev-intent-routed` | task intent + evidence-structure 概率 + Worker 概率 | Harness 按版本化阈值策略选架构和 Worker |

新实验臂只向 Jev 发起**一次** `POST /v1/systemone` 请求。它仅接收公开或合成 benchmark 的问题文本，不读取 gold labels、任务标签、Evidence 或真实用户记录。

## 4. Jev typed 输出

同一请求包含以下 named questions：

### 4.1 主要任务意图

增加一个 `Choice` 问题 `primary_task_intent`，选项如下：

- `direct_lookup`：查找一个直接事实或建议
- `general_explanation`：解释单一概念或主题
- `multi_topic_synthesis`：综合多个主题或子问题
- `cross_authority_comparison`：比较不同机构或辖区的建议
- `current_guideline_lookup`：问题要求当前或近期指南
- `conflicting_guidance_review`：用户要求解释或梳理不同来源的差异
- `serial_follow_up`：问题有明确的先后依赖
- `other`：以上都不适用或无法判断

`primary_task_intent` 用于逐例审计和分 slice 评估。因为不同任务特征可以同时出现，最终 policy 不从这个单选标签直接推断全部需求。

### 4.2 独立证据结构信号

增加以下 `Noul` 问题，读取 yes 概率而不是只读取最高选项：

- `needs_multi_topic_coverage`
- `needs_independent_sources`
- `requires_cross_source_comparison`
- `needs_current_guidance`
- `requires_conflict_review`
- `requires_serial_dependency`

冲突信号只表示“问题要求检查或解释来源差异”。它不代表 Jev 已证实来源真的冲突。

### 4.3 现有路由输出

保留现有 `architecture` 和四个 Worker `Noul` 问题，使新实验可以沿用概率解析、fallback 和执行接口。所有问题继续基于原始问题文本；不把 benchmark 的 `task_family`、`required_evidence_groups`、参考答案或人工标签送给 Jev。

## 5. Harness deterministic policy

实现一个具版本号的 `task-intent-policy-v1`，阈值初值沿用当前 0.5，并写入每次运行的 config identity。

1. 当现有 `P(PARALLEL_TEAM) >= 0.5` 时运行 Team。
2. 当 `P(needs_independent_sources)`、`P(requires_cross_source_comparison)` 或 `P(requires_conflict_review)` 任一达到 0.5 时，Team 是最低架构。
3. 单独的 `needs_multi_topic_coverage`、`needs_current_guidance` 或 `requires_serial_dependency` 不强制并行 Team：多个主题可能来自同一来源，时效查询也可能只需一个来源；串行依赖不适合被误解为并行工作。这些信号写入结果供分析，之后可设计专门的多主题、时效检索或串行执行实验。
4. Worker 先取概率不低于 0.5 的来源角色。Team 路由若没有角色越过阈值，则选择概率最高的 Worker。若 `needs_independent_sources`、`requires_cross_source_comparison` 或 `requires_conflict_review` 越过阈值，而入选 Worker 少于两个，则按概率从高到低补足至两个。
5. Jev 请求失败、输出缺字段、类型错误或概率越界时，保守回退至现有全量 Team，并保存 fallback reason；不静默降为 Single。

最终架构、入选 Worker、触发路由的 policy reason codes、所有原始概率都写入 case artifact。这样可区分 Jev 的判断和 Python 的执行规则。

## 6. 实现文件

| 文件 | 改动 |
|---|---|
| `src/health_ai_copilot/routing/task_intent.py` | 意图枚举、typed decision、Jev 问题 schema 与答案解析校验。 |
| `src/health_ai_copilot/routing/architecture.py` | 新增 `JevTaskIntentRouter`；复用 `JevClient`，单次请求组合意图、证据结构和 Worker 问题；保留现有 `JevArchitectureRouter` 行为。 |
| `src/health_ai_copilot/routing/__init__.py` | 导出新 Router 和 decision 类型。 |
| `tools/run_resume_multiagent.py` | 新增 `--architecture jev-intent-routed`；调用新 Router，执行现有 `run_single` / `run_team`，并记录 task intent 与 policy 结果。 |
| `docs/jev_experimental_adapters.md` | 新实验臂的使用方式、输出字段与数据范围。 |
| `tests/test_jev_task_intent.py` | TRD 建议的离线单元测试，尚未创建。 |
| `tests/test_resume_jev_intent.py` | TRD 建议的 runner 集成测试，尚未创建。 |

不改 `pipeline.py`、`safety.py`、Medical verifier、M8 Team runtime、ContextManager 默认配置、MIRAGE 数据、冻结 benchmark 原文件或现有 `jev-routed` 对照语义。

## 7. 数据边界

- CLI 必须显式提供 `--jev-data-classification public|synthetic`。
- state 只包含 benchmark 原始 question；不包含姓名、病历、会话历史、Evidence、gold label、答案或检索卡内容。
- 产品调用链不会启用这个 Router。紧急/用药规则和后续医疗判断不交给 Jev。
- `primary_task_intent` 和概率只能用于任务编排、实验分析；不能作为医学证据或引用来源。

## 8. 评估方案

在相同的已冻结 case、相同 answer provider、相同预算模式下，对比：

1. `always-single`
2. `always-team`
3. 现有 `jev-routed`
4. 新增 `jev-intent-routed`

保留现有 `EvidenceGroupCoverage`、All Required Groups Covered、Citation Integrity、unsafe OOD rate、provider calls、tokens、P50/P95 latency；另报告 intent facet 的逐项 precision/recall、架构路由率、worker-source 选择情况及按 task family 的 slice。报告 Jev 自身额外调用和 token 成本。

当前 `research_architecture_v2_candidate` 仍是 review candidate：只能做标注清楚的 DEV 探索。TEST 运行必须等待该 benchmark 按现有规则冻结并获批，不能用候选标签宣称意图识别准确率。

## 9. 后续验收标准

- 同一 case 的新增 Router 只发送一次 Jev 请求，且请求中不含 benchmark gold/标签。
- 所有输出由命名 typed answer 解析；缺字段、wrong type 或非法概率会进入保守 fallback。
- 固定 Jev 响应重复运行时，阈值 policy 产生相同架构、角色和 reason codes。
- 多来源/跨来源信号达到阈值时不会路由到 Single；跨来源信号达到阈值时至少有两个不同角色，除非 fallback 直接选择全队。
- `jev-routed`、`single`、`team`、默认产品路径、safety gate、ContextManager 未受新 policy 影响。
- 每条已完成或失败的 case artifact 都记录 Jev 调用、tokens、意图概率、最终路由和 fallback 状态。

## 10. 实施顺序与剩余验证

1. task-intent typed schema、`JevTaskIntentRouter` 和 deterministic policy 已实现。
2. Resume runner 已加入 `jev-intent-routed`、逐例输出、路由失败计数和 intent summary。
3. 后续按批准的测试范围补充离线 parser/policy/runner 测试。
4. 之后在 DEV 合成/公开 cases 做 smoke；TEST 仍须等待 benchmark 正式冻结与批准。
