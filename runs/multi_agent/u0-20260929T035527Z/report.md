# U0 Run Report

Run ID：20260929T035527Z
Base commit：9e023dbddc1f8025e3608bfae4c5390c1a7957ef；结束时 HEAD：5f666b241fefc0dbcd7524bd4ea6c85bbeaa6bb9（并行 Memory audit commit，不属于 U0）
分支：mem-b0-benchmark-routing-20260929
本轮新增 U0 文件尚未提交；审计期间共享分支由独立 Memory 工作推进到 5f666b2（只涉及 docs/research/memory 与 runs/memory），未被本轮修改；原有无关未跟踪文件保留。

## 结论

Memory external：MedMemoryBench（有条件采用）。
MAS external primary：MedAgentBoard（有条件采用；license 与公平适配先行）。
MAS secondary：MedMASLab（REFERENCE_ONLY）。
Unified backbone：ESL patient state + custom versioned HC environment。
Whole-agent transfer：AgentClinic 与 HealthAgentBench，均不进 E2。
Post-training compatibility：PARTIAL。
推荐下一步：D，先做只读 Unified Environment prototype 设计；不启动 E2-B/L4、正式评测或训练。

## TEST 边界

未打开 ESL evaluation/TEST 行；未读取 MedAgentBoard test 数据；未打开 HealthAgentBench tasks/**/tests/**；未调用 evaluator。唯一 parser smoke 解析 AgentClinic 无官方 split 的公开 JSONL 第 0 行，状态记为 SPLIT_POLICY_UNRESOLVED；没有保存 query、gold、原始行或哈希。Smoke 不是 benchmark reproduction。

## 文件

benchmark_matrix.json 记录六个候选 provenance、license、split、能力、基线、资源和角色。adapter_smoke.json 仅记录 schema parse。decision.json 记录推荐。manifest.json 记录范围和 gates；两份规范矩阵副本位于 docs/research/multi_agent。
