# U0 Decision

## 选择

- Memory external benchmark：MedMemoryBench，CONDITIONAL_ADOPT。
- Multi-Agent external primary：MedAgentBoard，CONDITIONAL_ADOPT；先解决 task data rights 并实现同模型/同能力公平适配。
- MAS secondary/reference：MedMASLab，REFERENCE_ONLY。
- Unified environment backbone：ESL-Bench patient state 加自有、版本化 Health-Copilot integration environment；ESL 不含 native external medical evidence。
- Whole-agent transfer：AgentClinic（交互诊断）和 HealthAgentBench（Harness/terminal）；都不进 E2。
- Post-training source：需要 custom integration environment，不能直接将公开 benchmark gold/test 作为训练集。
- Next recommendation：D. Proceed first to Unified Environment prototype。

## 不进主线

MedMASLab 因 code license/组件 data terms 未解且模型/prompt/topology 混杂，仅作参考。AgentClinic 的多角色主要模拟诊疗环境，不是 team orchestration。HealthAgentBench 不是 native MAS/memory，且本地有公开 gold-file path 差异。R2MED/MIRAGE 只作 RAG subsystem characterization，不是 longitudinal backbone 或现成 corpus。

## 统一能力约束

纵向 patient state：版本化 ESL profile/timeline/exam。外部医学证据：必须是独立许可和版本均冻结的 public-health/guideline/literature namespace；本轮没有采集。MEMORY_READ 访问 patient/session history，EXTERNAL_RETRIEVAL 查询独立 external medical knowledge。Single 的 source/tool/corpus 总权限必须等于 Team workers 并集，且模型、query、预算与 evaluator 完全相同。

原生/适配/不可用 arms：见 u0_counterfactual_arm_matrix.json。ESL 支持 NONE；MEMORY、TEAM、MEMORY+TEAM 可适配；RAG 与涉及 RAG 的组合当前 NOT_AVAILABLE。MedMemory 官方 BM25/Embedding history retrieval 归 MEMORY_READ。

## OOD 和 post-training

ESL manifest 所见六批 persona ID 无重叠，但这不排除模板、事实、预训练污染。未来 proposal：IID patient-disjoint；OOD_PATIENT、OOD_TASK_FAMILY、OOD_TEMPORAL；OOD_SOURCE_FAMILY 需冻结 source groups；OOD_COMPOSITION 需自建 sibling-arm integration protocol。本轮没有冻结 split。

POST_TRAIN_COMPATIBILITY=PARTIAL：SFT 缺成功 Harness action trace；GRPO 缺同 episode G actions 的真实 Harness outcome；OPD 缺 privileged teacher/student context isolation 和 trajectories。还需成本账本、失败 taxonomy、锁定 split 和外部语料许可。

## 严格先后（只推荐，不启动）

1. 做 schema/metadata 级只读 Unified Environment prototype 与 provenance/split manifest。
2. 在新建 locked HC split 上检查 episode isolation/replay。
3. 引用现有已 eligible PUBLIC_HEALTH snapshot，落实 Single-Team capability equality。
4. 再单独决策是否处理 GUIDELINE/LITERATURE eligibility；在此之前不启动 E2-B/L4。

若 split/license/evidence manifest 无法解决，下一阶段改为 E（insufficient evidence + 具体 blocker）。
