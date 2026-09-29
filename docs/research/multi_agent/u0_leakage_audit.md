# U0 Leakage Audit

范围：公开文档、README、source/config、manifest、路径名与文件大小。未运行 evaluator；不声称可以证明 foundation models 没见过数据。

| 候选 | 项目级泄漏通道 | 控制与处置 |
|---|---|---|
| ESL-Bench | 公开 evaluation/test 批次；退役批次可放出答案；共享生成模板、KG/query family、相同事实族的风险。六批 manifest subject IDs 无交叉仅是有利迹象 | HC future split 按 subject/persona 互斥；public eval/released answers 记为 exposed，不进 train/heldout |
| MedMemoryBench | persona 间划分未定义；同 persona sessions 必然相连；query paraphrase/noise/trap 与 gold 可公开 | eval path 不等于正式 split；不使用其公开 gold 做 post-training；先按 persona/time 设计隔离 |
| MedAgentBoard | 多 public sources/prompts/results；源码指 test file names；EHR subject 或问法重复未知；Zenodo release | 获取获许可资产后做 task/source/patient 去重和 split 审计；public labels 不作训练奖励 |
| MedMASLab | 11 个组件来源可交叉，问法/患者/答案事实可能跨库重复；public methods/results | 仅 reference；逐组件 source/split/hash audit 后才可选 task |
| AgentClinic | 无官方 split；base/extended sets 可能重叠；NEJM/MedQA/MIMIC 来源重复风险；Moderator 可见 gold | 本轮 smoke 不保存内容/gold；以后按 source case 去重并隔离 Moderator；不进 E2 |
| HealthAgentBench | 54 task public；本地 tracked tests paths 有 gold.txt / labels.csv，与 README “运行后获取 labels”声明有差异；source datasets可能与训练语料重合 | 本轮只读路径名，不打开 tasks/**/tests/**；先核实 Harbor label boundary；禁止训练/执行这些 tests |

## Cross-split checks proposal

记录 benchmark/source revision、subject/patient id、split、query family/template id、source document/corpus revision。优先 patient-disjoint，再检查 scenario template、paraphrase、answer fact、external evidence group/document overlap。Counterfactual sibling episodes 必须整体进入同一个 split，不能把相同 patient/query 的不同 action arms 拆进 train 与 test。

Evaluator/IntegrationGold/Moderator/released answers 必须与 runtime/student namespaces 分离。Student 不读取 gold、future temporal facts、counterfactual siblings 或 task labels；teacher privileged context 必须显式标权。

项目级 controls 无法回答 foundation-model pretraining contamination unknown。

## U0 TEST 访问声明

未打开 ESL JSON/JSONL evaluation 行/答案；未读取 MedAgentBoard test 数据；未打开 HealthAgentBench tasks/**/tests/**/gold.txt 或 labels.csv；没有 evaluator calls。MedMemoryBench 未采样。AgentClinic parser smoke 来自无官方 split 的公开 JSONL，标记 SPLIT_POLICY_UNRESOLVED，不能被描述为官方 train/dev/test 或 external validation。runs bundle 未包含原始 benchmark 行、question、gold 或测试文件内容。
