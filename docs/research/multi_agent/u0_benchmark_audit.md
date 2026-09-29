# U0 公开医疗 Benchmark 可行性审计

审计日：2026-09-29。起始 HEAD：9e023dbddc1f8025e3608bfae4c5390c1a7957ef；结束观察 HEAD：5f666b241fefc0dbcd7524bd4ea6c85bbeaa6bb9（审计期间共享分支上的独立 Memory audit commit）。本轮 U0 文档与 run bundle 未提交，且没有修改该 Memory commit。正式 benchmark execution、LLM evaluation、训练、prompt/retriever 调参均未执行。

## 结论摘要

- Memory external：有条件采用 MedMemoryBench。纵向 persona/session stream、query timing 与多类记忆问题匹配医疗记忆迁移；但数据许可声明冲突，且 eval 不是正式 train/dev/test contract，先解决许可并冻结 HC split。
- MAS external primary：MedAgentBoard。它明确比较 single LLM、MAS 与传统系统，但模型、prompt、knowledge/tools、topology 同时变化，不是 orchestration-only 因果实验；数据许可/资产还需核清。只作为外部迁移验证。
- MAS secondary：MedMASLab 作为方法/架构参考，不承诺执行。
- Unified backbone：ESL-Bench 是最好的 longitudinal patient-state 候选，不是完整 unified environment。具备 profile/timeline/event/device/exam 与 KG ground truth；无独立外部医学知识或工具环境。公开评测批次不能直接当训练数据。
- Whole-agent transfer：AgentClinic 面向交互诊断；HealthAgentBench 面向长期 Harness/terminal tasks。均不进 E2。
- Post-training：PARTIAL。没有现成的 HC state/action/trajectory/cost/failure/counterfactual sibling 全链路。ESL 仅可支撑经许可、隔离 split 后构建自有环境。
- 唯一下一阶段建议：D，先做 Unified Environment prototype 设计/只读 adapter；不启动 E2-B 或 L4。

详细 provenance、license、split、baselines 和状态见同目录的 compatibility matrix；counterfactual arm 状态见 counterfactual arm matrix。

## ESL-Bench：纵向状态骨架

本地 manifest v16，更新时间 2026-09-07，latest 指向 202608/sample320-20260830.jsonl；canonical HF repo 按任务指定为 mirobody/ESL-Bench，审计记录 revision 5626c5bd939e70706a739502baf3291fcf48c568。本地镜像没有 Git commit。六个 batch 共 1,274 个文件、4,787,079,805 bytes；只检查 manifest、目录、文件名与文件大小，不打开 benchmark JSON/JSONL 行。用户/persona ID 在六批 manifest 中无交叉，但这不证明模板和生成事实独立。

README 的 schema 描述：profile 有 demographics 与 health profile；timeline 按时间记录 event、wearable/device measurement、exam indicator；exam_data 另有 clinical records；KG evaluation query 引用 patient event-indicator-time relationships。可形成 PersistentPatientState，多个源类型有真实语义，适合切 worker domain。此处的 KG/timeline 都是 patient state，不是外部知识。

MEMORY_READ 可通过受限 read API 实现，不能把整份历史塞入 context。EXTERNAL_RETRIEVAL_NATIVE=NO。未来挂独立 medical corpus 时，必须记录源 namespace、jurisdiction、version/date、license、document/evidence IDs。README 描述数值/exact/list 类可程序化评分，explanation 类有额外 rubric/LLM judging。U0 未运行 evaluator。

Split proposal：未来 HC 按 persona/subject 分组，先 IID subject-disjoint，再提议 OOD_PATIENT、OOD_TASK_FAMILY、OOD_TEMPORAL、OOD_SOURCE_FAMILY 与 OOD_COMPOSITION；当前六批都是已公开 evaluation 来源，不在本轮冻结任何新 split。

## MedMemoryBench：医疗域 Memory 转移

本地 repo commit 7227bc105b84a1a9f7a75861eb9e1be3ea502882，clean；代码 Apache-2.0。GitHub README 声明 data CC BY 4.0，HF dataset card 声明 CC BY-NC-SA 4.0。未解决前不得把许可证等同，也不将其用于派生训练。

配置指向每 persona 的 eval dialogue/query/noise 文件；evaluator 按顺序注入 sessions，默认每 10 sessions 做 query checkpoint。任务覆盖实体回忆、时间定位、状态更新、选择、推理、多跳临床推断。计分混合 string/option matching 与 LLM judge。没有共用的官方 train/dev/test split；eval 目录名称不能擅自改叫 test，也不表示可用于 training。

官方 BM25/Embedding RAG 的 memorize() 把 session text 切块并建索引，query 再检索同 persona/session history。因此必须记为 MEMORY_READ。Personal/Episodic Retrieval != External Knowledge Retrieval。它不提供 external guideline/literature index。

裁决：CONDITIONAL_ADOPT，仅作 Memory external validation，先 resolution license 与 HC-held-out split。

## MedAgentBoard：MAS external primary candidate

本地 commit b09dcbb11cb2c908e23b83b8b71c566be41326c5，clean。README 指向 Zenodo 22299495；本地只有 MedQA、lay-summary、EHR 代码，不含数据；clinical workflow 代码在独立仓库。根目录无 LICENSE。MIMIC 相关数据/结果需要 PhysioNet authorization；Zenodo 具体资产 revision/license/大小没有本地验证。

覆盖医疗 QA（含视觉）、lay summary、EHR prediction、clinical workflow。评分是 task-specific，包括结构预测指标、ROUGE/SARI 和 LLM judge。代码引用 test 命名路径；只审源码和文件名，没有读数据行。它比较的系统除 orchestration 外还改变 model、prompt、knowledge/tools、topology，因此不能把原 paper rank 解释成 orchestration effect。

可公平的窄适配：先选固定文本 task、同一 model family/checkpoint、question/prompt、知识和工具总集合、budget 与 evaluator；Team workers union 必须等于 Single 的全 capability。L1/L3/L4 可写 task adapter，但当前不是可直接接入的公平比较。外部验证优先，不把公开结果数据作为 training。

## MedMASLab：方法参考

本地 commit 709eccbe019e6692fc359f0581374c0ee62ae004，clean；本地无 dataset payload，没找到 code LICENSE。论文/框架在 11 个医疗 datasets、多模态设置下比较 MAS；HF dataset MIT 元数据只对应该 HF repo，不覆盖底层组件数据 rights。论文规模 VLM/judge 计算成本高。没有纵向 patient state、标准 external RAG 或工具 environment。最好价值是架构/方法 catalog 与评测方法参考，不是 unified benchmark 或 orchestration-only ablation。

裁决：REFERENCE_ONLY。

## AgentClinic：交互诊断 whole-agent transfer

本地 commit b6570edefb940857a7c334350656b29f9d984f24，代码 MIT。四个本地 JSONL 共 847,130 bytes；MIMIC-IV 需 PhysioNet，case-source rights 不由代码 license 覆盖。无官方 split，状态 SPLIT_POLICY_UNRESOLVED。

Doctor 与 Patient、Measurement、Moderator 构成交互临床模拟。multi-agent 主要是 environment actor，不是分布式 doctor team。Moderator 持有 correct diagnosis，必须 evaluator-only。不是 E2 主 benchmark，但以后可让 Doctor backend 适配 Single/Team，作为 whole Harness transfer。

本轮只用一条无官方 split 的公开 JSONL 做 JSON/schema parse smoke，不记录 query、gold、原始行或 row hash，也不调用模拟角色/evaluator。结果见 run bundle；不是 benchmark reproduction。

## HealthAgentBench：Harness/terminal transfer

本地 commit bcbb8085fd549469e2dc7455f4bfd68a1b98895a，代码 MIT。README 列 54 Harbor terminal task、七类任务，完整数据准备至少 30 GB。依赖 Docker/container + Harbor 0.8.0；MIMIC/PhysioNet、EHRSHOT/Redivis、CT-RATE/OpenRAIL 等各有数据条款。部分 X-ray verifier 用 GPT-5.4/API。Harbor cost/time 可记录。它没有 native MAS comparison，也没有统一跨 session memory。

本地 tracked file names 中 tasks/**/tests/** 有 gold.txt 和 labels.csv；README 声称运行后下载 labels，二者存在需单独核查的差异。本轮未打开 tests 内容，不能开始 HealthAgentBench execution。它适合作为未来 Harness external validation，不进 E2。

## Unified Environment、Fairness 与 arms

纵向 patient state 取自冻结版本的 ESL synthetic profile/timeline/exam。外部医学证据须单独作为许可审查、版本化 namespace；E2-A 只有 PUBLIC_HEALTH eligible，GUIDELINE/LITERATURE 仍 NOT_YET_ELIGIBLE。MEMORY_READ 读个人/会话纵向 state；EXTERNAL_RETRIEVAL 读 independent external medical knowledge。禁止把患者 history retrieval 叫 external RAG。

Single vs Team 必须满足 Capability(Single) == Union(Capability(Team workers))：同一 patient state、memory scope、external corpus snapshot、tool/API 集合、model family、query、gold/evaluator、总 token/API/tool/cost budget 和 deadline。允许改变 orchestration、parallelism、worker context partition/contract、通信方式。违规不能解释为 orchestration gain。细节见 fairness contract。

ESL 可做 NONE native、MEMORY/TEAM/MEMORY+TEAM adaptable；所有需 external RAG 的 arm 当前 NOT_AVAILABLE。MedMemory 的两个官方 history-RAG baseline 归 MEMORY，不提供 RAG arm。其他 benchmark 的八臂 feasibility 逐项记录在 counterfactual matrix。不同 subsystem metric 不得线性相加成 OverallScore。

## Post-training 与 OOD

SFT：现有数据有 query/gold，却缺成功 Harness action、near-minimal execution trace 和统一成本/失败标签。
GRPO：缺同一 episode 上多 policy/action 经真实 Harness 执行得到的 outcome reward。
OPD：可以把 gold evidence、完整患者 state、counterfactual siblings、successful trajectories 放入 privileged teacher context，但需强隔离，student 不可读。
结论：POST_TRAIN_COMPATIBILITY=PARTIAL，不是可直接训练的数据集。

OOD proposal：IID subject-disjoint；OOD_PATIENT、OOD_TASK_FAMILY、OOD_TEMPORAL 可围绕 ESL split 候选设计。OOD_SOURCE_FAMILY 需要来源版本与分组；OOD_COMPOSITION 要用自有集成环境把 MEMORY/RAG/TEAM 组合成 sibling arms 后 hold out。当前外部 datasets 不原生提供可验证的 composition split；本轮未冻结 split、未构造 gold。

## Source links

[ESL-Bench dataset](https://huggingface.co/datasets/mirobody/ESL-Bench) · [paper](https://arxiv.org/abs/2604.02834)
[MedMemoryBench repo](https://github.com/AQ-MedAI/MedMemoryBench) · [dataset card](https://huggingface.co/datasets/Cyan27/MedMemoryBench) · [paper](https://arxiv.org/abs/2605.11814)
[MedAgentBoard repo](https://github.com/yhzhu99/MedAgentBoard) · [Zenodo data/results](https://zenodo.org/records/22299495) · [paper](https://arxiv.org/abs/2505.12371)
[MedMASLab repo](https://github.com/NUS-Project/MedMASLab) · [paper](https://arxiv.org/abs/2603.09909)
[AgentClinic repo](https://github.com/SamuelSchmidgall/AgentClinic) · [paper](https://arxiv.org/abs/2405.07960)
[HealthAgentBench repo](https://github.com/microsoft/HealthAgentBench) · [paper](https://arxiv.org/abs/2606.31179)

本地 ScientificData2021_HealthGym 是旧项目，不等同于 2026 Healthcare AI GYM。本轮未评估后者。
