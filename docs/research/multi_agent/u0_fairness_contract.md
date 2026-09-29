# U0 Single vs Team 公平性合同

Capability namespace：MEMORY_READ、EXTERNAL_RETRIEVAL、TOOL_USE、TEAM_ORCHESTRATION。

## 硬约束

Capability(Single) == Union(Capability(Team workers))

对同一 episode，Single 与 Team 必须使用相同的：

- patient/profile/timeline/exam observable state；
- MEMORY_READ 的 subject scope、历史时间窗、查询 API、读预算；
- EXTERNAL_RETRIEVAL corpus snapshot、document IDs、版本/辖区/日期、索引与 evidence 总量；
- tools/API endpoint、参数约束、allowlist、总调用/成本上限；
- model family/checkpoint/API revision；不同模型只能作为独立 model ablation；
- query、system policy、answerability、gold/evaluator 与成功标准；
- 总 token、总调用、总成本、wall-clock deadline；
- evaluator-only 信息隔离。

Single 可读的来源/工具集合必须等于 Team workers 权限并集。分区之后，Team union 只能与 Single 同能力，不能多拿 corpus、API、图像或更长历史。

## 允许改变

固定总资源后，允许作为 treatment 改变 orchestration、parallelism、执行顺序、worker context partition、specialty contract、消息路由、委派/重试/聚合/停止策略。

## 禁止改变

不允许仅给 Team 增加知识、工具、数据、配额、context、模型能力、gold 或评分准则。若有一项发生变化，结果只能报告为完整系统配置比较，不能称为 orchestration gain。

## Harness 检查

每 episode 生成 canonical capability manifest hash。Single manifest 与 Team worker-union manifest 不一致即拒绝运行。记录 source/corpus revision、model/prompt hash、tool allowlist、预算、memory/external retrieval/tool access log、Team delegation、evaluator version、patient snapshot hash。

AgentClinic 的 Patient、Measurement、Moderator 是 environment actors，不计作 HC Team workers；Moderator gold diagnosis 仅 evaluator 可见。MedMemoryBench 的 BM25/Embedding history retrieval 是 MEMORY_READ，不是 EXTERNAL_RETRIEVAL。ESL profile/event/device/exam 分区后，Single 仍须通过等价接口获得全部来源。

违反时 episode 标 FAIRNESS_INVALID，不纳入因果 outcome，也不用于 SFT/GRPO/OPD。RAG nDCG、Memory F1、Team EvidenceCoverage 不可线性合并成 OverallScore。
