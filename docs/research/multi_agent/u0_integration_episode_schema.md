# U0 Integration Episode Schema Proposal

本 schema 是只读 adapter contract 提案，不接 production pipeline；不实现 router、生成、Team execution、RL environment。无官方共同 split 的来源填 SPLIT_POLICY_UNRESOLVED，不自行赋值 train/test。

## Runtime Episode

    @dataclass(frozen=True)
    class IntegrationEpisode:
        episode_id: str
        benchmark_source: str
        split: str
        subject_id: str | None
        query: str
        observable_state_ref: str
        longitudinal_history_ref: str | None
        external_evidence_space_ref: str | None
        tool_surface_ref: str | None
        answerability: str
        evaluator_id: str
        cost_profile_id: str
        metadata: Mapping[str, Any]

episode_id 和 refs 指向被冻结的 source/version/hash。observable_state_ref 是当前可见状态；longitudinal_history_ref 只能通过受控 MEMORY_READ 访问，不代表全历史注入。external_evidence_space_ref 独立于患者 state；无语料时是 null。tool_surface_ref 无环境时为 null。runtime view 不得携带答案、answer key、moderator private context、hidden label 或未来事实。

## Evaluator-only Gold

    @dataclass(frozen=True)
    class IntegrationGold:
        episode_id: str
        expected_answer: object | None
        required_evidence_refs: tuple[str, ...]
        temporal_dependency: bool
        external_knowledge_dependency: bool
        historical_dependency: bool
        independent_subtask_count: int | None
        task_family: str

IntegrationGold 只能放 evaluator namespace，用于 evaluator、counterfactual oracle、teacher privileged context 和 failure attribution。绝不可进入 runtime/student context、retrieval index 或普通 transcript。teacher 权限必须显式标记且 student projector fail closed。

## Future execution records

    ExecutionAction =
        NONE
        MEMORY_READ(query, time_scope, budget)
        EXTERNAL_RETRIEVAL(query, corpus_snapshot, budget)
        TOOL_USE(tool_id, arguments, budget)
        DELEGATE(worker_id, scoped_capability, budget)
        AGGREGATE(worker_outputs, policy)

    ExecutionTrajectory:
        episode_id
        capability_manifest_hash
        ordered_actions
        observed_state_transitions
        final_outcome
        cost_breakdown
        latency_ms
        failure_categories
        evaluator_version
        sibling_group_id

Personal/Episodic Retrieval != External Knowledge Retrieval。两者使用不同 namespace、tool identifier、access log 与 cost bucket。Sibling executions 为同一 episode 不同 capability arm；其他 sibling outcome 只进入 oracle/privileged teacher。

answerability proposal：ANSWERABLE_FROM_STATE、ANSWERABLE_FROM_MEMORY、REQUIRES_EXTERNAL_EVIDENCE、REQUIRES_INTERACTION、INSUFFICIENT_EVIDENCE、OUT_OF_SCOPE。

failure category proposal：MISSING_MEMORY、WRONG_TEMPORAL_SCOPE、MISSING_EXTERNAL_EVIDENCE、RETRIEVAL_MISS、SOURCE_VERSION_MISMATCH、TOOL_FAILURE、DELEGATION_FAILURE、AGGREGATION_FAILURE、UNSUPPORTED_CLAIM、ABSTENTION_ERROR、BUDGET_EXHAUSTED、FAIRNESS_INVALID、EVALUATOR_ERROR。

成本记录 input/output tokens、API/tool calls、retrieval calls、worker count/parallel wall-time、container cost。不要合并 subsystem nDCG、Memory F1、EvidenceCoverage 成 OverallScore。

## Adapter smoke boundary

只读 loader 验证 manifest/schema 与 required keys，不改原始 benchmark 文件。smoke 只记录 schema/result metadata，不输出 raw query、gold 或 row content。本轮未新增生产 adapter 代码。
