# Health-Copilot Agent Notes

本文档沉淀在 M10/M10.1 开发、debug、验证和交付过程中反复证明有价值的过程性经验。它不是产品需求替代品；遇到更具体的模块文档或测试契约时，以代码与测试为准。

## 开发前先确认的边界

- 先查看 `git status --short`。这个仓库经常带有历史评测产物、pytest 临时目录和其他未提交文件；除非明确属于本次任务，否则不要删除、重置、覆盖或暂存它们。
- 变更前确认当前 profile、组件 manifest 和已有冻结 hash。M3、M8、M9 的 profile hash 是兼容性证据，不应因为 M10 改动而漂移。
- M10 是 opt-in harness substrate，产品默认仍是 memory-off 的 M3 profile。不要把实验性的 session/memory/context 行为偷偷接入默认路径。
- 不要把本地 SQLite 状态当成 EHR、临床纵向记录或生产级多租户存储；测试只使用合成/公开 fixture。

## M10 的核心设计不变量

### 状态平面必须分开

`RunContext`、单次运行的 `AgentSession`、跨运行的 `Persistent Session`、一次模型调用的 `ContextPlan`/provider context，以及 `MemoryStore` 是不同状态平面。常见错误是把它们互相当作 transcript 或 memory：

- `AgentSession` 只保存当前 bounded execution 的 typed messages。
- `SessionStore` 只保存可重放的交互事件，不保存 hidden reasoning。
- `MemoryStore` 只保存经过显式 policy 校验的 typed state；assistant output、tool output、MCP output、retrieved web text 不能自动成为 memory。
- `ContextManager` 只负责选择，不调用 provider、tool 或 memory write。
- `ContextProjector` 是唯一把 `ContextPlan` 转成实际 provider-visible message tuple 的边界。

### Plan 和 provider request 必须一一对应

每次模型调用前都重新生成 plan，包括工具执行后的第二轮调用。provider 看到的 persistent history、selected memory、current Evidence 和 current tool exchange 必须都能在 plan 中找到；plan 选中的 provider-visible 项也必须实际投影出去。不能只记录一个“看起来正确”的 plan，却继续把未裁剪的 `AgentSession.messages` 传给模型。

正确顺序是：

1. 对原始 current question 做 safety gate。
2. 读取/查询允许的 session 和 memory state。
3. 执行 retrieval，拿到真实 Evidence。
4. 使用真实 Evidence 构建当前 model turn 的 ContextPlan。
5. 通过 ContextProjector 生成 provider messages。
6. provider 返回 tool call 后执行 tool，追加结构化 observation，再为下一轮重新规划。

历史和 memory 必须使用 data-bearing、untrusted message；不能拼接进 system prompt，也不能让它们改变 tool permission、profile、KnowledgeScope、MemoryPolicy 或 safety authority。当前用户、当前 Evidence、system pins 和完整的 tool call/result exchange 是保护项；保护内容放不下时要 fail closed。

### 工具交换必须原子

工具调用和结果必须成对进入 plan、provider context 和持久化 turn。持久化的 tool-call payload 是嵌套的 `tool_calls[{"id": ...}]`，不是顶层 `call_id`；配对校验必须同时理解这两种形状。只看到一个 call 或一个 result 时，应抛出 context atomicity failure，而不是静默丢弃一半。

### Safety 必须早于所有状态副作用

`answer_in_session()` 对 raw current question 的 safety route 必须先于 session resume、memory query、retrieval 和 provider call。这样即使 session/memory 状态损坏，urgent-care 或 human-review 路由仍然可用。普通 M10 state error、projection error、budget error、revision conflict 和 commit error 都要返回明确的 abstain/fail-closed reason。

### Session turn 必须原子提交

一个 turn 应作为单个 revision-checked batch 提交，典型顺序为：`USER_INPUT`、`TOOL_CALL`、`TOOL_RESULT`、`ASSISTANT_OUTPUT`。第二个事件校验失败、SQLite insert 失败或 expected revision 过期时，不能留下前半个 turn。In-memory store 要先 materialize/validate 全部事件再 mutation；SQLite store 要在 transaction 中完成并 rollback。

## 本轮 debug 中最值得复用的故障模式

### 1. Runtime export 引发 circular import

问题链路是 `runtime.__init__` eager export projector，projector 顶层导入 agent messages，agent package 初始化又导入 loop，loop 再导入 projector，最终出现 partially initialized module。

解决方案：

- projector 顶层只保留 `TYPE_CHECKING` 类型导入。
- 运行时确实需要的 message class 放到方法内部 lazy import。
- 修复后必须直接测试 `from health_ai_copilot.runtime import ContextProjector, SessionContextProjector`，不能只跑不触发该 import path 的单测。

### 2. 局部 import 解决循环后被静态检查抓到名称未定义

`UserMessage` 在函数体中局部导入，但返回注解仍被 ruff 判定为未定义。解决方案是同时在 `TYPE_CHECKING` 中声明类型，并保持 `from __future__ import annotations`。经验是：循环依赖修复后立刻跑 ruff，不要等到全量测试才发现静态问题。

### 3. 把 pipeline 状态误读成 RuntimeComponents 状态

`answer_in_session()` 中曾经使用 `self.last_agent_run`，但 `last_agent_run` 属于本次新建的 `HealthCopilotPipeline`，不属于长生命周期的 `RuntimeComponents`。结果是模型调用完成后才抛 `AttributeError`。

解决方案：保留 `active_pipeline` 局部变量，并从 `active_pipeline.last_agent_run` 读取 `context_plans` 和 session messages。类似的 per-run 状态都应该从 execution-local object 读取，不能挂到 shared component graph 上。

### 4. safety fixture 的语言假设错误

fixture 使用英文 “chest pain”，但当前 deterministic safety policy 只匹配中文 marker `胸痛`/`呼吸困难`，导致集成测试错误地把 safety precedence 判成失败。解决方案不是在 fixture 中手写另一套 marker，而是让 fixture 调用真实的 `route_question()`，并使用与当前 policy 契约一致的合成输入。

### 5. Replay state identity 只传了一半

回放 metadata 同时包含 `context_plan_hash` 和 `context_plan_hashes` 时，测试只传了 hash 列表，matching replay 仍然被正确拒绝。解决方案是把 session revision、memory snapshot hash、current/initial plan identity 作为一个完整 state tuple 比较；测试必须同时覆盖 matching 和 changed-state mismatch。

### 6. 持久化 tool-call 的 group id 提取不完整

只从 `payload["call_id"]` 或 `payload["tool_call_id"]` 提取 group id 会漏掉产品实际写入的 `payload["tool_calls"][0]["id"]`。解决方案是统一规范化 call id，再检查 group 内是否同时存在 call 和 result；不能仅用“group 成员数量大于 1”代替配对校验。

### 7. 只验证 plan metadata，不验证真实 provider messages

M10 初期可以生成“正确”的 plan，但 provider 仍然收到旧的 `AgentSession`，这会造成 dropped context leakage、history 未到达 provider、工具第二轮缺 observation 等隐蔽错误。解决方案是增加 provider-capture fake，直接断言 provider request 中的 message IDs/content 与 `ContextPlan` 的 selected IDs 对齐。

### 8. 兼容性字段演进要保留旧调用形状

`SessionAnswer` 从单个 `context_plan` 演进为多个 model-turn plans 后，保留 `.context_plan`/`.initial_context_plan` property，并在 `__post_init__` 兼容旧的四参数构造方式。类似 API 演进应优先增加字段和兼容 property，不要无必要地破坏已有 positional caller。

## 测试与评测经验

- 快速回归：

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q tests\test_m101_context_projection.py tests\test_m10.py tests\test_agent.py tests\test_runtime_replay.py -p no:cacheprovider
  ```

- 全量回归：

  ```powershell
  .\.venv\Scripts\ruff.exe check src tests
  .\.venv\Scripts\python.exe -m pytest -q --basetemp=.tmp-pytest-check -p no:cacheprovider
  ```

- `--basetemp` 可以隔离 pytest 产物；本仓库历史上存在 ACL 受限的 pytest 目录，出现 `Permission denied` warning 时不要把它们当成产品代码失败，也不要递归删除整个 workspace。
- 全量测试失败且报 frozen fixture 没有匹配行时，先确认 Git LFS 文件是否仍是三行 pointer 文本；本轮 `runs/integration/u2f-owned-v1-55955b2eff38/latent_worlds.jsonl` 实际对应 205 MB LFS 对象。用 `git lfs checkout -- <精确路径>` 物化缓存对象，并按 pointer 的 `oid sha256` 校验；不要因此改 U3R/U2F 评测代码或删除历史运行目录。
- M10.1 集成套件必须是离线、确定性、合成数据，并直接捕获 provider messages；live provider smoke 不是离线 suite 的替代品。
- 每个 M10.1 指标单独报告 numerator、denominator、definition version；不要用一个 magic aggregate 掩盖 projection、leakage、safety、atomicity、replay 的差异。
- 保持 `m10-memory-v1` 的 24-case frozen contract 不变。新增 context suite 时不要修改旧 dataset、旧 grader 语义或旧 profile hash。
- 验证 component/profile identity 时，确认 `context-manager-v2` / `m10-context-v2` 已进入 manifest，而旧 M3/M8/M9 hash 仍保持原值。

## 提交与交付经验

- 只用明确文件路径 `git add`，不要用 `git add .`，因为 workspace 中有大量历史 `runs/`、pytest 临时目录和其他用户文件。
- 提交前至少核对 `git diff --cached --check`、`git status --short` 和 staged diff stat。
- 用户明确要求推送时，提交后执行 `git push origin main`，再核对 `git log -2 --oneline --decorate` 与 `git rev-list --left-right --count origin/main...main`；理想结果为 `0 0`。
- 最终报告要区分：已验证的离线证据、未执行的 live/provider smoke，以及工作区中保留的无关未跟踪文件。不要把“测试通过”扩写成临床安全或真实 provider 质量结论。

## 2026-10-05 L40 / vLLM / Ninja 排查记录（暂停点）

### 已确认的环境事实

- Qwen3-8B 本地权重路径为 `/root/gpufree-share/data/Qwen3-8B`。
- 当前复用的后训练环境是 `/root/gpufree-data/Health-Copilot/training/posttrain/.venv`，Python 3.11，vLLM 为开发构建 `0.30.1rc1.dev622+gf03026a54`；不要在实验报告里称为 stable。
- `ninja` 可执行文件已在该环境的 `.venv/bin/ninja`，版本 `1.13.2.git.kitware.jobserver-pipe-1`；但默认 shell 的 `PATH` 不含 `.venv/bin`，所以裸跑 `command -v ninja` 会误报缺失。启动 vLLM 前显式将该目录加入 `PATH`。
- `/usr/local/cuda/bin/nvcc` 存在，CUDA Toolkit 13.0；Python `flashinfer` 包存在，但 `flashinfer_cubin` 不存在。缺少 cubin 时会走其他可用后端或 JIT 路径，不能单凭这个包缺失断定启动失败。
- H0 未提交实现和运行产物原先在临时目录 `/tmp/health-copilot-h0-203b2cdc`。为防止关机清掉 `/tmp`，已原样复制到持久目录 `/root/gpufree-data/Health-Copilot-harness-v1-consolidation-20261004`；明天从该目录继续，分支 `harness-v1-consolidation-20261004`，HEAD `203b2cdcfe3765695f405e4afe6c0704da07d80f`，现有未提交文件和 `runs/common_eval/` checkpoint 均已核对保留。不要从主 checkout 的 `main` 误接着改。

### 故障判别顺序

1. 先查 `nvidia-smi`、服务进程、目标端口和 `PATH`，再决定是否归因显存。此前 GPU 空闲时，vLLM 子进程在受限 sandbox 中报 `Can't initialize NVML`、`cudaGetDeviceCount Error 304` 和 `Failed to infer device type`；同样的模型与参数在可访问 GPU 的执行环境里可以启动。这种错误先按执行环境隔离排查，不要先当作 OOM。
2. 先执行 `command -v ninja`、`.venv/bin/ninja --version` 和 `nvcc --version`。当前 Ninja 本身已安装，关键是启动 vLLM 的进程能从 `PATH` 找到它。
3. 只有看到明确 CUDA OOM 日志并同时记录当时 GPU 占用，才把失败归因显存不足。曾经的 8001 nohup 日志文件是 0 字节，不能拿它们证明 OOM 或 Ninja 编译失败；后台启动要确认日志确实有内容，诊断时优先保留前台完整 stderr/stdout。
4. 不要停止或重配 Memory 会话的 8000 服务。Multi-Agent/H0 的临时 vLLM 诊断使用 loopback 的 8001；每次长跑前重新查看 GPU 占用，避免和 Memory 任务争用。

### 已验证与未验证的边界

- 在完整 GPU 访问环境中，设置 `PATH=/root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin:$PATH` 后，Qwen3-8B 可在 8001 以 BF16 启动；`--gpu-memory-utilization 0.88` 启动记录约有 23.47 GiB KV cache，并成功完成一条 temperature=0 的本地聊天请求。
- 那次诊断启动使用了 `--enforce-eager`。vLLM 明确说明该参数会禁用 `torch.compile`、CUDAGraph 和 JIT kernel warmup，因此它只验证了模型加载、GPU 推理和 OpenAI 兼容接口，**没有验证 Ninja 驱动的 JIT warmup**。
- 下一次验证应保持 `.venv/bin` 在 `PATH` 中，使用正常编译路径（不加 `--enforce-eager`），保存完整日志，再判断 Ninja/JIT 是否通过。若明确报编译错误，先保存 traceback 和编译命令；若明确 OOM，再结合 `nvidia-smi` 调整显存预算。
- vLLM 启动后从同一网络命名空间检查 `/health`、`/v1/models`，再发一条很小的确定性请求。受限 sandbox 的 curl 曾无法访问本机 loopback，即使服务实际已启动；这表示网络命名空间隔离，不能据此判定服务未监听。
- 这次单题是合成算术 smoke，只证明接口可以返回结果，不构成医学准确率证据。之前 H0 的 B2 运行仍标记为基础设施失败；不能把它们改称有效成绩。修好服务后，先核对原失败 checkpoint/manifest，再只恢复确实失败且配置身份匹配的样本。

### 明日继续时的最短检查

```bash
git status --short
nvidia-smi
ps -eo pid,ppid,pgid,etime,stat,cmd
VLLM_ENV=/root/gpufree-data/Health-Copilot/training/posttrain/.venv
PATH="$VLLM_ENV/bin:$PATH" command -v ninja
"$VLLM_ENV/bin/ninja" --version
/usr/local/cuda/bin/nvcc --version
```

启动前确认 8000 属于 Memory 且不碰它；在 GPU 空闲或用户明确允许的剩余预算内，只启动绑定 `127.0.0.1:8001` 的服务。不要为这次 Ninja 排查重装整个共享环境、改锁文件或覆盖现有运行结果。

### 2026-10-05 正常 JIT 启动与断点续跑补充

- 在此执行环境中，`nohup ... &` 返回 PID 后进程会随命令会话结束而消失，日志仍为 0 字节；不要据此判断 vLLM 参数错误。用持久的前台 `exec` 会话承载服务，并通过 `tee -a <log>` 保存完整启动输出。现在已验证该方式可正常运行。
- 首次在 API Server 启动的同一秒探测 `/health` 可能因端口尚未开始监听而连接失败；等待看到 `Application startup complete` 后重试，后续 `/health` 返回 200，`/v1/models` 返回预期 served model。
- 2026-10-05 的 8001 正常编译启动实证：Ninja 可用，`torch.compile` 33.06 秒完成，CUDA graph 与 JIT warmup 完成；BF16 Qwen3-8B 在 16.3 GiB 左右权重/运行占用下分配 22.01 GiB FP8 KV cache。不要因之前的 Ninja 怀疑绕过编译或加 `--enforce-eager`。
- 从中断的评测恢复时，先核对 `cases.jsonl` 唯一 ID 数、冻结选择文件 hash、source revision、profile、模型 hash，再同步更新 resume manifest 的实现/runtime identity；把之前的 execution segment 原样保留并记下其 runtime config。旧行最终应通过独立 rescore 全部统一到冻结的 parser revision。
- 目标重试文件可能只包含原全量运行中传输失败的 ID。这是 `SelectedCaseAdapter` 的显式选择，不代表基础 adapter 只读到了这些候选项；源全量结果保留其余有效行，最终由 retry merger 按 ID 合成完整冻结集合。
