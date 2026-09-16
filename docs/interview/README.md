# 面试准备文档

这组文档把面试回答绑定到可回到仓库核对的事实，而不是背技术名词。

## 两份文档的边界

- [project.md](project.md)：Health-Copilot 在 `main@1bea0f37f91aaff7dead8e9a01acecfba360af3e` 的项目专属追问，覆盖 M0、已合并的 M1、源码、测试、冻结评测、bad case、指标和 M2 边界。M1 相关内容已经按真实实现重写；M2 仍标为 `[M2 PLANNED]`。
- [bagua.md](bagua.md)：独立的 Agent Runtime、RAG/Retrieval、LLM 结构与推理、Serving、分布式训练、后训练和 Eval 通用八股。它可以解释项目没有使用的概念，但不会把它们写成 Health-Copilot 的已实现能力。

## 推荐阅读路径

1. 先读 `project.md` 的 30 秒版本，再按 Q1–Q57 练习：项目是什么 → 请求怎么走 → 为什么这样设计 → bad case → M1 loop → 指标 → 工程边界 → M2。
2. 面试官从项目事实跳到通用概念时，再打开 `bagua.md`；每个主题按短答、原理/公式、trade-off、追问组织。
3. 需要核对事实时，优先查看 `src/`、`tests/`、`evals/`、`runs/m1/` 和 `.github/workflows/ci.yml`；外部参考链接只用于通用技术事实，不替代项目证据。

## 标记含义

- `[M0 已实现]` / `[M1 已实现]`：当前基线源码和测试/评测可以核对。
- `[M2 PLANNED]`：后续方向，当前没有实现，不能作为简历中的已完成能力。
- 通用知识：与项目实现状态解耦；是否用在 Health-Copilot，必须回到 `project.md` 判断。
