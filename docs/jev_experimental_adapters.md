# Jev research adapters

Jev is an opt-in research adapter. The default M3/M8 paths, frozen benchmark files, claim verification, citation authority, and medical-answer generation do not call Jev.

## Architecture routing

The multi-agent resume runner supports `--architecture jev-routed`. Jev receives the case question and returns one typed choice plus four independent worker probabilities. Python applies fixed thresholds, then executes the existing Single or heterogeneous team runner. If a Jev request or response fails, the routing fallback selects all team workers and records the fallback reason. Jev does not decide what the medical answer should say.

For example, run it against a public or synthetic DEV split:

```powershell
python tools/run_resume_multiagent.py `
  --benchmark-dir benchmarks/research_architecture_v2_candidate `
  --architecture jev-routed `
  --split DEV `
  --budget-mode native `
  --model YOUR_EXISTING_ANSWER_MODEL `
  --jev-data-classification public `
  --output-dir runs/jev_routed_dev
```

The runner requires an explicit `--jev-data-classification public|synthetic` before sending question text to TypeSafe. It reads `JEV_API_KEY` from the process environment or the repository `.env`; optional settings are `JEV_MODEL` (default `jev-latest`), `JEV_BASE_URL`, and `JEV_TIMEOUT_SECONDS`. It also uses the existing `HEALTH_COPILOT_MULTIAGENT_PROVIDER` adapter and answer-model setting. The default architecture and worker thresholds are both `0.5`. Results record route probabilities, chosen worker roles, Jev latency, Jev token usage, and total provider calls. `single` and `team` modes keep their prior behavior.

## Memory and history priority advice

`JevContextSelector` only returns high/low priority hints. The existing `ContextManager` still applies token budgets, protection rules, atomic tool grouping, and final plan construction. Current user, current evidence, system pins, and protected items are never sent as candidates. A completed tool exchange is evaluated as one group and receives one shared hint.

Use only explicitly classified public or synthetic material. The selector sends candidate content to TypeSafe to judge relevance, so do not pass real user health records or private session content.

```python
candidates = context_manager.priority_candidates(
    memory_records=memory_records,
    history=session_history,
    research_data_classification="synthetic",
)
advice = await JevContextSelector().advise(
    current_task=research_question,
    candidates=candidates,
    data_classification="synthetic",
)
plan = context_manager.build_plan(
    session_id=session_id,
    session_revision=session_revision,
    current_user=research_question,
    memory_records=memory_records,
    history=session_history,
    priority_hints=advice.priority_hints,
)
```

Priority hints affect only those unprotected memory/history candidates. They do not authorize memory writes, replace evidence, or directly drop context. The normal context planner may still exclude low-priority items under its configured budget.

The context candidate builder leaves data unclassified by default. Both the builder and selector require an explicit matching `public` or `synthetic` label before candidate content can be sent to Jev; do not apply that label to real or private health data.

## TypeSafe API

The client uses TypeSafe's documented `POST /v1/systemone` endpoint with bearer authentication and named typed questions. See the [official API schema](https://api.typesafe.ai/docs).
