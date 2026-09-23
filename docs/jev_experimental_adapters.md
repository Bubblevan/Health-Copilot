# Jev research adapters

Jev is an opt-in research adapter. The default M3/M8 paths, frozen benchmark files, claim verification, citation authority, and medical-answer generation do not call Jev.

## Architecture routing

The multi-agent resume runner supports both `--architecture jev-routed` and `--architecture jev-intent-routed`. The existing `jev-routed` arm returns an architecture choice and four independent worker probabilities. The new intent arm adds a primary task intent and independent evidence-structure probabilities in the same Jev request; Python applies a versioned policy and then executes the existing Single or heterogeneous team runner. If a Jev request or response fails, routing falls back to all team workers and records the fallback reason. Jev does not decide what the medical answer should say.

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

For the task-intent experiment, run the same public or synthetic DEV split with `--architecture jev-intent-routed`. Its `task_intent` output records primary intent counts and mean probabilities for each evidence-structure facet; each case stores the full probability map and the deterministic policy reasons. Independent-source, cross-source-comparison, or conflict-review probability at or above `0.5` raises the minimum route to Team. When independent-source, cross-source-comparison, or conflict-review work is indicated, the policy selects at least two source workers. Topic breadth, freshness, and serial-dependency signals are recorded but do not alone force parallel execution.

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

## Task-intent routing

The task-intent experiment outputs a primary label (`direct_lookup`, `general_explanation`, `multi_topic_synthesis`, `cross_authority_comparison`, `current_guideline_lookup`, `conflicting_guidance_review`, `serial_follow_up`, or `other`) and probabilities for topic breadth, independent sources, cross-source comparison, current guidance, conflict review, and serial dependency. The conflict signal means the question asks to review apparent differences; it does not establish that sources actually disagree. The deterministic route policy and its version are stored in the run configuration and per-case result.

The context candidate builder leaves data unclassified by default. Both the builder and selector require an explicit matching `public` or `synthetic` label before candidate content can be sent to Jev; do not apply that label to real or private health data.

## TypeSafe API

The client uses TypeSafe's documented `POST /v1/systemone` endpoint with bearer authentication and named typed questions. See the [official API schema](https://api.typesafe.ai/docs).
