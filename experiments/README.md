# Experiments: execution status

This file records which parts of the study have been executed and which are pending, so that no table in the repository can be mistaken for a result that was never measured.

Protocol v1 was frozen on 2026-11-22 ([`manifest.json`](manifest.json)). No LLM provider is recorded in it.

## Executed (offline, no credentials)

| Study | Split | Result directory |
|---|---|---|
| Rule-based signal classifier vs generator labels | all 150 records | [`results/signal-classifier-rules.json`](results/signal-classifier-rules.json) |
| Evidence-gate calibration, `hashing-v1` | dev | [`results/calibration-dev-hashing.json`](results/calibration-dev-hashing.json) |
| Retrieval ablation (vector vs rules) | dev | [`results/ablation-dev.json`](results/ablation-dev.json) |
| Configurations A and B, 3 repetitions | dev | [`results/dev-ab/`](results/dev-ab/summary.md) |
| Retrieval ablation (vector vs rules) | holdout | [`results/ablation-holdout.json`](results/ablation-holdout.json) |
| Configurations A and B, 3 repetitions | holdout | [`results/holdout-ab/`](results/holdout-ab/summary.md) |

## Pending: require an OpenAI-compatible LLM provider

| Study | Status | What is needed to run it |
|---|---|---|
| Configuration C (vector + LLM) | **not executed** | `DIRAG_LLM_*` variables; dev run first |
| Configuration D (rules + LLM) | **not executed** | same |
| Configuration E (LLM intent + LLM) | **not executed** | same |
| Configuration F (LLM intent + extractive) | **not executed** | same |
| Retrieval ablation with `llm_intent` | **not executed** | same |
| LLM correctness judge and claim verification | **not executed** | `DIRAG_JUDGE_MODEL`, preferably from another model family |
| LLM signal classifier (`--classifier llm`) | **not executed** | same as C–F; optional |

These components are implemented and covered by tests that use a scripted model (`ScriptedLLM`) in place of a provider. Scripted outputs only exercise the code paths. They are never written to `results/` and are not results.

Order of execution once a provider is available:

1. Development: `dirag run --split dev --configs C,D,E,F --name llm-dev`, then `dirag judge --name llm-dev`.
2. Provider freeze: `dirag amend-providers --note "…"`. This records the model, judge, temperature, seed and response format in the manifest, once and with a timestamp. Any other change to code, data, prompts or parameters keeps the holdout locked.
3. Holdout: `dirag run --split holdout --configs C,D,E,F --name llm-holdout`, then `dirag judge --name llm-holdout`.
4. Report all four configurations, including operational failures and spend, whether or not they improve on A and B.
