# Experiments: execution status

This file records what has been executed under each protocol and what is
pending, so that no table in the repository can be mistaken for a result that
was never measured.

| Protocol | Status | Artefacts | Manifest | Verify |
|---|---|---|---|---|
| **v2** | current; frozen on 2026-11-22; holdout run once for A and B | `data/v2/`, `experiments/v2/` | [`v2/manifest.json`](v2/manifest.json) | `dirag verify-manifest` |
| **v1** | historical, superseded; holdout no longer blind | `data/synthetic/`, `experiments/{configurations,results}/` | [`manifest.json`](manifest.json) | `dirag verify-v1` |

The current code runs protocol v2 only. Protocol v1 is never recomputed with
it; see [`v1/README.md`](v1/README.md).

## Protocol v2: executed (offline, no credentials)

| Study | Split | Result |
|---|---|---|
| Evidence-gate calibration, `hashing-v1` | dev | [`v2/results/calibration-dev-hashing.json`](v2/results/calibration-dev-hashing.json) |
| Retrieval ablation (vector vs rules) | dev | [`v2/results/ablation-dev.json`](v2/results/ablation-dev.json) |
| Configurations A and B, one run each | dev | [`v2/results/dev-ab/`](v2/results/dev-ab/summary.md) |
| Retrieval ablation (vector vs rules) | holdout | [`v2/results/ablation-holdout.json`](v2/results/ablation-holdout.json) |
| Configurations A and B, one run each | holdout | [`v2/results/holdout-ab/`](v2/results/holdout-ab/summary.md) |
| Rule-based signal classifier vs generator labels | all 180 records | [`v2/results/signal-classifier-rules.json`](v2/results/signal-classifier-rules.json) |

Results and their interpretation: [`v2/README.md`](v2/README.md). These results are
**exploratory**: the holdout was unseen by the system, but the same author wrote
the system, the corpus and the questions.

## Protocol v2: pending (require an OpenAI-compatible LLM provider)

| Study | Status | What is needed to run it |
|---|---|---|
| Configuration C (vector + LLM) | **not executed** | `DIRAG_LLM_*` variables, token prices (or `DIRAG_BUDGET_USD=none`); dev run first |
| Configuration D (rules + LLM) | **not executed** | same |
| Configuration E (LLM intent + LLM) | **not executed** | same |
| Configuration F (LLM intent + extractive) | **not executed** | same |
| Retrieval ablation with `llm_intent` | **not executed** | same |
| LLM correctness judge and claim verification | **not executed** | `DIRAG_JUDGE_MODEL`, preferably from another model family |
| LLM signal classifier (`--classifier llm`) | **not executed** | same as C–F; optional |

These components are implemented and covered by tests that use a scripted model
(`ScriptedLLM`) in place of a provider. Scripted outputs only exercise the code
paths. They are never written to `results/` and are not results.

Order of execution once a provider is available:

1. Development: `dirag run --split dev --configs C,D,E,F --name llm-dev`, then `dirag judge --name llm-dev`.
2. Provider freeze: `dirag amend-providers --note "…"`. This records the model, judge, temperature, seed,
   response format and endpoint in the manifest, once and with a timestamp.
3. Holdout: `dirag run --split holdout --configs C,D,E,F --name llm-holdout`, then
   `dirag judge --name llm-holdout`.
4. Report all four configurations, including operational failures, unknown-cost calls and spend, whether or
   not they improve on A and B.
