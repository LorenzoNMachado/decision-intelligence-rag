# Experiments: execution status

This file records what has been executed under each protocol and what is
pending, so that no table in the repository can be mistaken for a result that
was never measured. Protocols use different corpora, questions and code;
their numbers are **not comparable** with one another.

| Protocol | Status | Artefacts | Verify |
|---|---|---|---|
| **v3** | current; frozen 2026-11-22; holdout run once for A–H | `data/v3/`, `experiments/v3/` | `dirag verify-manifest` |
| **v2** | historical; holdout inspected while v3 was designed | `data/v2/`, `experiments/v2/` | `dirag verify-v2` (against commit `9153e5d`) |
| **v1** | historical; holdout inspected during the first audit | `data/synthetic/`, `experiments/{configurations,results}/` | `dirag verify-v1` (against commit `30bce65`) |

The current code runs protocol v3 only. Historical protocols are never
recomputed with it; see [`v1/README.md`](v1/README.md) and
[`v2/README.md`](v2/README.md).

## Protocol v3: executed (offline, no credentials)

| Study | Split | Result |
|---|---|---|
| Calibration of the gate (hashing, BM25) and topical band | dev | [`v3/results/calibration-dev.json`](v3/results/calibration-dev.json) |
| Configurations A–H, development runs (`dev`, `dev-r2`, `dev-r3`, `dev-r4`) | dev | [`v3/results/dev-r4/`](v3/results/dev-r4/summary.md) (final) |
| Retrieval ablation (vector, bm25, rules-v2, rules-v3-flat, rules-v3) | dev, holdout | [`v3/results/ablation-holdout.json`](v3/results/ablation-holdout.json) |
| Configurations A–H, one run each | holdout | [`v3/results/holdout/`](v3/results/holdout/summary.md) |
| Rule classifiers (rules-v2, rules-v3) vs generator labels | all 180 records | [`v3/results/`](v3/results/) |
| Adversarial audit (not independent; analysis only, no tuning) | holdout + 37 new cases | [`../docs/audits/v3/`](../docs/audits/v3/README.md) |

Plan, hypotheses and the list of changes made after the data was written:
[`../docs/v3/protocol.md`](../docs/v3/protocol.md). Results and threats to
validity: [`v3/README.md`](v3/README.md). These results are **exploratory**.

## Protocol v3: pending (require an OpenAI-compatible LLM provider)

| Study | Status | What is needed to run it |
|---|---|---|
| Configuration L (rules-v3 + sufficiency + LLM generator) | **not executed** | `DIRAG_LLM_*` variables, token prices (or `DIRAG_BUDGET_USD=none`); dev run first |
| Strategy 3 (LLM intent) | **not executed**, not part of the v3 configurations | same |
| LLM correctness judge and claim verification | **not executed** | `DIRAG_JUDGE_MODEL`, preferably from another model family |

These components are tested with a scripted model (`ScriptedLLM`) only;
scripted outputs are never written to `results/` and are not results. Order
once a provider is available: dev run, `dirag amend-providers`, holdout run,
`dirag judge`, report everything including failures and spend.
