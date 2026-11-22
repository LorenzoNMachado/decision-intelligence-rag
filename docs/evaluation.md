# Evaluation

## Metrics

A *row* is one execution of (configuration, repetition, question). Metrics are
aggregated over all rows of a configuration.

| Metric | Population | Definition |
|---|---|---|
| Recall@4 | answerable rows | \|expected ∩ top-4 retrieved\| / \|expected\| |
| Supporting-source precision | answerable rows that answered with ≥ 1 citation | \|cited ∩ expected\| / \|cited\| |
| Expected-source coverage | same as above | \|cited ∩ expected\| / \|expected\| |
| Answer correctness | answerable rows | rows labelled CORRECT / answerable rows without a judge failure. The state rule applies first: ABSTAIN and ERROR are INCORRECT whatever a judge says |
| Citation faithfulness (quote) | model-written claims of ANSWER rows | claims whose quote occurs literally in a cited record's body / model-written claims. Extractive quotes are copied, so they are excluded (reported as —) |
| Citation faithfulness (LLM) | claims of LLM ANSWER rows | claims judged SUPPORTED / judged claims (experimental) |
| Correct abstention rate | unanswerable rows | ABSTAIN / unanswerable rows |
| False-answer rate | unanswerable rows | ANSWER / unanswerable rows |
| False-abstention rate | answerable rows | ABSTAIN / answerable rows |
| Latency p50 / p95 | all rows | end-to-end milliseconds, including intent and generation calls |
| Cost per question | all rows | mean USD of intent and generation calls (judge calls excluded); `None` when any row's cost is unknown (`cost_unknown_rows` counts them) |
| Paired comparison | questions | per question, configuration B better / worse / tied than A on Recall@4 and strict correctness, with an exact two-sided sign test; repetitions are averaged, never counted as observations |
| Operational failures | all rows | rows with status ERROR (timeouts, network, truncation, budget) |
| Irrelevant sources cited | answerable ANSWER rows | cited records not in the expected set, per repetition |

Correct abstention + false answer + operational failure = 1 on unanswerable rows.

## Judges

All judges run after the **state rule** (`judges.grade`): unanswerable
questions are CORRECT exactly when the system abstained; answerable questions
that abstained or failed are INCORRECT. Only answered answerable rows reach a
judge, and the text a judge sees is the composed answer of validated claims,
never discarded model output.

* **Lexical key-fact judge (proxy, deterministic).** CORRECT when every key fact
  of the reference appears in the answer and no forbidden fact does. For list
  questions, the forbidden facts are the titles of every non-expected record;
  for lookups, the other values of the same attribute. It cannot recognise
  paraphrases, so it under-credits fluent LLM answers. Tables label it as a proxy.
* **LLM correctness judge (`judge_correctness_v1`).** Sees the question, the
  reference and the answer, but not the sources. A judge from a different model
  family than the generator is recommended.
* **Claim verifier (`claim_verification_v1`, experimental extension).** Sees one
  claim and the full text of its cited sources and labels it SUPPORTED, PARTIAL,
  NOT_SUPPORTED or CONTRADICTED. It has not been validated against human
  judgements in this repository.

## Studies

| Study | Command | Needs credentials |
|---|---|---|
| Signal classifier accuracy | `dirag signals-evaluate` | no (rules); yes for `--classifier llm` |
| Gate calibration (dev) | `dirag calibrate` | no |
| Retrieval ablation | `dirag ablation --split dev --strategies vector,rules[,llm_intent]` | only for `llm_intent` |
| Configurations A–F | `dirag run --split … --configs …` | C, D, E, F |
| LLM judging | `dirag judge --name …` | yes |
| Quote-gate ablation | automatic in `report` (replay) | no extra calls |

## Outputs

Each run directory under `experiments/v2/results/<name>/` contains:

* `metadata.json` — split, protocol, embedder, threshold, pipeline fingerprint
  and one entry per session (configurations, provider settings without
  secrets, manifest hash and warnings, timestamps, rows written, error rows,
  spend split into confirmed and estimated, calls with unknown cost);
* `intents.jsonl` — stored LLM intent interpretations (Strategy 3 only);
* `rows.jsonl` — one row per execution with intent, ranked documents, gate
  decisions, answer, claims, rejected citations, latency and cost;
* `judgements.jsonl` — LLM verdicts, when `dirag judge` was run;
* `summary.json`, `summary.csv`, `summary.md` — metrics per configuration.
