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

## Formulas and denominators

Notation, per row (one configuration, repetition and question): *E* is the set of
expected records, *T* the top-4 retrieved records (admitted or not), *C* the
records cited by the answer's claims, and *s* ∈ {ANSWER, ABSTAIN, ERROR} the
final status. *Q⁺* are the answerable questions and *Q⁻* the unanswerable ones.

| Metric | Formula | Denominator |
|---|---|---|
| Recall@4 | mean over Q⁺ of \|E ∩ T\| / \|E\| | answerable rows |
| Supporting-source precision | mean of \|C ∩ E\| / \|C\| | answerable rows with s = ANSWER and \|C\| ≥ 1 |
| Expected-source coverage | mean of \|C ∩ E\| / \|E\| | same as precision |
| Strict correctness | #CORRECT / #judged | answerable rows whose label is not a judge failure |
| Partial correctness | #PARTIAL / #judged | same |
| Quote verification rate | #claims with a verified quote / #claims | model-written claims of ANSWER rows |
| Claim support (LLM) | #SUPPORTED / #judged claims | claims assessed by the claim verifier |
| Correct abstention | #(s = ABSTAIN) / \|rows of Q⁻\| | unanswerable rows |
| … in-domain / out-of-domain (v3) | the same, split by category | in-domain: a real record, information not stated; out-of-domain: no record on the topic |
| Unsupported answer rate (v3) | #(s = ANSWER and (q ∈ Q⁻ or C ∩ E = ∅)) / #(s = ANSWER) | ANSWER rows |
| False answer | #(s = ANSWER) / \|rows of Q⁻\| | unanswerable rows |
| False abstention | #(s = ABSTAIN) / \|rows of Q⁺\| | answerable rows |
| Operational failures | #(s = ERROR) | all rows (counted, never folded into abstention) |

On unanswerable rows, correct abstention + false answer + operational
failures = 1.

**Intervals (v3).** Proportions get 95 % Wilson score intervals. Recall@4 gets
a seeded percentile bootstrap over questions (2 000 resamples; repetitions are
averaged per question first, so they never count as observations). Paired
comparisons report the exact two-sided sign test and a bootstrap 95 % interval
of the mean per-question difference.

## What each grounding check shows, and what it does not

| Check | It shows | It does **not** show |
|---|---|---|
| **Recall** | The expected records were among the top four retrieved | That they were admitted, cited, or used correctly |
| **Supporting-source precision** | Which share of the cited records were expected ones | That the statements about them are true |
| **Correctness** (lexical proxy) | That the expected key facts appear in the answer and no forbidden ones do | That a paraphrase is right (it is not recognised), or that the answer is well grounded |
| **Citation validity** | Every cited id is a record that passed the evidence gate | That the record supports the statement |
| **Quote verification** | The quote occurs verbatim in the cited record's body | That the statement follows from the quote |
| **Faithfulness / support** (LLM verifier, experimental) | An automated judgement that the cited records entail the claim | A human-validated judgement; this verifier has not been calibrated |
| **Evidence sufficiency** (v3) | An admitted record asserts a value of the expected type about the requested attribute, or asserts every required condition | That the value is correct; it trusts the record. Paraphrased attributes are missed (false abstention) |
| **Abstention** | The system declined: nothing admitted, insufficient evidence, or no claim survived | That it declines whenever it should; v3 catches in-domain unanswerables only for statements its rules recognise |

A valid citation with a verified quote makes a claim **traceable**, not
**true**. Correctness is graded only on the composed answer of validated
claims, and only after the state rule below.

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
