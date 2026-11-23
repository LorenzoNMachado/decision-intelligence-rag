# Methodology (protocol v2, historical)

> This document describes protocol v2, now historical. The current protocol is v3: [`docs/v3/protocol.md`](v3/protocol.md). The research-method section below (DSRM and the relation to the earlier study) applies unchanged to v3.

Protocol v1 is historical (see `experiments/v1/README.md`). This document
describes protocol v2, which is the only protocol the current code runs.

## Research questions

1. **RQ1.** Does a semantic-signal layer (rule-based or LLM-based question
   interpretation) improve retrieval over a pure vector baseline?
2. **RQ2.** How much of answer quality comes from the generator rather than the retriever?
3. **RQ3.** How well do an evidence gate and citation validation prevent answers
   to questions that the corpus cannot answer?

## Research method and relation to the earlier study

The original applied research, carried out at PUCPR, followed **Design Science
Research Methodology** (DSRM; Peffers et al., 2007). Its steps are: identify
the problem, define objectives, design and develop an artefact, demonstrate it,
evaluate it, and communicate. This repository is an **independent
reimplementation** of the concepts studied there: semantic signals for
decision records, evidence-gated answers, and citation grounding. It keeps the
DSRM shape: the artefact is the pipeline, the demonstration is the CLI and the
Streamlit app, and the evaluation is the frozen protocol below. However:

* no code, data, prompt or result of the earlier study is reused;
* the corpus is synthetic and generated here, not the study's data;
* the numbers reported here measure this implementation only and make no claim
  about, and are not comparable with, the earlier study's results.

## Strategies A–F in one sentence each

* **A**: vector similarity only, extractive answer. The baseline.
* **B**: A plus the rule-based semantic layer. Isolates rule-based signals without an LLM.
* **C**: A's retrieval with the LLM generator. Isolates the generator.
* **D**: B's retrieval with the LLM generator. Rule-based signals combined with an LLM.
* **E**: LLM intent interpretation and the LLM generator.
* **F**: LLM intent with the extractive generator. Isolates LLM intent from LLM generation.

Only A and B have been run. C–F need a provider and are pending.

## Why a second protocol

The v1 holdout was run once and then inspected during an audit, so it can no
longer serve as an unseen test set. The audit also found defects that affected
the v1 results: parser negation scope, circularity between the rule lexicon and
the generator's templates, a saturated Recall@4, and abstention measured only
out of domain. Protocol v2 has its own corpus, questions, paths and manifest.
v1 artefacts are never regenerated or recomputed with the current code.

## Order of work (verifiable in git)

1. The rule lexicon and parser were fixed (commit `00d44a6`) **before** any v2
   phrase pool or question existed. Neither has been edited since.
2. The v2 phrase pools and the 70 question specifications were written
   (`83740d1`), and the data generated (`ea47514`).
3. The evidence gate was calibrated and A/B were run on **dev only** (`58f57ba`).
   No system change followed from the dev results.
4. The protocol was frozen (`1674075`).
5. The holdout was run once for A and B (`9137cc0`). Defects found afterwards
   are documented, not fixed (see `experiments/v2/README.md`).

## Corpus

`dirag corpus-generate` builds 180 synthetic decision records (10 domains × 3
subjects × 6 actions, seed 4242). Each record has a title, decision, context,
rationale and metadata. Organisations, vendors and regulations are fictional,
and public audit frameworks appear only as vocabulary.

Each of the five signals gets an *intended* label per record, drawn
independently: PRESENT 40 %, ABSENT stated explicitly 22 %, UNCLEAR 38 %. About
30 % of the UNCLEAR labels come with a hedged mention. The record text is
written from those labels. To limit circularity with the rule lexicon, the v2
pools mix:

* sentences that use the lexicon's cue words;
* paraphrases without any cue word ("implementation work is subcontracted to …",
  "staff were losing hours every week to re-entry");
* distractors that use cue words without asserting a signal ("the vendor
  management office reviewed the business case", "the security of next year's
  funding was confirmed").

Six records per subject (up from five) mean that retrieving all of a subject's
records no longer fills the top four. Chunking is section-based (summary,
context, rationale), which gives 540 chunks, each starting with the record title
and id.

## Questions

`data/v2/question_specs.json` holds 70 hand-written phrasings, each used once.
`build_question_set` (question seed 1717) assigns each filter a subject drawn at
random among those with 1–4 matching records, and each lookup or in-domain
unanswerable a random record. Expected evidence comes from the generator's
labels.

| Category | Count | Expected evidence |
|---|---:|---|
| provider / regulation / framework lookup | 8 / 6 / 6 | the one record whose attribute is asked for |
| signal_filter | 14 | records of the subject with the signal PRESENT |
| negated_filter | 8 | records of the subject that state the signal is ABSENT |
| conjunction_filter | 6 | records satisfying both constraints (PRESENT/PRESENT or PRESENT/ABSENT) |
| disjunction_filter | 4 | records satisfying either constraint |
| hedged_filter | 4 | records with the signal PRESENT ("might", "may", "perhaps" are read as a preference) |
| unanswerable_in_domain | 8 | none: a real record, but information no record contains (budget, owner, date, value) |
| unanswerable_out_of_domain | 6 | none: topics no record touches |

Of the 36 filter questions, 27 contain at least one lexicon cue word and 9 none.
The split is drawn per category in proportion (30 % dev, at least one question
per category in each split): 20 dev questions (16 answerable, 4 unanswerable)
and 50 holdout questions (40 + 10). `corpus-validate` checks that every key fact
occurs in its expected evidence and that each split covers every category.

## Retrieval strategies

| Strategy | Intent source | Signal index |
|---|---|---|
| 1 · Vector baseline | none | not used |
| 2 · Rule-based enrichment | lexicon with local negation, coordination, hedging, entity masking | rule-based classifier (`rules-v2`) |
| 3 · LLM semantic intent | LLM, validated deterministically | rule-based classifier (`rules-v2`) |

Scoring: `final = cos + 0.06 · matched REQUIRED + 0.02 · matched PREFERRED − 0.12 · contradicted REQUIRED`,
over a candidate pool of the 20 most similar records. These weights were
inherited from v1 and were not tuned on v2 data. A record is admitted when its
cosine similarity reaches the evidence threshold and it does not contradict a
REQUIRED constraint.

The rule parser (Strategy 2) makes a signal REQUIRED unless it is hedged or
offered as an alternative to another signal ("a vendor *or* a legal
obligation"), in which case it is PREFERRED. A signal mentioned with both
polarities is dropped and reported. Negation is local: a negator governs a short
window after it, stopping at commas, semicolons, "but", "while" and similar
words. A predicate negation ("played no role") reaches back to its subject.
Names of subjects, departments and job titles are masked before analysis.

LLM intent validation: every constraint must quote at least two words of the
question verbatim, and at most two constraints stay REQUIRED. Malformed output or
provider errors produce status UNAVAILABLE, which falls back to the baseline.

## Generators and answer states

* **Extractive.** No model. For each admitted record, it quotes the sentence with
  the largest token overlap with the question, excluding title words. It
  abstains only when the gate admits nothing.
* **LLM.** Prompt `answer_v2`, strict JSON schema, per-claim `source_ids` and
  verbatim `quote`, explicit ABSTAIN option. Protocol v2 uses
  `quote_policy: "enforce"`: a claim survives only if it cites admitted records,
  names no record it does not cite, and has a quote of at least four words found
  literally in a cited record's body. The answer shown and judged is composed of
  the surviving claims only. The model's free text is kept for audit as
  `discarded_text`.
* **States.** `models.TRANSITIONS` fixes the allowed final status per outcome.
  An answer that loses every claim becomes ABSTAIN. Provider, schema and pipeline
  failures become ERROR, never ABSTAIN.

## Configurations

| ID | Retrieval | Generator | Isolates |
|---|---|---|---|
| A | Vector | Extractive | baseline |
| B | Rules | Extractive | effect of rule-based signals, without an LLM |
| C | Vector | LLM | effect of the LLM generator |
| D | Rules | LLM | rule-based signals with an LLM |
| E | LLM intent | LLM | LLM intent with an LLM |
| F | LLM intent | Extractive | LLM intent without an LLM generator |

E and F share the same intent interpretation per (question, repetition). It is
stored in the run's `intents.jsonl`, so the sharing also holds across resumed
sessions.

## Evidence-gate calibration

The threshold depends on the embedder's cosine scale. It is selected on the
**dev split only** with `dirag calibrate`. The sweep maximises Youden's J for
"some evidence admitted" (answerable vs unanswerable), breaks ties by the share
of expected records admitted, and returns the median of the optimal range.

For `hashing-v1` the v2 dev range was 0.08–0.27, with J = 0.5. The gate admits
evidence for half of the dev unanswerables (the in-domain ones) at every
threshold that keeps the answerable ones. The selected threshold is **0.18**.
Four unanswerable dev questions make this a coarse calibration.

## Protocol and holdout lock

1. Develop and calibrate on the dev split only.
2. `dirag freeze` writes `experiments/v2/manifest.json` with two hash groups:
   * **pipeline**: corpus, chunks, questions, question specifications, signal
     index, protocol, configurations, system prompts and all package source except
     the evaluation files;
   * **evaluation**: metrics, judges, report and statistics modules, the claim
     verifier, and the judge and verifier prompts.

   It also records the environment (Python, numpy and httpx versions, SHA-256 of
   `uv.lock`) and the providers (backend, model, endpoint; never a credential).
3. The runner refuses the holdout unless both groups, the lock file and the
   providers match. Python or package version differences are warnings.
4. `freeze` never replaces a manifest that holdout results were produced under.
5. LLM configurations also need the LLM provider recorded, once, with
   `dirag amend-providers`.
6. Evaluation code may be corrected after a holdout run only with
   `dirag amend-evaluation --note …`, which logs the old and new hashes. The
   pipeline group cannot be amended; changing it requires a new protocol version.

## Statistical limits

* **Unit of analysis.** The question. There are 50 holdout questions and 3–10
  per category, so category-level differences are descriptive.
* **Test.** An exact two-sided sign test on per-question differences between
  two configurations; ties are excluded. With 11 untied questions, as for A vs B
  on Recall@4, only a split of 10–1 or more extreme would reach p < 0.05.
* **No independent repetitions** for deterministic configurations. LLM
  configurations will have three repetitions per question, which estimate
  provider variability, not sampling variability over questions.
* **No correction for multiple comparisons.** None is needed so far, since a
  single pre-specified comparison (A vs B on Recall@4) was made. Per-category
  comparisons are exploratory.
* **No confidence intervals** are reported for the means. With this sample, an
  interval on a per-category mean would span most of [0, 1].

## Repetitions and reproducibility

Deterministic configurations (A and B) run once
(`deterministic_repetitions: 1`), because repeating them only duplicates rows.
LLM configurations run three repetitions. Temperature 0 and a seed are sent when
supported, but providers do not guarantee deterministic sampling. Configurations
are compared per question with an exact two-sided sign test, which is reported
in every summary.

Results are appended row by row to `rows.jsonl`. A run directory is reused only
with `--resume`, and only if split, embedder, threshold, protocol and pipeline
fingerprint (and, on the holdout, the manifest) are unchanged. Every session is
logged in `metadata.json`. Timeouts, network and schema errors, truncation and
budget refusals are ERROR rows, never abstentions.
