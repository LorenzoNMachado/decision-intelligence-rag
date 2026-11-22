# Methodology

## Research questions

1. **RQ1.** Does a semantic-signal layer (rule-based or LLM-based question
   interpretation) improve retrieval over a pure vector baseline?
2. **RQ2.** How much of answer quality comes from the generator rather than the retriever?
3. **RQ3.** How well do an evidence gate and citation validation prevent answers
   to questions that the corpus cannot answer?

## Corpus

`dirag corpus-generate` builds 150 synthetic decision records (10 domains × 3
subjects × 5 actions) from seeded phrase pools. Each record has a title,
decision, context, rationale and metadata. Organisations, vendors and
regulations are fictional; public audit frameworks (ISO/IEC 27001, SOC 2,
PCI DSS, COSO) appear only as vocabulary.

Each of the five signals receives an *intended* label per record, drawn
independently (PRESENT 40 %, ABSENT stated explicitly 22 %, UNCLEAR 38 %, of
which about 30 % contain a hedged mention such as "may be considered later").
The record text is written from those labels. The labels are used only to
build the question set and to measure classifier accuracy; retrieval never
reads them.

Chunking is section-based (summary, context, rationale), with a sentence-aware
split for sections longer than 800 characters, giving 450 chunks. Every chunk
starts with the record title and identifier.

A demo preset (`--size demo`, 30 records) exists for quick experimentation.

## Questions

`build_question_set` produces 50 questions over the experimental corpus:

| Category | Count | Expected evidence |
|---|---:|---|
| provider_lookup | 8 | the one record whose vendor is asked for |
| regulation_lookup | 6 | the one record whose regulation is asked for |
| framework_lookup | 6 | the one record whose audit framework is asked for |
| signal_filter | 12 | records of one subject with the signal PRESENT (1–4) |
| negated_filter | 5 | records of one subject that state the signal is ABSENT |
| multi_signal_filter | 3 | records of one subject with two signals PRESENT |
| unanswerable | 10 | none (topics absent from the corpus) |

Signal filters use three paraphrases per signal, several of which are outside
the rule lexicon, so the rule-based parser is not guaranteed to recognise every
constraint. The split is stratified: 10 dev questions (8 answerable, 2
unanswerable) and 40 holdout questions (32 + 8).

Every reference is checked by `corpus-validate`: each key fact must occur
literally in the expected evidence, so no reference asks for information that
the corpus does not contain.

## Retrieval strategies

| Strategy | Intent source | Signal index |
|---|---|---|
| 1 · Vector baseline | none | not used |
| 2 · Rule-based enrichment | cue lexicon with clause-level negation | rule-based classifier |
| 3 · LLM semantic intent | LLM, validated deterministically | rule-based classifier |

Scoring: `final = cos + 0.06 · matched REQUIRED + 0.02 · matched PREFERRED − 0.12 · contradicted REQUIRED`,
computed over a candidate pool of the 20 most similar documents. A document is
admitted when its cosine similarity reaches the evidence threshold and it does
not contradict a REQUIRED constraint.

LLM intent validation: every constraint must quote at least two words of the
question verbatim; at most two constraints remain REQUIRED; malformed output or
provider errors produce status UNAVAILABLE, which falls back to the baseline.

## Generators

* **Extractive** — no model. For each admitted record, the sentence with the
  largest token overlap with the question (excluding title words) is quoted.
  It abstains only when the gate admits nothing.
* **LLM** — prompt `answer_v1`, strict JSON schema, per-claim `source_ids` and
  verbatim `quote`, explicit ABSTAIN option. Output is validated by the
  grounding layer before it is returned.

## Configurations

| ID | Retrieval | Generator | Isolates |
|---|---|---|---|
| A | Vector | Extractive | baseline |
| B | Rules | Extractive | effect of rule-based signals, without an LLM |
| C | Vector | LLM | effect of the LLM generator |
| D | Rules | LLM | rule-based signals with an LLM |
| E | LLM intent | LLM | LLM intent with an LLM |
| F | LLM intent | Extractive | LLM intent without an LLM generator |

E and F share the same intent interpretation per (question, repetition), so
they differ only in the generator.

## Evidence-gate calibration

The threshold depends on the embedder's cosine scale. It is selected on the
**dev split only** with `dirag calibrate`: the sweep maximises Youden's J for
"some evidence admitted" (answerable vs unanswerable), then the share of
expected records admitted, and returns the median of the optimal range (the
largest margin on both sides).

For the offline `hashing-v1` embedder the optimal dev range was 0.11–0.25 and
the selected threshold is **0.18**. The design default for dense neural
embeddings remains 0.65, which must be recalibrated in the same way before it
is used.

## Protocol and holdout lock

1. Develop and calibrate on the dev split only.
2. `dirag freeze` writes `experiments/manifest.json` with SHA-256 hashes of the
   corpus, chunks, questions, signal index, protocol, configurations, every
   prompt and the package source, plus the embedder and (when configured) LLM
   provider settings and the git commit.
3. The runner refuses the holdout split unless `verify-manifest` reports no
   change. Any edit to code, data, prompts or parameters locks the holdout.
4. LLM configurations additionally require the LLM provider to be recorded in
   the manifest. A manifest frozen without one can be amended exactly once with
   `dirag amend-providers --note "…"`. The amendment is timestamped in the
   manifest and cannot alter any frozen input.

## Repetitions and reproducibility

Each configuration runs three repetitions. Deterministic configurations (A, B)
produce identical repetitions by construction; the repetitions matter for
LLM-based configurations. Temperature 0 and a seed are sent when the provider
supports them, but providers do not guarantee deterministic sampling.

Results are appended row by row to `rows.jsonl`; an interrupted run resumes
where it stopped. Network errors, timeouts, truncation and budget exhaustion
are recorded as operational failures (status ERROR) and never silently retried
into a different outcome.
