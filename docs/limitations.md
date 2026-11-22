# Limitations

These limitations bound what the reported numbers can support. They are listed
so that readers can weigh the results, not as a to-do list.

## Authorship and independence

* **One author for system, data and questions.** The corpus generator, its
  phrase pools, the rule lexicon, the question phrasings and the references
  were all written by the same author. Protocol v2 reduces the coupling: the
  lexicon was frozen before the v2 pools and questions were written, the pools
  include cue-free paraphrases and distractors, and subjects and splits are
  drawn at random. It does not remove the coupling. The v2 holdout was unseen
  by the system but not independently authored, so **v2 results are
  exploratory, not confirmatory.**
* **References come from the generator.** Expected evidence is derived from the
  generator's intended labels. Those labels are not independent annotations;
  they are only as valid as the phrase pools that express them.

## Data

* **Synthetic corpus.** Records come from phrase pools and are shorter and more
  regular than real decision records. Nothing here measures performance on
  organisational documents.
* **Small evaluation set.** There are 70 questions (50 in the holdout), and 3–10
  holdout questions per category. A per-question sign test is reported, but most
  differences between configurations are within what chance can produce. Category
  breakdowns are descriptive.
* **Hedged questions use a convention.** "Which decisions might involve a
  vendor?" counts the records with the signal PRESENT as its answer. Another
  reader could include hedged (UNCLEAR) records.
* **Cosmetic generator flaw.** Titles derived from "usage-based pricing model"
  read "an usage-based …". The flaw was found on dev and left in place, to avoid
  regenerating data during development.

## Signals and retrieval

* **The rule classifier is weak on v2 text** (78.3 % macro accuracy against the
  generator's labels; 93.9 % on v1, where the lexicon matched the templates).
  Cue-free paraphrases are missed (UNCLEAR), and distractor sentences produce
  false PRESENT labels.
* **PRESENT overrides ABSENT** when a record has both kinds of mention. Combined
  with the contradiction penalty and the bonus for matching ABSENT, this makes
  negated filters fail: on the v2 holdout, Strategy 2 drops their Recall@4 from
  1.000 to 0.333 and admits records about other subjects. This was diagnosed
  after the holdout run and is deliberately not fixed in v2.
* **Retrieval weights were inherited** from v1 (bonus 0.06 / 0.02, penalty 0.12,
  candidate pool 20) and never tuned on v2 data.
* **The LLM signal classifier** (`signals-classify --classifier llm`) can be
  evaluated against the generator's labels, but retrieval always uses the
  rule-based index.

## Embeddings and the evidence gate

* **The offline embedder is lexical.** `hashing-v1` measures word and bigram
  overlap, not meaning, so every published result is a lexical-retrieval
  result. A dense model can be used through
  `DIRAG_EMBEDDING_BACKEND=openai-compatible`, but needs its own calibration and
  freeze.
* **The gate cannot detect in-domain unanswerable questions.** A question that
  names a real record matches it lexically whatever it asks. On the v2 holdout,
  A and B answered all six in-domain unanswerables. One out-of-domain question
  ("What menu changes were approved…") was also admitted, through the words
  "changes" and "approved". The gate was calibrated on four dev unanswerables.

## Judging and metrics

* **Lexical correctness is a proxy.** It requires the literal key facts and
  forbids literal mentions of non-expected records. It under-credits paraphrases
  and penalises, by design, the extractive generator's multi-record answers.
* **Faithfulness of extractive quotes is not a measurement.** They are copied
  from the record, so they are excluded from the metric (shown as "—").
* **Traceability is not truth.** A valid citation plus a literally matched quote
  shows that a claim points at admitted evidence. Only the experimental LLM claim
  verifier assesses whether the statement is supported, and it has not been
  validated against human labels.
* **Latency** is measured on one machine within one run. A warm-up precedes
  timing, but sub-millisecond differences remain noise and are not evidence that
  one configuration is faster.

## LLM configurations and cost

* **Not executed.** C–F, Strategy 3, the LLM judge and the claim verifier
  require credentials. They are tested only with scripted models; no LLM result
  exists in this repository.
* **The spending limit has conditions.** It is enforced against an upper bound
  computed from the configured prices and refuses to run without prices. It
  holds only if those prices are correct and the provider counts at most one
  token per byte and honours `max_tokens`. Requests rejected with an HTTP error
  are assumed unbilled. The provider's billing is authoritative.
* **Provider determinism.** Temperature 0 and a seed are requested when
  supported, but hosted models do not guarantee identical outputs, which is why
  LLM configurations run three repetitions.

## Scope

* This is an independent implementation of concepts studied during applied
  research in Computer Science at PUCPR. It does not reproduce, and makes no
  claim about, the results of that earlier study.
* Cost per question excludes judge and claim-verification calls, which are
  evaluation overhead rather than system cost.
