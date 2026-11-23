# Protocol v3 — evidence sufficiency and retrieval reliability

This document was committed **before** any calibration, dev run or holdout run
of protocol v3. Everything after the "Results" heading of
`experiments/v3/README.md` is added later and refers back to this plan.

## Scientific problem

Protocol v2 showed three failures (`docs/v3/diagnosis.md`): the evidence gate
admits records that name the right decision but do not contain the requested
information, so in-domain unanswerable questions are answered; signal bonuses
lift records of other subjects and a single distractor mention outweighs an
explicit negation; and the extractive generator quotes the sentence with the
most shared words, not the one that answers.

## Hypotheses

* **H1 (sufficiency).** An evidence-sufficiency check between retrieval and
  generation raises the correct-abstention rate on in-domain unanswerable
  questions, at a measurable cost in false abstentions on answerable ones.
* **H2 (retrieval).** Scoped signal labels and topical eligibility (rules-v3)
  remove the v2 regression on negated filters without losing Recall@4
  elsewhere, compared with the v2 semantic layer (rules-v2).
* **H3 (generation).** Quoting the sentence found by the sufficiency check
  (extractive-v2) raises strict correctness over the overlap-based quote
  (extractive-v1) on the same sufficient records.
* **H4 (baseline).** BM25 and the hashing embedder perform similarly on this
  lexical corpus; no claim about semantic retrieval follows from either.

None of these is assumed to hold. A null or negative result is reported as such.

## Configurations

| ID | Retrieval | Sufficiency | Generator | Isolates |
|---|---|---|---|---|
| A | vector | — | extractive-v1 | baseline |
| B | rules-v2 | — | extractive-v1 | v2 semantic layer |
| C | rules-v3 | — | extractive-v1 | corrected semantic layer (vs B) |
| D | rules-v2 | ✓ | extractive-v1 | sufficiency without the new ranking (vs B) |
| E | rules-v3 | ✓ | extractive-v1 | sufficiency with the new ranking (vs C, vs D) |
| F | rules-v3 | ✓ | extractive-v2 | the generator (vs E) |
| G | bm25 | — | extractive-v1 | lexical baseline (vs A) |
| H | vector | ✓ | extractive-v2 | sufficiency + generator without signals (vs A, vs F) |
| L | rules-v3 | ✓ | llm | LLM generator — **not run** (needs a provider) |

Retrieval-only ablation: vector, bm25, rules-v2, rules-v3-flat (scoped labels,
flat ranking), rules-v3 (scoped labels, topical ranking).

## Pre-specified comparisons

Primary (holdout, per question):

1. **E vs C** — correct abstention on in-domain unanswerables
   (`entity_missing` + `fact_missing`, 40 questions) and false abstention on
   answerable questions (100). Tests H1.
2. **C vs B** — Recall@4 on answerable questions, overall and on
   `negated_filter`. Tests H2.
3. **F vs E** — strict correctness (lexical proxy) on answerable questions. Tests H3.
4. **G vs A** — Recall@4. Tests H4.

Secondary: D vs B, E vs D, H vs A, F vs H, and every per-category table.
Paired differences use the exact two-sided sign test and a seeded bootstrap
95 % interval of the mean difference; proportions get 95 % Wilson intervals.
There is no correction for multiple comparisons; the four primary comparisons
are fixed here, and everything else is exploratory.

## Sizes, fixed before the experiment

8 question categories × (8 dev + 20 holdout) = 224 questions, 160 in the
holdout. Reasons:

* 20 holdout questions per category bound a 95 % Wilson interval to about
  ±0.22 at a rate of 0.5 and ±0.13 at 0.9 — enough to tell "almost never"
  from "almost always" per category, not to rank close configurations;
* the primary comparisons pool categories (40 in-domain unanswerable, 100
  answerable questions), where a consistent effect on 10 or more untied
  questions out of 10 reaches p < 0.01 in a sign test;
* 8 dev questions per category (64) keep every category represented in
  calibration while leaving most questions to the holdout.

## Calibration, dev only

* **Evidence gate** per scorer (hashing-v1 for vector/rules, bm25-v1):
  maximise Youden's J of "some record admitted" between answerable and
  *out-of-domain* dev questions, ties broken by the share of expected records
  admitted; take the median of the optimal set. In-domain unanswerables are not
  used: they name real records, so separating them is the sufficiency check's
  job, not the gate's.
* **Topical band** of rules-v3: maximise mean Recall@4 over answerable dev
  questions on the grid 0.02, 0.04, …, 0.20; median of the optimal set.
* Bonuses, penalty, candidate pool and top-k are inherited from v1/v2 and not tuned.

## Holdout rules

The protocol is frozen (`dirag freeze`) after calibration and dev runs. The
holdout is run once for A–H. No change to the system follows from holdout
results; defects found afterwards are documented as such.

## Independence and its limits

* The system (sufficiency check, assertion scope, lexicon v3, topical ranking,
  extractive-v2) was committed (`53dbec6`) before the v3 generator and
  questions were written.
* The documents (generator), the questions (specifications) and the gold
  standard (derived only from the generator's recorded facts and labels, and
  audited by `corpus-validate`) are separate steps, but **one author wrote all
  of them and the system**. No independent human review exists. Results are
  therefore **exploratory**, not confirmatory.
* `*_missing` questions reuse the phrasings of `*_lookup` questions, so a
  system cannot tell answerable from unanswerable questions by wording.

## Changes made after the v3 data was written

| Commit | Change | Found by |
|---|---|---|
| `4277b90` | Entity words no longer removed from the question focus ("budget planning tool" removed "budget") | interface smoke test on one corpus record, before calibration |
| (dev wiring fix) | The runner now passes the sufficiency checker to the pipeline when a configuration enables it; before, D/E ran without the check and F/H abstained on everything | first dev run (D = B, E = C exactly; F, H false abstention 1.0); no system logic changed, the dev run was repeated |

| (dev fix) | A copular "who is the <type noun>" question asks for that type ("who is the external supplier" → organisation, not person) | dev-r2: false abstentions on two dev entity lookups |
| (dev fix) | The checker masks the referenced record's title as a phrase instead of removing every title word from the focus, so "budget" survives in "the decision to introduce a budget planning tool" | dev-r2: false abstention on one dev budget lookup |
| (dev fix) | A capitalised name counts as an organisation only in a sentence related to the question or asserting a third-party dependency (a consulted person's name had been accepted as "the" organisation) | dev-r3: one in-domain unanswerable vendor question answered with a person's name |

Observed on dev and deliberately **not** fixed (fixing them would mean tuning
the lexicon to the v3 phrasings; they are reported as limitations):

* paraphrases without lexicon words ("because the current setup keeps
  failing", "rely on another company") leave the condition not understood,
  so the check abstains;
* counterfactual modality ("without the change the organisation would
  breach …") is read as hypothetical;
* attribute synonyms ("staff" for "employees", "affects" for "use") do not
  match the question's focus;
* the cue "compliance" also occurs in legal sentences ("compliance with the
  … Regulation"), which makes some records ambiguous on compliance.

Dev runs are kept as they were produced: `dev` (runner wiring defect), `dev-r2`
and `dev-r3` (before the dev fixes above) and `dev-r4`, the last dev iteration.
The protocol was frozen right after `dev-r4`, whatever it showed, to stop
iterating on dev.
