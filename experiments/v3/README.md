# Protocol v3: results

Plan registered before any run: [`docs/v3/protocol.md`](../../docs/v3/protocol.md)
(`beb1c7a`). Frozen at `6ec3481` (manifest committed in `6ac8892`, 2026-11-22
22:53:58 UTC). The holdout was run once, 6 seconds after the freeze, for A–H,
with no change to the system afterwards (`765e420`). Embedder `hashing-v1`
(gate 0.2115), BM25 gate 7.522, topical band 0.12, all from dev calibration.
Correctness is the lexical key-fact proxy. Each configuration is deterministic
and ran once; the question is the unit of comparison.

**Status: exploratory.** The system was frozen before the corpus and questions
were written, but one author wrote everything and no independent human review
exists (see "Threats to validity"). An adversarial audit run after the holdout
([`docs/audits/v3/`](../../docs/audits/v3/README.md); not independent, same AI
agent as the implementation) qualifies several results; its findings are
summarised below. Nothing in the protocol or the stored results was changed.

## Holdout (160 questions: 100 answerable, 40 in-domain unanswerable, 20 out of domain)

| Metric | A | B | C | D | E | F | G | H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Retrieval | vector | rules-v2 | rules-v3 | rules-v2 | rules-v3 | rules-v3 | bm25 | vector |
| Sufficiency check | — | — | — | ✓ | ✓ | ✓ | — | ✓ |
| Generator | ext-v1 | ext-v1 | ext-v1 | ext-v1 | ext-v1 | ext-v2 | ext-v1 | ext-v2 |
| Recall@4 | 0.791 | 0.828 | **0.953** | 0.828 | 0.953 | 0.953 | 0.824 | 0.791 |
| Supporting-source precision | 0.306 | 0.343 | 0.424 | 0.943 | 0.952 | 0.952 | 0.300 | 0.948 |
| Strict correctness (lexical) | 0.110 | 0.100 | 0.190 | 0.350 | 0.530 | **0.700** | 0.100 | 0.480 |
| Correct abstention, in-domain (40) | 0/40 | 0/40 | 0/40 | 40/40 | 40/40 | 40/40 | 0/40 | 40/40 |
| Correct abstention, out-of-domain (20) | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| False abstention, answerable (100) | 1 | 2 | 1 | 22 | 14 | 14 | 0 | 24 |
| Unsupported answers (share of ANSWER) | 0.367 | 0.362 | 0.309 | 0.013 | 0.012 | 0.012 | 0.343 | 0.013 |

95 % intervals: Recall@4 of C/E/F [0.914, 0.987], of A [0.723, 0.855]; strict
correctness of F [0.604, 0.781], of E [0.433, 0.625]; in-domain abstention with
the check [0.912, 1.000]; false abstention of E [0.085, 0.221]. Full tables,
including per-category results and all paired comparisons:
[`results/holdout/summary.md`](results/holdout/summary.md).

## Pre-specified primary comparisons

| # | Comparison | Metric | Result | Hypothesis |
|---|---|---|---|---|
| 1 | E vs C | in-domain correct abstention | 0/40 → 40/40 | **H1 supported, with a cost** |
| | | false abstention (answerable) | 1/100 → 14/100 | |
| 2 | C vs B | Recall@4 | 0.828 → 0.953; 21 better, 1 worse, p < 0.001; mean +0.126 [0.073, 0.183] | **H2 supported on this corpus** |
| | | negated filters | 0.579 → 0.933 | |
| 3 | F vs E | strict correctness (100 answerable) | 0.53 → 0.70; 17 better, 0 worse, p < 0.001; +0.17 [0.10, 0.25] | **H3 supported** |
| 4 | G vs A | Recall@4 | 0.791 vs 0.824; 13 vs 9, p = 0.52; +0.033 [−0.032, 0.098] | **no difference (H4)** |

Denominators. Strict correctness and its paired differences are over the 100
answerable questions, as pre-registered. The stored summary
(`results/holdout/summary.json`, `paired`) compares the "correct" label over
all 160 questions, unanswerable ones included: there, F vs E is +0.106
[0.063, 0.156] (the same 17 questions divided by 160). Earlier versions of
this file quoted that figure next to 0.53 → 0.70; the interval [0.10, 0.25]
is the audit's bootstrap over the 100 answerable questions (5,000 resamples,
seed 1). The stored results are unchanged.

Secondary: sufficiency on the v2 retrieval (D vs B) has the same abstention
effect but more false abstentions (22 vs 14), because the v2 ranking admits
fewer expected records. Without signals (H vs F), Recall@4 is 0.1625 lower and
strict correctness 0.22 lower (22 of 100 answerable questions; −0.1375 over
all 160 in the stored summary).

## What changed what (retrieval ablation, holdout)

| Strategy | Recall@4 | All expected records admitted | Non-expected records admitted |
|---|---:|---:|---:|
| vector | 0.791 | 0.68 | 434 |
| bm25 | 0.824 | 0.70 | 440 |
| rules-v2 | 0.828 | 0.69 | 403 |
| rules-v3-flat (scoped labels, flat ranking) | 0.953 | 0.90 | 375 |
| rules-v3 (scoped labels, topical ranking) | 0.953 | 0.90 | 376 |

**The Recall@4 gain of C over B comes mostly from the scoped signal index,
then from the v3 parser, not from topical eligibility.** The audit separated
them (Recall@4, holdout): v2 parser with the v2 index 0.828, v2 parser with
the scoped index 0.913, v3 parser with the v2 index 0.833, both v3 0.953;
topical ranking adds 0.000–0.005 in every combination. On this corpus, rules-v3 and
rules-v3-flat are indistinguishable, and the dev calibration of the band was
flat over 0.04–0.20. The mechanism that lifted foreign records in v2 needs
false labels on the subject's own records (v2 diagnosis). With scoped labels
those are rare, so the band had nothing left to correct. Topical eligibility
stays as a safeguard, but this experiment gives no evidence for it.

Classifiers against the generator's intended labels (all 180 records):
rules-v2 0.661 macro accuracy (third-party 0.444), rules-v3 0.893
(third-party 0.983). This is corpus-level, not a question split. The
third-party difference comes mostly from dependencies stated only through a
named organisation. On the generator's own phrase pool rules-v3 labels 28 of
33 phrases; on 24 new paraphrases written for the audit it labels 10 (rules-v2:
9), so the classifier's advantage is specific to this corpus' wording.

## Where the sufficiency check fails (E, holdout, post-run reading of the rows)

Of the 14 false abstentions:

* 6 fact lookups: the value was present, but the sentence did not name the
  requested attribute in the question's words ("funding" or "capped" for
  "budget", "staff" for "employees", "goes live" for "take effect");
* 2 signal filters: a cue-free paraphrase ("rely on another company"), so the
  condition was not understood;
* 2 multi-condition and 1 negated filter: an expected record was not retrieved
  or not admitted;
* 1 multi-condition: an operational motivation stated without cue words;
* 1 entity lookup: a regulation stated as a counterfactual ("would breach") is
  read as hypothetical;
* 1 negated filter: "Which budget planning tool decisions …" is classified as a
  money question, because "budget" in the subject name looks like the answer
  type.

These are documented, not fixed. The last two are general defects for a future
protocol; the others are limits of a lexical approach.

## Threats to validity

* **What the 40/40 measures.** In 29 of the 40 in-domain unanswerable
  targets there is no value of the requested type at all. Audit ablations of
  F: without assertion-scope classification the check still abstains on
  39/40 (13/100 false abstentions); without scope and without the
  attribute-word requirement, on 30/40 (9/100). Scope classification changed
  the outcome of one question. The 40/40 shows that the check rejects records
  lacking the requested type of value, and some whose value concerns another
  attribute; it does **not** show robust interpretation of scope or
  generalisation.
* **Same author, same intuitions.** The sufficiency rules (scope cues such as
  "previous", "may", "planned"; person and organisation patterns) were
  committed before the corpus, but the corpus author knew them: the person
  patterns are the generator's lead sentences, and the near misses (9 of the
  20 `fact_missing` holdout targets carry one) use the cues the rules reject.
* **Adversarial cases.** On 37 cases written for the audit with wording the
  generator does not use, F passes 12: it declines 6 of the 13 it should
  decline and answers the other 7 without support (a rejected budget, a
  withdrawn approval, savings taken as the budget, contradictory values, an
  impossible date), and answers 6 of the 24 answerable cases correctly. The
  cases were written by an AI agent to find failures; they are not a random
  sample and not an estimate of performance.
* **Structured metadata.** The check's topicality test uses the records'
  `subject` and `domain` fields, and the gold standard of the filter
  questions is defined from the same `subject` field. Without them, F's
  supporting-source precision is 0.873 instead of 0.952 and its strict
  correctness 62 instead of 70 answerable questions.
* **Dev and holdout share phrasings.** 89 of the 140 in-domain holdout
  questions use a phrasing that also occurs in dev, where three defects were
  found and fixed. The holdout is a frozen set of unseen questions, not an
  independent test of new question formats.
* **Templates.** Questions use 6–12 phrasings per category, each 2–3 times;
  documents come from phrase pools. Paraphrase gaps of real text are
  under-represented, though not absent (they cause most false abstentions).
* **Small samples per category** (20); per-category results are descriptive.
* **Lexical judge.** Strict correctness rewards quoting the exact value or
  title; it does not check that the rest of an answer is right.
* **No LLM.** Configuration L was not run.
