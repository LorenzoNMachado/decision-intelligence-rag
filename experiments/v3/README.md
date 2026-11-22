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
exists (see "Threats to validity").

## Holdout (160 questions: 100 answerable, 40 in-domain unanswerable, 20 out of domain)

| Metric | A | B | C | D | E | F | G | H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Retrieval | vector | rules-v2 | rules-v3 | rules-v2 | rules-v3 | rules-v3 | bm25 | vector |
| Sufficiency check | — | — | — | ✓ | ✓ | ✓ | — | ✓ |
| Generator | ext-v1 | ext-v1 | ext-v1 | ext-v1 | ext-v1 | ext-v2 | ext-v1 | ext-v2 |
| Recall@4 | 0.791 | 0.828 | **0.953** | 0.828 | 0.953 | 0.953 | 0.824 | 0.791 |
| Supporting-source precision | 0.306 | 0.343 | 0.424 | 0.943 | 0.952 | 0.952 | 0.300 | 0.949 |
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
| 3 | F vs E | strict correctness | 0.53 → 0.70; 17 better, 0 worse, p < 0.001; +0.106 [0.063, 0.156] | **H3 supported** |
| 4 | G vs A | Recall@4 | 0.791 vs 0.824; 13 vs 9, p = 0.52; +0.033 [−0.032, 0.098] | **no difference (H4)** |

Secondary: sufficiency on the v2 retrieval (D vs B) has the same abstention
effect but more false abstentions (22 vs 14), because the v2 ranking admits
fewer expected records. Without signals (H vs F), Recall@4 is 0.162 lower and
strict correctness 0.138 lower.

## What changed what (retrieval ablation, holdout)

| Strategy | Recall@4 | All expected records admitted | Non-expected records admitted |
|---|---:|---:|---:|
| vector | 0.791 | 0.68 | 434 |
| bm25 | 0.824 | 0.70 | 440 |
| rules-v2 | 0.828 | 0.69 | 403 |
| rules-v3-flat (scoped labels, flat ranking) | 0.953 | 0.90 | 375 |
| rules-v3 (scoped labels, topical ranking) | 0.953 | 0.90 | 376 |

**The Recall@4 gain of C over B comes from the scoped labels and the v3
parser, not from topical eligibility.** On this corpus, rules-v3 and
rules-v3-flat are indistinguishable, and the dev calibration of the band was
flat over 0.04–0.20. The mechanism that lifted foreign records in v2 needs
false labels on the subject's own records (v2 diagnosis). With scoped labels
those are rare, so the band had nothing left to correct. Topical eligibility
stays as a safeguard, but this experiment gives no evidence for it.

Classifiers against the generator's intended labels (all 180 records):
rules-v2 0.661 macro accuracy (third-party 0.444), rules-v3 0.893
(third-party 0.983). This is corpus-level, not a question split. The
third-party difference comes mostly from dependencies stated only through a
named organisation.

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

* **Same author, same intuitions.** The sufficiency rules (scope cues such as
  "previous", "may", "planned"; person and organisation patterns) were
  committed before the corpus, but the corpus author knew them. Every near
  miss in the corpus is the kind of statement the rules were built to reject
  (9 of the 20 `fact_missing` holdout targets carry one). **40/40 in-domain
  abstention is therefore an optimistic estimate**: it shows the mechanism
  works on statements it was designed for, not that it generalises.
* **Templates.** Questions use 6–12 phrasings per category, each 2–3 times;
  documents come from phrase pools. Paraphrase gaps of real text are
  under-represented, though not absent (they cause most false abstentions).
* **Small samples per category** (20); per-category results are descriptive.
* **Lexical judge.** Strict correctness rewards quoting the exact value or
  title; it does not check that the rest of an answer is right.
* **No LLM.** Configuration L was not run.
