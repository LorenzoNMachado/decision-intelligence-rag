# Protocol v2: results (A and B)

Frozen at commit `58f57ba` (manifest committed in `1674075`, 2026-11-22
20:37 UTC). The holdout was run once (`9137cc0`), eight seconds after the
freeze, with no change to the system before or after. Embedder `hashing-v1`,
evidence threshold 0.18 (dev calibration), correctness by the lexical key-fact
proxy. Each configuration is deterministic and ran once; the question is the
unit of comparison.

**Status: exploratory.** The holdout questions were unseen by the system, but
the author of the system also wrote the corpus, the lexicon and the questions.

## Holdout (50 questions)

| Metric | A · vector | B · rules |
|---|---:|---:|
| Recall@4 (40 answerable) | 0.865 | 0.810 |
| Supporting-source precision | 0.369 | 0.381 |
| Strictly correct / partially correct | 0.050 / 0.675 | 0.100 / 0.525 |
| Abstained, in-domain unanswerable | 0/6 | 0/6 |
| Abstained, out-of-domain unanswerable | 3/4 | 3/4 |
| Abstained, answerable | 0/40 | 0/40 |
| Non-expected records cited | 101 | 98 |
| Questions with an active rule constraint | — | 27/50 |

Per question, A → B: Recall@4 better 6, worse 5, tied 29 (sign test p = 1.00);
strict correctness better 2, worse 0, tied 48 (p = 0.50).

| Recall@4 by category | n | A | B |
|---|---:|---:|---:|
| provider / regulation / framework lookup | 6 / 4 / 4 | 1.000 | 1.000 |
| signal_filter | 10 | 0.708 | 0.842 |
| conjunction_filter | 4 | 0.875 | 1.000 |
| hedged_filter | 3 | 0.667 | 0.750 |
| disjunction_filter | 3 | 0.667 | 0.583 |
| negated_filter | 6 | 1.000 | 0.333 |

Dev (20 questions, used for calibration only): Recall@4 0.797 (A) vs 0.812 (B),
with B better on 1, worse on 1 and tied on 14.

Rule classifier on all 180 records: 78.3 % macro accuracy against the
generator's labels (third-party 72.8 %, operational 73.9 %, legal 83.9 %,
security 72.2 %, compliance 88.9 %).

## Post-run diagnosis (not acted upon)

The following was found **after** the holdout run by reading `rows.jsonl`. It is
reported so it can inform a future protocol, and the system was not changed in
response: doing so would tune v2 on its own holdout.

* **Negated filters fail under Strategy 2** (1.000 → 0.333), for three reasons:
  1. Records whose intended label is ABSENT often also contain a distractor
     ("A former legal adviser drafted the original proposal"). The classifier
     then labels them PRESENT, because PRESENT wins over ABSENT, and the
     contradiction penalty pushes them out of the top four.
  2. The +0.06 bonus for matching ABSENT, applied over a 20-record candidate
     pool, promotes records of **other subjects** (e.g. pricing-model records
     for a contract-repository question), which then pass the gate.
  3. "excluded from audit work" contains no negator the lexicon knows, so it is
     read as compliance PRESENT.
* **Positive filters gain** when the question contains a cue word and the
  relevant records state the signal with cue words. Cue-free questions (9 of 36
  filter questions) produce no constraint, so B falls back to A.
* **The gate cannot reject in-domain unanswerables.** A question that names a
  real record matches it lexically whatever it asks. One out-of-domain question
  ("What menu changes were approved for the staff canteen?") was also admitted,
  through the words "changes" and "approved" (similarity 0.222 > 0.18).

Candidate changes for a future protocol, to be developed on new dev data and
evaluated on a new holdout: restrict signal adjustments to the
similarity band of the subject; do not let a PRESENT mention override an
explicit ABSENT statement; and add a question–evidence coverage check before
answering.
