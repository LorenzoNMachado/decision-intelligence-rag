# Protocol v3 — diagnosis of the v2 failures

Reproduced on 2026-11-22 from the v2 code (`src/` at the v2 freeze) and the v2
holdout rows. The v2 holdout is **not** blind any more and was used here only
as diagnostic evidence; nothing in v3 is calibrated on it.

| # | Failure | Mechanism in the code | Evidence from the v2 rows |
|---|---|---|---|
| 1 | All 6 in-domain unanswerable questions were answered | The evidence gate compares only the best-chunk cosine with a threshold (`retriever.py`, `if similarity < evidence_threshold`). A question naming a real record matches it lexically whatever it asks; the extractive generator answers whenever one record is admitted | Every one of the 6 questions had 4 admitted records, best similarity 0.38–0.58 against a threshold of 0.18 |
| 2 | Negated filters: Recall@4 1.000 → 0.333 under B | Signal bonuses are applied over a 20-record candidate pool with no notion of topic (`final = similarity + bonus − penalty`), so a record of another subject whose label matches the constraint can overtake the subject's own records | Under B, 10 of the 24 records admitted for negated filters belonged to another subject |
| 3 | PRESENT overrides ABSENT | `RuleBasedSignalClassifier` keeps the highest-ranked label over all mentions (`rank = {UNCLEAR: 0, ABSENT: 1, PRESENT: 2}`), so one distractor ("a former legal adviser drafted…") outweighs an explicit "there is no legal requirement" | 24 of the 192 intended ABSENT labels were classified PRESENT (37 more UNCLEAR) |
| 4 | Off-topic records promoted | Same as 2: no separation between topical eligibility and constraint compatibility | see 2 |
| 5 | Extractive citations that do not answer | `ExtractiveGenerator` quotes, per admitted record, the sentence with the largest token overlap with the question (title words excluded) — overlap, not the requested fact | On lookups, only 3 of the 14 claims citing the expected record quoted the sentence containing the requested value |
| 6 | Tests | Existing tests covered negation scope, the answer state machine and claim validation, but nothing checked whether admitted evidence contains the requested information, assertion scope (historical, hypothetical, contextual), or topical relevance | — |

Consequences for v3 (implemented in this branch):

1. an **evidence-sufficiency** step between retrieval and generation that
   decides, from the question text and the admitted records only, whether a
   record contains the requested information;
2. **assertion scope** (factual, negated, hypothetical, historical,
   contextual) and a **contradiction** label instead of "PRESENT wins";
3. **topical eligibility** before signal re-ranking;
4. an extractive generator that quotes the sentence the sufficiency check
   found, and abstains otherwise;
5. a new protocol (corpus, questions, dev/holdout) with an ablation that
   isolates each change.
