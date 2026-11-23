# Protocol v3: known issues

Defects found by the v3 audit ([`README.md`](README.md)). **None is fixed in
protocol v3**: v3 is frozen, and changing the pipeline would invalidate its
manifest and holdout. Each item is input for a future protocol, with new data
written by someone who does not know the rules.

Case identifiers refer to `evidence/cases/cases.py`; observed behaviour is
configuration F (rules-v3 + sufficiency + extractive-v2) unless stated, as
recorded in `evidence/outputs/`. Line numbers refer to commit `ff0655c`.
The cases were written by the AI agent that implemented v3 and are not a
random sample: they show that a failure exists, not how often it occurs.

Priority: **High** — the system answers without support, or the defect affects
a reported headline result; **Medium** — false abstention or wrong set on
plausible input; **Low** — cosmetic or rare.

## Unsupported answers (high priority)

### 1. A rejected amount is accepted as the approved one
* **Expected:** "What budget was approved…" → ABSTAIN when the record says the board *rejected* a budget.
* **Observed:** ANSWER "The board rejected a budget of €900,000 for the work."
* **Case:** B04.
* **Module:** `sufficiency/check.py:184-188` (`_attribute`): any focus-word overlap ("budget") with a FACTUAL money value makes the record SUFFICIENT; the status of the decision or of the amount is not considered.
* **Impact:** wrong answer with a valid citation. **Priority:** High.

### 2. A withdrawn approval is treated as current
* **Expected:** ABSTAIN when an approval was later withdrawn.
* **Observed:** ANSWER "A budget of €220,000 was approved in March." (the next sentence withdraws it).
* **Case:** B20.
* **Module:** `sufficiency/check.py` (`_attribute`); `semantic_signals/scope.py` assigns assertion type per sentence, with no relation between sentences.
* **Impact:** outdated value presented as current. **Priority:** High.

### 3. Contradictory values are not detected
* **Expected:** ABSTAIN or flag the conflict when a record states two approved budgets.
* **Observed:** ANSWER with the first value (€200,000); the second (€260,000) is ignored.
* **Case:** B12.
* **Module:** `sufficiency/check.py:184-188`: the best-overlap sentence wins; AMBIGUOUS is only produced for signal polarity (`_list`) and for several named entities without overlap.
* **Impact:** arbitrary choice between conflicting facts. **Priority:** High.

### 4. Savings taken as the budget
* **Expected:** ABSTAIN: the record states expected savings, not a budget.
* **Observed:** ANSWER "The budget owner estimates savings of €75,000 per year."
* **Case:** B14.
* **Module:** `sufficiency/check.py:184-188`: the word "budget" in "budget owner" satisfies the attribute requirement.
* **Impact:** wrong attribute answered. **Priority:** High.

### 5. Right subject, wrong fact
* **Expected:** ABSTAIN when the only matching value belongs to something else (a vendor's certificate is not the decision's framework; a compliance team copied on an e-mail is not a compliance requirement).
* **Observed:** B09a ANSWER "ISO 9001"; B09b lists the record as needed for audit or certification. Also B14.
* **Module:** `sufficiency/check.py:189-192` (a single named entity is SUFFICIENT without attribute overlap); scoped classifier cue "certification".
* **Impact:** unsupported answers on records that are on topic. **Priority:** High.

### 6. Unsupported answers in general
* **Expected:** decline whenever no record asserts the requested information.
* **Observed:** on the 13 adversarial cases that should be declined, 7 were answered (B04, B09a, B09b, B12, B14, B20, B24). The holdout's unsupported-answer rate (0.012 for E and F) does not transfer to wording outside the generator's templates. See also issue 10.
* **Module:** `sufficiency/check.py`, `sufficiency/facts.py`.
* **Impact:** the check's main purpose is only met on template-like text. **Priority:** High.

## Semantic interpretation

### 7. Double negation in a question is read as single negation
* **Expected:** "Which decisions … do **not not** involve an external supplier?" = decisions that involve one (DEC-001).
* **Observed:** returns DEC-002, DEC-003, DEC-005, the records *without* a supplier; the parsed intent is `third_party_dependency = ABSENT`.
* **Case:** `evidence/outputs/query-probes.txt`.
* **Module:** `intent/parsers.py` (`RuleBasedIntentParser`) with the lexicon's negation scope (`semantic_signals/lexicon.py:370`).
* **Impact:** opposite answer set. **Priority:** Medium (unusual phrasing).

### 8. An explicit exception is read as complete absence
* **Expected:** "No external party is involved, except Pinecrest Data, which removes the old barriers." → a third-party dependency exists.
* **Observed:** the record is listed as having no external vendor.
* **Case:** C02 (ADV-35).
* **Module:** exception cues are in `semantic_signals/lexicon.py:245-248` but do not reverse the negated mention in this construction (`semantic_signals/scope.py`).
* **Impact:** wrong record in a negated filter; a documented v3 feature does not work here. **Priority:** Medium.

### 9. Nested subject names make every record OFF_TOPIC
* **Expected:** a question about the "staff canteen ordering app" keeps that subject's records.
* **Observed:** every admitted record is OFF_TOPIC when another subject's name ("canteen ordering app") is a substring of the question's subject.
* **Case:** C03–C05 in `evidence/cases/cases.py` (original run).
* **Module:** `sufficiency/check.py:122-125`: every subject or domain found as a substring of the question must match the record's own field.
* **Impact:** total false abstention for such subjects. **Priority:** Medium.

### 10. A question without a subject gets an arbitrary answer
* **Expected:** "What budget was approved??" → ABSTAIN (which decision?).
* **Observed:** ANSWER with the budget of DEC-097 ("Introduce a budget planning tool"), matched through "budget" in the subject name.
* **Case:** `evidence/outputs/query-probes.txt`, `robust.txt`.
* **Module:** `sufficiency/check.py:113-126`: without a reference or a named scope, every admitted record is topical.
* **Impact:** unsupported answer to an underspecified question. **Priority:** High.

### 11. Synonyms and paraphrases are not recognised
* **Expected / observed:**
  * "The funding allocation … stands at €410,000" (B01) → ABSTAIN; the attribute is not named in the question's words;
  * a paraphrased decision title, "roll out" for "introduce" (B15a) → every record OFF_TOPIC;
  * "How much money was set aside for …" (B15b) → ABSTAIN (configuration C cites the right record but quotes a sentence without the amount);
  * the current budget stated as "Funding of €140,000 has been set aside" (B21) → ABSTAIN;
  * signal paraphrases: rules-v3 10/24 (`coupling.txt`), for example "self-operated", "multi-factor login", "depend on an outside company" (C02, C03, C05).
* **Module:** `sufficiency/need.py` (`analyse_need`, focus words), `sufficiency/check.py:113-121` (title must be a token subset of the question), the v3 lexicon.
* **Impact:** false abstentions on ordinary wording. **Priority:** Medium.

### 12. Valid dates in other formats are not recognised
* **Expected:** 03/15/2026, 15.03.2026 and "spring 2026" are dates.
* **Observed:** no DATE found → ABSTAIN.
* **Case:** B03a, B03b, B03c.
* **Module:** `sufficiency/facts.py:48` (DATE pattern: "D Month YYYY", ISO, quarters, "end of YYYY").
* **Impact:** false abstention. **Priority:** Medium.

### 13. Impossible dates are accepted
* **Expected:** "31 February 2026" is not a valid go-live date → ABSTAIN or flag.
* **Observed:** ANSWER with that date.
* **Case:** B24.
* **Module:** `sufficiency/facts.py:48` (pattern only, no calendar validation).
* **Impact:** invalid fact presented as an answer. **Priority:** Low.

### 14. People and organisations outside the known templates
* **Expected:** "Day-to-day responsibility for the rollout sits with Marta Kowalski" names the lead (B02); "Orchard Systems was contracted to run the service" names the vendor (B06).
* **Observed:** ABSTAIN in both.
* **Module:** `sufficiency/facts.py:70` (`PERSON_PATTERNS`: role verbs identical to the generator's `LEAD_PHRASES`); `sufficiency/check.py:154-169` (an organisation counts only with focus overlap or a known third-party cue); `semantic_signals/scope.py:109`.
* **Impact:** false abstention; evidence of coupling to the generator. **Priority:** Medium.

## Generation

### 15. Compound questions are answered partially
* **Expected:** "Which vendor hosts the service, and at what annual cost, …" → vendor and amount.
* **Observed:** only the vendor sentence is quoted.
* **Case:** B11b.
* **Module:** `sufficiency/need.py:241` (one answer type per question); `generation/generators.py:138` (`ExtractiveGeneratorV2` quotes the supporting sentence(s) of that type).
* **Impact:** incomplete answer judged as an answer. **Priority:** Medium.

### 16. Quotes show former and current values together
* **Expected:** the current lead (Omar Haddad) or the corrected budget (€450,000) alone.
* **Observed:** the whole sentence is quoted: "The work was previously led by Hana Ito; it is now led by Omar Haddad." (B18); "The project budget is not €500,000 as first reported; it is €450,000." (B23). A careful reader gets the right value; the lexical judge would score PARTIAL.
* **Module:** `generation/generators.py:138` (sentence-level quotes).
* **Impact:** low for human readers, misleading for automatic scoring. **Priority:** Low.

### 17. Sentence selection when several sentences compete
* **Expected:** B21: the current amount, in a sentence that does not repeat the question's words; B12: no single choice between contradictory values.
* **Observed:** B21 ABSTAIN (the right sentence has no focus overlap, and the overlapping one is historical); B12 the first matching sentence is quoted.
* **Module:** `sufficiency/check.py:182-188` (largest focus overlap wins); the generator quotes what the check selected.
* **Impact:** wrong or missing evidence in the quote. **Priority:** Medium.

## Other observations

* **Counterfactual legal driver.** "Without this control the company would be in breach of …" is read as hypothetical (B17, also in the holdout); `semantic_signals/lexicon.py` treats "would" as a hedge. Medium.
* **Answer-type noun inside a subject name** ("Which budget planning tool decisions …" read as a money question), found in the holdout. Not reproduced with "cost reporting dashboard" (B16 passes). Medium.
* **Conditional dependency ranked first.** "An external vendor would be engaged only if …" is the top record for "depend on an external vendor" in configurations B and C, while the double-negation and exception records fall outside the top 4 (C01). Medium.
* **Misleading abstention reason.** When every admitted record is OFF_TOPIC, the reason reads "no admitted record contains the requested MONEY" even if the record contains one (`sufficiency/check.py:290`). Low.
* **Coupling of rules and corpus.** The near-miss sentences of the corpus use exactly the cues the rules reject; see the audit's finding 1. A future protocol needs near misses and paraphrases written independently of the rules.
