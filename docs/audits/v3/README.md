# Protocol v3 audit

An adversarial audit of protocol v3, run on 2026-11-23 against commit `ff0655c`
(branch `feat/protocol-v3-grounding`) after the v3 holdout had been run once.
This folder keeps the audit's findings, its pre-registered adversarial cases, the
scripts that produced every number below and their recorded outputs.

> **This audit is not independent.** It was carried out by the same AI agent
> that implemented protocol v3, in the same working session. The adversarial
> cases were written by that agent, not by independent human annotators. They
> were **not sampled at random**: they were built to find failures, so their
> pass rates are **not an estimate of performance** on any population of
> questions or records. Treat them as evidence of known failure modes.

Nothing in the v3 protocol was changed because of this audit: the system,
data, configurations, calibration, manifest and stored results are exactly as
frozen (`dirag verify-manifest`). Known defects are catalogued in
[`known-issues.md`](known-issues.md) for a future protocol.

## What was verified

* **Integrity.** `verify-manifest`, `verify-v2` and `verify-v1` pass; 220 tests
  pass on Python 3.12 and 3.13; `ruff` and strict `mypy` are clean; the v3 data
  regenerates byte for byte; nothing tracked by the manifest changed after the
  freeze; the full git history contains no credentials.
* **Order of events.** Plan registered (22:49:03 UTC), calibration (22:49:04),
  dev runs (from 22:49:11), freeze (22:53:58), holdout (22:54:04), from run
  metadata and the session log. This is not externally timestamped: the branch
  had not been pushed.
* **Run count.** 8 configurations (A–H) × 160 holdout questions = 1,280 rows.
* **Metrics.** Every stored holdout metric (A–H) was recomputed with
  independent code (`evidence/scripts/recompute.py`, standard library only)
  and matches to four decimals. One reported effect used the wrong population
  (next section).
* **Leakage.** No inference code reads a reference answer, an expected record,
  a question category, the generator's `metadata` or `ground_truth_signals`.
  It does read the records' structured `subject` and `domain` fields (below).
* **Invariants.** On all 1,792 stored v3 rows (dev-r4 and holdout), every
  ANSWER has ≥ 1 claim citing only admitted records with verified quotes, and
  every ABSTAIN has no claims; there are no ERROR rows. Empty, very long,
  non-English, control-character and SQL-like questions raise no exception.

## Findings

### 1. What the 40/40 in-domain abstention measures

Configurations D, E, F and H abstain on all 40 in-domain unanswerable holdout
questions. Audit-only ablations of F on the holdout (nothing stored, no tuning;
`evidence/scripts/ablate.py`, output `evidence/outputs/ablate.txt`):

| Variant of the sufficiency check (F) | In-domain correct abstention | False abstention |
|---|---:|---:|
| As frozen | 40/40 | 14/100 |
| Without assertion scope (every value treated as FACTUAL) | 39/40 | 13/100 |
| Without assertion scope and without the attribute-word requirement | 30/40 | 9/100 |

In **29 of the 40** target records there is no value of the requested type at
all (no amount, date, person, vendor, regulation or framework), so abstaining
only requires recognising that the type is absent. The attribute-word
requirement accounts for about 10 more. **Assertion-scope classification
(negated, hypothetical, historical) changed the outcome of one question.**
The 40/40 therefore shows that the check rejects records lacking the
requested type of value, and some records whose value is about another
attribute; it does not show robust interpretation of scope.

The check's patterns are also coupled to the corpus generator: the person
patterns (`sufficiency/facts.py:70`) are the generator's three `LEAD_PHRASES`;
the near-miss cues ("previous", "may") are lexicon cues. The corpus was written
after the rules, by the same author.

### 2. Adversarial cases (37, pre-registered)

Expectations were written before any run (`evidence/cases/cases.py`, SHA-256
`79ec94c866919b85289cbb23cb3f3b55ec5c8474e92d2626cd1805942ff24b24`, 01:05:07 UTC)
with new subjects and wording not used by the corpus generator. Results for
configuration F (rules-v3 + sufficiency + extractive-v2), end to end over the
v3 corpus plus the 36 adversarial records:

| Group | Result |
|---|---|
| All cases | **12/37 pass** |
| Cases where the system should decline (13) | **6/13** correct abstentions; **7/13** answered without support |
| Cases where the system should answer (24) | **6/24** answered correctly |

Unsupported answers: a budget the board *rejected* (B04), an approval later
*withdrawn* (B20), *savings* taken as the budget (B14), a vendor's ISO 9001
certificate taken as the decision's framework (B09a, B09b), two contradictory
budgets (B12, the first one is answered), an impossible date (B24). False
abstentions: "funding allocation" for budget, dates written 03/15/2026,
15.03.2026 or "spring 2026", a lead described without a role verb, a
contracted vendor without a known cue, a paraphrased decision title, a
"how much money was set aside" question (configuration C cites the right
record but quotes a sentence without the amount; F abstains). The audit's
first report stated that C answered this case correctly; the recorded output
shows it did not, and this document is corrected accordingly. Per-case detail: `evidence/outputs/cases-summary.txt` and
`cases-results.json`; catalogue: [`known-issues.md`](known-issues.md).

**Fixture defect, corrected post hoc.** Cases C03–C05 first ran with two
adversarial subjects of which one name contains the other ("canteen ordering
app", "staff canteen ordering app"). That exposed a real defect (every record
becomes OFF_TOPIC, known issue 9) but stopped those cases from testing what
they were written for. `evidence/cases/cases_posthoc.py` renames the first
subject only; expectations are unchanged. C03–C05 still fail (outputs
`cases-summary-posthoc.txt`). The counts above use the original run.

### 3. Effect of the generator (F vs E): denominators

Strict correctness rises from 0.53 to 0.70 on the **100 answerable** holdout
questions, the population the plan pre-registered (`docs/v3/protocol.md`,
comparison 3): **+0.17**, 95 % bootstrap interval **[0.10, 0.25]** (17 questions
better, 0 worse; auditor's bootstrap, 5,000 resamples, seed 1).

The stored summary (`experiments/v3/results/holdout/summary.json`, `paired`)
compares the "correct" label over **all 160** questions, unanswerable ones
included: +0.106 [0.063, 0.156] is 17/160. Earlier documentation quoted that
figure next to 0.53 → 0.70. The stored results are unchanged; the
documentation now reports the answerable-only effect and names the
denominator of the stored figure. The same applies to H vs F
(strict correctness 0.48 vs 0.70: −0.22 on 100 answerable, −0.1375 on 160).

### 4. Dependence on structured `subject` and `domain` fields

The sufficiency check's topicality test (`sufficiency/check.py:113-126`) marks a
record OFF_TOPIC when the question names a subject or domain (substring match
against the set of all `subject` and `domain` values, `config.py:401`) and the
record's own `subject`/`domain` field is different. These are structured fields
written by the corpus generator, and the gold standard of the filter questions
is defined from the same `subject` field (records of one subject whose
intended labels satisfy the conditions). The system therefore does **not**
process only unstructured text.

Audit-only ablation of F on the holdout (`evidence/scripts/metadata_ablation.py`):

| F | Supporting-source precision | Strict correctness | False abstention |
|---|---:|---:|---:|
| With `subject`/`domain` (as frozen) | 0.952 | 70/100 | 14/100 |
| Without them (`scopes=()`) | 0.873 | 62/100 | 12/100 |

Definitions as in `evaluation/metrics.py` and the lexical judge: precision is
the mean, over answerable questions answered with ≥ 1 citation, of the share
of cited records that are expected; strict correctness is all key facts and
no forbidden fact in the answer, over the 100 answerable questions (an
abstention is incorrect). The frozen row reproduces the stored F metrics
exactly.

### 5. Dev and holdout share phrasings

Questions are built from 6–12 phrasings per category, used round-robin across
both splits. **89 of the 140 in-domain holdout questions** use a phrasing that
also occurs in the dev split (`evidence/scripts/template_overlap.py`). The
defects fixed on dev (copular "who is the …", masking the record title as a
phrase, a person's name accepted as an organisation) were found on those
phrasings. The v3 holdout is a frozen, unseen set of *questions*; it is not
an independent test of new question *formats*, nor of new authors.

### 6. Classifier and retrieval

| Signal classifier | Generator's own phrase pool (33) | Audit paraphrases (24, labels fixed before the run) |
|---|---:|---:|
| rules-v2 | 24/33 | 9/24 |
| rules-v3 | 28/33 | 10/24 |

Recall@4 on the holdout's 100 answerable questions, decomposing C vs B
(`ablate.py`):

| Intent parser | Signal index | Flat ranking | Topical ranking |
|---|---|---:|---:|
| v2 | rules-v2 | 0.828 | 0.833 |
| v2 | rules-v3 | 0.913 | 0.913 |
| v3 | rules-v2 | 0.833 | 0.838 |
| v3 | rules-v3 | 0.953 | 0.953 |

Most of the gain comes from the scoped signal index (+0.086 with the v2
parser); the v3 parser adds about +0.04 on top; topical ranking adds 0.000–0.005.
The index's advantage over v2 nearly disappears on paraphrases outside the
generator's phrase pool (10/24 vs 9/24), so the Recall@4 gain is specific to
this corpus. Other probes: a doubly negated question ("do not not involve an
external supplier") returns the records *without* a supplier; "What budget
was approved??" (no subject) is answered with one arbitrary record's budget.

## Evidence and provenance

| Path | What | Provenance |
|---|---|---|
| `evidence/cases/cases.py` | 36 adversarial records, 37 cases with expectations and reasons | written before any run; SHA-256 `79ec94c8…f24b24` |
| `evidence/cases/cases_posthoc.py` | same, one subject renamed (fixture defect) | post hoc, after the first run; SHA-256 `e3b84544…daf451a` |
| `evidence/cases/paraphrases.json` | 24 signal paraphrases with labels | written before the run; SHA-256 `861b24ef…de4f85151` |
| `evidence/scripts/run_cases.py` | harness: v3 corpus + adversarial records, configurations A, B, C, E, F, isolated and end-to-end | audit file, unchanged |
| `evidence/scripts/recompute.py` | independent metric recomputation | audit file, unchanged |
| `evidence/scripts/ablate.py` | scope/attribute ablation, type-presence count, retrieval decomposition | audit file, unchanged |
| `evidence/scripts/coupling.py` | classifiers on generator phrases vs paraphrases | audit file (signal-name mapping added during the audit, before results were reported) |
| `evidence/scripts/robust.py`, `sanity.py` | invariants and malformed inputs; harness sanity (64/64 dev-r4 rows reproduced) | audit files, unchanged |
| `evidence/scripts/metadata_ablation.py`, `template_overlap.py`, `query_probes.py` | subject/domain ablation, dev/holdout phrasing overlap, query probes | run inline during the audit; text recovered verbatim from the session log |
| `evidence/outputs/` | recorded outputs of every script | `cases-results*.json` are the audit's original files; the `.txt` files come from a re-run of the same scripts on 2026-11-23 at `ff0655c`, which reproduced every number reported by the audit |

Environment: Python 3.12.16, dependencies from `uv.lock` at `ff0655c`, hashing
embedder, offline, no credentials, no LLM.

## Reproducing

```bash
uv sync --locked
docs/audits/v3/reproduce.sh /tmp/v3-audit   # or no argument: a new temporary directory
```

The script is read-only for the repository, takes well under a minute, and
reports whether each output is identical to the recorded one. It is **not**
part of `make check` or CI and is not expected to "pass": v3 fails most
adversarial cases by design of this audit. A change in its output means the
system's behaviour changed. `make audit-v3` runs the same script.
