# Changelog

All notable changes to this project. Results are only ever reported for the
protocol under which they were measured.

## [Unreleased] — v3 audit remediation

Documentation only; protocol v3, its data, code and stored results are unchanged.

### Added
- `docs/audits/v3/`: the adversarial audit of protocol v3 (not independent:
  same AI agent as the implementation), with its pre-registered cases,
  scripts, recorded outputs, `reproduce.sh` (`make audit-v3`, not part of CI)
  and a catalogue of known issues for a future protocol.

### Changed
- README and v3 documentation: all eight configurations in the results
  table; F vs E reported as +0.17 on the 100 answerable questions (the
  stored +0.106 is over all 160); what the 40/40 in-domain abstention does
  and does not show; the check's dependence on structured `subject`/`domain`
  fields; phrasings shared by dev and holdout; the decomposition of the
  retrieval gain; adversarial results. H's supporting-source precision is
  0.948 (0.94846), not 0.949.

## [0.3.0] — protocol v3

### Added
- Evidence sufficiency check between retrieval and generation
  (`decision_intelligence.sufficiency`): information need, typed facts with
  assertion scope, per-record decisions; insufficient evidence → ABSTAIN.
- Assertion scope (factual, negated, hypothetical, historical, contextual) and
  a scoped classifier (rules-v3) with a CONTRADICTORY label.
- Versioned lexicon (v3 adds implicit negation, exceptions, conditional hedges).
- Topical eligibility ranking, BM25 baseline, extractive generator v2.
- Protocol v3: new corpus, 224 questions with in-domain unanswerables that
  reuse the answerable phrasings, configurations A–H (+ L pending), dev-only
  calibration, pre-registered plan, holdout run once.
- Metrics: in/out-of-domain abstention, unsupported answers, Wilson and
  bootstrap intervals, per-category tables.
- `dirag verify-v2`: protocol v2 preserved as a verifiable historical record.

### Changed
- v2 is historical; the code runs v3 only. The v2 semantic layer is kept
  unchanged as ablation configuration B.

## [Unreleased — release preparation]

### Added
- Real interface screenshots and SVG figures generated from the measured v2
  results (`scripts/make_figures.py`, checked by the test suite).
- `docs/results.md`: interpretation of the v2 holdout results.
- `Makefile`, `CONTRIBUTING.md`, `SECURITY.md`, issue templates.
- CI: package build, linting of `scripts/`; security workflow with a
  full-history secret scan (gitleaks) and a dependency audit (pip-audit).

### Changed
- README restructured for first-time readers; technical documentation expanded
  (metric formulas, research method, architecture with optional LLM stages).

No change to the pipeline, the data, the prompts or any frozen result.

## [0.2.0] — protocol v2

### Fixed
- Rule parser: local negation scope, coordination, hedging, entity masking
  (`outside` no longer read as a negation).
- Spending limit fails closed; unknown cost is never zero.
- Explicit answer states; abstentions can no longer be graded correct.
- Validation of claims, provider responses and embeddings; failing rows become
  ERROR rows instead of ending a run.

### Added
- Protocol v2: new corpus (180 records), 70 hand-written questions including
  in-domain unanswerables, proportional dev/holdout split, two-group manifest,
  resumable runs, paired sign tests. Holdout run once for A and B.
- `dirag verify-v1`: protocol v1 preserved as a verifiable historical record.

## [0.1.0] — protocol v1 (historical)

Initial implementation; holdout results for A and B. Superseded after an
independent audit (see `experiments/v1/README.md`).
