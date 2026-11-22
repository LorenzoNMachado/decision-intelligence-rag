# Changelog

All notable changes to this project. Results are only ever reported for the
protocol under which they were measured.

## [Unreleased]

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
