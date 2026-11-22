# Contributing

Thanks for your interest. This is a research codebase with a frozen
experimental protocol, so a few rules matter more than usual.

## Setup

```bash
uv sync --locked --extra app   # Python 3.12 or 3.13
make check                     # lint, mypy, tests, corpus and manifest verification
```

No API key, account or paid service is needed for development; the test suite
removes every `DIRAG_*` variable and mocks all providers.

## The frozen protocol

`experiments/v3/manifest.json` hashes everything that can change a result:

* **pipeline group** — `src/decision_intelligence/**` (except the evaluation
  files below), `data/v3/*`, `experiments/v3/configurations/*` and the system
  prompts;
* **evaluation group** — `evaluation/{metrics,judges,report,statistics}.py`,
  `grounding/claim_verification.py` and the judge/verifier prompts;
* the environment — `uv.lock`.

`dirag verify-manifest` must stay green. Therefore:

* Changes to documentation, tests, `app/`, `scripts/`, CI and `pyproject.toml`
  metadata are welcome at any time (check that `uv.lock` does not change).
* A **bug fix in the evaluation group** is recorded with
  `dirag amend-evaluation --note "…"`, which logs old and new hashes. Explain
  the correction in the pull request.
* A change to the **pipeline group or dependencies** cannot be merged into
  protocol v3. It belongs to a new protocol version with new dev and holdout
  data. Open an issue first.
* Protocol v2 (`data/v2`, `experiments/v2`) is historical too and must stay
  byte-identical (`dirag verify-v2`).
* Never use `dirag freeze --force` or edit manifest hashes to make a check pass.
* Never overwrite files under `experiments/*/results/`; commands refuse to.
* Protocol v1 (`data/synthetic`, `experiments/{configurations,results}`,
  `experiments/manifest.json`) is historical and must stay byte-identical
  (`dirag verify-v1`).

## Pull requests

* Keep them small and focused, with tests for behaviour changes.
* Run `make check` (and `uv build`) before pushing.
* Report results only if they were measured, with their protocol, split and
  sample size. Do not add numbers produced with scripted models.
* Never commit credentials, `.env` files or provider output that contains them.
