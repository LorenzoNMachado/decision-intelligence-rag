# Convenience targets. Every target is a thin wrapper around a documented uv command.
Q ?= Which decisions about the feature flag service have security dependencies?

.PHONY: help install ask app test lint typecheck verify figures check

help:  ## list targets
	@grep -E "^[a-z]+:.*##" $(MAKEFILE_LIST) | sed "s/:.*##/ -/"

install:  ## install the package, the demo app and the dev tools (locked versions)
	uv sync --locked --extra app

ask:  ## answer one question offline, e.g. make ask Q="..."
	uv run dirag ask "$(Q)" --retrieval rules

app:  ## start the Streamlit demo on http://localhost:8501
	uv run streamlit run app/streamlit_app.py

test:  ## run the test suite
	uv run pytest

lint:  ## ruff lint and format check
	uv run ruff check src tests app scripts
	uv run ruff format --check src tests app scripts

typecheck:  ## mypy in strict mode
	uv run mypy

verify:  ## corpus integrity and both protocol manifests
	uv run dirag corpus-validate
	uv run dirag verify-manifest
	uv run dirag verify-v1

figures:  ## regenerate docs/assets/*.svg from the measured results
	uv run python scripts/make_figures.py

check: lint typecheck test verify  ## everything CI checks locally, except the build
