"""The documentation figures are generated from the measured results and kept in sync."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("make_figures", ROOT / "scripts" / "make_figures.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_figures_match_the_measured_results() -> None:
    # --check recomputes every value from rows.jsonl, cross-checks summary.json and compares bytes.
    assert _module().main(["--check"]) == 0


def test_figures_use_full_scale_and_state_denominators() -> None:
    figures = _module().render()
    for name, svg in figures.items():
        assert ">0</text>" in svg and ">1</text>" in svg, name  # axis from 0 to 1
        assert "n = " in svg, name
        assert "<desc" in svg and "prefers-color-scheme: dark" in svg, name
