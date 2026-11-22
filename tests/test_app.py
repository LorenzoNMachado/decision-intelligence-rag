"""Headless test of the Streamlit interface without any provider credentials."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [e.value for e in at.error]
    return "\n".join(str(p) for p in parts)


def test_app_runs_offline_and_explains_missing_provider() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    text = _all_text(at)
    assert "offline demo (no API credentials)" in text
    assert "DIRAG_LLM_" in text
    # LLM-only options are hidden when no provider is configured.
    retrieval_options = list(at.radio[0].options)
    generator_options = list(at.radio[1].options)
    assert any("Strategy 1" in o for o in retrieval_options)
    assert not any("Strategy 3" in o for o in retrieval_options)
    assert not any("LLM" in o for o in generator_options)


def test_app_answers_and_abstains_offline() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.text_input[0].input("Which decisions about the feature flag service have security dependencies?").run()
    at.radio[0].set_value("rules").run()
    at.button[0].click().run()
    assert not at.exception
    assert "ANSWER" in _all_text(at)

    at.text_input[0].input("Which catering company was selected for the staff cafeteria contract?").run()
    at.button[0].click().run()
    assert not at.exception
    text = _all_text(at)
    assert "ABSTAIN" in text and "This is not an abstention" not in text
    assert "no document passed the evidence gate" in text


def test_app_labels_results_by_protocol_and_hides_unmeasured_llm_results() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    text = _all_text(at)
    assert "Protocol v2 · holdout" in text and "Protocol v1 · holdout · A and B (historical)" in text
    assert "have not been executed" in text
    assert "sign test" in text
