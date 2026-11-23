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
    assert any("Vector baseline" in o for o in retrieval_options)
    assert not any("LLM" in o for o in retrieval_options)
    assert not any("LLM" in o for o in generator_options)


def _budget_questions() -> tuple[str, str, str]:
    """A budget question the corpus answers, its value, and the same question for a record without one."""

    from decision_intelligence.corpus.io import load_documents

    docs = load_documents(APP.parents[1] / "data" / "v3" / "corpus.jsonl")
    with_budget = next(d for d in docs if d.metadata.get("budget"))
    without = next(d for d in docs if not d.metadata.get("budget"))

    def ask(title: str) -> str:
        return f"What budget was approved for the decision to {title[:1].lower() + title[1:]}?"

    return ask(with_budget.title), with_budget.metadata["budget"], ask(without.title)


def test_app_answers_and_abstains_offline() -> None:
    answerable, value, missing = _budget_questions()
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.text_input[0].input(answerable).run()
    at.button[0].click().run()
    assert not at.exception
    text = _all_text(at)
    assert 'status-ANSWER">ANSWER' in text
    assert value.replace(".", "\\.") in text
    assert "Live demonstration" in text and "Evidence sufficiency (v3 mechanism)" in text

    at.text_input[0].input(missing).run()
    at.button[0].click().run()
    assert not at.exception
    text = _all_text(at)
    assert 'status-ABSTAIN">ABSTAIN' in text and "INSUFFICIENT" in text

    at.text_input[0].input("Which architect designed the new headquarters atrium?").run()
    at.button[0].click().run()
    assert not at.exception
    assert "no record passed the evidence gate" in _all_text(at)


def test_app_labels_results_by_protocol_and_hides_unmeasured_llm_results() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    text = _all_text(at)
    assert "Protocol v3 · holdout" in text
    assert "Protocol v2 · holdout · A and B (historical)" in text
    assert "Protocol v1 · holdout · A and B (historical)" in text
    assert "have not been executed" in text
    assert "sign test" in text
