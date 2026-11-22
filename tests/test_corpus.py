from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from decision_intelligence.corpus.chunking import chunk_corpus, chunk_document
from decision_intelligence.corpus.generator import generate_corpus
from decision_intelligence.corpus.questions import CATEGORIES, QuestionSpecError, build_question_set
from decision_intelligence.corpus.validation import validate_corpus
from decision_intelligence.models import DecisionDocument


def test_generator_is_deterministic() -> None:
    assert [d.to_dict() for d in generate_corpus("demo")] == [d.to_dict() for d in generate_corpus("demo")]
    assert generate_corpus("demo", seed=1)[0].to_dict() != generate_corpus("demo", seed=2)[0].to_dict()


def test_experimental_corpus_has_180_documents_and_540_chunks(corpus: list[DecisionDocument]) -> None:
    chunks = chunk_corpus(corpus)
    assert len(corpus) == 180
    assert len(chunks) == 540
    assert len(generate_corpus("demo")) == 36
    subjects = Counter(d.subject for d in corpus)
    assert set(subjects.values()) == {6}  # six records per subject


def test_chunking_is_deterministic_and_prefixes_the_title(corpus: list[DecisionDocument]) -> None:
    doc = corpus[0]
    first, second = chunk_document(doc), chunk_document(doc)
    assert first == second
    assert [c.section for c in first] == ["summary", "context", "rationale"]
    assert all(c.text.startswith(doc.title) for c in first)


def test_long_sections_are_split_on_sentence_boundaries(corpus: list[DecisionDocument]) -> None:
    doc = replace(corpus[0], context="First sentence here. " * 60)
    chunks = chunk_document(doc, max_chars=200)
    context_chunks = [c for c in chunks if c.section == "context"]
    assert len(context_chunks) > 1
    assert all(c.text.rstrip().endswith(".") for c in context_chunks)


def test_question_set_shape(corpus: list[DecisionDocument], specs: list[dict[str, Any]]) -> None:
    questions = build_question_set(corpus, specs)
    assert len(questions) == 70
    splits = Counter((q.split, q.answerable) for q in questions)
    assert splits == {("dev", True): 16, ("dev", False): 4, ("holdout", True): 40, ("holdout", False): 10}
    assert all(1 <= len(q.expected_doc_ids) <= 4 for q in questions if q.answerable)
    # Every category appears in both splits, in proportion.
    for category in CATEGORIES:
        counts = Counter(q.split for q in questions if q.category == category)
        assert counts["dev"] >= 1 and counts["holdout"] > counts["dev"], category
    # Phrasings are not reused, and in-domain unanswerables name a real record.
    assert len({q.text for q in questions}) == len(questions)
    titles = {d.title[:1].lower() + d.title[1:] for d in corpus}
    in_domain = [q for q in questions if q.category == "unanswerable_in_domain"]
    assert in_domain and all(any(t in q.text for t in titles) for q in in_domain)


def test_question_set_is_deterministic_and_split_is_disjoint(
    corpus: list[DecisionDocument], specs: list[dict[str, Any]]
) -> None:
    first, second = build_question_set(corpus, specs), build_question_set(corpus, specs)
    assert first == second
    dev = {q.text for q in first if q.split == "dev"}
    holdout = {q.text for q in first if q.split == "holdout"}
    assert dev and holdout and not dev & holdout


def test_question_specs_reject_reused_phrasings(
    corpus: list[DecisionDocument], specs: list[dict[str, Any]]
) -> None:
    with pytest.raises(QuestionSpecError):
        build_question_set(corpus, [specs[0], specs[0]])


def test_validation_passes_on_generated_artefacts(
    corpus: list[DecisionDocument], specs: list[dict[str, Any]]
) -> None:
    report = validate_corpus(corpus, chunk_corpus(corpus), build_question_set(corpus, specs))
    assert report.ok, report.errors


def test_validation_detects_unreachable_key_facts(
    corpus: list[DecisionDocument], specs: list[dict[str, Any]]
) -> None:
    questions = build_question_set(corpus, specs)
    broken = replace(next(q for q in questions if q.answerable), key_facts=("a fact that appears nowhere",))
    report = validate_corpus(corpus, chunk_corpus(corpus), [broken])
    assert any("key fact" in e for e in report.errors)


def test_validation_detects_duplicates_and_missing_chunks(corpus: list[DecisionDocument]) -> None:
    docs = [*corpus[:3], corpus[0]]
    report = validate_corpus(docs, chunk_corpus(corpus[:2]))
    assert "duplicate document ids" in report.errors
    assert any("chunks cover" in e for e in report.errors)
