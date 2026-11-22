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


def test_question_set_shape(corpus: list[DecisionDocument], specs: dict[str, Any]) -> None:
    questions = build_question_set(corpus, specs)
    assert len(questions) == 8 * 28  # sizes fixed in the specification: 8 dev + 20 holdout per category
    for category in CATEGORIES:
        counts = Counter(q.split for q in questions if q.category == category)
        assert counts == {"dev": 8, "holdout": 20}, category
    assert all(1 <= len(q.expected_doc_ids) <= 4 for q in questions if q.answerable)
    assert all(not q.expected_doc_ids for q in questions if not q.answerable)
    assert len({q.text for q in questions}) == len(questions)


def test_missing_information_questions_reuse_the_lookup_phrasings(
    corpus: list[DecisionDocument], specs: dict[str, Any]
) -> None:
    """Answerability must depend on the evidence, not on the wording."""

    questions = build_question_set(corpus, specs)
    templates = [t["text"].split("{title}")[0] for t in specs["categories"]["fact_lookup"]["templates"]]
    missing = [q for q in questions if q.category == "fact_missing"]
    present = [q for q in questions if q.category == "fact_lookup"]
    assert missing and present
    for group in (missing, present):
        assert all(any(q.text.startswith(t) for t in templates) for q in group)
    by_id = {d.id: d for d in corpus}
    for q in missing:
        target, field = q.reference_answer.split()[1], q.reference_answer.split()[-1].rstrip(".")
        assert field not in by_id[target].metadata and by_id[target].title[1:] in q.text


def test_question_set_is_deterministic_and_split_is_disjoint(
    corpus: list[DecisionDocument], specs: dict[str, Any]
) -> None:
    first, second = build_question_set(corpus, specs), build_question_set(corpus, specs)
    assert first == second
    dev = {q.text for q in first if q.split == "dev"}
    holdout = {q.text for q in first if q.split == "holdout"}
    assert dev and holdout and not dev & holdout


def test_question_specs_must_cover_exactly_the_categories(
    corpus: list[DecisionDocument], specs: dict[str, Any]
) -> None:
    broken = {**specs, "categories": {k: v for k, v in specs["categories"].items() if k != "out_of_domain"}}
    with pytest.raises(QuestionSpecError):
        build_question_set(corpus, broken)


def test_validation_passes_on_generated_artefacts(
    corpus: list[DecisionDocument], specs: dict[str, Any]
) -> None:
    report = validate_corpus(corpus, chunk_corpus(corpus), build_question_set(corpus, specs))
    assert report.ok, report.errors


def test_validation_detects_unreachable_key_facts(
    corpus: list[DecisionDocument], specs: dict[str, Any]
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
