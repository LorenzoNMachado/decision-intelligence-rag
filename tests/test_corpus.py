from __future__ import annotations

from collections import Counter
from dataclasses import replace

from decision_intelligence.corpus.chunking import chunk_corpus, chunk_document
from decision_intelligence.corpus.generator import generate_corpus
from decision_intelligence.corpus.questions import build_question_set
from decision_intelligence.corpus.validation import validate_corpus
from decision_intelligence.models import DecisionDocument


def test_generator_is_deterministic() -> None:
    assert [d.to_dict() for d in generate_corpus("demo")] == [d.to_dict() for d in generate_corpus("demo")]
    assert generate_corpus("demo", seed=1)[0].to_dict() != generate_corpus("demo", seed=2)[0].to_dict()


def test_experimental_corpus_has_150_documents_and_450_chunks(corpus: list[DecisionDocument]) -> None:
    chunks = chunk_corpus(corpus)
    assert len(corpus) == 150
    assert len(chunks) == 450
    assert len(generate_corpus("demo")) == 30


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


def test_question_set_shape(corpus: list[DecisionDocument]) -> None:
    questions = build_question_set(corpus)
    assert len(questions) == 50
    splits = Counter((q.split, q.answerable) for q in questions)
    assert splits == {("dev", True): 8, ("dev", False): 2, ("holdout", True): 32, ("holdout", False): 8}
    assert all(1 <= len(q.expected_doc_ids) <= 4 for q in questions if q.answerable)


def test_validation_passes_on_generated_artefacts(corpus: list[DecisionDocument]) -> None:
    report = validate_corpus(corpus, chunk_corpus(corpus), build_question_set(corpus))
    assert report.ok, report.errors


def test_validation_detects_unreachable_key_facts(corpus: list[DecisionDocument]) -> None:
    questions = build_question_set(corpus)
    broken = replace(next(q for q in questions if q.answerable), key_facts=("a fact that appears nowhere",))
    report = validate_corpus(corpus, chunk_corpus(corpus), [broken])
    assert any("key fact" in e for e in report.errors)


def test_validation_detects_duplicates_and_missing_chunks(corpus: list[DecisionDocument]) -> None:
    docs = [*corpus[:3], corpus[0]]
    report = validate_corpus(docs, chunk_corpus(corpus[:2]))
    assert "duplicate document ids" in report.errors
    assert any("chunks cover" in e for e in report.errors)
