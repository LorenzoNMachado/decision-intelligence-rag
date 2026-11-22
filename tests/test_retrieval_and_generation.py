from __future__ import annotations

from typing import Any

import numpy as np

from decision_intelligence.corpus.chunking import chunk_corpus
from decision_intelligence.embeddings.hashing import HashingEmbedder
from decision_intelligence.generation.generators import ExtractiveGenerator, LLMGenerator
from decision_intelligence.grounding.evidence import RawClaim, quote_in, render_document, validate_claims
from decision_intelligence.intent.parsers import RuleBasedIntentParser
from decision_intelligence.llm.client import LLMError, ScriptedLLM
from decision_intelligence.models import (
    AnswerStatus,
    DecisionDocument,
    IntentSignal,
    Polarity,
    QueryIntent,
    Requirement,
    Signal,
    SignalLabel,
)
from decision_intelligence.retrieval.retriever import RetrievalParams, Retriever
from decision_intelligence.semantic_signals.classifier import RuleBasedSignalClassifier
from decision_intelligence.semantic_signals.index import SignalIndex, build_signal_index


def test_hashing_embedder_is_normalised_and_deterministic() -> None:
    embedder = HashingEmbedder(dimension=256)
    a = embedder.embed(["external vendor contract", "external vendor contract", "encryption at rest"])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(a[0], a[1])
    assert float(a[0] @ a[2]) < float(a[0] @ a[1])


def _retriever(
    corpus: list[DecisionDocument], strategy: str, threshold: float = 0.18, **kw: Any
) -> Retriever:
    index, _ = build_signal_index(corpus, RuleBasedSignalClassifier())
    return Retriever(
        strategy=strategy,
        chunks=chunk_corpus(corpus),
        embedder=HashingEmbedder(),
        params=RetrievalParams(evidence_threshold=threshold, **kw),
        intent_parser=RuleBasedIntentParser() if strategy == "rules" else None,
        signal_index=None if strategy == "vector" else index,
    )


def test_vector_retrieval_returns_top_k_sorted(corpus: list[DecisionDocument]) -> None:
    result = _retriever(corpus, "vector").retrieve(corpus[5].title)
    assert len(result.documents) == 4
    assert result.documents[0].doc_id == corpus[5].id
    scores = [d.final_score for d in result.documents]
    assert scores == sorted(scores, reverse=True)


def test_gate_rejects_low_similarity(corpus: list[DecisionDocument]) -> None:
    result = _retriever(corpus, "vector", threshold=0.99).retrieve("quantum lab sponsorship")
    assert result.admitted() == ()
    assert {d.gate_reason for d in result.documents} == {"below_threshold"}


def test_rules_without_constraints_equal_vector_baseline(corpus: list[DecisionDocument]) -> None:
    question = "What happened to the meeting room booking tool?"
    vector = _retriever(corpus, "vector").retrieve(question)
    rules = _retriever(corpus, "rules").retrieve(question)
    assert rules.intent.status == "NONE"
    assert [(d.doc_id, d.final_score, d.admitted) for d in rules.documents] == [
        (d.doc_id, d.final_score, d.admitted) for d in vector.documents
    ]


def test_signal_constraints_rerank_and_veto(corpus: list[DecisionDocument]) -> None:
    retriever = _retriever(corpus, "rules", veto_contradictions=True)
    subject = corpus[0].subject
    question = f"Which decisions about the {subject} depend on an external vendor?"
    result = retriever.retrieve(question)
    assert retriever.signal_index is not None
    for d in result.documents:
        label = retriever.signal_index.label(d.doc_id, Signal.THIRD_PARTY)
        if label is SignalLabel.PRESENT:
            assert "third_party_dependency" in d.matched_signals
        if label is SignalLabel.ABSENT:
            assert d.gate_reason in {"signal_contradiction", "below_threshold"}
            assert not d.admitted


def test_unclear_and_error_labels_are_neutral(corpus: list[DecisionDocument]) -> None:
    doc = corpus[0]
    labels = {doc.id: dict.fromkeys(Signal, SignalLabel.ERROR)}
    retriever = _retriever(corpus, "rules")
    retriever.signal_index = SignalIndex("test", labels=labels, errors={})
    intent = QueryIntent("t", "OK", (IntentSignal(Signal.LEGAL, Requirement.REQUIRED, Polarity.PRESENT),))
    result = retriever.retrieve(doc.title, intent=intent)
    top = next(d for d in result.documents if d.doc_id == doc.id)
    assert top.final_score == top.similarity
    assert top.matched_signals == () and top.contradicted_signals == ()


def test_quote_verification_is_literal_and_normalised(corpus: list[DecisionDocument]) -> None:
    text = render_document(corpus[0])
    sentence = corpus[0].decision
    assert quote_in(sentence.upper(), [text])
    assert quote_in("  " + sentence.replace(" ", "   ") + " ", [text])
    assert not quote_in("A sentence that is not in the document at all.", [text])
    assert not quote_in("the", [text])  # too short to be evidence


def test_claim_validation_removes_unadmitted_and_leaked_ids(corpus: list[DecisionDocument]) -> None:
    admitted = {corpus[0].id: corpus[0]}
    validated = validate_claims(
        [
            RawClaim("ok", (corpus[0].id, "DEC-999"), corpus[0].decision),
            RawClaim("only foreign", ("DEC-777",), "anything"),
            RawClaim("bad quote", (corpus[0].id,), "This quote was invented by the model."),
        ],
        admitted,
        discarded_text=f"See {corpus[0].id} and also DEC-555.",
        enforce_quotes=False,
    )
    assert [c.statement for c in validated.claims] == ["ok", "bad quote"]
    assert [c.quote_verified for c in validated.claims] == [True, False]
    assert set(validated.rejected_citations) == {"DEC-999", "DEC-777", "DEC-555"}
    enforced = validate_claims(
        [RawClaim("bad", (corpus[0].id,), "invented quote text here")], admitted, "", enforce_quotes=True
    )
    assert enforced.claims == () and enforced.dropped_claims == 1


def test_extractive_generator_answers_or_abstains(corpus: list[DecisionDocument]) -> None:
    documents = {d.id: d for d in corpus}
    retriever = _retriever(corpus, "vector")
    answer = ExtractiveGenerator().generate(corpus[3].title, retriever.retrieve(corpus[3].title), documents)
    assert answer.status is AnswerStatus.ANSWER
    assert all(c.quote_verified for c in answer.claims)
    gated = _retriever(corpus, "vector", threshold=0.99).retrieve("unrelated question about drones")
    assert ExtractiveGenerator().generate("q", gated, documents).status is AnswerStatus.ABSTAIN


def test_llm_generator_validates_output(corpus: list[DecisionDocument]) -> None:
    documents = {d.id: d for d in corpus}
    retrieval = _retriever(corpus, "vector").retrieve(corpus[3].title)
    admitted = retrieval.admitted()[0].doc_id

    def answer(system: str, user: str) -> dict[str, Any]:
        assert "EVIDENCE" in user and admitted in user
        return {
            "status": "ANSWER",
            "answer": f"Per {admitted} and DEC-999.",
            "claims": [
                {"statement": "s1", "source_ids": [admitted], "quote": documents[admitted].decision},
                {"statement": "s2", "source_ids": ["DEC-999"], "quote": "made up"},
            ],
            "abstain_reason": "",
        }

    result = LLMGenerator(ScriptedLLM(answer)).generate("q", retrieval, documents)
    assert result.status is AnswerStatus.ANSWER
    assert [c.source_ids for c in result.claims] == [(admitted,)]
    assert "DEC-999" in result.rejected_citations


def test_llm_generator_abstains_errors_and_skips_calls(corpus: list[DecisionDocument]) -> None:
    documents = {d.id: d for d in corpus}
    retrieval = _retriever(corpus, "vector").retrieve(corpus[3].title)
    invented = {
        "status": "ANSWER",
        "answer": "x",
        "claims": [{"statement": "s", "source_ids": ["DEC-999"], "quote": "q"}],
        "abstain_reason": "",
    }
    assert (
        LLMGenerator(ScriptedLLM(lambda s, u: invented)).generate("q", retrieval, documents).status
        is AnswerStatus.ABSTAIN
    )
    abstain = {"status": "ABSTAIN", "answer": "", "claims": [], "abstain_reason": "not in evidence"}
    result = LLMGenerator(ScriptedLLM(lambda s, u: abstain)).generate("q", retrieval, documents)
    assert result.status is AnswerStatus.ABSTAIN and result.abstain_reason == "not in evidence"
    error = LLMGenerator(ScriptedLLM(lambda s, u: LLMError("t/o", kind="timeout"))).generate(
        "q", retrieval, documents
    )
    assert error.status is AnswerStatus.ERROR and error.error_kind == "timeout"

    scripted = ScriptedLLM(lambda s, u: abstain)
    gated = _retriever(corpus, "vector", threshold=0.99).retrieve("drones")
    assert LLMGenerator(scripted).generate("q", gated, documents).status is AnswerStatus.ABSTAIN
    assert scripted.calls == []  # no admitted evidence ⇒ no model call
