"""Protocol v3 retrieval: assertion scope, signal aggregation, topical eligibility, parser, BM25."""

from __future__ import annotations

import numpy as np
import pytest

from decision_intelligence.embeddings.hashing import HashingEmbedder
from decision_intelligence.generation.generators import ExtractiveGenerator, ExtractiveGeneratorV2
from decision_intelligence.intent.parsers import RuleBasedIntentParser
from decision_intelligence.models import (
    ABSTAIN_TEXT,
    AnswerStatus,
    Chunk,
    DecisionDocument,
    IntentSignal,
    Outcome,
    Polarity,
    QueryIntent,
    Requirement,
    Signal,
    SignalLabel,
)
from decision_intelligence.pipeline import Pipeline
from decision_intelligence.retrieval.bm25 import BM25Scorer
from decision_intelligence.retrieval.retriever import RetrievalParams, Retriever
from decision_intelligence.semantic_signals.classifier import (
    RuleBasedSignalClassifier,
    ScopedSignalClassifier,
)
from decision_intelligence.semantic_signals.index import SignalIndex
from decision_intelligence.semantic_signals.lexicon import LEXICON_V3
from decision_intelligence.sufficiency import EvidenceSufficiencyChecker

TP, LEGAL, COMP, SEC = Signal.THIRD_PARTY, Signal.LEGAL, Signal.COMPLIANCE, Signal.SECURITY
P, A = Polarity.PRESENT, Polarity.ABSENT
REQ, PREF = Requirement.REQUIRED, Requirement.PREFERRED


def _doc(doc_id: str, subject: str, body: str, action: str = "Adopt the") -> DecisionDocument:
    return DecisionDocument(
        id=doc_id,
        title=f"{action} {subject}",
        domain="Operations",
        subject=subject,
        date="2025-01-01",
        status="approved",
        owner_role="Head of Operations",
        decision=f"The Head of Operations approved the proposal to {action.lower()} {subject}.",
        context=body,
        rationale="Usage doubled last year.",
    )


# --- scoped classifier -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "signal", "v2", "v3"),
    [
        # A distractor (cue word modifying an organisational unit) is context, not a dependency.
        (
            "The vendor management office reviewed the business case.",
            TP,
            SignalLabel.PRESENT,
            SignalLabel.UNCLEAR,
        ),
        # A former state is not the current one.
        (
            "A former legal adviser drafted the proposal. There is no legal requirement behind it.",
            LEGAL,
            SignalLabel.PRESENT,
            SignalLabel.ABSENT,
        ),
        # v2 has no conditional hedges, so it reads a hypothetical as PRESENT.
        ("If the pilot succeeds, a vendor could be engaged.", TP, SignalLabel.PRESENT, SignalLabel.UNCLEAR),
        # Factual statements on both sides are a contradiction, not a PRESENT.
        (
            "Delivery relies on a vendor. No external vendor is involved in operations.",
            TP,
            SignalLabel.PRESENT,
            SignalLabel.CONTRADICTORY,
        ),
        (
            "Delivery relies on a service contract with Halvard Systems.",
            TP,
            SignalLabel.PRESENT,
            SignalLabel.PRESENT,
        ),
        ("The work is subcontracted to Halvard Systems.", TP, SignalLabel.UNCLEAR, SignalLabel.PRESENT),
        ("The activity is excluded from audit work.", COMP, SignalLabel.PRESENT, SignalLabel.ABSENT),
    ],
)
def test_scoped_aggregation_versus_v2(body: str, signal: Signal, v2: SignalLabel, v3: SignalLabel) -> None:
    doc = _doc("DEC-001", "payroll platform", body)
    assert RuleBasedSignalClassifier().classify(doc).labels[signal] is v2
    assert ScopedSignalClassifier().classify(doc).labels[signal] is v3


# --- parser v3 ---------------------------------------------------------------------------


def _parse(question: str) -> set[tuple[Signal, Polarity, Requirement]]:
    intent = RuleBasedIntentParser(lexicon=LEXICON_V3).parse(question)
    return {(s.signal, s.polarity, s.requirement) for s in intent.signals}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which decisions lack an outside vendor?", {(TP, A, REQ)}),
        ("Which decisions are free of external dependencies?", {(TP, A, REQ)}),
        ("Which decisions are excluded from audit work?", {(COMP, A, REQ)}),
        ("Which decisions are exempt from regulation?", {(LEGAL, A, REQ)}),
        ("All decisions except those involving a vendor?", {(TP, A, REQ)}),
        (
            "Which decisions, other than those with a legal basis, involve a vendor?",
            {(LEGAL, A, REQ), (TP, P, REQ)},
        ),
        ("Which decisions would require an external vendor?", {(TP, P, PREF)}),
        ("Which decisions depend on a vendor or a legal obligation?", {(TP, P, PREF), (LEGAL, P, PREF)}),
        ("Which decisions depend on a vendor and are relevant for audits?", {(TP, P, REQ), (COMP, P, REQ)}),
        # A paraphrase without any lexicon word yields no constraint (documented limitation).
        ("Which decisions hinge on another company?", set()),
    ],
)
def test_parser_v3(question: str, expected: set[tuple[Signal, Polarity, Requirement]]) -> None:
    assert _parse(question) == expected


def test_v2_parser_behaviour_is_unchanged() -> None:
    assert RuleBasedIntentParser().name == "rules-v2"
    assert RuleBasedIntentParser().parse("Which decisions lack an outside vendor?").signals[0].polarity is P


# --- topical eligibility -----------------------------------------------------------------


def _index(docs: list[DecisionDocument], labels: dict[str, SignalLabel]) -> SignalIndex:
    return SignalIndex(
        classifier="test",
        labels={
            d.id: {
                s: labels.get(d.id, SignalLabel.UNCLEAR) if s is TP else SignalLabel.UNCLEAR for s in Signal
            }
            for d in docs
        },
        errors={},
    )


def _retriever(docs: list[DecisionDocument], index: SignalIndex, ranking: str) -> Retriever:
    chunks = [Chunk(f"{d.id}#c0", d.id, "summary", 0, f"{d.title}. {d.context}") for d in docs]
    params = RetrievalParams(
        top_k=2, candidate_pool=20, evidence_threshold=0.0, ranking=ranking, topical_band=0.1
    )
    return Retriever("rules", chunks, HashingEmbedder(), params, signal_index=index)


def test_signal_bonus_cannot_lift_an_off_topic_record_above_the_subject() -> None:
    on = [
        _doc(f"DEC-00{i}", "payroll platform", f"Variant {i} of the payroll platform work.") for i in (1, 2)
    ]
    off = _doc("DEC-009", "loyalty scheme", "No external vendor is involved.")
    docs = [*on, off]
    # As in the v2 failure: the subject's records carry (false) PRESENT labels and are penalised,
    # while the foreign record matches the ABSENT constraint and gets the bonus.
    labels = {"DEC-009": SignalLabel.ABSENT, "DEC-001": SignalLabel.PRESENT, "DEC-002": SignalLabel.PRESENT}
    index = _index(docs, labels)
    intent = QueryIntent("t", "OK", (IntentSignal(TP, REQ, A),))
    question = "Which decisions about the payroll platform involve no outside vendor?"
    flat = _retriever(docs, index, "flat").retrieve(question, intent=intent)
    topical = _retriever(docs, index, "topical").retrieve(question, intent=intent)
    assert "DEC-009" in flat.top_ids()  # the v2 behaviour: the bonus promotes the foreign record
    assert "DEC-009" not in topical.top_ids()


def test_signals_still_reorder_records_within_the_subject_band() -> None:
    docs = [
        _doc(f"DEC-00{i}", "payroll platform", f"Variant {i} of the payroll platform work.")
        for i in (1, 2, 3)
    ]
    index = _index(docs, {"DEC-003": SignalLabel.PRESENT})
    intent = QueryIntent("t", "OK", (IntentSignal(TP, REQ, P),))
    result = _retriever(docs, index, "topical").retrieve("payroll platform vendor", intent=intent)
    assert result.top_ids()[0] == "DEC-003"


def test_contradictory_labels_are_neutral_in_ranking() -> None:
    docs = [_doc(f"DEC-00{i}", "payroll platform", "Same text.") for i in (1, 2)]
    index = _index(docs, {"DEC-001": SignalLabel.CONTRADICTORY})
    intent = QueryIntent("t", "OK", (IntentSignal(TP, REQ, P),))
    result = _retriever(docs, index, "topical").retrieve("payroll platform", intent=intent)
    first = next(d for d in result.documents if d.doc_id == "DEC-001")
    assert first.final_score == first.similarity and first.admitted


# --- BM25 --------------------------------------------------------------------------------


def test_bm25_prefers_rare_matching_terms_and_is_deterministic() -> None:
    chunks = [
        Chunk("a#0", "a", "summary", 0, "payroll platform budget approved"),
        Chunk("b#0", "b", "summary", 0, "payroll platform rollout schedule"),
        Chunk("c#0", "c", "summary", 0, "loyalty scheme budget"),
    ]
    scorer = BM25Scorer(chunks)
    scores = scorer.scores("payroll budget")
    assert scores.shape == (3,) and np.argmax(scores) == 0
    assert np.array_equal(scores, BM25Scorer(chunks).scores("payroll budget"))
    assert scorer.scores("nothing in common").max() == 0.0
    assert scorer.name.startswith("bm25")


# --- generator v2 and pipeline ---------------------------------------------------------------


def _budget_docs() -> dict[str, DecisionDocument]:
    doc = DecisionDocument(
        id="DEC-001",
        title="Adopt the payroll platform",
        domain="People Operations",
        subject="payroll platform",
        date="2025-01-01",
        status="approved",
        owner_role="VP of People",
        decision="The VP of People approved the proposal to adopt the payroll platform.",
        context="The payroll platform budget review was approved by the board for the payroll platform.",
        rationale="The approved budget is €240,000.",
    )
    return {doc.id: doc}


def _pipeline(generator: object, docs: dict[str, DecisionDocument]) -> Pipeline:
    chunks = [
        Chunk(f"{d.id}#c0", d.id, "summary", 0, f"{d.title}. {d.context} {d.rationale}")
        for d in docs.values()
    ]
    retriever = Retriever("vector", chunks, HashingEmbedder(), RetrievalParams(evidence_threshold=0.0))
    checker = EvidenceSufficiencyChecker(entities=("payroll platform",), scopes=("payroll platform",))
    return Pipeline(retriever, generator, docs, checker=checker)  # type: ignore[arg-type]


QUESTION = "What budget was approved for the decision to adopt the payroll platform?"


def test_v2_generator_quotes_the_sentence_that_contains_the_requested_fact() -> None:
    docs = _budget_docs()
    v1 = _pipeline(ExtractiveGenerator(), docs).run(QUESTION).answer
    v2 = _pipeline(ExtractiveGeneratorV2(), docs).run(QUESTION).answer
    assert v1.status is v2.status is AnswerStatus.ANSWER
    assert "€240,000" not in v1.claims[0].quote  # lexical overlap prefers the wordier sentence
    assert v2.claims[0].quote == "The approved budget is €240,000."
    assert v2.claims[0].quote_origin == "extracted" and v2.claims[0].support == "NOT_CHECKED"


def test_insufficient_evidence_abstains_before_generation() -> None:
    docs = _budget_docs()
    doc = docs["DEC-001"]
    docs = {doc.id: DecisionDocument(**{**doc.to_dict(), "rationale": "The steering group met twice."})}
    result = _pipeline(ExtractiveGeneratorV2(), docs).run(QUESTION)
    assert result.answer.status is AnswerStatus.ABSTAIN
    assert result.answer.outcome is Outcome.INSUFFICIENT_EVIDENCE and result.answer.text == ABSTAIN_TEXT
    assert "MONEY" in result.answer.abstain_reason
    assert result.sufficiency is not None and result.to_dict()["sufficiency"]["status"] == "INSUFFICIENT"
