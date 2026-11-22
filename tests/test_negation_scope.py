"""Linguistic behaviour of the rule-based intent parser and document classifier.

These cases are general English constructions (negation scope, coordination,
hedging, entity names). They were written to reproduce the defects found by
the 2026-11-22 audit: ``outside`` read as a negation, negation applied to a
whole clause, ``but``/``or`` not delimiting scope, and words inside a subject
name ("supplier onboarding process") read as constraints.
"""

from __future__ import annotations

import pytest

from decision_intelligence.intent.parsers import RuleBasedIntentParser
from decision_intelligence.models import DecisionDocument, Polarity, Requirement, Signal, SignalLabel
from decision_intelligence.semantic_signals.classifier import RuleBasedSignalClassifier

TP, OP, LEGAL, SEC, COMP = (
    Signal.THIRD_PARTY,
    Signal.OPERATIONAL,
    Signal.LEGAL,
    Signal.SECURITY,
    Signal.COMPLIANCE,
)
P, A = Polarity.PRESENT, Polarity.ABSENT
REQ, PREF = Requirement.REQUIRED, Requirement.PREFERRED

ENTITIES = (
    "supplier onboarding process",
    "backup and restore service",
    "Security Engineering",
    "Legal Operations",
    "Chief Information Security Officer",
)


def constraints(question: str) -> set[tuple[Signal, Polarity, Requirement]]:
    intent = RuleBasedIntentParser(entities=ENTITIES).parse(question)
    return {(s.signal, s.polarity, s.requirement) for s in intent.signals}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # "outside" as an adjective is not a negation (audit regression Q-038).
        ("Which decisions rely on an outside supplier or partner?", {(TP, P, REQ)}),
        ("Which decisions depend on outside contractors?", {(TP, P, REQ)}),
        # "outside … scope" is a negation of the signal it scopes.
        ("Which activities fall outside the audit scope?", {(COMP, A, REQ)}),
        # Negation is local: it does not reach constraints on the other side of a coordinator.
        (
            "Which decisions have security dependencies and no legal requirement?",
            {(SEC, P, REQ), (LEGAL, A, REQ)},
        ),
        (
            "Which decisions depend on a vendor but have no security implications?",
            {(TP, P, REQ), (SEC, A, REQ)},
        ),
        (
            "Which decisions do not depend on vendors but are relevant for audits?",
            {(TP, A, REQ), (COMP, P, REQ)},
        ),
        # A negated coordinated noun phrase stays negated as a whole.
        ("Which decisions have no legal or regulatory requirement?", {(LEGAL, A, REQ)}),
        ("Which decisions involve neither vendors nor audits?", {(TP, A, REQ), (COMP, A, REQ)}),
        # Predicate negation reaches back to its subject.
        ("Which decisions say operational concerns played no role?", {(OP, A, REQ)}),
        ("Where was security found to be not relevant?", {(SEC, A, REQ)}),
        ("Which projects were completed without any vendor?", {(TP, A, REQ)}),
        # Inherently negative cues, and their double negation.
        ("Which decisions keep the work in-house?", {(TP, A, REQ)}),
        ("Which decisions are not run in-house?", {(TP, P, REQ)}),
        # A disjunction of different signals cannot make both REQUIRED.
        ("Which decisions involve a vendor or a legal obligation?", {(TP, P, PREF), (LEGAL, P, PREF)}),
        # … but a disjunction inside one signal is still one REQUIRED constraint.
        ("Which decisions are required by law or regulation?", {(LEGAL, P, REQ)}),
        # Hedged wording is a preference, not a requirement.
        ("Which decisions might involve an external vendor?", {(TP, P, PREF)}),
        (
            "Which decisions depend on a vendor and may need a security review?",
            {(TP, P, REQ), (SEC, P, PREF)},
        ),
        # Words inside a known entity name are not constraints (audit regression Q-012).
        (
            "Which decisions about the supplier onboarding process explicitly state that there is no legal "
            "requirement?",
            {(LEGAL, A, REQ)},
        ),
        ("Which decisions about the backup and restore service depend on a vendor?", {(TP, P, REQ)}),
        ("Which decisions about the supplier onboarding process are there?", set()),
    ],
)
def test_rule_parser_scopes(question: str, expected: set[tuple[Signal, Polarity, Requirement]]) -> None:
    assert constraints(question) == expected


def test_contradictory_constraints_are_dropped_and_reported() -> None:
    intent = RuleBasedIntentParser().parse(
        "Which decisions depend on a vendor and do not depend on a vendor?"
    )
    assert intent.signals == ()
    assert intent.status == "NONE"
    assert "contradict" in intent.notes


def test_parser_without_entities_still_reads_positive_questions() -> None:
    intent = RuleBasedIntentParser().parse(
        "Which decisions about the payroll processing system depend on an external vendor?"
    )
    assert [(s.signal, s.polarity, s.requirement) for s in intent.signals] == [(TP, P, REQ)]
    assert RuleBasedIntentParser().parse("What happened to the meeting room booking tool?").status == "NONE"


def _doc(body: str) -> DecisionDocument:
    return DecisionDocument(
        id="DEC-001",
        title="Adopt a backup and restore service",
        domain="Infrastructure",
        subject="backup and restore service",
        date="2024-01-01",
        status="approved",
        owner_role="Head of Platform Engineering",
        decision=body,
        context="",
        rationale="",
    )


@pytest.mark.parametrize(
    ("body", "signal", "label"),
    [
        ("The team will build and run it in-house, with no outside supplier.", TP, SignalLabel.ABSENT),
        ("Delivery relies on Halvard Systems as an outside partner.", TP, SignalLabel.PRESENT),
        ("Delivery relies on a partner, and no regulation applies.", TP, SignalLabel.PRESENT),
        ("Delivery relies on a partner, and no regulation applies.", LEGAL, SignalLabel.ABSENT),
        ("There is no legal or regulatory requirement behind it.", LEGAL, SignalLabel.ABSENT),
        ("Stakeholders from Security Engineering were consulted.", SEC, SignalLabel.UNCLEAR),
        ("The Chief Information Security Officer signed it off.", SEC, SignalLabel.UNCLEAR),
        ("Counsel has not yet confirmed whether any regulation applies.", LEGAL, SignalLabel.UNCLEAR),
        ("Security review found nothing that affects this change.", SEC, SignalLabel.ABSENT),
        ("This item is out of scope for audit and certification work.", COMP, SignalLabel.ABSENT),
    ],
)
def test_rule_classifier_scopes(body: str, signal: Signal, label: SignalLabel) -> None:
    classifier = RuleBasedSignalClassifier(entities=ENTITIES)
    assert classifier.classify(_doc(body)).labels[signal] is label
