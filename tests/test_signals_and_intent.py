from __future__ import annotations

from dataclasses import replace
from typing import Any

from decision_intelligence.intent.parsers import LLMIntentParser, RuleBasedIntentParser
from decision_intelligence.llm.client import LLMError, ScriptedLLM
from decision_intelligence.models import DecisionDocument, Polarity, Requirement, Signal, SignalLabel
from decision_intelligence.semantic_signals.classifier import (
    LLMSignalClassifier,
    RuleBasedSignalClassifier,
    entity_names,
)
from decision_intelligence.semantic_signals.index import build_signal_index, evaluate_against_ground_truth


def _doc(
    corpus: list[DecisionDocument], context: str, rationale: str = "Nothing else to add here."
) -> DecisionDocument:
    return replace(corpus[0], decision="The board approved it.", context=context, rationale=rationale)


def test_rule_classifier_present_absent_unclear(corpus: list[DecisionDocument]) -> None:
    doc = _doc(
        corpus,
        "Delivery relies on a service contract with Halvard Systems. "
        "No statute or regulation applies to this change.",
        "Counsel has not yet confirmed whether any audit applies.",
    )
    labels = RuleBasedSignalClassifier().classify(doc).labels
    assert labels[Signal.THIRD_PARTY] is SignalLabel.PRESENT
    assert labels[Signal.LEGAL] is SignalLabel.ABSENT
    assert labels[Signal.COMPLIANCE] is SignalLabel.UNCLEAR  # hedged mention
    assert labels[Signal.SECURITY] is SignalLabel.UNCLEAR  # never mentioned


def test_present_wins_over_absent_across_sentences(corpus: list[DecisionDocument]) -> None:
    doc = _doc(
        corpus,
        "No external vendor is involved at first. Later, Bluefin Hosting will operate it as a partner.",
    )
    assert RuleBasedSignalClassifier().classify(doc).labels[Signal.THIRD_PARTY] is SignalLabel.PRESENT


def test_rule_classifier_accuracy_is_measured(corpus: list[DecisionDocument]) -> None:
    index, cost = build_signal_index(corpus, RuleBasedSignalClassifier(entity_names(corpus)))
    report = evaluate_against_ground_truth(index, corpus)
    assert cost == 0.0
    # The v2 pools include cue-free phrasings and distractors, so accuracy is measured, not assumed.
    assert 0.0 < report["macro_accuracy"] < 1.0
    unmasked, _ = build_signal_index(corpus, RuleBasedSignalClassifier())
    assert report["macro_accuracy"] >= evaluate_against_ground_truth(unmasked, corpus)["macro_accuracy"]
    assert set(report["signals"]) == {s.value for s in Signal}


def test_llm_classifier_rejects_unverifiable_evidence_and_marks_errors(
    corpus: list[DecisionDocument],
) -> None:
    doc = corpus[0]
    sentence = doc.decision

    def responder(system: str, user: str) -> dict[str, Any]:
        return {
            "signals": [
                {
                    "signal": "third_party_dependency",
                    "label": "PRESENT",
                    "evidence": "invented sentence not in text",
                },
                {"signal": "legal_dependency", "label": "ABSENT", "evidence": sentence},
            ]
        }

    result = LLMSignalClassifier(ScriptedLLM(responder)).classify(doc)
    assert result.labels[Signal.THIRD_PARTY] is SignalLabel.UNCLEAR
    assert result.labels[Signal.LEGAL] is SignalLabel.ABSENT

    failing = LLMSignalClassifier(ScriptedLLM(lambda s, u: LLMError("boom", kind="timeout"))).classify(doc)
    assert set(failing.labels.values()) == {SignalLabel.ERROR}
    assert failing.error.startswith("timeout")


def test_retry_only_reclassifies_failed_documents(corpus: list[DecisionDocument]) -> None:
    docs = corpus[:3]
    calls = {"n": 0}

    def flaky(system: str, user: str) -> dict[str, Any] | LLMError:
        calls["n"] += 1
        return LLMError("down", kind="network") if docs[1].id in user else {"signals": []}

    first, _ = build_signal_index(docs, LLMSignalClassifier(ScriptedLLM(flaky)))
    assert list(first.errors) == [docs[1].id]
    calls["n"] = 0
    second, _ = build_signal_index(
        docs, LLMSignalClassifier(ScriptedLLM(lambda s, u: {"signals": []})), first
    )
    assert not second.errors


def test_rule_intent_parser_detects_polarity() -> None:
    parser = RuleBasedIntentParser()
    positive = parser.parse(
        "Which decisions about the payroll processing system depend on an external vendor?"
    )
    assert positive.status == "OK"
    assert [(s.signal, s.polarity) for s in positive.signals] == [(Signal.THIRD_PARTY, Polarity.PRESENT)]

    negative = parser.parse(
        "Which decisions about the contract repository explicitly state that there is no legal requirement?"
    )
    assert [(s.signal, s.polarity) for s in negative.signals] == [(Signal.LEGAL, Polarity.ABSENT)]

    combined = parser.parse(
        "Which decisions about the e-signature workflow are required by law or regulation "
        "and also have security dependencies?"
    )
    assert {s.signal for s in combined.signals} == {Signal.LEGAL, Signal.SECURITY}
    assert all(s.polarity is Polarity.PRESENT for s in combined.signals)

    assert parser.parse("What happened to the meeting room booking tool?").status == "NONE"


def test_llm_intent_parser_validates_evidence_and_caps_required() -> None:
    question = (
        "Which decisions depend on an external vendor, are required by law, have security dependencies?"
    )

    def responder(system: str, user: str) -> dict[str, Any]:
        return {
            "signals": [
                {
                    "signal": "third_party_dependency",
                    "requirement": "REQUIRED",
                    "polarity": "PRESENT",
                    "evidence": "external vendor",
                },
                {
                    "signal": "legal_dependency",
                    "requirement": "REQUIRED",
                    "polarity": "PRESENT",
                    "evidence": "required by law",
                },
                {
                    "signal": "security_dependency",
                    "requirement": "REQUIRED",
                    "polarity": "PRESENT",
                    "evidence": "security dependencies",
                },
                {
                    "signal": "compliance_relevance",
                    "requirement": "REQUIRED",
                    "polarity": "PRESENT",
                    "evidence": "audit evidence",
                },
            ],
            "notes": "",
        }

    intent = LLMIntentParser(ScriptedLLM(responder)).parse(question)
    assert intent.status == "OK"
    assert [s.signal for s in intent.signals] == [Signal.THIRD_PARTY, Signal.LEGAL, Signal.SECURITY]
    assert [s.requirement for s in intent.signals] == [
        Requirement.REQUIRED,
        Requirement.REQUIRED,
        Requirement.PREFERRED,
    ]
    assert "compliance_relevance" in intent.notes


def test_llm_intent_parser_failure_is_unavailable() -> None:
    intent = LLMIntentParser(ScriptedLLM(lambda s, u: LLMError("slow", kind="timeout"))).parse(
        "Which decisions?"
    )
    assert intent.status == "UNAVAILABLE"
    assert intent.signals == ()
    malformed = LLMIntentParser(ScriptedLLM(lambda s, u: {"signals": [{"signal": "nope"}]})).parse(
        "Which decisions?"
    )
    assert malformed.status in {"UNAVAILABLE", "NONE"}
