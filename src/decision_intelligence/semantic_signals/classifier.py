"""Document-level semantic signal classifiers.

``RuleBasedSignalClassifier`` labels each signal mention by mention, using the
local negation and hedging scope of ``semantic_signals.lexicon``: a hedged
mention gives UNCLEAR, a negated mention ABSENT and a plain mention PRESENT.
Across mentions PRESENT wins over ABSENT, and ABSENT over UNCLEAR.

``LLMSignalClassifier`` asks an LLM for the five labels with a verbatim
evidence quote. A label whose evidence is not found in the document is
downgraded to UNCLEAR, and any provider failure yields ERROR for all signals,
which the ranking treats as neutral.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from decision_intelligence.generation.prompts import load_prompt
from decision_intelligence.llm.client import LLMClient, LLMError
from decision_intelligence.models import DecisionDocument, Polarity, Signal, SignalLabel
from decision_intelligence.semantic_signals.lexicon import analyse
from decision_intelligence.semantic_signals.scope import Assertion, scoped_mentions

_SENTENCES = re.compile(r"(?<=[.!?;])\s+")

SignalLabels = dict[Signal, SignalLabel]


@dataclass(frozen=True)
class ClassificationResult:
    labels: SignalLabels
    evidence: dict[Signal, str]
    error: str = ""
    cost_usd: float | None = 0.0


class SignalClassifier(Protocol):
    @property
    def name(self) -> str: ...

    def classify(self, doc: DecisionDocument) -> ClassificationResult: ...


def _body_text(doc: DecisionDocument) -> str:
    return f"{doc.decision} {doc.context} {doc.rationale}"


class RuleBasedSignalClassifier:
    """Labels each signal from its mentions, with local negation and hedging.

    A hedged mention gives UNCLEAR, a negated mention ABSENT and a plain mention
    PRESENT. Across mentions PRESENT wins over ABSENT, and ABSENT over UNCLEAR.
    ``entities`` (department names, job titles, subjects) are masked first.
    """

    name = "rules-v2"

    def __init__(self, entities: Iterable[str] = ()) -> None:
        self.entities = tuple(entities)

    def classify(self, doc: DecisionDocument) -> ClassificationResult:
        rank = {SignalLabel.UNCLEAR: 0, SignalLabel.ABSENT: 1, SignalLabel.PRESENT: 2}
        labels: SignalLabels = dict.fromkeys(Signal, SignalLabel.UNCLEAR)
        evidence: dict[Signal, str] = {}
        for sentence in _SENTENCES.split(_body_text(doc)):
            for mention in analyse(sentence, self.entities):
                if mention.hedged:
                    label = SignalLabel.UNCLEAR
                elif mention.polarity is Polarity.ABSENT:
                    label = SignalLabel.ABSENT
                else:
                    label = SignalLabel.PRESENT
                if rank[label] > rank[labels[mention.signal]]:
                    labels[mention.signal] = label
                    evidence[mention.signal] = sentence
        return ClassificationResult(labels=labels, evidence=evidence)


class ScopedSignalClassifier:
    """Protocol v3 classifier: aggregates *asserted* statements instead of letting PRESENT win.

    Per signal, only FACTUAL (current, positive) and NEGATED (current, negative)
    mentions count (``semantic_signals.scope``); hypothetical, historical and
    contextual mentions do not. The label is PRESENT or ABSENT when the record
    asserts only one polarity, CONTRADICTORY when it asserts both, and UNCLEAR
    otherwise.
    """

    name = "rules-v3"

    def __init__(self, entities: Iterable[str] = ()) -> None:
        self.entities = tuple(entities)

    def classify(self, doc: DecisionDocument) -> ClassificationResult:
        positive: dict[Signal, str] = {}
        negative: dict[Signal, str] = {}
        for mention in scoped_mentions(_body_text(doc), self.entities):
            if mention.assertion is Assertion.FACTUAL and mention.polarity is Polarity.PRESENT:
                positive.setdefault(mention.signal, mention.sentence)
            elif mention.assertion is Assertion.NEGATED:
                negative.setdefault(mention.signal, mention.sentence)
        labels: SignalLabels = {}
        evidence: dict[Signal, str] = {}
        for signal in Signal:
            if signal in positive and signal in negative:
                labels[signal] = SignalLabel.CONTRADICTORY
                evidence[signal] = f"{positive[signal]} | {negative[signal]}"
            elif signal in positive:
                labels[signal], evidence[signal] = SignalLabel.PRESENT, positive[signal]
            elif signal in negative:
                labels[signal], evidence[signal] = SignalLabel.ABSENT, negative[signal]
            else:
                labels[signal] = SignalLabel.UNCLEAR
        return ClassificationResult(labels=labels, evidence=evidence)


def entity_names(documents: Iterable[DecisionDocument]) -> list[str]:
    """Names that identify records rather than describe them: subjects, domains, owner roles."""

    names = {name for d in documents for name in (d.subject, d.domain, d.owner_role)}
    return sorted(n for n in names if n)


SIGNAL_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["signals"],
    "properties": {
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["signal", "label", "evidence"],
                "properties": {
                    "signal": {"type": "string", "enum": [s.value for s in Signal]},
                    "label": {"type": "string", "enum": ["PRESENT", "ABSENT", "UNCLEAR"]},
                    "evidence": {"type": "string"},
                },
            },
        }
    },
}


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


class LLMSignalClassifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.prompt = load_prompt("signal_classification_v1")

    @property
    def name(self) -> str:
        return f"llm-{self.prompt.version}:{self.llm.name}"

    def classify(self, doc: DecisionDocument) -> ClassificationResult:
        text = _body_text(doc)
        user = f"Decision record {doc.id}: {doc.title}\n\n{text}"
        try:
            response = self.llm.complete_json(
                self.prompt.system,
                user,
                SIGNAL_CLASSIFICATION_SCHEMA,
                "signal_classification",
                max_tokens=800,
            )
        except LLMError as exc:
            return ClassificationResult(
                labels=dict.fromkeys(Signal, SignalLabel.ERROR), evidence={}, error=f"{exc.kind}: {exc}"
            )
        labels: SignalLabels = dict.fromkeys(Signal, SignalLabel.UNCLEAR)
        evidence: dict[Signal, str] = {}
        normalised = _normalise(text)
        try:
            items = response.parsed["signals"]
            for item in items:
                signal = Signal(item["signal"])
                label = SignalLabel(item["label"])
                quote = str(item.get("evidence", ""))
                if label is not SignalLabel.UNCLEAR and (not quote or _normalise(quote) not in normalised):
                    label = SignalLabel.UNCLEAR  # unverifiable evidence
                labels[signal] = label
                if quote:
                    evidence[signal] = quote
        except (KeyError, TypeError, ValueError) as exc:
            return ClassificationResult(
                labels=dict.fromkeys(Signal, SignalLabel.ERROR),
                evidence={},
                error=f"schema: {type(exc).__name__}",
                cost_usd=response.cost_usd,
            )
        return ClassificationResult(labels=labels, evidence=evidence, cost_usd=response.cost_usd)
