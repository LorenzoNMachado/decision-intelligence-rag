"""Document-level semantic signal classifiers.

``RuleBasedSignalClassifier`` labels each signal sentence by sentence: a hedged
mention gives UNCLEAR, a negated mention ABSENT and a plain mention PRESENT.
Across sentences PRESENT wins over ABSENT, and ABSENT over UNCLEAR.

``LLMSignalClassifier`` asks an LLM for the five labels with a verbatim
evidence quote. A label whose evidence is not found in the document is
downgraded to UNCLEAR, and any provider failure yields ERROR for all signals,
which the ranking treats as neutral.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from decision_intelligence.generation.prompts import load_prompt
from decision_intelligence.llm.client import LLMClient, LLMError
from decision_intelligence.models import DecisionDocument, Signal, SignalLabel
from decision_intelligence.semantic_signals.lexicon import is_hedged, is_negated, signals_mentioned

_SENTENCES = re.compile(r"(?<=[.!?;])\s+")

SignalLabels = dict[Signal, SignalLabel]


@dataclass(frozen=True)
class ClassificationResult:
    labels: SignalLabels
    evidence: dict[Signal, str]
    error: str = ""
    cost_usd: float = 0.0


class SignalClassifier(Protocol):
    @property
    def name(self) -> str: ...

    def classify(self, doc: DecisionDocument) -> ClassificationResult: ...


def _body_text(doc: DecisionDocument) -> str:
    return f"{doc.decision} {doc.context} {doc.rationale}"


class RuleBasedSignalClassifier:
    name = "rules-v1"

    def classify(self, doc: DecisionDocument) -> ClassificationResult:
        rank = {SignalLabel.UNCLEAR: 0, SignalLabel.ABSENT: 1, SignalLabel.PRESENT: 2}
        labels: SignalLabels = dict.fromkeys(Signal, SignalLabel.UNCLEAR)
        evidence: dict[Signal, str] = {}
        for sentence in _SENTENCES.split(_body_text(doc)):
            for signal, _cue in signals_mentioned(sentence):
                if is_hedged(sentence):
                    label = SignalLabel.UNCLEAR
                elif is_negated(sentence):
                    label = SignalLabel.ABSENT
                else:
                    label = SignalLabel.PRESENT
                if rank[label] > rank[labels[signal]]:
                    labels[signal] = label
                    evidence[signal] = sentence
        return ClassificationResult(labels=labels, evidence=evidence)


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
