"""Persisted mapping document → signal labels, plus classifier evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_intelligence.corpus.io import read_json, write_json
from decision_intelligence.models import DecisionDocument, Signal, SignalLabel, add_costs
from decision_intelligence.semantic_signals.classifier import SignalClassifier, SignalLabels


@dataclass
class SignalIndex:
    classifier: str
    labels: dict[str, SignalLabels]
    errors: dict[str, str]

    def label(self, doc_id: str, signal: Signal) -> SignalLabel:
        return self.labels.get(doc_id, {}).get(signal, SignalLabel.UNCLEAR)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier": self.classifier,
            "labels": {
                d: {s.value: lab.value for s, lab in v.items()} for d, v in sorted(self.labels.items())
            },
            "errors": dict(sorted(self.errors.items())),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignalIndex:
        return cls(
            classifier=data["classifier"],
            labels={
                d: {Signal(s): SignalLabel(lab) for s, lab in v.items()} for d, v in data["labels"].items()
            },
            errors=dict(data.get("errors", {})),
        )

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> SignalIndex:
        return cls.from_dict(read_json(path))


def build_signal_index(
    documents: list[DecisionDocument], classifier: SignalClassifier, previous: SignalIndex | None = None
) -> tuple[SignalIndex, float | None]:
    """Classify every document. With ``previous``, only documents that had errors are retried."""

    labels: dict[str, SignalLabels] = {}
    errors: dict[str, str] = {}
    cost: float | None = 0.0
    for doc in documents:
        if previous is not None and doc.id in previous.labels and doc.id not in previous.errors:
            labels[doc.id] = previous.labels[doc.id]
            continue
        result = classifier.classify(doc)
        labels[doc.id] = result.labels
        cost = add_costs(cost, result.cost_usd)
        if result.error:
            errors[doc.id] = result.error
    return SignalIndex(classifier=classifier.name, labels=labels, errors=errors), cost


def evaluate_against_ground_truth(index: SignalIndex, documents: list[DecisionDocument]) -> dict[str, Any]:
    """Per-signal accuracy and confusion counts against the generator's intended labels."""

    report: dict[str, Any] = {"classifier": index.classifier, "documents": len(documents), "signals": {}}
    total_correct = 0
    total = 0
    for signal in Signal:
        confusion: dict[str, dict[str, int]] = {}
        correct = 0
        for doc in documents:
            truth = doc.ground_truth_signals[signal.value]
            predicted = index.label(doc.id, signal).value
            confusion.setdefault(truth, {}).setdefault(predicted, 0)
            confusion[truth][predicted] += 1
            correct += truth == predicted
        report["signals"][signal.value] = {
            "accuracy": round(correct / len(documents), 4),
            "confusion_truth_to_predicted": confusion,
        }
        total_correct += correct
        total += len(documents)
    report["macro_accuracy"] = round(total_correct / total, 4) if total else None
    return report
