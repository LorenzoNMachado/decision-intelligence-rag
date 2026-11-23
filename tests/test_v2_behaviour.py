"""The v2 components kept for the v3 ablation must behave exactly as they did under protocol v2."""

from __future__ import annotations

import json
from pathlib import Path

from decision_intelligence.corpus.io import load_documents
from decision_intelligence.semantic_signals.classifier import RuleBasedSignalClassifier, entity_names
from decision_intelligence.semantic_signals.index import build_signal_index

ROOT = Path(__file__).resolve().parents[1]


def test_rules_v2_reproduces_the_frozen_v2_signal_index() -> None:
    documents = load_documents(ROOT / "data" / "v2" / "corpus.jsonl")
    index, _ = build_signal_index(documents, RuleBasedSignalClassifier(entity_names(documents)))
    assert index.to_dict() == json.loads((ROOT / "data" / "v2" / "signal_index.rules.json").read_text())
