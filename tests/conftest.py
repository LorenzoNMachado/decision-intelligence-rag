from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from decision_intelligence.config import Environment, Workspace
from decision_intelligence.corpus.chunking import chunk_corpus
from decision_intelligence.corpus.generator import generate_corpus
from decision_intelligence.corpus.io import save_questions, write_jsonl
from decision_intelligence.corpus.questions import build_question_set
from decision_intelligence.models import DecisionDocument
from decision_intelligence.semantic_signals.classifier import RuleBasedSignalClassifier
from decision_intelligence.semantic_signals.index import build_signal_index

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite hermetic: provider variables of the developer's shell never reach a test.

    Without this, a machine with real DIRAG_LLM_* credentials exported could make
    paid network calls from the CLI tests.
    """

    import os

    for name in list(os.environ):
        if name.startswith("DIRAG_"):
            monkeypatch.delenv(name)


@pytest.fixture(scope="session")
def corpus() -> list[DecisionDocument]:
    return generate_corpus("experimental")


@pytest.fixture()
def workspace(tmp_path: Path, corpus: list[DecisionDocument]) -> Workspace:
    """A throw-away workspace with freshly generated artefacts and the real configurations."""

    ws = Workspace(tmp_path)
    write_jsonl(ws.corpus_path, (d.to_dict() for d in corpus))
    write_jsonl(ws.chunks_path, (c.to_dict() for c in chunk_corpus(corpus)))
    save_questions(ws.questions_path, build_question_set(corpus))
    index, _ = build_signal_index(corpus, RuleBasedSignalClassifier())
    index.save(ws.signal_index_path)
    shutil.copytree(REPO_ROOT / "experiments" / "configurations", ws.config_dir)
    return ws


@pytest.fixture()
def offline_env() -> Environment:
    return Environment()


def write_protocol(ws: Workspace, **retrieval_overrides: object) -> None:
    data = json.loads(ws.protocol_path.read_text())
    data["retrieval"].update(retrieval_overrides)
    ws.protocol_path.write_text(json.dumps(data))
