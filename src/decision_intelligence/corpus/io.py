"""Reading and writing corpus artefacts as UTF-8 JSON Lines / JSON."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from decision_intelligence.models import Chunk, DecisionDocument, Question


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_documents(path: Path) -> list[DecisionDocument]:
    return [DecisionDocument.from_dict(row) for row in read_jsonl(path)]


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.from_dict(row) for row in read_jsonl(path)]


def load_questions(path: Path) -> list[Question]:
    return [Question.from_dict(row) for row in read_json(path)["questions"]]


def save_questions(path: Path, questions: list[Question]) -> None:
    write_json(path, {"version": 1, "questions": [q.to_dict() for q in questions]})
