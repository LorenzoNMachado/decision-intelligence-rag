"""Integrity checks for the corpus, chunks and question set."""

from __future__ import annotations

import re
import re as _re
from dataclasses import dataclass, field

from decision_intelligence.corpus.questions import CATEGORIES, MISSING
from decision_intelligence.models import Chunk, DecisionDocument, Question, Signal, SignalLabel

_ID_PATTERN = re.compile(r"^DEC-\d{3}$")


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_corpus(
    documents: list[DecisionDocument], chunks: list[Chunk], questions: list[Question] | None = None
) -> ValidationReport:
    report = ValidationReport()
    ids = [d.id for d in documents]
    if len(ids) != len(set(ids)):
        report.errors.append("duplicate document ids")
    titles = [d.title for d in documents]
    if len(titles) != len(set(titles)):
        report.errors.append("duplicate document titles")
    known_signals = {s.value for s in Signal}
    allowed_labels = {SignalLabel.PRESENT.value, SignalLabel.ABSENT.value, SignalLabel.UNCLEAR.value}
    for doc in documents:
        if not _ID_PATTERN.match(doc.id):
            report.errors.append(f"{doc.id}: malformed id")
        for name in ("title", "decision", "context", "rationale"):
            if not getattr(doc, name).strip():
                report.errors.append(f"{doc.id}: empty {name}")
        if set(doc.ground_truth_signals) != known_signals:
            report.errors.append(f"{doc.id}: ground truth must cover the five signals")
        if not set(doc.ground_truth_signals.values()) <= allowed_labels:
            report.errors.append(f"{doc.id}: invalid ground-truth label")
        for key in ("vendor", "regulation", "framework"):
            value = doc.metadata.get(key)
            if value and value.lower() not in doc.full_text().lower():
                report.errors.append(f"{doc.id}: metadata {key} not present in text")

    id_set = set(ids)
    chunk_ids = [c.id for c in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        report.errors.append("duplicate chunk ids")
    covered = {c.doc_id for c in chunks}
    if covered != id_set:
        report.errors.append(f"chunks cover {len(covered)} of {len(id_set)} documents")
    for chunk in chunks:
        if chunk.doc_id not in id_set:
            report.errors.append(f"{chunk.id}: unknown document")
        if not chunk.text.strip():
            report.errors.append(f"{chunk.id}: empty text")

    if questions is not None:
        qids = [q.id for q in questions]
        if len(qids) != len(set(qids)):
            report.errors.append("duplicate question ids")
        by_id = {d.id: d for d in documents}
        for q in questions:
            if q.category not in CATEGORIES:
                report.errors.append(f"{q.id}: unknown category {q.category}")
            if q.split not in {"dev", "holdout"}:
                report.errors.append(f"{q.id}: invalid split {q.split}")
            missing = [d for d in q.expected_doc_ids if d not in id_set]
            if missing:
                report.errors.append(f"{q.id}: unknown expected documents {missing}")
            if q.answerable and not q.expected_doc_ids:
                report.errors.append(f"{q.id}: answerable question without expected documents")
            if not q.answerable and q.expected_doc_ids:
                report.errors.append(f"{q.id}: unanswerable question with expected documents")
            # In-domain unanswerables: the target record must really lack the attribute (gold audit,
            # from the generator's fact table only).
            if q.category in MISSING:
                target = _re.match(r"ABSTAIN: (DEC-\d+) states no (\w+)\.", q.reference_answer)
                if target is None or target.group(1) not in by_id:
                    report.errors.append(f"{q.id}: missing-information question without a valid target")
                elif by_id[target.group(1)].metadata.get(target.group(2)):
                    report.errors.append(f"{q.id}: target {target.group(1)} does state {target.group(2)}")
            # Every key fact must be literally recoverable from the expected evidence,
            # otherwise the reference cannot be reached by any grounded system.
            evidence = " ".join(by_id[d].full_text() for d in q.expected_doc_ids if d in by_id).lower()
            for fact in q.key_facts:
                if fact.lower() not in evidence:
                    report.errors.append(f"{q.id}: key fact {fact!r} absent from expected evidence")
        report.stats["questions"] = len(questions)
        report.stats["dev"] = sum(q.split == "dev" for q in questions)
        report.stats["holdout"] = sum(q.split == "holdout" for q in questions)
        report.stats["unanswerable"] = sum(not q.answerable for q in questions)
        for split in ("dev", "holdout"):
            categories = {q.category for q in questions if q.split == split}
            missing_categories = sorted({q.category for q in questions} - categories)
            if missing_categories:
                report.errors.append(f"{split} split lacks categories: {missing_categories}")

    report.stats["documents"] = len(documents)
    report.stats["chunks"] = len(chunks)
    return report
