"""Evidence sufficiency check (protocol v3).

Runs after retrieval and the evidence gate, before generation. For every
admitted record it decides one of:

* ``OFF_TOPIC`` — the record is not about what the question names: for a
  question about one decision ("the decision to adopt the payroll platform"),
  the record's title is not that decision; for a question scoped to a subject
  or domain, the record belongs to another one;
* ``SUFFICIENT`` — the record *asserts* what is asked: a current (FACTUAL)
  value of the expected answer type in a sentence about the requested
  attribute; for a list question, every REQUIRED condition asserted with the
  requested polarity; for a "why" question, a stated reason;
* ``PARTIAL`` — a value of the right type exists but is about another
  attribute, or is only hypothetical, historical or negated;
* ``AMBIGUOUS`` — the record asserts both polarities of a required condition;
* ``INSUFFICIENT`` — nothing in the record answers the question.

The overall status is SUFFICIENT when at least one record is; otherwise
AMBIGUOUS if one record is ambiguous, else INSUFFICIENT. Anything but
SUFFICIENT leads to ABSTAIN: an ambiguous record is a reason to decline, not
to guess.

The checker sees only the question text, the retrieval result and the
records. It never reads a reference answer, an expected record or a category.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from decision_intelligence.embeddings.hashing import tokenize
from decision_intelligence.models import DecisionDocument, Polarity, Requirement, RetrievalResult
from decision_intelligence.semantic_signals.scope import Assertion, scoped_mentions
from decision_intelligence.sufficiency.facts import AnswerType, SentenceFacts, extract_facts
from decision_intelligence.sufficiency.need import InformationNeed, analyse_need, lemmas, related

# Generic value types need the sentence to name the requested attribute; named entities do not
# when the record contains exactly one of them.
_GENERIC_TYPES = frozenset({AnswerType.MONEY, AnswerType.DATE, AnswerType.QUANTITY, AnswerType.PERSON})


class Sufficiency(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"
    AMBIGUOUS = "AMBIGUOUS"
    OFF_TOPIC = "OFF_TOPIC"


@dataclass(frozen=True)
class DocumentSufficiency:
    doc_id: str
    status: Sufficiency
    reason: str
    supporting: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "status": self.status.value,
            "reason": self.reason,
            "supporting": list(self.supporting),
        }


@dataclass(frozen=True)
class SufficiencyReport:
    need: InformationNeed
    status: Sufficiency
    reason: str
    documents: tuple[DocumentSufficiency, ...]

    def sufficient(self) -> tuple[DocumentSufficiency, ...]:
        return tuple(d for d in self.documents if d.status is Sufficiency.SUFFICIENT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "need": self.need.to_dict(),
            "documents": [d.to_dict() for d in self.documents],
        }


def _body(doc: DecisionDocument) -> str:
    return f"{doc.decision} {doc.context} {doc.rationale}"


def _content(text: str) -> set[str]:
    return set(tokenize(text))


class EvidenceSufficiencyChecker:
    name = "sufficiency-v1"

    def __init__(self, entities: Iterable[str] = (), scopes: Iterable[str] = ()) -> None:
        self.entities = tuple(entities)
        self.scopes = tuple(s for s in scopes if s.strip())

    # -- topicality ---------------------------------------------------------------------------

    def _topical(self, need: InformationNeed, doc: DecisionDocument) -> str | None:
        """None when topical, else the reason the record is off topic."""

        question = need.text.lower()
        if need.reference:
            title = _content(doc.title)
            if not title <= _content(need.text):
                return f"the question is about another decision than '{doc.title}'"
            return None
        named = [s for s in self.scopes if s.lower() in question]
        for scope in named:
            if scope.lower() not in {doc.subject.lower(), doc.domain.lower()}:
                return f"the question is scoped to '{scope}'"
        return None

    # -- attribute and reason questions ---------------------------------------------------------

    @staticmethod
    def _focus(need: InformationNeed, doc: DecisionDocument) -> list[str]:
        title = set(lemmas(doc.title))
        return [f for f in need.focus if f not in title]

    @staticmethod
    def _overlap(focus: list[str], sentence: str) -> int:
        words = lemmas(sentence)
        return sum(any(related(f, w) for w in words) for f in focus)

    def _attribute(self, need: InformationNeed, doc: DecisionDocument) -> DocumentSufficiency:
        facts = extract_facts(_body(doc), self.entities)
        focus = self._focus(need, doc)
        candidates: list[tuple[int, SentenceFacts]] = []
        partial = ""
        for sentence in facts:
            typed = sentence.of_type(*need.accepted_types)
            if not typed:
                continue
            factual = [m for m in typed if m.assertion is Assertion.FACTUAL]
            overlap = self._overlap(focus, sentence.text)
            if not factual:
                partial = f"{typed[0].type.value} stated only as {typed[0].assertion.value.lower()}"
                continue
            candidates.append((overlap, sentence))
        types = "/".join(t.value for t in need.accepted_types)
        if not candidates:
            if partial:
                return DocumentSufficiency(doc.id, Sufficiency.PARTIAL, partial)
            return DocumentSufficiency(doc.id, Sufficiency.INSUFFICIENT, f"no {types} value in the record")
        about = [c for c in candidates if c[0] > 0]
        generic = need.answer_type in _GENERIC_TYPES
        if about:
            best = max(about, key=lambda c: c[0])[1]
            return DocumentSufficiency(
                doc.id, Sufficiency.SUFFICIENT, f"{types} value about the requested attribute", (best.text,)
            )
        if not generic and len(candidates) == 1:
            return DocumentSufficiency(
                doc.id, Sufficiency.SUFFICIENT, f"the record names one {types}", (candidates[0][1].text,)
            )
        if not generic:
            return DocumentSufficiency(
                doc.id, Sufficiency.AMBIGUOUS, f"several {types} values, none about the question"
            )
        return DocumentSufficiency(
            doc.id,
            Sufficiency.PARTIAL,
            f"{types} value present but not about {', '.join(focus) or 'the question'}",
        )

    def _reason(self, need: InformationNeed, doc: DecisionDocument) -> DocumentSufficiency:
        for sentence in extract_facts(_body(doc), self.entities):
            if any(m.assertion is Assertion.FACTUAL for m in sentence.of_type(AnswerType.REASON)):
                return DocumentSufficiency(
                    doc.id, Sufficiency.SUFFICIENT, "the record states a reason", (sentence.text,)
                )
        return DocumentSufficiency(doc.id, Sufficiency.INSUFFICIENT, "no stated reason in the record")

    # -- list questions --------------------------------------------------------------------------

    def _list(self, need: InformationNeed, doc: DecisionDocument) -> DocumentSufficiency:
        required = [c for c in need.constraints if c.requirement is Requirement.REQUIRED]
        if not required:
            if need.focus:
                return DocumentSufficiency(
                    doc.id, Sufficiency.INSUFFICIENT, "the question's condition is not understood"
                )
            return DocumentSufficiency(
                doc.id, Sufficiency.SUFFICIENT, "no condition: every record of the scope"
            )
        mentions = scoped_mentions(_body(doc), self.entities)
        supporting: list[str] = []
        for constraint in required:
            own = [m for m in mentions if m.signal is constraint.signal]
            positive = [m for m in own if m.assertion is Assertion.FACTUAL and m.polarity is Polarity.PRESENT]
            negative = [m for m in own if m.assertion is Assertion.NEGATED]
            if positive and negative:
                return DocumentSufficiency(
                    doc.id, Sufficiency.AMBIGUOUS, f"{constraint.signal.value} asserted both ways"
                )
            wanted = positive if constraint.polarity is Polarity.PRESENT else negative
            if not wanted:
                state = "contradicted" if (negative if wanted is positive else positive) else "not asserted"
                return DocumentSufficiency(
                    doc.id,
                    Sufficiency.INSUFFICIENT,
                    f"{constraint.signal.value}={constraint.polarity.value} {state}",
                )
            supporting.append(wanted[0].sentence)
        return DocumentSufficiency(
            doc.id,
            Sufficiency.SUFFICIENT,
            "every required condition is asserted",
            tuple(dict.fromkeys(supporting)),
        )

    # -- entry point -------------------------------------------------------------------------------

    def check(
        self, question: str, retrieval: RetrievalResult, documents: dict[str, DecisionDocument]
    ) -> SufficiencyReport:
        need = analyse_need(question, self.entities)
        admitted = [documents[d.doc_id] for d in retrieval.admitted() if d.doc_id in documents]
        if not admitted:
            return SufficiencyReport(need, Sufficiency.INSUFFICIENT, "no record passed the evidence gate", ())
        if need.kind == "unknown":
            reports = tuple(
                DocumentSufficiency(
                    d.id, Sufficiency.INSUFFICIENT, "the requested information is not understood"
                )
                for d in admitted
            )
            return SufficiencyReport(
                need, Sufficiency.INSUFFICIENT, "the requested information is not understood", reports
            )
        results = []
        for doc in admitted:
            off_topic = self._topical(need, doc)
            if off_topic is not None:
                results.append(DocumentSufficiency(doc.id, Sufficiency.OFF_TOPIC, off_topic))
            elif need.kind == "list":
                results.append(self._list(need, doc))
            elif need.kind == "reason":
                results.append(self._reason(need, doc))
            else:
                results.append(self._attribute(need, doc))
        documents_report = tuple(results)
        if any(r.status is Sufficiency.SUFFICIENT for r in results):
            status, reason = (
                Sufficiency.SUFFICIENT,
                "at least one admitted record contains the requested information",
            )
        elif any(r.status is Sufficiency.AMBIGUOUS for r in results):
            status, reason = Sufficiency.AMBIGUOUS, "the admitted evidence is contradictory or ambiguous"
        else:
            expected = "/".join(t.value for t in need.accepted_types)
            status = Sufficiency.INSUFFICIENT
            reason = f"no admitted record contains the requested {expected}"
            if need.kind == "list":
                reason = "no admitted record asserts the question's conditions"
        return SufficiencyReport(need, status, reason, documents_report)
