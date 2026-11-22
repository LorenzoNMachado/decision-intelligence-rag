"""Core data types shared by every component.

All types are frozen dataclasses or enums so that pipeline stages cannot mutate
their inputs and results can be serialised deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Signal(StrEnum):
    """The five semantic signals studied in this project."""

    THIRD_PARTY = "third_party_dependency"
    OPERATIONAL = "operational_motivation"
    LEGAL = "legal_dependency"
    SECURITY = "security_dependency"
    COMPLIANCE = "compliance_relevance"


class SignalLabel(StrEnum):
    """Classification outcome for one (document, signal) pair.

    ``ERROR`` is reserved for classifier failures (network error, invalid output)
    and is treated as neutral by every ranking component.
    """

    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNCLEAR = "UNCLEAR"
    ERROR = "ERROR"


class Polarity(StrEnum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"


class Requirement(StrEnum):
    REQUIRED = "REQUIRED"
    PREFERRED = "PREFERRED"


class AnswerStatus(StrEnum):
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class DecisionDocument:
    """A synthetic organisational decision record."""

    id: str
    title: str
    domain: str
    subject: str
    date: str
    status: str
    owner_role: str
    decision: str
    context: str
    rationale: str
    metadata: dict[str, str] = field(default_factory=dict)
    # Generator-side truth, used only to evaluate classifiers. Never read by retrieval.
    ground_truth_signals: dict[str, str] = field(default_factory=dict)

    def full_text(self) -> str:
        return f"{self.title}\n{self.decision}\n{self.context}\n{self.rationale}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "domain": self.domain,
            "subject": self.subject,
            "date": self.date,
            "status": self.status,
            "owner_role": self.owner_role,
            "decision": self.decision,
            "context": self.context,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
            "ground_truth_signals": dict(self.ground_truth_signals),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionDocument:
        return cls(
            id=data["id"],
            title=data["title"],
            domain=data["domain"],
            subject=data["subject"],
            date=data["date"],
            status=data["status"],
            owner_role=data["owner_role"],
            decision=data["decision"],
            context=data["context"],
            rationale=data["rationale"],
            metadata=dict(data.get("metadata", {})),
            ground_truth_signals=dict(data.get("ground_truth_signals", {})),
        )


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    section: str
    position: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "section": self.section,
            "position": self.position,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            id=data["id"],
            doc_id=data["doc_id"],
            section=data["section"],
            position=int(data["position"]),
            text=data["text"],
        )


@dataclass(frozen=True)
class Question:
    id: str
    split: str  # "dev" | "holdout"
    category: str
    text: str
    answerable: bool
    expected_doc_ids: tuple[str, ...]
    reference_answer: str
    key_facts: tuple[str, ...] = ()
    # Facts whose presence in an answer indicates an incorrect inclusion.
    forbidden_facts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "category": self.category,
            "text": self.text,
            "answerable": self.answerable,
            "expected_doc_ids": list(self.expected_doc_ids),
            "reference_answer": self.reference_answer,
            "key_facts": list(self.key_facts),
            "forbidden_facts": list(self.forbidden_facts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        return cls(
            id=data["id"],
            split=data["split"],
            category=data["category"],
            text=data["text"],
            answerable=bool(data["answerable"]),
            expected_doc_ids=tuple(data["expected_doc_ids"]),
            reference_answer=data["reference_answer"],
            key_facts=tuple(data.get("key_facts", [])),
            forbidden_facts=tuple(data.get("forbidden_facts", [])),
        )


@dataclass(frozen=True)
class IntentSignal:
    signal: Signal
    requirement: Requirement
    polarity: Polarity
    evidence: str = ""


@dataclass(frozen=True)
class QueryIntent:
    """Semantic constraints extracted from a question.

    ``status`` is ``OK`` when the parser produced a usable interpretation,
    ``NONE`` when it found no constraint and ``UNAVAILABLE`` when it failed;
    in the last case ranking falls back to pure vector similarity.
    """

    parser: str
    status: str
    signals: tuple[IntentSignal, ...] = ()
    notes: str = ""
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    def required(self) -> tuple[IntentSignal, ...]:
        return tuple(s for s in self.signals if s.requirement is Requirement.REQUIRED)

    def preferred(self) -> tuple[IntentSignal, ...]:
        return tuple(s for s in self.signals if s.requirement is Requirement.PREFERRED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser": self.parser,
            "status": self.status,
            "signals": [
                {
                    "signal": s.signal.value,
                    "requirement": s.requirement.value,
                    "polarity": s.polarity.value,
                    "evidence": s.evidence,
                }
                for s in self.signals
            ],
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RetrievedDocument:
    doc_id: str
    rank: int
    similarity: float  # max cosine over the document's chunks
    final_score: float
    best_chunk_id: str
    admitted: bool  # passed the evidence gate
    gate_reason: str
    matched_signals: tuple[str, ...] = ()
    contradicted_signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "rank": self.rank,
            "similarity": round(self.similarity, 6),
            "final_score": round(self.final_score, 6),
            "best_chunk_id": self.best_chunk_id,
            "admitted": self.admitted,
            "gate_reason": self.gate_reason,
            "matched_signals": list(self.matched_signals),
            "contradicted_signals": list(self.contradicted_signals),
        }


@dataclass(frozen=True)
class RetrievalResult:
    strategy: str
    question: str
    intent: QueryIntent
    documents: tuple[RetrievedDocument, ...]

    def top_ids(self) -> list[str]:
        return [d.doc_id for d in self.documents]

    def admitted(self) -> tuple[RetrievedDocument, ...]:
        return tuple(d for d in self.documents if d.admitted)


@dataclass(frozen=True)
class Claim:
    statement: str
    source_ids: tuple[str, ...]
    quote: str
    quote_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "source_ids": list(self.source_ids),
            "quote": self.quote,
            "quote_verified": self.quote_verified,
        }


@dataclass(frozen=True)
class GeneratedAnswer:
    generator: str
    status: AnswerStatus
    text: str
    claims: tuple[Claim, ...] = ()
    abstain_reason: str = ""
    rejected_citations: tuple[str, ...] = ()
    error: str = ""
    error_kind: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    prompt_version: str = ""

    def cited_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for claim in self.claims:
            for source in claim.source_ids:
                seen.setdefault(source, None)
        return list(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "status": self.status.value,
            "text": self.text,
            "claims": [c.to_dict() for c in self.claims],
            "abstain_reason": self.abstain_reason,
            "rejected_citations": list(self.rejected_citations),
            "error": self.error,
            "error_kind": self.error_kind,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": self.cost_usd,
            "prompt_version": self.prompt_version,
        }
