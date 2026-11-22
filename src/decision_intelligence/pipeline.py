"""End-to-end question answering: retrieval → evidence gate → [sufficiency] → generation.

With a sufficiency checker (protocol v3), a question whose admitted evidence
does not contain the requested information is answered with ABSTAIN
(outcome ``INSUFFICIENT_EVIDENCE``) before any generator runs, and the
generator only receives the records judged sufficient.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from decision_intelligence.generation.generators import Generator
from decision_intelligence.models import (
    ABSTAIN_TEXT,
    AnswerStatus,
    DecisionDocument,
    GeneratedAnswer,
    Outcome,
    RetrievalResult,
    add_costs,
)
from decision_intelligence.retrieval.retriever import Retriever
from decision_intelligence.sufficiency import EvidenceSufficiencyChecker, Sufficiency, SufficiencyReport


@dataclass(frozen=True)
class PipelineResult:
    question: str
    retrieval: RetrievalResult
    answer: GeneratedAnswer
    latency_ms: float
    cost_usd: float | None
    sufficiency: SufficiencyReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "strategy": self.retrieval.strategy,
            "intent": self.retrieval.intent.to_dict(),
            "retrieved": [d.to_dict() for d in self.retrieval.documents],
            "sufficiency": self.sufficiency.to_dict() if self.sufficiency is not None else None,
            "answer": self.answer.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": self.cost_usd,
        }


class Pipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        documents: dict[str, DecisionDocument],
        checker: EvidenceSufficiencyChecker | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.documents = documents
        self.checker = checker

    def answer(
        self, question: str, retrieval: RetrievalResult
    ) -> tuple[GeneratedAnswer, SufficiencyReport | None]:
        """Sufficiency check (when configured) and generation for an existing retrieval result."""

        report = None
        if self.checker is not None:
            started = time.perf_counter()
            report = self.checker.check(question, retrieval, self.documents)
            if report.status is not Sufficiency.SUFFICIENT:
                answer = GeneratedAnswer(
                    generator=self.generator.name,
                    status=AnswerStatus.ABSTAIN,
                    text=ABSTAIN_TEXT,
                    abstain_reason=f"{report.status.value}: {report.reason}",
                    latency_ms=(time.perf_counter() - started) * 1000,
                    outcome=Outcome.INSUFFICIENT_EVIDENCE,
                )
                return answer, report
        return self.generator.generate(question, retrieval, self.documents, sufficiency=report), report

    def run(self, question: str) -> PipelineResult:
        started = time.perf_counter()
        retrieval = self.retriever.retrieve(question)
        answer, report = self.answer(question, retrieval)
        return PipelineResult(
            question=question,
            retrieval=retrieval,
            answer=answer,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=add_costs(retrieval.intent.cost_usd, answer.cost_usd),
            sufficiency=report,
        )
