"""End-to-end question answering: retrieval → evidence gate → generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from decision_intelligence.generation.generators import Generator
from decision_intelligence.models import DecisionDocument, GeneratedAnswer, RetrievalResult, add_costs
from decision_intelligence.retrieval.retriever import Retriever


@dataclass(frozen=True)
class PipelineResult:
    question: str
    retrieval: RetrievalResult
    answer: GeneratedAnswer
    latency_ms: float
    cost_usd: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "strategy": self.retrieval.strategy,
            "intent": self.retrieval.intent.to_dict(),
            "retrieved": [d.to_dict() for d in self.retrieval.documents],
            "answer": self.answer.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": self.cost_usd,
        }


class Pipeline:
    def __init__(
        self, retriever: Retriever, generator: Generator, documents: dict[str, DecisionDocument]
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.documents = documents

    def run(self, question: str) -> PipelineResult:
        started = time.perf_counter()
        retrieval = self.retriever.retrieve(question)
        answer = self.generator.generate(question, retrieval, self.documents)
        return PipelineResult(
            question=question,
            retrieval=retrieval,
            answer=answer,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=add_costs(retrieval.intent.cost_usd, answer.cost_usd),
        )
