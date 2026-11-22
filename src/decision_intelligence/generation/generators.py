"""Answer generators.

``ExtractiveGenerator`` needs no model: for each admitted document it selects
the sentence with the largest token overlap with the question (ignoring words of
the record title) and cites it.
It abstains only when the evidence gate admits nothing, which makes it a
useful baseline for measuring what the gate alone achieves.

``LLMGenerator`` asks an LLM for a structured answer with per-claim citations
and verbatim quotes, then validates every citation against the admitted set
(``grounding.evidence.validate_claims``). An ANSWER with no surviving claim is
converted to ABSTAIN.
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Any, Protocol

from decision_intelligence.embeddings.hashing import tokenize
from decision_intelligence.generation.prompts import load_prompt
from decision_intelligence.grounding.evidence import render_document, validate_claims
from decision_intelligence.llm.client import LLMClient, LLMError
from decision_intelligence.models import (
    AnswerStatus,
    Claim,
    DecisionDocument,
    GeneratedAnswer,
    RetrievalResult,
)

_SENTENCES = re.compile(r"(?<=[.!?])\s+")


class Generator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(
        self, question: str, retrieval: RetrievalResult, documents: dict[str, DecisionDocument]
    ) -> GeneratedAnswer: ...


def _admitted_documents(
    retrieval: RetrievalResult, documents: dict[str, DecisionDocument]
) -> list[DecisionDocument]:
    return [documents[d.doc_id] for d in retrieval.admitted()]


class ExtractiveGenerator:
    name = "extractive-v1"

    def generate(
        self, question: str, retrieval: RetrievalResult, documents: dict[str, DecisionDocument]
    ) -> GeneratedAnswer:
        started = time.perf_counter()
        admitted = _admitted_documents(retrieval, documents)
        if not admitted:
            return GeneratedAnswer(
                generator=self.name,
                status=AnswerStatus.ABSTAIN,
                text="The available records do not contain enough evidence to answer.",
                abstain_reason="no document passed the evidence gate",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        query_tokens = set(tokenize(question))
        lines = []
        claims = []
        for doc in admitted:
            sentences = [s for s in _SENTENCES.split(f"{doc.decision} {doc.context} {doc.rationale}") if s]
            # Words shared with the record title identify the record, not the information
            # requested, so they are ignored when choosing the sentence to quote.
            focus = query_tokens - set(tokenize(doc.title))
            best = max(sentences, key=lambda s: (len(focus & set(tokenize(s))), -sentences.index(s)))
            lines.append(f"- {doc.title} ({doc.id}): {best}")
            claims.append(Claim(statement=best, source_ids=(doc.id,), quote=best, quote_verified=True))
        return GeneratedAnswer(
            generator=self.name,
            status=AnswerStatus.ANSWER,
            text="Relevant decisions:\n" + "\n".join(lines),
            claims=tuple(claims),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "answer", "claims", "abstain_reason"],
    "properties": {
        "status": {"type": "string", "enum": ["ANSWER", "ABSTAIN"]},
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["statement", "source_ids", "quote"],
                "properties": {
                    "statement": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                    "quote": {"type": "string"},
                },
            },
        },
        "abstain_reason": {"type": "string"},
    },
}


class LLMGenerator:
    def __init__(self, llm: LLMClient, enforce_quotes: bool = False, max_tokens: int = 1500) -> None:
        self.llm = llm
        self.enforce_quotes = enforce_quotes
        self.max_tokens = max_tokens
        self.prompt = load_prompt("answer_v1")

    @property
    def name(self) -> str:
        return f"llm-{self.prompt.version}:{self.llm.name}"

    def generate(
        self, question: str, retrieval: RetrievalResult, documents: dict[str, DecisionDocument]
    ) -> GeneratedAnswer:
        started = time.perf_counter()
        admitted = _admitted_documents(retrieval, documents)
        if not admitted:
            # Nothing to ground an answer in: abstain without spending a model call.
            return GeneratedAnswer(
                generator=self.name,
                status=AnswerStatus.ABSTAIN,
                text="The available records do not contain enough evidence to answer.",
                abstain_reason="no document passed the evidence gate",
                prompt_version=self.prompt.version,
            )
        evidence = "\n\n".join(render_document(d) for d in admitted)
        user = f"QUESTION: {question}\n\nEVIDENCE:\n{evidence}"
        try:
            response = self.llm.complete_json(
                self.prompt.system, user, ANSWER_SCHEMA, "grounded_answer", max_tokens=self.max_tokens
            )
        except LLMError as exc:
            return GeneratedAnswer(
                generator=self.name,
                status=AnswerStatus.ERROR,
                text="",
                error=str(exc),
                error_kind=exc.kind,
                latency_ms=(time.perf_counter() - started) * 1000,
                prompt_version=self.prompt.version,
            )
        parsed = response.parsed
        base = GeneratedAnswer(
            generator=self.name,
            status=AnswerStatus.ERROR,
            text="",
            latency_ms=(time.perf_counter() - started) * 1000,
            cost_usd=response.cost_usd,
            prompt_version=self.prompt.version,
        )
        status = str(parsed.get("status", ""))
        answer_text = str(parsed.get("answer", ""))
        if status == "ABSTAIN":
            return replace(
                base,
                status=AnswerStatus.ABSTAIN,
                text=answer_text,
                abstain_reason=str(parsed.get("abstain_reason", "")) or "model abstained",
            )
        if status != "ANSWER" or not isinstance(parsed.get("claims"), list):
            return replace(base, error="schema violation", error_kind="schema")
        raw_claims = []
        for item in parsed["claims"]:
            if not isinstance(item, dict):
                continue
            sources = item.get("source_ids") or []
            raw_claims.append(
                (str(item.get("statement", "")), [str(s) for s in sources], str(item.get("quote", "")))
            )
        by_id = {d.id: d for d in admitted}
        validated = validate_claims(raw_claims, by_id, answer_text, enforce_quotes=self.enforce_quotes)
        if not validated.claims:
            return replace(
                base,
                status=AnswerStatus.ABSTAIN,
                text=answer_text,
                abstain_reason="no claim survived citation validation",
                rejected_citations=validated.rejected_citations,
            )
        return replace(
            base,
            status=AnswerStatus.ANSWER,
            text=answer_text,
            claims=validated.claims,
            rejected_citations=validated.rejected_citations,
        )
