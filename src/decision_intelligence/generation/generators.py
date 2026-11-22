"""Answer generators.

``ExtractiveGenerator`` needs no model: for each admitted document it selects
the sentence with the largest token overlap with the question (ignoring words of
the record title) and cites it. Its quotes are copied from the record, so they
are verified by construction (``quote_origin="extracted"``). It abstains only
when the evidence gate admits nothing, which makes it a useful baseline for
measuring what the gate alone achieves.

``LLMGenerator`` asks an LLM for a structured answer with per-claim citations
and verbatim quotes, then validates every claim deterministically
(``grounding.evidence.validate_claims``). The answer shown and judged is
composed from the surviving claims only; the model's free-text ``answer`` is
kept as ``discarded_text`` for audit. Final states follow
``models.TRANSITIONS``: an answer with no surviving claim becomes ABSTAIN, and
provider or schema failures are ERROR, never ABSTAIN.
"""

from __future__ import annotations

import re
import time
from typing import Any, Protocol

from decision_intelligence.embeddings.hashing import tokenize
from decision_intelligence.generation.prompts import load_prompt
from decision_intelligence.grounding.evidence import (
    RawClaim,
    compose_answer,
    render_document,
    validate_claims,
)
from decision_intelligence.llm.client import LLMClient, LLMError
from decision_intelligence.models import (
    ABSTAIN_TEXT,
    AnswerStatus,
    Claim,
    DecisionDocument,
    GeneratedAnswer,
    Outcome,
    RetrievalResult,
)

_SENTENCES = re.compile(r"(?<=[.!?])\s+")
MAX_DISCARDED_CHARS = 2000
MAX_REASON_CHARS = 300
UNBILLED_ERRORS = frozenset({"budget", "config"})


class Generator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(
        self, question: str, retrieval: RetrievalResult, documents: dict[str, DecisionDocument]
    ) -> GeneratedAnswer: ...


def _admitted_documents(
    retrieval: RetrievalResult, documents: dict[str, DecisionDocument]
) -> list[DecisionDocument]:
    return [documents[d.doc_id] for d in retrieval.admitted() if d.doc_id in documents]


def _no_evidence(generator: str, started: float, prompt_version: str = "") -> GeneratedAnswer:
    return GeneratedAnswer(
        generator=generator,
        status=AnswerStatus.ABSTAIN,
        text=ABSTAIN_TEXT,
        abstain_reason="no document passed the evidence gate",
        latency_ms=(time.perf_counter() - started) * 1000,
        prompt_version=prompt_version,
        outcome=Outcome.NO_EVIDENCE,
    )


class ExtractiveGenerator:
    name = "extractive-v1"

    def generate(
        self, question: str, retrieval: RetrievalResult, documents: dict[str, DecisionDocument]
    ) -> GeneratedAnswer:
        started = time.perf_counter()
        query_tokens = set(tokenize(question))
        lines = []
        claims = []
        for doc in _admitted_documents(retrieval, documents):
            text = f"{doc.decision} {doc.context} {doc.rationale}"
            sentences = [s for s in _SENTENCES.split(text) if s.strip()]
            if not sentences:
                continue
            # Words shared with the record title identify the record, not the information
            # requested, so they are ignored when choosing the sentence to quote.
            focus = query_tokens - set(tokenize(doc.title))
            best = max(
                enumerate(sentences), key=lambda item: (len(focus & set(tokenize(item[1]))), -item[0])
            )[1]
            lines.append(f"- {doc.title} ({doc.id}): {best}")
            claims.append(
                Claim(
                    statement=best,
                    source_ids=(doc.id,),
                    quote=best,
                    quote_verified=True,
                    quote_origin="extracted",
                )
            )
        if not claims:
            return _no_evidence(self.name, started)
        return GeneratedAnswer(
            generator=self.name,
            status=AnswerStatus.ANSWER,
            text="Relevant decisions:\n" + "\n".join(lines),
            claims=tuple(claims),
            latency_ms=(time.perf_counter() - started) * 1000,
            outcome=Outcome.EXTRACTED,
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


def _raw_claims(items: list[Any]) -> list[RawClaim]:
    """Keep well-typed claim items; malformed ones are ignored rather than fatal."""

    claims = []
    for item in items:
        if not isinstance(item, dict):
            continue
        statement, quote, sources = item.get("statement"), item.get("quote"), item.get("source_ids")
        if not isinstance(statement, str) or not isinstance(quote, str) or not isinstance(sources, list):
            continue
        claims.append(RawClaim(statement, tuple(s for s in sources if isinstance(s, str)), quote))
    return claims


class LLMGenerator:
    def __init__(self, llm: LLMClient, enforce_quotes: bool = True, max_tokens: int = 1500) -> None:
        self.llm = llm
        self.enforce_quotes = enforce_quotes
        self.max_tokens = max_tokens
        self.prompt = load_prompt("answer_v2")

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
            return _no_evidence(self.name, started, self.prompt.version)
        evidence = "\n\n".join(render_document(d) for d in admitted)
        user = f"QUESTION: {question}\n\nEVIDENCE:\n{evidence}"
        common: dict[str, Any] = {"generator": self.name, "prompt_version": self.prompt.version}
        try:
            response = self.llm.complete_json(
                self.prompt.system, user, ANSWER_SCHEMA, "grounded_answer", max_tokens=self.max_tokens
            )
        except LLMError as exc:
            return GeneratedAnswer(
                status=AnswerStatus.ERROR,
                text="",
                error=str(exc),
                error_kind=exc.kind,
                latency_ms=(time.perf_counter() - started) * 1000,
                outcome=Outcome.PROVIDER_ERROR,
                # A call refused before sending costs nothing; any other failure may have been
                # billed, so its cost is unknown (the client charged its budget conservatively).
                cost_usd=0.0 if exc.kind in UNBILLED_ERRORS else None,
                **common,
            )
        common["cost_usd"] = response.cost_usd
        parsed = response.parsed
        status, answer_text, items = parsed.get("status"), parsed.get("answer", ""), parsed.get("claims")
        if not isinstance(answer_text, str):
            answer_text = None
        if (
            status not in ("ANSWER", "ABSTAIN")
            or answer_text is None
            or (status == "ANSWER" and not isinstance(items, list))
        ):
            return GeneratedAnswer(
                status=AnswerStatus.ERROR,
                text="",
                error="response violates the answer schema",
                error_kind="schema",
                latency_ms=(time.perf_counter() - started) * 1000,
                outcome=Outcome.SCHEMA_ERROR,
                **common,
            )
        discarded = answer_text[:MAX_DISCARDED_CHARS]
        if status == "ABSTAIN":
            reason = parsed.get("abstain_reason")
            return GeneratedAnswer(
                status=AnswerStatus.ABSTAIN,
                text=ABSTAIN_TEXT,
                abstain_reason=(
                    reason[:MAX_REASON_CHARS] if isinstance(reason, str) and reason else "model abstained"
                ),
                discarded_text=discarded,
                latency_ms=(time.perf_counter() - started) * 1000,
                outcome=Outcome.MODEL_ABSTAIN,
                **common,
            )
        assert isinstance(items, list)
        validated = validate_claims(
            _raw_claims(items), {d.id: d for d in admitted}, answer_text, enforce_quotes=self.enforce_quotes
        )
        latency_ms = (time.perf_counter() - started) * 1000
        if not validated.claims:
            return GeneratedAnswer(
                status=AnswerStatus.ABSTAIN,
                text=ABSTAIN_TEXT,
                abstain_reason="no claim survived citation validation",
                rejected_citations=validated.rejected_citations,
                discarded_text=discarded,
                latency_ms=latency_ms,
                outcome=Outcome.MODEL_ANSWER,
                **common,
            )
        return GeneratedAnswer(
            status=AnswerStatus.ANSWER,
            text=compose_answer(validated.claims),
            claims=validated.claims,
            rejected_citations=validated.rejected_citations,
            discarded_text=discarded,
            latency_ms=latency_ms,
            outcome=Outcome.MODEL_ANSWER,
            **common,
        )
