"""Evidence rendering and deterministic claim validation.

The LLM generator sees each admitted record rendered by ``render_document``.
Its claims are then validated without any model:

1. **Citation** — a cited id is valid only if the record passed the evidence
   gate. Invalid ids are removed and recorded in ``rejected_citations``; a
   claim left without a valid citation is dropped.
2. **Statement references** — a claim whose statement names a record id that
   is not among its valid citations is dropped (the statement points at
   evidence the claim does not cite).
3. **Quote** — the quote must occur literally (after whitespace, case and
   quote-mark normalisation) in the *body* (decision, context, rationale) of a
   cited record and be at least ``MIN_QUOTE_WORDS`` words long. Header fields
   such as the title or status line do not count as evidence. Under the
   ``enforce`` policy an unverified claim is dropped; under ``record`` it is
   kept with ``quote_verified=False``.

A valid citation and a verified quote show that the claim is traceable to an
admitted record. They do not show that the statement is entailed by it: that
is the ``support`` label, which only the (experimental) LLM claim verifier sets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from decision_intelligence.models import Claim, DecisionDocument

DOC_ID_PATTERN = re.compile(r"\bDEC-\d{3,}\b")
MIN_QUOTE_WORDS = 4


def render_document(doc: DecisionDocument) -> str:
    return (
        f"[{doc.id}] {doc.title}\n"
        f"Status: {doc.status}; Date: {doc.date}; Domain: {doc.domain}\n"
        f"Decision: {doc.decision}\n"
        f"Context: {doc.context}\n"
        f"Rationale: {doc.rationale}"
    )


def evidence_body(doc: DecisionDocument) -> str:
    return f"{doc.decision} {doc.context} {doc.rationale}"


def normalise(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return " ".join(text.lower().split()).strip(" .\"'")


def quote_in(quote: str, texts: list[str]) -> bool:
    needle = normalise(quote)
    if len(needle.split()) < MIN_QUOTE_WORDS:
        return False
    return any(needle in normalise(t) for t in texts)


@dataclass(frozen=True)
class RawClaim:
    statement: str
    source_ids: tuple[str, ...]
    quote: str


@dataclass(frozen=True)
class ValidatedClaims:
    claims: tuple[Claim, ...]
    rejected_citations: tuple[str, ...]
    dropped_claims: int


def validate_claims(
    raw_claims: list[RawClaim],
    admitted: dict[str, DecisionDocument],
    discarded_text: str,
    enforce_quotes: bool,
) -> ValidatedClaims:
    """Apply the citation, statement-reference and quote rules (see the module docstring)."""

    rejected: list[str] = []

    def reject(doc_id: str) -> None:
        if doc_id not in rejected:
            rejected.append(doc_id)

    claims: list[Claim] = []
    dropped = 0
    for raw in raw_claims:
        valid_ids: list[str] = []
        for source in raw.source_ids:
            if source in admitted:
                if source not in valid_ids:
                    valid_ids.append(source)
            else:
                reject(source)
        stray = [d for d in DOC_ID_PATTERN.findall(raw.statement) if d not in valid_ids]
        for doc_id in stray:
            if doc_id not in admitted:
                reject(doc_id)
        if not valid_ids or stray or not raw.statement.strip():
            dropped += 1
            continue
        verified = quote_in(raw.quote, [evidence_body(admitted[s]) for s in valid_ids])
        if enforce_quotes and not verified:
            dropped += 1
            continue
        claims.append(
            Claim(
                statement=raw.statement.strip(),
                source_ids=tuple(valid_ids),
                quote=raw.quote,
                quote_verified=verified,
                quote_origin="model",
            )
        )
    for leaked in DOC_ID_PATTERN.findall(discarded_text):
        if leaked not in admitted:
            reject(leaked)
    return ValidatedClaims(claims=tuple(claims), rejected_citations=tuple(rejected), dropped_claims=dropped)


def compose_answer(claims: tuple[Claim, ...]) -> str:
    """The answer shown and judged: validated claims only, each with its citations."""

    return "\n".join(f"- {c.statement} [{', '.join(c.source_ids)}]" for c in claims)
