"""Evidence rendering and citation validation.

The generator sees each admitted document rendered by ``render_document``;
quotes are verified against that same rendering, after whitespace and case
normalisation, so a quote counts as verified only if it is literally present in
a document the generator was allowed to cite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from decision_intelligence.models import Claim, DecisionDocument

DOC_ID_PATTERN = re.compile(r"\bDEC-\d{3}\b")


def render_document(doc: DecisionDocument) -> str:
    return (
        f"[{doc.id}] {doc.title}\n"
        f"Status: {doc.status}; Date: {doc.date}; Domain: {doc.domain}\n"
        f"Decision: {doc.decision}\n"
        f"Context: {doc.context}\n"
        f"Rationale: {doc.rationale}"
    )


def normalise(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return " ".join(text.lower().split()).strip(" .\"'")


def quote_in(quote: str, texts: list[str]) -> bool:
    needle = normalise(quote)
    if len(needle) < 8:
        return False
    return any(needle in normalise(t) for t in texts)


@dataclass(frozen=True)
class ValidatedClaims:
    claims: tuple[Claim, ...]
    rejected_citations: tuple[str, ...]
    dropped_claims: int


def validate_claims(
    raw_claims: list[tuple[str, list[str], str]],
    admitted: dict[str, DecisionDocument],
    answer_text: str,
    enforce_quotes: bool,
) -> ValidatedClaims:
    """Keep only citations to admitted documents and verify each quote.

    * A citation to a document outside the admitted set is removed and recorded.
    * A claim left without any admitted citation is dropped.
    * With ``enforce_quotes`` a claim whose quote is not literally present in a
      cited document is dropped; otherwise it is kept with ``quote_verified=False``.
    * Document ids mentioned in the answer prose but not admitted are recorded.
    """

    rendered = {doc_id: render_document(doc) for doc_id, doc in admitted.items()}
    rejected: list[str] = []
    claims: list[Claim] = []
    dropped = 0
    for statement, source_ids, quote in raw_claims:
        valid_ids = []
        for source in source_ids:
            if source in admitted:
                if source not in valid_ids:
                    valid_ids.append(source)
            elif source not in rejected:
                rejected.append(source)
        if not valid_ids:
            dropped += 1
            continue
        verified = quote_in(quote, [rendered[s] for s in valid_ids])
        if enforce_quotes and not verified:
            dropped += 1
            continue
        claims.append(
            Claim(statement=statement, source_ids=tuple(valid_ids), quote=quote, quote_verified=verified)
        )
    for leaked in DOC_ID_PATTERN.findall(answer_text):
        if leaked not in admitted and leaked not in rejected:
            rejected.append(leaked)
    return ValidatedClaims(claims=tuple(claims), rejected_citations=tuple(rejected), dropped_claims=dropped)
