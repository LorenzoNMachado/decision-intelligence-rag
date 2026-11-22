"""Deterministic section-based chunking.

Each record yields one chunk per section (summary, context, rationale). A
section longer than ``max_chars`` is split on sentence boundaries. Every chunk
starts with the record title so that a chunk is interpretable in isolation.
"""

from __future__ import annotations

import re

from decision_intelligence.models import Chunk, DecisionDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str, max_chars: int) -> list[str]:
    sentences = [s for s in _SENTENCE_BOUNDARY.split(text.strip()) if s]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_document(doc: DecisionDocument, max_chars: int = 800) -> list[Chunk]:
    header = f"{doc.title} [{doc.id}] | Domain: {doc.domain} | Status: {doc.status} | Date: {doc.date}"
    sections = [("summary", doc.decision), ("context", doc.context), ("rationale", doc.rationale)]
    chunks: list[Chunk] = []
    for section, body in sections:
        if not body.strip():
            continue
        for piece in _split_sentences(body, max_chars):
            position = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{doc.id}#c{position}",
                    doc_id=doc.id,
                    section=section,
                    position=position,
                    text=f"{header}\n{section.capitalize()}: {piece}",
                )
            )
    return chunks


def chunk_corpus(documents: list[DecisionDocument], max_chars: int = 800) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, max_chars=max_chars))
    return chunks
