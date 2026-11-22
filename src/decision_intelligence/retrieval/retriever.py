"""Document retrieval with an optional semantic-signal adjustment.

All three strategies run the same steps so that comparisons are controlled:

1. Embed the question and score every chunk by cosine similarity.
2. Aggregate to documents (score = best chunk) and keep a candidate pool.
3. Parse the question intent (strategy-specific; Strategy 1 parses nothing).
4. Adjust each candidate's score with the signal index:
   ``final = similarity + bonus × matched REQUIRED + preferred_bonus × matched PREFERRED
   − penalty × contradicted REQUIRED``. UNCLEAR and ERROR labels are neutral.
5. Return the top-k by final score (ties broken by document id).
6. Apply the evidence gate: a document is admitted when its similarity reaches
   the threshold and (optionally) it does not contradict a REQUIRED signal.

With no intent constraints, steps 4 and 6 reduce exactly to the vector baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from decision_intelligence.embeddings.base import Embedder, VectorIndex
from decision_intelligence.intent.parsers import IntentParser, NoIntentParser
from decision_intelligence.models import (
    Chunk,
    Polarity,
    QueryIntent,
    Requirement,
    RetrievalResult,
    RetrievedDocument,
    SignalLabel,
)
from decision_intelligence.semantic_signals.index import SignalIndex


@dataclass(frozen=True)
class RetrievalParams:
    top_k: int = 4
    candidate_pool: int = 20
    evidence_threshold: float = 0.65
    required_bonus: float = 0.06
    preferred_bonus: float = 0.02
    contradiction_penalty: float = 0.12
    veto_contradictions: bool = True


class Retriever:
    def __init__(
        self,
        strategy: str,
        chunks: list[Chunk],
        embedder: Embedder,
        params: RetrievalParams,
        intent_parser: IntentParser | None = None,
        signal_index: SignalIndex | None = None,
        index: VectorIndex | None = None,
    ) -> None:
        self.strategy = strategy
        self.chunks = chunks
        self.embedder = embedder
        self.params = params
        self.intent_parser: IntentParser = intent_parser or NoIntentParser()
        self.signal_index = signal_index
        self.index = index or VectorIndex([c.id for c in chunks], embedder.embed([c.text for c in chunks]))
        self._chunk_doc = np.array([c.doc_id for c in chunks])

    def similarities(self, question: str) -> dict[str, tuple[float, str]]:
        """Best chunk similarity per document: doc_id → (similarity, chunk_id)."""

        scores = self.index.similarities(self.embedder.embed([question])[0])
        best: dict[str, tuple[float, str]] = {}
        for chunk, score in zip(self.chunks, scores.tolist(), strict=True):
            current = best.get(chunk.doc_id)
            if current is None or score > current[0]:
                best[chunk.doc_id] = (float(score), chunk.id)
        return best

    def retrieve(self, question: str, intent: QueryIntent | None = None) -> RetrievalResult:
        if intent is None:
            intent = self.intent_parser.parse(question)
        best = self.similarities(question)
        pool = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[: self.params.candidate_pool]
        scored = []
        for doc_id, (similarity, chunk_id) in pool:
            matched: list[str] = []
            contradicted: list[str] = []
            final = similarity
            if self.signal_index is not None and intent.status == "OK":
                for constraint in intent.signals:
                    label = self.signal_index.label(doc_id, constraint.signal)
                    if label in (SignalLabel.UNCLEAR, SignalLabel.ERROR):
                        continue
                    wanted = (
                        SignalLabel.PRESENT if constraint.polarity is Polarity.PRESENT else SignalLabel.ABSENT
                    )
                    required = constraint.requirement is Requirement.REQUIRED
                    if label is wanted:
                        matched.append(constraint.signal.value)
                        final += self.params.required_bonus if required else self.params.preferred_bonus
                    elif required:
                        contradicted.append(constraint.signal.value)
                        final -= self.params.contradiction_penalty
            scored.append((doc_id, similarity, final, chunk_id, tuple(matched), tuple(contradicted)))
        scored.sort(key=lambda row: (-row[2], row[0]))
        documents = []
        for rank, (doc_id, similarity, final, chunk_id, matched_ids, contradicted_ids) in enumerate(
            scored[: self.params.top_k], start=1
        ):
            if similarity < self.params.evidence_threshold:
                admitted, reason = False, "below_threshold"
            elif contradicted_ids and self.params.veto_contradictions:
                admitted, reason = False, "signal_contradiction"
            else:
                admitted, reason = True, "admitted"
            documents.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    rank=rank,
                    similarity=similarity,
                    final_score=final,
                    best_chunk_id=chunk_id,
                    admitted=admitted,
                    gate_reason=reason,
                    matched_signals=matched_ids,
                    contradicted_signals=contradicted_ids,
                )
            )
        return RetrievalResult(
            strategy=self.strategy, question=question, intent=intent, documents=tuple(documents)
        )
