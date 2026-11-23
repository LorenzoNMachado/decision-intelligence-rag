"""BM25 over chunks: an independent lexical baseline (protocol v3).

Okapi BM25 (Robertson & Zaragoza, 2009) with the same tokenizer as the
hashing embedder (lower-casing, stop words, light stemming), so the two
baselines differ only in the scoring function: BM25 weights terms by inverse
document frequency and saturates term frequency, while the hashing embedder
compares fixed-size signed feature vectors by cosine. Scores are raw BM25
values (not in [0, 1]); the evidence-gate threshold for BM25 is therefore
calibrated separately, on the dev split.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import numpy.typing as npt

from decision_intelligence.embeddings.hashing import tokenize
from decision_intelligence.models import Chunk


class BM25Scorer:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._terms = [Counter(tokenize(c.text)) for c in chunks]
        self._lengths = np.array([sum(t.values()) for t in self._terms], dtype=np.float64)
        self._average = float(self._lengths.mean()) if len(chunks) else 0.0
        frequency: Counter[str] = Counter()
        for terms in self._terms:
            frequency.update(terms.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in frequency.items()}

    @property
    def name(self) -> str:
        return f"bm25-v1:k1={self.k1}:b={self.b}"

    def scores(self, question: str) -> npt.NDArray[np.float32]:
        query = [t for t in dict.fromkeys(tokenize(question)) if t in self._idf]
        scores = np.zeros(len(self._terms), dtype=np.float64)
        if not query or self._average == 0:
            return scores.astype(np.float32)
        norm = self.k1 * (1 - self.b + self.b * self._lengths / self._average)
        for term in query:
            tf = np.array([terms.get(term, 0) for terms in self._terms], dtype=np.float64)
            scores += self._idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return scores.astype(np.float32)
