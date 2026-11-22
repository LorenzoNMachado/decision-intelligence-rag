"""Deterministic local embedder based on signed feature hashing.

This backend needs no network access and no model download, which makes it
suitable for the offline demo, for tests and for CI. It captures lexical
overlap (unigrams and bigrams of lightly stemmed tokens) rather than meaning;
results obtained with it are reported as such and are not comparable to dense
neural embeddings. Its cosine scale is also different, so the evidence-gate
threshold must be calibrated per embedder (see ``docs/methodology.md``).
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re

import numpy as np

from decision_intelligence.embeddings.base import Matrix, l2_normalise

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
        "will",
        "what",
        "who",
        "whom",
        "whose",
        "did",
        "do",
        "does",
        "about",
        "any",
        "also",
        "there",
        "their",
        "them",
        "they",
        "these",
        "those",
        "our",
        "we",
        "you",
        "your",
        "than",
        "then",
        "so",
        "such",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "not",
        "no",
    ]
)
# Negation words are kept as tokens even though they are stopwords elsewhere.
KEEP = frozenset({"not", "no", "without"})


def _stem(token: str) -> str:
    for suffix in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in STOPWORDS and raw not in KEEP:
            continue
        tokens.append(_stem(raw))
    return tokens


class HashingEmbedder:
    def __init__(self, dimension: int = 2048, use_bigrams: bool = True) -> None:
        if dimension < 64:
            raise ValueError("dimension must be at least 64")
        self.dimension = dimension
        self.use_bigrams = use_bigrams

    @property
    def name(self) -> str:
        return f"hashing-v1:d{self.dimension}:{'uni+bi' if self.use_bigrams else 'uni'}"

    def _features(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        features: dict[str, float] = {}
        for token in tokens:
            features[token] = features.get(token, 0.0) + 1.0
        if self.use_bigrams:
            for left, right in itertools.pairwise(tokens):
                key = f"{left}_{right}"
                features[key] = features.get(key, 0.0) + 0.5
        return {k: 1.0 + math.log(v) if v >= 1 else v for k, v in features.items()}

    def embed(self, texts: list[str]) -> Matrix:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text).items():
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                column = value % self.dimension
                sign = 1.0 if (value >> 63) & 1 else -1.0
                matrix[row, column] += sign * weight
        return l2_normalise(matrix)
