"""Embedding interface and a vector index with cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float32]


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend fails; ``kind`` classifies the failure."""

    def __init__(self, message: str, kind: str = "provider") -> None:
        super().__init__(message)
        self.kind = kind


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier recorded in manifests (backend + model + dimension)."""
        ...

    def embed(self, texts: list[str]) -> Matrix:
        """Return one L2-normalised row per input text."""
        ...


def l2_normalise(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


@dataclass
class VectorIndex:
    """Exact (brute-force) cosine index. Adequate for corpora of a few thousand chunks."""

    ids: list[str]
    vectors: Matrix

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or len(self.ids) != self.vectors.shape[0]:
            raise EmbeddingError("ids and vectors must have the same length", kind="dimension_mismatch")
        if not np.all(np.isfinite(self.vectors)):
            raise EmbeddingError("index vectors contain NaN or infinite values", kind="invalid_vector")
        self.vectors = l2_normalise(self.vectors.astype(np.float32))

    def similarities(self, query: Matrix) -> npt.NDArray[np.float32]:
        flat = query.reshape(-1)
        if flat.shape[0] != self.vectors.shape[1]:
            raise EmbeddingError(
                f"query dimension {flat.shape[0]} differs from index dimension {self.vectors.shape[1]}",
                kind="dimension_mismatch",
            )
        if not np.all(np.isfinite(flat)):
            raise EmbeddingError("query vector contains NaN or infinite values", kind="invalid_vector")
        q = l2_normalise(flat.reshape(1, -1).astype(np.float32))
        scores: npt.NDArray[np.float32] = (self.vectors @ q[0]).astype(np.float32)
        return scores
