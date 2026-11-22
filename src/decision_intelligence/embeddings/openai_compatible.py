"""Embeddings from any server implementing the OpenAI ``/embeddings`` endpoint.

Works with hosted providers and with local servers that expose the same API.
Results are cached on disk by (endpoint, model, text) so that experiments never
embed the same text twice and can be re-run offline once the cache is
populated; the endpoint is part of the key, so two providers serving a model
under the same name never share vectors.

Every vector is validated before it is cached or used: the response must be a
JSON object whose ``data`` holds exactly one item per input with a unique
index, and every embedding must be a non-empty list of finite numbers of one
consistent dimension. Violations raise ``EmbeddingError`` (never a bare
``KeyError``/``ValueError``), with messages that contain no credentials.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np

from decision_intelligence.embeddings.base import EmbeddingError, Matrix, l2_normalise


def endpoint_identity(base_url: str) -> str:
    """``scheme://host[:port]/path`` without credentials, query or fragment."""

    parts = urlsplit(base_url.strip())
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}{parts.path.rstrip('/')}"


def validate_vector(value: Any, dimension: int | None = None) -> list[float]:
    if not isinstance(value, list) or not value:
        raise EmbeddingError("embedding is not a non-empty list", kind="schema")
    if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in value):
        raise EmbeddingError("embedding contains non-numeric values", kind="schema")
    vector = [float(x) for x in value]
    if not all(math.isfinite(x) for x in vector):
        raise EmbeddingError("embedding contains NaN or infinite values", kind="invalid_vector")
    if dimension is not None and len(vector) != dimension:
        raise EmbeddingError(
            f"embedding dimension {len(vector)} differs from {dimension}", kind="dimension_mismatch"
        )
    return vector


class EmbeddingCache:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._data: dict[str, list[float]] = {}
        if path is not None and path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise EmbeddingError(f"unreadable embedding cache {path.name}", kind="cache") from exc
            if not isinstance(raw, dict):
                raise EmbeddingError(f"invalid embedding cache {path.name}", kind="cache")
            self._data = {str(k): validate_vector(v) for k, v in raw.items()}

    @staticmethod
    def key(provider: str, model: str, text: str) -> str:
        return hashlib.sha256(f"{provider}\x00{model}\x00{text}".encode()).hexdigest()

    def get(self, provider: str, model: str, text: str) -> list[float] | None:
        return self._data.get(self.key(provider, model, text))

    def put(self, provider: str, model: str, text: str, vector: list[float]) -> None:
        self._data[self.key(provider, model, text)] = vector

    def dimension(self) -> int | None:
        dims = {len(v) for v in self._data.values()}
        if len(dims) > 1:
            raise EmbeddingError("embedding cache mixes dimensions", kind="dimension_mismatch")
        return next(iter(dims), None)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data), encoding="utf-8")
        tmp.replace(self.path)


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None,
        cache_path: Path | None = None,
        batch_size: int = 64,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        offline: bool = False,
        transport: httpx.BaseTransport | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.provider = endpoint_identity(base_url)
        self._api_key = api_key
        self.cache = EmbeddingCache(cache_path)
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.offline = offline
        self._transport = transport
        self._sleep = sleep

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}"

    def _vectors(self, response: httpx.Response, inputs: list[str]) -> list[list[float]]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise EmbeddingError("provider returned a non-JSON body", kind="schema") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(inputs):
            raise EmbeddingError("provider returned a different number of vectors", kind="schema")
        by_index: dict[int, Any] = {}
        for item in data:
            index = item.get("index") if isinstance(item, dict) else None
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(inputs):
                raise EmbeddingError("embedding item has an invalid index", kind="schema")
            if index in by_index:
                raise EmbeddingError("embedding indices are not unique", kind="schema")
            by_index[index] = item.get("embedding")
        dimension = self.cache.dimension()
        vectors = []
        for index in range(len(inputs)):
            vector = validate_vector(by_index[index], dimension)
            dimension = len(vector)
            vectors.append(vector)
        return vectors

    def _request(self, inputs: list[str]) -> list[list[float]]:
        if self.offline:
            raise EmbeddingError("offline mode: text missing from the embedding cache", kind="cache_miss")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        last_error: EmbeddingError | None = None
        with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(
                        f"{self.base_url}/embeddings",
                        headers=headers,
                        json={"model": self.model, "input": inputs},
                    )
                except httpx.TimeoutException as exc:
                    last_error = EmbeddingError(f"timeout: {type(exc).__name__}", kind="timeout")
                except httpx.TransportError as exc:
                    last_error = EmbeddingError(f"network error: {type(exc).__name__}", kind="network")
                else:
                    if response.status_code == 200:
                        return self._vectors(response, inputs)
                    last_error = EmbeddingError(f"HTTP {response.status_code}", kind="http")
                    if response.status_code not in (408, 429) and response.status_code < 500:
                        raise last_error
                if attempt < self.max_retries:
                    self._sleep(min(2.0**attempt, 10.0))
        assert last_error is not None
        raise last_error

    def embed(self, texts: list[str]) -> Matrix:
        missing = [t for t in dict.fromkeys(texts) if self.cache.get(self.provider, self.model, t) is None]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            for text, vector in zip(batch, self._request(batch), strict=True):
                self.cache.put(self.provider, self.model, text, vector)
        if missing:
            self.cache.save()
        rows = [self.cache.get(self.provider, self.model, t) for t in texts]
        return l2_normalise(np.asarray(rows, dtype=np.float32).reshape(len(texts), -1))
