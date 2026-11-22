"""Embeddings from any server implementing the OpenAI ``/embeddings`` endpoint.

Works with hosted providers and with local servers that expose the same API.
Results are cached on disk by (model, text) so that experiments never embed the
same text twice and can be re-run offline once the cache is populated.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx
import numpy as np

from decision_intelligence.embeddings.base import EmbeddingError, Matrix, l2_normalise


class EmbeddingCache:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._data: dict[str, list[float]] = {}
        if path is not None and path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        return self._data.get(self.key(model, text))

    def put(self, model: str, text: str, vector: list[float]) -> None:
        self._data[self.key(model, text)] = vector

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
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.cache = EmbeddingCache(cache_path)
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.offline = offline
        self._transport = transport
        self._dimension: int | None = None

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}"

    def _request(self, inputs: list[str]) -> list[list[float]]:
        if self.offline:
            raise EmbeddingError("offline mode: text missing from the embedding cache", kind="cache_miss")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        last_error: Exception | None = None
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
                        data = response.json().get("data", [])
                        if len(data) != len(inputs):
                            raise EmbeddingError(
                                "provider returned a different number of vectors", kind="schema"
                            )
                        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
                    last_error = EmbeddingError(f"HTTP {response.status_code}", kind="http")
                    if response.status_code not in (408, 429) and response.status_code < 500:
                        raise last_error
                if attempt < self.max_retries:
                    time.sleep(min(2.0**attempt, 10.0))
        assert last_error is not None
        raise last_error

    def embed(self, texts: list[str]) -> Matrix:
        missing = [t for t in dict.fromkeys(texts) if self.cache.get(self.model, t) is None]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            for text, vector in zip(batch, self._request(batch), strict=True):
                self.cache.put(self.model, text, vector)
        if missing:
            self.cache.save()
        rows = [self.cache.get(self.model, t) for t in texts]
        return l2_normalise(np.asarray(rows, dtype=np.float32))
