"""Malformed provider responses, invalid embeddings and failure isolation (audit P2-2).

Every provider is a mock transport; nothing leaves the process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import httpx
import numpy as np
import pytest

from decision_intelligence.config import ExperimentConfig
from decision_intelligence.embeddings.base import EmbeddingError, VectorIndex
from decision_intelligence.embeddings.openai_compatible import (
    EmbeddingCache,
    OpenAICompatibleEmbedder,
    endpoint_identity,
)
from decision_intelligence.evaluation.runner import _execute
from decision_intelligence.generation.generators import ExtractiveGenerator
from decision_intelligence.llm.client import Budget, LLMError, OpenAICompatibleLLM, Pricing, sanitise
from decision_intelligence.models import Question

KEY = "sk-fake-ABCDEFGH12345678"


def _llm(handler: Any, budget: Budget | None = None) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        "m",
        "https://llm.example/v1",
        KEY,
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
        max_retries=0,
        pricing=Pricing(1.0, 1.0),
        budget=budget,
    )


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>gateway error</html>"),
        httpx.Response(200, json=[1, 2]),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": ["text"]}),
        httpx.Response(200, json={"choices": [{"message": {"content": {"a": 1}}}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": [1]}),
        httpx.Response(
            200, json={"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": None}}
        ),
        httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
    ],
)
def test_malformed_llm_responses_raise_classified_errors_only(response: httpx.Response) -> None:
    budget = Budget(limit_usd=1.0)
    try:
        result = _llm(lambda r: response, budget).complete_json("s", "u", {}, "x", max_tokens=10)
    except LLMError as exc:
        assert exc.kind == "parse"
    else:  # malformed usage is tolerated when the content itself is valid
        assert result.parsed == {} and result.cost_source == "upper_bound"
    assert budget.spent_usd > 0  # a 200 may have been billed: it is charged conservatively


def test_connection_failures_are_network_errors_and_not_charged() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    budget = Budget(limit_usd=1.0)
    with pytest.raises(LLMError) as err:
        _llm(refuse, budget).complete_json("s", "u", {}, "x")
    assert err.value.kind == "network" and budget.spent_usd == 0


@pytest.mark.parametrize(
    "body",
    [
        f"invalid key: Bearer {KEY}",
        f'{{"error": "bad key {KEY}"}}',
        "Authorization: Bearer abc.def.ghi",
        "api_key=s3cr3tvalue123",
    ],
)
def test_authentication_errors_never_echo_credentials(body: str) -> None:
    with pytest.raises(LLMError) as err:
        _llm(lambda r: httpx.Response(401, text=body)).complete_json("s", "u", {}, "x")
    message = str(err.value)
    assert KEY not in message and "abc.def.ghi" not in message and "s3cr3tvalue123" not in message
    assert "[redacted]" in message


def test_sanitise_removes_configured_secret_even_without_a_known_prefix() -> None:
    assert "plainsecret" not in sanitise("leaked plainsecret here", secrets=["plainsecret"])


def _embedder(handler: Any, tmp_path: Path | None = None, base_url: str = "https://emb.example/v1") -> Any:
    return OpenAICompatibleEmbedder(
        "e",
        base_url,
        "k",
        cache_path=None if tmp_path is None else tmp_path / "cache.json",
        transport=httpx.MockTransport(handler),
        max_retries=0,
        sleep=lambda s: None,
    )


def _data(*vectors: Any) -> dict[str, Any]:
    return {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (httpx.Response(200, text="oops"), "schema"),
        (httpx.Response(200, json={"data": [{"index": 0}, {"index": 1}]}), "schema"),
        (httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]}), "schema"),  # count
        (httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}] * 2}), "schema"),  # index
        (httpx.Response(200, json=_data([1.0, 2.0], [1.0])), "dimension_mismatch"),
        (httpx.Response(200, json=_data(["a"], [1.0])), "schema"),
        (
            httpx.Response(
                200,
                content=b'{"data":[{"index":0,"embedding":[NaN,1]},{"index":1,"embedding":[1,1]}]}',
                headers={"content-type": "application/json"},
            ),
            "invalid_vector",
        ),
    ],
)
def test_invalid_embeddings_are_rejected(response: httpx.Response, kind: str) -> None:
    with pytest.raises(EmbeddingError) as err:
        _embedder(lambda r: response).embed(["a", "b"])
    assert err.value.kind == kind


def test_embedding_dimension_must_match_the_cache(tmp_path: Path) -> None:
    first = _embedder(lambda r: httpx.Response(200, json=_data([1.0, 0.0])), tmp_path)
    first.embed(["a"])
    second = _embedder(lambda r: httpx.Response(200, json=_data([1.0, 0.0, 0.0])), tmp_path)
    with pytest.raises(EmbeddingError) as err:
        second.embed(["b"])
    assert err.value.kind == "dimension_mismatch"


def test_cache_key_depends_on_the_endpoint(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_data([1.0, 0.0]))

    _embedder(handler, tmp_path, "https://a.example/v1").embed(["same text"])
    _embedder(handler, tmp_path, "https://b.example/v1").embed(["same text"])
    assert calls["n"] == 2
    assert EmbeddingCache.key("p1", "m", "t") != EmbeddingCache.key("p2", "m", "t")
    assert endpoint_identity("https://user:pw@h.example:8443/v1/?key=x") == "https://h.example:8443/v1"


def test_vector_index_rejects_non_finite_and_mismatched_vectors() -> None:
    with pytest.raises(EmbeddingError):
        VectorIndex(["a"], np.array([[np.nan, 1.0]], dtype=np.float32))
    index = VectorIndex(["a"], np.array([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(EmbeddingError):
        index.similarities(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    with pytest.raises(EmbeddingError):
        index.similarities(np.array([np.inf, 0.0], dtype=np.float32))


class _FailingRetriever:
    intent_parser = None

    def retrieve(self, question: str, intent: Any = None) -> Any:
        raise EmbeddingError("timeout: ReadTimeout", kind="timeout")


class _Pipeline:
    retriever = _FailingRetriever()
    generator = ExtractiveGenerator()
    documents: ClassVar[dict[str, Any]] = {}


def test_a_failing_row_becomes_an_error_row_not_an_abstention() -> None:
    question = Question("Q-1", "dev", "signal_filter", "q?", True, ("DEC-001",), "ref")
    config = ExperimentConfig("A", "vector", "extractive", "d")
    row = _execute(_Pipeline(), config, 1, question, {})  # type: ignore[arg-type]
    assert row["answer"]["status"] == "ERROR"
    assert row["answer"]["outcome"] == "PIPELINE_ERROR" and row["answer"]["error_kind"] == "timeout"
    assert row["retrieved"] == [] and row["cost_usd"] is None
    json.dumps(row)  # still serialisable
