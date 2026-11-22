from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from decision_intelligence.embeddings.base import EmbeddingError
from decision_intelligence.embeddings.openai_compatible import OpenAICompatibleEmbedder
from decision_intelligence.llm.client import (
    Budget,
    LLMError,
    OpenAICompatibleLLM,
    Pricing,
    extract_json,
    sanitise,
)

SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False, "required": []}


def _completion(content: str, finish: str = "stop", usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "model": "m",
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 500},
    }


def _client(handler: Any, **kwargs: Any) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        "m",
        "https://llm.example/v1",
        "sk-test-1234567890",
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
        **kwargs,
    )


def test_successful_call_parses_json_and_computes_cost() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer sk-test-1234567890"
        return httpx.Response(200, json=_completion('```json\n{"ok": true}\n```'))

    client = _client(handler, pricing=Pricing(1.0, 2.0), seed=7)
    response = client.complete_json("sys", "user", SCHEMA, "s")
    assert response.parsed == {"ok": True}
    assert response.cost_usd == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    assert seen["temperature"] == 0.0 and seen["seed"] == 7
    assert seen["response_format"]["type"] == "json_schema"


def test_provider_reported_cost_takes_precedence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("{}", usage={"prompt_tokens": 1, "cost": 0.0042}))

    assert _client(handler, pricing=Pricing(100, 100)).complete_json("s", "u", SCHEMA, "s").cost_usd == 0.0042


def test_retries_rate_limits_then_fails_with_kind() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, text="slow down")

    with pytest.raises(LLMError) as err:
        _client(handler, max_retries=2).complete_json("s", "u", SCHEMA, "s")
    assert err.value.kind == "rate_limit" and calls["n"] == 3


def test_client_errors_are_not_retried_and_secrets_are_redacted() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="invalid key sk-test-1234567890")

    with pytest.raises(LLMError) as err:
        _client(handler).complete_json("s", "u", SCHEMA, "s")
    assert calls["n"] == 1
    assert "1234567890" not in str(err.value)


def test_timeouts_truncation_and_invalid_json_are_classified() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError) as err:
        _client(timeout, max_retries=1).complete_json("s", "u", SCHEMA, "s")
    assert err.value.kind == "timeout"

    with pytest.raises(LLMError) as err:
        _client(lambda r: httpx.Response(200, json=_completion("{}", finish="length"))).complete_json(
            "s", "u", SCHEMA, "s"
        )
    assert err.value.kind == "truncated"

    with pytest.raises(LLMError) as err:
        _client(lambda r: httpx.Response(200, json=_completion("not json"))).complete_json(
            "s", "u", SCHEMA, "s"
        )
    assert err.value.kind == "parse"


def test_budget_blocks_calls_once_exhausted() -> None:
    budget = Budget(limit_usd=0.001)
    handler = lambda r: httpx.Response(200, json=_completion("{}", usage={"cost": 0.002}))  # noqa: E731
    client = _client(handler, budget=budget, pricing=Pricing(0.0, 0.0))
    client.complete_json("s", "u", SCHEMA, "s")
    with pytest.raises(LLMError) as err:
        client.complete_json("s", "u", SCHEMA, "s")
    assert err.value.kind == "budget"


def test_missing_key_is_a_config_error() -> None:
    client = OpenAICompatibleLLM("m", "https://x", None)
    with pytest.raises(LLMError) as err:
        client.complete_json("s", "u", SCHEMA, "s")
    assert err.value.kind == "config"


def test_helpers() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    with pytest.raises(LLMError):
        extract_json("[1, 2]")
    assert "[redacted]" in sanitise("token key_ABCDEFGH12345 leaked")


def test_embedder_caches_and_supports_offline_replay(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        inputs = json.loads(request.content)["input"]
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": [float(i + 1), 1.0]} for i in range(len(inputs))]}
        )

    cache = tmp_path / "cache.json"
    embedder = OpenAICompatibleEmbedder(
        "e", "https://x", "k", cache_path=cache, transport=httpx.MockTransport(handler)
    )
    first = embedder.embed(["a", "b", "a"])
    assert calls["n"] == 1 and first.shape == (3, 2)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)

    offline = OpenAICompatibleEmbedder("e", "https://x", None, cache_path=cache, offline=True)
    assert np.allclose(offline.embed(["b"]), first[1:2])
    with pytest.raises(EmbeddingError) as err:
        offline.embed(["never seen"])
    assert err.value.kind == "cache_miss"
