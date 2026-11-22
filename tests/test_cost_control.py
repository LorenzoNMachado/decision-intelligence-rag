"""Fail-closed spending control (audit finding P1-3). All providers are mocked."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from decision_intelligence.config import ConfigurationError, Environment
from decision_intelligence.llm.client import Budget, LLMError, OpenAICompatibleLLM, Pricing

SCHEMA: dict[str, Any] = {"type": "object"}


def _completion(usage: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
    if usage is not ...:
        payload["usage"] = usage
    return payload


class Provider:
    def __init__(self, usage: Any = None, error: Exception | None = None) -> None:
        self.calls = 0
        self.usage = {"prompt_tokens": 100, "completion_tokens": 50} if usage is None else usage
        self.error = error

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.error is not None:
            raise self.error
        # json.dumps writes inf as Infinity, which some providers emit and Python parses.
        body = json.dumps(_completion(self.usage)).encode()
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})


def _client(provider: Provider, budget: Budget, pricing: Pricing | None = None) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        "m",
        "https://llm.example/v1",
        "sk-test-1234567890",
        transport=httpx.MockTransport(provider),
        sleep=lambda s: None,
        max_retries=0,
        pricing=pricing,
        budget=budget,
    )


PRICED = Pricing(prompt_per_mtok=1.0, completion_per_mtok=2.0)


def test_unknown_pricing_blocks_a_limited_budget_before_any_request() -> None:
    provider = Provider()
    with pytest.raises(LLMError) as err:
        _client(provider, Budget(limit_usd=1.0), pricing=Pricing()).complete_json("s", "u", SCHEMA, "x")
    assert err.value.kind == "budget" and "pricing" in str(err.value)
    assert provider.calls == 0


def test_zero_prices_are_accepted_only_when_set_explicitly() -> None:
    assert not Pricing().configured
    assert Pricing(0.0, 0.0).configured  # e.g. a local server: explicitly free


def test_the_maximum_cost_is_checked_before_the_request() -> None:
    provider = Provider()
    budget = Budget(limit_usd=0.0001)
    client = _client(provider, budget, pricing=PRICED)
    # max_tokens alone would cost 1500 × 2 / 1e6 = 0.003 > 0.0001.
    with pytest.raises(LLMError) as err:
        client.complete_json("s", "u", SCHEMA, "x", max_tokens=1500)
    assert err.value.kind == "budget" and provider.calls == 0


def test_limit_exceeded_after_accumulated_spend() -> None:
    provider = Provider(usage={"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.004})
    budget = Budget(limit_usd=0.01)
    client = _client(provider, budget, pricing=PRICED)
    client.complete_json("s", "u", SCHEMA, "x", max_tokens=100)
    client.complete_json("s", "u", SCHEMA, "x", max_tokens=100)
    assert budget.spent_usd == pytest.approx(0.008)
    with pytest.raises(LLMError) as err:
        client.complete_json("s", "u", SCHEMA, "x", max_tokens=1500)  # bound 0.003 > remaining 0.002
    assert err.value.kind == "budget" and provider.calls == 2


@pytest.mark.parametrize(
    ("reported", "anomaly"),
    [(-100, True), ("0.5", True), (math.inf, True), (True, True), (None, False)],
)
def test_invalid_provider_cost_is_never_trusted(reported: Any, anomaly: bool) -> None:
    usage: dict[str, Any] = {"prompt_tokens": 100, "completion_tokens": 50}
    if reported is not None:
        usage["cost"] = reported
    budget = Budget(limit_usd=1.0)
    response = _client(Provider(usage=usage), budget, pricing=PRICED).complete_json("s", "u", SCHEMA, "x")
    expected = (100 * 1.0 + 50 * 2.0) / 1e6
    assert response.cost_source == "usage_estimate"
    assert response.cost_usd == pytest.approx(expected)
    assert budget.spent_usd == pytest.approx(expected) and budget.spent_usd >= 0
    assert response.cost_anomaly == anomaly


def test_valid_provider_cost_is_confirmed() -> None:
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0042}
    budget = Budget(limit_usd=1.0)
    response = _client(Provider(usage=usage), budget, pricing=PRICED).complete_json("s", "u", SCHEMA, "x")
    assert (response.cost_usd, response.cost_source) == (0.0042, "provider")
    assert budget.confirmed_usd == 0.0042 and budget.estimated_usd == 0.0


def test_missing_usage_is_charged_at_the_upper_bound() -> None:
    budget = Budget(limit_usd=1.0)
    client = _client(Provider(usage=...), budget, pricing=PRICED)
    response = client.complete_json("s", "u", SCHEMA, "x", max_tokens=100)
    assert response.cost_source == "upper_bound"
    assert response.cost_usd is not None and response.cost_usd >= 100 * 2.0 / 1e6
    assert budget.spent_usd == response.cost_usd


def test_a_timeout_is_charged_at_the_upper_bound_because_it_may_have_been_billed() -> None:
    budget = Budget(limit_usd=1.0)
    provider = Provider(error=httpx.ReadTimeout("slow"))
    with pytest.raises(LLMError) as err:
        _client(provider, budget, pricing=PRICED).complete_json("s", "u", SCHEMA, "x", max_tokens=100)
    assert err.value.kind == "timeout"
    assert budget.spent_usd >= 100 * 2.0 / 1e6


def test_explicitly_unlimited_budget_records_unknown_cost_not_zero() -> None:
    budget = Budget(limit_usd=None)
    response = _client(Provider(usage=...), budget, pricing=Pricing()).complete_json("s", "u", SCHEMA, "x")
    assert response.cost_usd is None and response.cost_source == "unknown"
    assert budget.unknown_cost_calls == 1


def test_pricing_and_budget_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        Pricing(prompt_per_mtok=-1.0, completion_per_mtok=1.0)
    with pytest.raises(ValueError):
        Budget(limit_usd=-0.5)


def test_environment_parses_budget_and_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    env = Environment.from_env()
    assert env.budget_usd == 1.0 and env.pricing == Pricing()
    monkeypatch.setenv("DIRAG_BUDGET_USD", "none")
    monkeypatch.setenv("DIRAG_PRICE_PROMPT_PER_MTOK", "0.5")
    monkeypatch.setenv("DIRAG_PRICE_COMPLETION_PER_MTOK", "1.5")
    env = Environment.from_env()
    assert env.budget_usd is None and env.pricing == Pricing(0.5, 1.5)
    monkeypatch.setenv("DIRAG_PRICE_PROMPT_PER_MTOK", "-1")
    with pytest.raises(ConfigurationError):
        Environment.from_env()
