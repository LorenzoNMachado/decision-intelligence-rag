"""Chat-completion clients.

``OpenAICompatibleLLM`` talks to any server implementing the OpenAI
``/chat/completions`` API (hosted gateways, vendor endpoints or local servers).
``ScriptedLLM`` returns pre-recorded responses and exists for tests and for
replaying recorded runs; it is never used to produce reported results.

Errors are classified (``timeout``, ``network``, ``http``, ``rate_limit``,
``truncated``, ``parse``, ``budget``) so that the experiment runner can record
operational failures separately from model behaviour. API keys are never
included in exceptions, logs or result files.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


class LLMError(RuntimeError):
    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class LLMResponse:
    content: str
    parsed: dict[str, Any]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = ""


class LLMClient(Protocol):
    @property
    def name(self) -> str: ...

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], schema_name: str, max_tokens: int = 1500
    ) -> LLMResponse: ...


@dataclass
class Pricing:
    """USD per one million tokens, used when the provider does not report cost."""

    prompt_per_mtok: float = 0.0
    completion_per_mtok: float = 0.0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.prompt_per_mtok + completion_tokens * self.completion_per_mtok
        ) / 1_000_000


@dataclass
class Budget:
    """Hard ceiling on cumulative spend for one process."""

    limit_usd: float
    spent_usd: float = 0.0

    def check(self) -> None:
        if self.spent_usd >= self.limit_usd:
            raise LLMError(f"budget of {self.limit_usd:.2f} USD exhausted", kind="budget")

    def add(self, amount: float) -> None:
        self.spent_usd += amount


_SECRET_PATTERN = re.compile(r"(sk|key|token)[-_][A-Za-z0-9\-_]{8,}", re.IGNORECASE)


def sanitise(message: str, limit: int = 240) -> str:
    return _SECRET_PATTERN.sub("[redacted]", message)[:limit]


def extract_json(content: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a surrounding Markdown code fence."""

    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"response is not valid JSON: {exc.msg}", kind="parse") from exc
    if not isinstance(value, dict):
        raise LLMError("response JSON is not an object", kind="parse")
    return value


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None,
        *,
        temperature: float | None = 0.0,
        seed: int | None = None,
        response_format: str = "json_schema",  # json_schema | json_object | none
        timeout_s: float = 60.0,
        max_retries: int = 2,
        pricing: Pricing | None = None,
        budget: Budget | None = None,
        extra_body: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.response_format = response_format
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.pricing = pricing or Pricing()
        self.budget = budget
        self.extra_body = extra_body or {}
        self._transport = transport
        self._sleep = sleep

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}"

    def _body(
        self, system: str, user: str, schema: dict[str, Any], schema_name: str, max_tokens: int
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.seed is not None:
            body["seed"] = self.seed
        if self.response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        elif self.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)
        return body

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], schema_name: str, max_tokens: int = 1500
    ) -> LLMResponse:
        if self.budget is not None:
            self.budget.check()
        if not self._api_key:
            raise LLMError("no API key configured for the LLM provider", kind="config")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        body = self._body(system, user, schema, schema_name, max_tokens)
        last: LLMError | None = None
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                except httpx.TimeoutException as exc:
                    last = LLMError(f"timeout ({type(exc).__name__})", kind="timeout")
                except httpx.TransportError as exc:
                    last = LLMError(f"network error ({type(exc).__name__})", kind="network")
                else:
                    if response.status_code == 200:
                        return self._parse(response.json(), started)
                    detail = sanitise(response.text)
                    if response.status_code == 429:
                        last = LLMError(f"rate limited: {detail}", kind="rate_limit")
                    elif response.status_code >= 500 or response.status_code == 408:
                        last = LLMError(f"HTTP {response.status_code}: {detail}", kind="http")
                    else:
                        raise LLMError(f"HTTP {response.status_code}: {detail}", kind="http")
                if attempt < self.max_retries:
                    self._sleep(min(2.0 ** (attempt + 1), 20.0))
        assert last is not None
        raise last

    def _parse(self, payload: dict[str, Any], started: float) -> LLMResponse:
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected response shape", kind="parse") from exc
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        reported = usage.get("cost")
        cost = (
            float(reported)
            if isinstance(reported, int | float)
            else self.pricing.cost(prompt_tokens, completion_tokens)
        )
        if self.budget is not None:
            self.budget.add(cost)
        finish = str(choice.get("finish_reason") or "")
        if finish == "length":
            raise LLMError("response truncated by max_tokens", kind="truncated")
        return LLMResponse(
            content=content,
            parsed=extract_json(content),
            model=str(payload.get("model", self.model)),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            finish_reason=finish,
        )


@dataclass
class ScriptedLLM:
    """Deterministic stand-in that answers from a callable or a queue of payloads."""

    responder: Callable[[str, str], dict[str, Any] | LLMError]
    model: str = "scripted"
    calls: list[tuple[str, str]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"scripted:{self.model}"

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], schema_name: str, max_tokens: int = 1500
    ) -> LLMResponse:
        self.calls.append((system, user))
        result = self.responder(system, user)
        if isinstance(result, LLMError):
            raise result
        return LLMResponse(content=json.dumps(result), parsed=result, model=self.model)
