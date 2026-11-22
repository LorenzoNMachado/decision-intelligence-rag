"""Chat-completion clients and fail-closed cost control.

``OpenAICompatibleLLM`` talks to any server implementing the OpenAI
``/chat/completions`` API (hosted gateways, vendor endpoints or local servers).
``ScriptedLLM`` returns pre-recorded responses and exists for tests and for
replaying recorded runs; it is never used to produce reported results.

Errors are classified (``timeout``, ``network``, ``http``, ``rate_limit``,
``truncated``, ``parse``, ``budget``, ``config``) so that the experiment runner
records operational failures separately from model behaviour. API keys and
authorisation headers are redacted from every error message.

Cost control
------------
A ``Budget`` with a limit authorises each request against an **upper bound**
of its cost, computed before the request is sent:

    bound = (request bytes + overhead) × prompt price + max_tokens × completion price

The bound assumes that the provider counts at most one prompt token per byte
of the request (true for byte-level BPE tokenisers), honours ``max_tokens`` and
bills at the configured prices. Without configured prices no bound exists, so
a limited budget refuses to send the request (fail closed). After a response,
the charge is, in order of preference: the provider-reported cost when it is a
finite non-negative number (``provider``), the configured prices applied to
the reported token usage (``usage_estimate``), or the upper bound
(``upper_bound``). A response that may have been billed but carries no usable
accounting (timeouts after sending, malformed bodies) is charged at the upper
bound. Requests rejected with an HTTP error status are assumed not to be
billed. The limit is therefore as reliable as the configured prices and these
assumptions; the effective charge always depends on the provider's billing.

``DIRAG_BUDGET_USD=none`` disables the limit explicitly. Costs that cannot be
determined are then recorded as unknown (``None``), never as zero.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeGuard

import httpx

REQUEST_OVERHEAD_TOKENS = 64


class LLMError(RuntimeError):
    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class LLMResponse:
    content: str
    parsed: dict[str, Any]
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = 0.0  # None: unknown
    cost_source: str = "scripted"  # provider | usage_estimate | upper_bound | unknown | scripted
    cost_anomaly: bool = False  # the provider reported an unusable cost
    latency_ms: float = 0.0
    finish_reason: str = ""


class LLMClient(Protocol):
    @property
    def name(self) -> str: ...

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], schema_name: str, max_tokens: int = 1500
    ) -> LLMResponse: ...


def _valid_amount(value: Any) -> TypeGuard[int | float]:
    return (
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    )


@dataclass(frozen=True)
class Pricing:
    """USD per one million tokens. ``None`` means not configured (different from an explicit 0)."""

    prompt_per_mtok: float | None = None
    completion_per_mtok: float | None = None

    def __post_init__(self) -> None:
        for value in (self.prompt_per_mtok, self.completion_per_mtok):
            if value is not None and not _valid_amount(value):
                raise ValueError("prices must be finite and non-negative")

    @property
    def configured(self) -> bool:
        return self.prompt_per_mtok is not None and self.completion_per_mtok is not None

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        if self.prompt_per_mtok is None or self.completion_per_mtok is None:
            return None
        return (
            prompt_tokens * self.prompt_per_mtok + completion_tokens * self.completion_per_mtok
        ) / 1_000_000

    def upper_bound(self, request_bytes: int, max_tokens: int) -> float | None:
        return self.cost(request_bytes + REQUEST_OVERHEAD_TOKENS, max_tokens)


@dataclass
class Budget:
    """Spending limit for one process. ``limit_usd=None`` disables the limit explicitly."""

    limit_usd: float | None
    confirmed_usd: float = 0.0  # provider-reported costs
    estimated_usd: float = 0.0  # usage estimates and upper bounds
    unknown_cost_calls: int = 0

    def __post_init__(self) -> None:
        if self.limit_usd is not None and not _valid_amount(self.limit_usd):
            raise ValueError("the budget limit must be finite and non-negative")

    @property
    def spent_usd(self) -> float:
        return self.confirmed_usd + self.estimated_usd

    def authorise(self, bound: float | None) -> None:
        if self.limit_usd is None:
            return
        if bound is None:
            raise LLMError(
                "cannot enforce the spending limit: token pricing is not configured "
                "(set DIRAG_PRICE_PROMPT_PER_MTOK and DIRAG_PRICE_COMPLETION_PER_MTOK, "
                "or DIRAG_BUDGET_USD=none to run without a limit)",
                kind="budget",
            )
        if self.spent_usd + bound > self.limit_usd:
            raise LLMError(
                f"budget of {self.limit_usd:.4f} USD would be exceeded "
                f"(spent {self.spent_usd:.4f}, request bound {bound:.4f})",
                kind="budget",
            )

    def charge(self, amount: float | None, source: str) -> None:
        if amount is None:
            self.unknown_cost_calls += 1
        elif source == "provider":
            self.confirmed_usd += amount
        else:
            self.estimated_usd += amount


_BEARER = re.compile(r"(?i)\bbearer\s+[^\s\"',;]+")
_KEY_VALUE = re.compile(r"(?i)\b(authorization|api[-_]?key|x-api-key)(\"?\s*[:=]\s*\"?)[^\s\"',;]+")
_KEY_LIKE = re.compile(r"(?i)\b(?:sk|key|token|pk|rk)[-_][A-Za-z0-9\-_]{8,}")


def sanitise(message: str, secrets: Iterable[str | None] = (), limit: int = 240) -> str:
    """Remove known secrets and credential-looking strings, then truncate."""

    for secret in secrets:
        if secret and len(secret) >= 4:
            message = message.replace(secret, "[redacted]")
    message = _BEARER.sub("Bearer [redacted]", message)
    message = _KEY_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", message)
    message = _KEY_LIKE.sub("[redacted]", message)
    return message[:limit]


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


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    if value < 0 or int(value) != value:
        return None
    return int(value)


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
        self.budget = budget if budget is not None else Budget(limit_usd=None)
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

    def _detail(self, response: httpx.Response) -> str:
        return sanitise(response.text, [self._api_key])

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], schema_name: str, max_tokens: int = 1500
    ) -> LLMResponse:
        if not self._api_key:
            raise LLMError("no API key configured for the LLM provider", kind="config")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        body = self._body(system, user, schema, schema_name, max_tokens)
        bound = self.pricing.upper_bound(len(json.dumps(body).encode("utf-8")), max_tokens)
        last: LLMError | None = None
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
            for attempt in range(self.max_retries + 1):
                self.budget.authorise(bound)
                try:
                    response = client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                    # Never reached the provider: nothing can have been billed.
                    last = LLMError(f"network error ({type(exc).__name__})", kind="network")
                except httpx.TimeoutException as exc:
                    self.budget.charge(bound, "upper_bound")
                    last = LLMError(f"timeout ({type(exc).__name__})", kind="timeout")
                except httpx.TransportError as exc:
                    self.budget.charge(bound, "upper_bound")
                    last = LLMError(f"network error ({type(exc).__name__})", kind="network")
                else:
                    if response.status_code == 200:
                        return self._parse(response, started, bound)
                    detail = self._detail(response)
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

    def _parse(self, response: httpx.Response, started: float, bound: float | None) -> LLMResponse:
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            payload = response.json()
        except ValueError as exc:
            self.budget.charge(bound, "upper_bound")
            raise LLMError("provider returned a non-JSON body", kind="parse") from exc
        if not isinstance(payload, dict):
            self.budget.charge(bound, "upper_bound")
            raise LLMError("unexpected response shape", kind="parse")

        raw_usage = payload.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        prompt_tokens = _token_count(usage.get("prompt_tokens"))
        completion_tokens = _token_count(usage.get("completion_tokens"))
        reported = usage.get("cost")
        anomaly = "cost" in usage and not _valid_amount(reported)
        cost: float | None
        if _valid_amount(reported):
            cost, source = float(reported), "provider"
        elif prompt_tokens is not None and completion_tokens is not None and self.pricing.configured:
            cost, source = self.pricing.cost(prompt_tokens, completion_tokens), "usage_estimate"
        elif bound is not None:
            cost, source = bound, "upper_bound"
        else:
            cost, source = None, "unknown"
        self.budget.charge(cost, source)

        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(choice, dict) or not isinstance(message, dict):
            raise LLMError("unexpected response shape", kind="parse")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise LLMError("message content is not text", kind="parse")
        finish = choice.get("finish_reason")
        finish = finish if isinstance(finish, str) else ""
        if finish == "length":
            raise LLMError("response truncated by max_tokens", kind="truncated")
        model = payload.get("model")
        return LLMResponse(
            content=content,
            parsed=extract_json(content),
            model=model if isinstance(model, str) else self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            cost_source=source,
            cost_anomaly=anomaly,
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
