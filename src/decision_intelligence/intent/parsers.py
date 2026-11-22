"""Question interpretation: which semantic signals constrain the answer.

Three parsers share the ``IntentParser`` interface:

* ``NoIntentParser`` (Strategy 1) always returns no constraint.
* ``RuleBasedIntentParser`` (Strategy 2) applies the cue lexicon with local
  negation scope, hedging and coordination (``semantic_signals.lexicon``).
* ``LLMIntentParser`` (Strategy 3) asks an LLM for REQUIRED/PREFERRED
  constraints with polarity, then validates the output deterministically:
  evidence must be a verbatim span of at least two words of the question,
  at most two constraints stay REQUIRED (the rest become PREFERRED), and any
  failure yields status UNAVAILABLE so retrieval falls back to similarity.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from typing import Any, Protocol

from decision_intelligence.generation.prompts import load_prompt
from decision_intelligence.llm.client import LLMClient, LLMError
from decision_intelligence.models import IntentSignal, Polarity, QueryIntent, Requirement, Signal
from decision_intelligence.semantic_signals.lexicon import analyse, disjunctive_signals

MAX_REQUIRED = 2


class IntentParser(Protocol):
    @property
    def name(self) -> str: ...

    def parse(self, question: str) -> QueryIntent: ...


class NoIntentParser:
    name = "none"

    def parse(self, question: str) -> QueryIntent:
        return QueryIntent(parser=self.name, status="NONE")


def _cap_required(signals: list[IntentSignal]) -> list[IntentSignal]:
    capped: list[IntentSignal] = []
    required = 0
    for s in signals:
        if s.requirement is Requirement.REQUIRED:
            required += 1
            if required > MAX_REQUIRED:
                s = IntentSignal(s.signal, Requirement.PREFERRED, s.polarity, s.evidence)
        capped.append(s)
    return capped


class RuleBasedIntentParser:
    """Strategy 2: the cue lexicon with local negation, hedging and coordination.

    * A signal is REQUIRED unless its mention is hedged or offered as an
      alternative to another signal ("a vendor or a legal obligation"), in
      which case it is PREFERRED.
    * Polarity comes from the mention's local negation scope (see
      ``semantic_signals.lexicon.analyse``).
    * A signal mentioned with both polarities is contradictory: it is dropped
      and reported in ``notes`` rather than guessed.
    * ``entities`` (subject names, departments, job titles) are masked first,
      so words inside a name never become constraints.
    """

    name = "rules-v2"

    def __init__(self, entities: Iterable[str] = ()) -> None:
        self.entities = tuple(entities)

    def parse(self, question: str) -> QueryIntent:
        mentions = analyse(question, self.entities)
        if not mentions:
            return QueryIntent(parser=self.name, status="NONE")
        alternatives = disjunctive_signals(question, mentions, self.entities)
        signals: list[IntentSignal] = []
        contradictory: list[str] = []
        for signal in Signal:
            own = [m for m in mentions if m.signal is signal]
            if not own:
                continue
            polarities = {m.polarity for m in own}
            if len(polarities) > 1:
                contradictory.append(signal.value)
                continue
            soft = all(m.hedged for m in own) or signal in alternatives
            signals.append(
                IntentSignal(
                    signal=signal,
                    requirement=Requirement.PREFERRED if soft else Requirement.REQUIRED,
                    polarity=own[0].polarity,
                    evidence=own[0].cue,
                )
            )
        signals.sort(key=lambda s: min(m.token for m in mentions if m.signal is s.signal))
        notes = f"contradictory constraints dropped: {', '.join(contradictory)}" if contradictory else ""
        if not signals:
            return QueryIntent(parser=self.name, status="NONE", notes=notes)
        return QueryIntent(parser=self.name, status="OK", signals=tuple(_cap_required(signals)), notes=notes)


INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["signals", "notes"],
    "properties": {
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["signal", "requirement", "polarity", "evidence"],
                "properties": {
                    "signal": {"type": "string", "enum": [s.value for s in Signal]},
                    "requirement": {"type": "string", "enum": ["REQUIRED", "PREFERRED"]},
                    "polarity": {"type": "string", "enum": ["PRESENT", "ABSENT"]},
                    "evidence": {"type": "string"},
                },
            },
        },
        "notes": {"type": "string"},
    },
}


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s-]", " ", text.lower()).split())


class LLMIntentParser:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.prompt = load_prompt("intent_v1")

    @property
    def name(self) -> str:
        return f"llm-{self.prompt.version}:{self.llm.name}"

    def parse(self, question: str) -> QueryIntent:
        started = time.perf_counter()
        try:
            response = self.llm.complete_json(
                self.prompt.system, f"QUESTION: {question}", INTENT_SCHEMA, "query_intent", max_tokens=600
            )
        except LLMError as exc:
            return QueryIntent(
                parser=self.name,
                status="UNAVAILABLE",
                notes=f"{exc.kind}: {exc}",
                # Refused before sending: free. Otherwise it may have been billed: unknown.
                cost_usd=0.0 if exc.kind in ("budget", "config") else None,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        normalised_question = _normalise(question)
        accepted: list[IntentSignal] = []
        rejected: list[str] = []
        try:
            items = response.parsed.get("signals", [])
            if not isinstance(items, list):
                raise TypeError("signals is not a list")
            for item in items:
                evidence = str(item.get("evidence", ""))
                span = _normalise(evidence)
                if len(span.split()) < 2 or span not in normalised_question:
                    rejected.append(str(item.get("signal")))
                    continue
                signal = Signal(item["signal"])
                if any(s.signal is signal for s in accepted):
                    continue
                accepted.append(
                    IntentSignal(
                        signal=signal,
                        requirement=Requirement(item["requirement"]),
                        polarity=Polarity(item["polarity"]),
                        evidence=evidence,
                    )
                )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            return QueryIntent(
                parser=self.name,
                status="UNAVAILABLE",
                notes=f"schema: {type(exc).__name__}",
                cost_usd=response.cost_usd,
                latency_ms=latency_ms,
            )
        notes = f"rejected unverifiable evidence for: {', '.join(rejected)}" if rejected else ""
        return QueryIntent(
            parser=self.name,
            status="OK" if accepted else "NONE",
            signals=tuple(_cap_required(accepted)),
            notes=notes,
            cost_usd=response.cost_usd,
            latency_ms=latency_ms,
        )
