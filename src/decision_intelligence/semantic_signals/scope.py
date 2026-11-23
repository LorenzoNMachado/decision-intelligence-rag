"""Assertion scope of statements in a record (protocol v3).

A cue word says that a topic is *mentioned*; it does not say that the record
*asserts* it about the decision. ``scoped_mentions`` gives every signal mention
an assertion type, decided from the clause around it rather than from word
counts:

* ``FACTUAL`` — a current statement about the decision ("delivery relies on …");
* ``NEGATED`` — a current statement that the signal does not hold (local
  negation scope, implicit negation such as "free of", "excluded from");
* ``HYPOTHETICAL`` — hedged, conditional or modal ("may", "if", "could be");
* ``HISTORICAL`` — about a former state ("until last year", "previously",
  "the former …");
* ``CONTEXTUAL`` — the cue word modifies an organisational unit or role
  ("the vendor management office", "a legal adviser"): it names who did
  something, not what the decision depends on.

Only FACTUAL and NEGATED mentions are evidence about the current decision.

A third-party dependency is also recognised when a *named organisation* is
the object of a dependency phrase ("relies on Halvard Systems", "subcontracted
to …"), since such sentences need not contain any cue word.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from decision_intelligence.models import Polarity, Signal
from decision_intelligence.semantic_signals.lexicon import (
    LEXICON_V3,
    Lexicon,
    ScopeFlags,
    analyse,
    scope_flags,
)


class Assertion(StrEnum):
    FACTUAL = "FACTUAL"
    NEGATED = "NEGATED"
    HYPOTHETICAL = "HYPOTHETICAL"
    HISTORICAL = "HISTORICAL"
    CONTEXTUAL = "CONTEXTUAL"


HISTORICAL_CUES = re.compile(
    r"\b(?:former(?:ly)?|previous(?:ly)?|earlier|original(?:ly)?|until|used to|in the past|prior to|"
    r"years ago|once relied|at the time)\b",
    re.IGNORECASE,
)
# Heads of organisational units and roles: a cue word directly before one of them names a unit.
ORGANISATIONAL_HEADS = frozenset(
    [
        "office",
        "team",
        "department",
        "committee",
        "adviser",
        "advisor",
        "advisers",
        "advisors",
        "manager",
        "managers",
        "officer",
        "officers",
        "board",
        "unit",
        "group",
        "function",
        "desk",
        "council",
        "panel",
        "staff",
        "lead",
    ]
)
DEPENDENCY_PHRASES = re.compile(
    r"\b(?:rel(?:y|ies|ied|ying) on|depend(?:s|ed|ing|ent)? on|delivered by|operated by|hosted by|"
    r"provided by|supplied by|run by|maintained by|handled by|built by|serviced by|"
    r"subcontracted to|outsourced to|(?:contract|agreement|partnership) with)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CAPITALISED_SPAN = re.compile(r"\b[A-Z][\w&'.-]*(?:\s+(?:of|and|&|[A-Z][\w&'.-]*))*")
_DETERMINERS = frozenset({"The", "A", "An", "This", "That", "These", "Those", "Its", "Their", "Our"})
_NOT_ORGANISATIONS = re.compile(
    r"\b(?:Act|Regulation|Directive|Code|Law|Ordinance|Rules|January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\b"
)


@dataclass(frozen=True)
class ScopedMention:
    signal: Signal
    polarity: Polarity
    assertion: Assertion
    cue: str
    sentence: str


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def organisations(sentence: str) -> list[tuple[str, int]]:
    """Named organisations: capitalised spans of two or more words, not sentence-initial words alone.

    Returns (name, character offset). Regulations and dates are excluded; known entity
    names should be masked by the caller.
    """

    found = []
    for match in _CAPITALISED_SPAN.finditer(sentence):
        words = match.group(0).split()
        start = match.start()
        while words and words[0] in _DETERMINERS:
            start += len(words[0]) + 1
            words = words[1:]
        while words and words[-1] in {"of", "and", "&"}:
            words = words[:-1]
        capitalised = [w for w in words if w[:1].isupper()]
        if len(capitalised) < 2 or _NOT_ORGANISATIONS.search(" ".join(words)):
            continue
        found.append((" ".join(words).rstrip(".,;:"), start))
    return found


def _segment_assertion(flags: ScopeFlags, position: int, historical_segments: set[int]) -> Assertion | None:
    segment = flags.segment_of(position)
    if segment in historical_segments:
        return Assertion.HISTORICAL
    if segment in flags.hedged_segments:
        return Assertion.HYPOTHETICAL
    return None


def _historical_segments(flags: ScopeFlags) -> set[int]:
    return {flags.segment_of(m.start()) for m in HISTORICAL_CUES.finditer(flags.masked)}


def assertion_at(
    text: str, position: int, entities: Iterable[str] = (), lexicon: Lexicon = LEXICON_V3
) -> Assertion:
    """Assertion type of whatever is stated at ``position`` (a character offset) in ``text``."""

    flags = scope_flags(text, entities, lexicon)
    scoped = _segment_assertion(flags, position, _historical_segments(flags))
    if scoped is not None:
        return scoped
    return Assertion.NEGATED if flags.is_negated(position) else Assertion.FACTUAL


def scoped_mentions(
    text: str, entities: Iterable[str] = (), lexicon: Lexicon = LEXICON_V3
) -> list[ScopedMention]:
    """Signal mentions of every sentence in ``text`` with their assertion type."""

    entities = tuple(entities)
    found: list[ScopedMention] = []
    for sentence in sentences(text):
        flags = scope_flags(sentence, entities, lexicon)
        historical = _historical_segments(flags)
        for mention in analyse(sentence, entities, lexicon):
            following = [w for w, _, _ in flags.tokens[mention.token + 1 : mention.token + 3]]
            if any(w in ORGANISATIONAL_HEADS for w in following):
                assertion = Assertion.CONTEXTUAL
            else:
                position = flags.tokens[mention.token][1]
                scoped = _segment_assertion(flags, position, historical)
                if scoped is not None:
                    assertion = scoped
                elif mention.hedged:
                    assertion = Assertion.HYPOTHETICAL
                elif mention.polarity is Polarity.ABSENT:
                    assertion = Assertion.NEGATED
                else:
                    assertion = Assertion.FACTUAL
            found.append(ScopedMention(mention.signal, mention.polarity, assertion, mention.cue, sentence))
        masked = flags.masked
        for phrase in DEPENDENCY_PHRASES.finditer(masked):
            segment = flags.segment_of(phrase.start())
            for name, offset in organisations(masked):
                if offset > phrase.start() and flags.segment_of(offset) == segment:
                    scoped = _segment_assertion(flags, phrase.start(), historical)
                    negated = flags.is_negated(phrase.start()) or flags.is_negated(offset)
                    assertion = scoped or (Assertion.NEGATED if negated else Assertion.FACTUAL)
                    polarity = Polarity.ABSENT if negated else Polarity.PRESENT
                    found.append(ScopedMention(Signal.THIRD_PARTY, polarity, assertion, name, sentence))
                    break
    return found
