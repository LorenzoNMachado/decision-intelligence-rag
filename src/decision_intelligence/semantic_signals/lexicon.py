"""Cue lexicon and scope analysis shared by the rule-based classifier and parser.

The cues are general domain vocabulary for each signal. ``analyse`` finds every
cue mention in a text and decides, for each mention separately:

* **polarity** — a mention is negated when a negator governs it. Negation is
  *local* (NegEx-style): a negator (``no``, ``not``, ``without``, ``neither``…)
  negates mentions in a short window after it, stopping at a scope terminator
  (comma, semicolon, ``but``, ``while``…). A negator that follows an auxiliary
  or a verb ("played no role", "is not relevant", "found nothing") is a
  predicate negation and also reaches back to the start of its clause, where
  ``and``/``or`` end a clause. Postposed negators ("unaffected", "out of
  scope") work in both directions. Some cues are inherently negative
  ("in-house"); negating them gives PRESENT.
* **hedging** — a mention is hedged when a hedge cue ("may", "whether",
  "planned"…) occurs in the same segment, where ``and``/``or`` also delimit
  segments.
* **disjunction** — two mentions of different signals separated by ``or`` are
  alternatives.

Known entity names (subjects, departments, job titles) are masked before
analysis, so "supplier onboarding process" or "Chief Information Security
Officer" are not read as signal mentions.

This lexicon was revised for protocol v2 before the v2 corpus and questions
were written, and was not edited afterwards. Its errors are measured against
the generator's intended labels by ``dirag signals-evaluate``.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from dataclasses import dataclass

from decision_intelligence.models import Polarity, Signal

SIGNAL_CUES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: [
        r"\bvendors?\b",
        r"\bsuppliers?\b",
        r"\bproviders?\b",
        r"\bpartners?\b",
        r"\bcontractors?\b",
        r"\boutsourc\w*",
        r"\bthird[- ]part(?:y|ies)\b",
        r"\bexternal(?:ly)?\b",
        r"\bservice contract\b",
        r"\bmanaged agreement\b",
    ],
    Signal.OPERATIONAL: [
        r"\boutages?\b",
        r"\bdowntime\b",
        r"\btoil\b",
        r"\bon-call\b",
        r"\bincidents?\b",
        r"\bthroughput\b",
        r"\bbottlenecks?\b",
        r"\befficiency\b",
        r"\bmanual\b",
        r"\bhandoffs?\b",
        r"\breliability\b",
        r"\boperations?\b",
        r"\boperational\b",
    ],
    Signal.LEGAL: [
        r"\blaws?\b",
        r"\blegal(?:ly)?\b",
        r"\bregulations?\b",
        r"\bregulatory\b",
        r"\bstatut\w*",
        r"\bAct\b",
        r"\bcounsel\b",
        r"\bobligations?\b",
    ],
    Signal.SECURITY: [
        r"\bsecurity\b",
        r"\bencrypt\w*",
        r"\baccess[- ]control\b",
        r"\baccess management\b",
        r"\bpenetration test\w*",
        r"\bvulnerabilit\w*",
        r"\bsingle sign-on\b",
        r"\bcredentials?\b",
        r"\bauthenticat\w*",
    ],
    Signal.COMPLIANCE: [
        r"\baudit\w*",
        r"\bcertification\b",
        r"\bcompliance\b",
        r"\bISO/IEC\b",
        r"\bSOC 2\b",
        r"\bPCI\b",
        r"\binternal control\b",
    ],
}

# Cues whose plain mention means the signal is ABSENT.
NEGATIVE_CUES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: [r"\bin-house\b", r"\bstays internal\b"],
}

# Negators that govern what follows them.
NEGATORS = [
    r"\bno\b",
    r"\bnot\b",
    r"n't\b",
    r"\bnever\b",
    r"\bnothing\b",
    r"\bnone\b",
    r"\bwithout\b",
    r"\bneither\b",
    r"\bnor\b",
]
# Postposed negators: they also negate the subject before them.
PREDICATE_NEGATORS = [
    r"\bunaffected\b",
    r"\bout of scope\b",
    r"\boutside (?:of )?(?:the |any |its |an? )?(?:[\w-]+ )?scope\b",
]
# An auxiliary or verb right before a negator makes it a predicate negation.
_PREDICATE_HEADS = frozenset(
    [
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "must",
        "played",
        "plays",
        "play",
        "found",
        "finds",
        "find",
        "poses",
        "pose",
        "posed",
        "raises",
        "raise",
        "raised",
        "makes",
        "make",
        "made",
        "involves",
        "involve",
        "involved",
        "requires",
        "require",
        "required",
    ]
)

HEDGE_CUES = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bnot yet\b",
    r"\bwhether\b",
    r"\bplanned\b",
    r"\bstill being\b",
    r"\bto be confirmed\b",
    r"\bpossibl[ey]\b",
    r"\bpotential(?:ly)?\b",
    r"\bperhaps\b",
]

# Scope terminators (end negation and hedge scope) and coordinators (end clauses).
_TERMINATORS = frozenset({",", ";", ":", "but", "however", "although", "though", "while", "whereas", "yet"})
_COORDINATORS = frozenset({"and", "or"})
FORWARD_WINDOW = 5
BACKWARD_WINDOW = 8

_FLAGS = re.IGNORECASE
_CASE_SENSITIVE = frozenset({r"\bAct\b"})
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'/][A-Za-z0-9]+)*|[,;:]")
ENTITY_PLACEHOLDER = "entityname"

Tokens = list[tuple[str, int, int]]


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) if p in _CASE_SENSITIVE else re.compile(p, _FLAGS) for p in patterns]


COMPILED_CUES = {signal: _compile(cues) for signal, cues in SIGNAL_CUES.items()}
COMPILED_NEGATIVE_CUES = {signal: _compile(cues) for signal, cues in NEGATIVE_CUES.items()}
COMPILED_NEGATORS = _compile(NEGATORS)
COMPILED_PREDICATE_NEGATORS = _compile(PREDICATE_NEGATORS)
COMPILED_HEDGES = _compile(HEDGE_CUES)


@dataclass(frozen=True)
class Mention:
    signal: Signal
    cue: str
    polarity: Polarity
    hedged: bool
    token: int  # index of the cue's first token


def mask_entities(text: str, entities: Iterable[str]) -> str:
    """Replace known entity names by a neutral placeholder (longest names first)."""

    names = sorted({e.strip() for e in entities if e.strip()}, key=len, reverse=True)
    if not names:
        return text
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)
    return pattern.sub(ENTITY_PLACEHOLDER, text)


def _tokens(text: str) -> Tokens:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _token_index(tokens: Tokens, position: int) -> int:
    for index, (_, start, end) in enumerate(tokens):
        if start <= position < end or position < start:
            return index
    return len(tokens) - 1


def _span(tokens: Tokens, match: re.Match[str]) -> tuple[int, int]:
    first = _token_index(tokens, match.start())
    last = _token_index(tokens, max(match.end() - 1, match.start()))
    return first, last


def _scope(tokens: Tokens, first: int, last: int, backward: bool) -> set[int]:
    """Token indices governed by a negator spanning tokens ``first..last``."""

    governed = set(range(first, last + 1))
    index = last + 1
    while index < len(tokens) and index <= last + FORWARD_WINDOW and tokens[index][0] not in _TERMINATORS:
        governed.add(index)
        index += 1
    if backward:
        index = first - 1
        while (
            index >= 0
            and index >= first - BACKWARD_WINDOW
            and tokens[index][0] not in _TERMINATORS
            and tokens[index][0] not in _COORDINATORS
        ):
            governed.add(index)
            index -= 1
    return governed


def _segments(tokens: Tokens) -> list[int]:
    """Segment id per token; terminators and coordinators start a new segment."""

    ids, current = [], 0
    for word, _, _ in tokens:
        if word in _TERMINATORS or word in _COORDINATORS:
            current += 1
        ids.append(current)
    return ids


def analyse(text: str, entities: Iterable[str] = ()) -> list[Mention]:
    """Every signal mention in ``text`` with its local polarity and hedging."""

    masked = mask_entities(text, entities)
    tokens = _tokens(masked)
    if not tokens:
        return []
    negated: dict[int, int] = {}  # token → number of negators governing it
    for pattern in COMPILED_NEGATORS:
        for match in pattern.finditer(masked):
            first, last = _span(tokens, match)
            word = tokens[first][0]
            # "not yet … whether" is hedging, not negation.
            if word == "not" and first + 1 < len(tokens) and tokens[first + 1][0] == "yet":
                continue
            previous = tokens[first - 1][0] if first > 0 else ""
            backward = "n't" in word or previous in _PREDICATE_HEADS
            for index in _scope(tokens, first, last, backward):
                negated[index] = negated.get(index, 0) + 1
    for pattern in COMPILED_PREDICATE_NEGATORS:
        for match in pattern.finditer(masked):
            first, last = _span(tokens, match)
            for index in _scope(tokens, first, last, backward=True):
                negated[index] = negated.get(index, 0) + 1
    segment = _segments(tokens)
    hedged_segments = {
        segment[_token_index(tokens, m.start())] for p in COMPILED_HEDGES for m in p.finditer(masked)
    }
    mentions: list[Mention] = []
    for signal in Signal:
        groups = [(p, False) for p in COMPILED_CUES[signal]]
        groups += [(p, True) for p in COMPILED_NEGATIVE_CUES.get(signal, [])]
        for pattern, inherently_negative in groups:
            for match in pattern.finditer(masked):
                first, _ = _span(tokens, match)
                flips = min(negated.get(first, 0), 1) + (1 if inherently_negative else 0)
                mentions.append(
                    Mention(
                        signal=signal,
                        cue=match.group(0),
                        polarity=Polarity.ABSENT if flips % 2 else Polarity.PRESENT,
                        hedged=segment[first] in hedged_segments,
                        token=first,
                    )
                )
    mentions.sort(key=lambda m: m.token)
    return mentions


def disjunctive_signals(text: str, mentions: list[Mention], entities: Iterable[str] = ()) -> set[Signal]:
    """Signals that appear as alternatives ("a vendor or a legal obligation")."""

    tokens = _tokens(mask_entities(text, entities))
    alternatives: set[Signal] = set()
    for left, right in itertools.pairwise(mentions):
        if left.signal is right.signal:
            continue
        between = [w for w, _, _ in tokens[left.token + 1 : right.token]]
        connectives = [w for w in between if w in _COORDINATORS or w in _TERMINATORS]
        if connectives and connectives[-1] == "or":
            alternatives.update({left.signal, right.signal})
    return alternatives
