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
from dataclasses import dataclass, field

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
# Digit groups ("50,000") are one token, so their comma is not read as a clause boundary.
_TOKEN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|[A-Za-z0-9]+(?:[-'/][A-Za-z0-9]+)*|[,;:]")
ENTITY_PLACEHOLDER = "entityname"

Tokens = list[tuple[str, int, int]]


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) if p in _CASE_SENSITIVE else re.compile(p, _FLAGS) for p in patterns]


@dataclass(frozen=True)
class Lexicon:
    """A versioned set of cues. ``LEXICON_V2`` is the protocol v2 lexicon, unchanged."""

    name: str
    signal_cues: dict[Signal, list[str]]
    negative_cues: dict[Signal, list[str]]
    negators: list[str]
    predicate_negators: list[str]
    hedges: list[str]
    cues: dict[Signal, list[re.Pattern[str]]] = field(init=False, repr=False)
    negative: dict[Signal, list[re.Pattern[str]]] = field(init=False, repr=False)
    compiled_negators: list[re.Pattern[str]] = field(init=False, repr=False)
    compiled_predicate_negators: list[re.Pattern[str]] = field(init=False, repr=False)
    compiled_hedges: list[re.Pattern[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cues", {s: _compile(c) for s, c in self.signal_cues.items()})
        object.__setattr__(self, "negative", {s: _compile(c) for s, c in self.negative_cues.items()})
        object.__setattr__(self, "compiled_negators", _compile(self.negators))
        object.__setattr__(self, "compiled_predicate_negators", _compile(self.predicate_negators))
        object.__setattr__(self, "compiled_hedges", _compile(self.hedges))


LEXICON_V2 = Lexicon("rules-v2", SIGNAL_CUES, NEGATIVE_CUES, NEGATORS, PREDICATE_NEGATORS, HEDGE_CUES)

# Protocol v3 additions, written before the v3 corpus and questions existed. They are
# general constructions, not phrasings of any question set:
# * implicit negation ("lacks", "free of", "excluded from", "exempt from", "no longer");
# * exceptions ("except", "other than", "apart from", "excluding");
# * conditional and modal hedging ("if", "unless", "would", "could", "considered", "proposed");
# * third-party dependency stated through subcontracting or an operator ("subcontracted").
LEXICON_V3 = Lexicon(
    "rules-v3",
    {
        **SIGNAL_CUES,
        Signal.THIRD_PARTY: [*SIGNAL_CUES[Signal.THIRD_PARTY], r"\bsubcontract\w*", r"\bthird parties\b"],
    },
    NEGATIVE_CUES,
    [
        *NEGATORS,
        r"\black(?:s|ing|ed)?\b",
        r"\bfree of\b",
        r"\bno longer\b",
        r"\bexcept(?: for)?\b",
        r"\bother than\b",
        r"\bapart from\b",
        r"\bexcluding\b",
    ],
    [
        *PREDICATE_NEGATORS,
        r"\bexcluded from\b",
        r"\bexempt from\b",
        r"\bindependent of\b",
        r"\bruled out\b",
    ],
    [
        *HEDGE_CUES,
        r"\bif\b",
        r"\bunless\b",
        r"\bwould\b",
        r"\bcould\b",
        r"\bconsidered\b",
        r"\bproposed\b",
        r"\bunder discussion\b",
    ],
)

# Module-level aliases of the v2 compiled cues (kept for callers of the v2 API).
COMPILED_CUES = LEXICON_V2.cues
COMPILED_NEGATIVE_CUES = LEXICON_V2.negative
COMPILED_NEGATORS = LEXICON_V2.compiled_negators
COMPILED_PREDICATE_NEGATORS = LEXICON_V2.compiled_predicate_negators
COMPILED_HEDGES = LEXICON_V2.compiled_hedges


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


def tokenize(text: str) -> Tokens:
    """Lower-cased tokens with character spans (punctuation that ends a scope is kept)."""

    return _tokens(text)


def segments(tokens: Tokens) -> list[int]:
    """Clause segment id per token (terminators and coordinators start a new segment)."""

    return _segments(tokens)


@dataclass(frozen=True)
class ScopeFlags:
    """Token-level scope of a text: clause segments, hedged segments and negation counts."""

    masked: str
    tokens: Tokens
    segment: list[int]
    hedged_segments: frozenset[int]
    negated: dict[int, int]  # token index → number of negators governing it

    def segment_of(self, position: int) -> int:
        return self.segment[_token_index(self.tokens, position)] if self.tokens else 0

    def is_negated(self, position: int) -> bool:
        return bool(self.tokens) and self.negated.get(_token_index(self.tokens, position), 0) > 0


def scope_flags(text: str, entities: Iterable[str] = (), lexicon: Lexicon = LEXICON_V2) -> ScopeFlags:
    masked = mask_entities(text, entities)
    tokens = _tokens(masked)
    negated: dict[int, int] = {}
    if not tokens:
        return ScopeFlags(masked, tokens, [], frozenset(), negated)
    for pattern in lexicon.compiled_negators:
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
    for pattern in lexicon.compiled_predicate_negators:
        for match in pattern.finditer(masked):
            first, last = _span(tokens, match)
            for index in _scope(tokens, first, last, backward=True):
                negated[index] = negated.get(index, 0) + 1
    segment = _segments(tokens)
    hedged = frozenset(
        segment[_token_index(tokens, m.start())] for p in lexicon.compiled_hedges for m in p.finditer(masked)
    )
    return ScopeFlags(masked, tokens, segment, hedged, negated)


def analyse(text: str, entities: Iterable[str] = (), lexicon: Lexicon = LEXICON_V2) -> list[Mention]:
    """Every signal mention in ``text`` with its local polarity and hedging."""

    flags = scope_flags(text, entities, lexicon)
    masked, tokens, negated, segment = flags.masked, flags.tokens, flags.negated, flags.segment
    if not tokens:
        return []
    mentions: list[Mention] = []
    for signal in Signal:
        groups = [(p, False) for p in lexicon.cues[signal]]
        groups += [(p, True) for p in lexicon.negative.get(signal, [])]
        for pattern, inherently_negative in groups:
            for match in pattern.finditer(masked):
                first, _ = _span(tokens, match)
                flips = min(negated.get(first, 0), 1) + (1 if inherently_negative else 0)
                mentions.append(
                    Mention(
                        signal=signal,
                        cue=match.group(0),
                        polarity=Polarity.ABSENT if flips % 2 else Polarity.PRESENT,
                        hedged=segment[first] in flags.hedged_segments,
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
