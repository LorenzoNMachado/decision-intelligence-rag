"""Cue lexicon shared by the rule-based document classifier and query parser.

The cues are general domain vocabulary for each signal. Negation cues flip a
signal mention to ABSENT and hedge cues make it UNCLEAR. The lexicon is
deliberately simple and auditable; its errors are measured against the
generator's intended labels by ``dirag signals evaluate``.
"""

from __future__ import annotations

import re

from decision_intelligence.models import Signal

SIGNAL_CUES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: [
        r"\bvendors?\b",
        r"\bsuppliers?\b",
        r"\bprovider\b",
        r"\bpartners?\b",
        r"\bcontractors?\b",
        r"\boutsourc\w*",
        r"\bthird[- ]party\b",
        r"\bexternal(?:ly)?\b",
        r"\bservice contract\b",
        r"\bmanaged agreement\b",
        r"\bin-house\b",
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

NEGATION_CUES = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnothing\b",
    r"\bnone\b",
    r"\bwithout\b",
    r"\bunaffected\b",
    r"\bout of scope\b",
    r"\boutside\b",
    r"\bin-house\b",
    r"\bstays internal\b",
]

HEDGE_CUES = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bnot yet\b",
    r"\bwhether\b",
    r"\bplanned\b",
    r"\bstill being\b",
    r"\bto be confirmed\b",
]

_FLAGS = re.IGNORECASE


def _compile(patterns: list[str], case_sensitive: frozenset[str] = frozenset()) -> list[re.Pattern[str]]:
    return [re.compile(p) if p in case_sensitive else re.compile(p, _FLAGS) for p in patterns]


# "Act" must be capitalised to denote a statute.
COMPILED_CUES = {signal: _compile(cues, frozenset({r"\bAct\b"})) for signal, cues in SIGNAL_CUES.items()}
COMPILED_NEGATIONS = _compile(NEGATION_CUES)
COMPILED_HEDGES = _compile(HEDGE_CUES)


def signals_mentioned(text: str) -> list[tuple[Signal, str]]:
    found: list[tuple[Signal, str]] = []
    for signal, patterns in COMPILED_CUES.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                found.append((signal, match.group(0)))
                break
    return found


def is_negated(text: str) -> bool:
    return any(p.search(text) for p in COMPILED_NEGATIONS)


def is_hedged(text: str) -> bool:
    return any(p.search(text) for p in COMPILED_HEDGES)
