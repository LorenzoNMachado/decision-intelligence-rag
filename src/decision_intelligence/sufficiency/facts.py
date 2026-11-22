"""Typed facts in record text (protocol v3).

``extract_facts`` splits a text into sentences and recognises, with general
patterns rather than a catalogue of this corpus' values, the kinds of facts a
question can ask for: amounts of money, dates, counts, named people,
organisations, regulations, assurance frameworks, and stated reasons. Every
mention carries the assertion type of its clause (``semantic_signals.scope``):
a budget that "may be requested next year" or a contract that "was worth" a
sum in the past is not a current fact.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from decision_intelligence.semantic_signals.lexicon import LEXICON_V3, mask_entities, scope_flags
from decision_intelligence.semantic_signals.scope import (
    HISTORICAL_CUES,
    Assertion,
    organisations,
    sentences,
)


class AnswerType(StrEnum):
    ORGANIZATION = "ORGANIZATION"
    REGULATION = "REGULATION"
    FRAMEWORK = "FRAMEWORK"
    MONEY = "MONEY"
    DATE = "DATE"
    PERSON = "PERSON"
    QUANTITY = "QUANTITY"
    REASON = "REASON"
    DECISIONS = "DECISIONS"
    UNKNOWN = "UNKNOWN"


_MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
PATTERNS: dict[AnswerType, re.Pattern[str]] = {
    AnswerType.MONEY: re.compile(
        rf"[€$£]\s?(?:{_NUMBER})(?:\s?(?:million|thousand|billion|m|k|bn)\b)?"
        rf"|\b(?:{_NUMBER})\s?(?:million |thousand )?(?:EUR|USD|GBP|euros?|dollars?|pounds?)\b"
    ),
    AnswerType.DATE: re.compile(
        rf"\b(?:\d{{1,2}}\s+)?{_MONTHS}(?:\s+\d{{1,2}})?,?\s+\d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b"
        r"|\b(?:Q[1-4]|(?:first|second|third|fourth) quarter(?: of)?)\s+\d{4}\b"
        r"|\b(?:end|start|beginning|middle) of\s+\d{4}\b"
    ),
    AnswerType.QUANTITY: re.compile(rf"\b(?:{_NUMBER})\s+(?:[a-z-]+\s+){{0,2}}?(?:[a-z]+s|staff|people)\b"),
    AnswerType.REGULATION: re.compile(
        r"\b(?:[A-Z][\w-]*\s+){1,6}(?:Act|Regulation|Directive|Code|Law|Ordinance|Rules)\b"
    ),
    AnswerType.FRAMEWORK: re.compile(
        r"\bISO(?:/IEC)?\s?\d{4,5}\b|\bSOC\s?[12](?:\s+Type\s+I{1,2})?\b|\bPCI[ -]DSS\b|\bCOSO\b"
        r"|\bNIST\s+[A-Z]*\s?[\d.-]+\b|\b[A-Z]{2,}(?:/[A-Z]{2,})?\s\d{3,5}\b"
    ),
    AnswerType.REASON: re.compile(
        r"\b(?:because|due to|owing to|driven by|motivated by|prompted by|in order to|so that|"
        r"to comply with|to meet|(?:is|are|was|were) required to|as a result of|"
        r"the (?:main )?(?:driver|reason|aim)|"
        r"triggered by|in response to)\b",
        re.IGNORECASE,
    ),
}
_PERSON_NAME = r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"
PERSON_PATTERNS = [
    re.compile(
        rf"\b(?:led|owned|managed|chaired|sponsored|headed|coordinated|supervised) by ({_PERSON_NAME})"
    ),
    re.compile(
        rf"\b({_PERSON_NAME}),? (?:is|was|will be|remains) (?:accountable|responsible|the owner|in charge)"
    ),
    re.compile(rf"\b({_PERSON_NAME}), the [a-z ]*(?:lead|owner|manager|sponsor|chair)\b"),
    re.compile(rf"\b({_PERSON_NAME}) (?:leads|owns|manages|chairs|sponsors|heads|coordinates)\b"),
]


@dataclass(frozen=True)
class TypedMention:
    type: AnswerType
    text: str
    assertion: Assertion


@dataclass(frozen=True)
class SentenceFacts:
    index: int
    text: str
    assertion: Assertion  # of the sentence as a whole
    mentions: tuple[TypedMention, ...]

    def of_type(self, *types: AnswerType) -> list[TypedMention]:
        return [m for m in self.mentions if m.type in types]


def _mention_assertion(sentence: str, position: int, entities: Iterable[str]) -> Assertion:
    flags = scope_flags(sentence, entities, LEXICON_V3)
    segment = flags.segment_of(position)
    if any(flags.segment_of(m.start()) == segment for m in HISTORICAL_CUES.finditer(flags.masked)):
        return Assertion.HISTORICAL
    if segment in flags.hedged_segments:
        return Assertion.HYPOTHETICAL
    return Assertion.NEGATED if flags.is_negated(position) else Assertion.FACTUAL


def _sentence_assertion(sentence: str, entities: Iterable[str]) -> Assertion:
    flags = scope_flags(sentence, entities, LEXICON_V3)
    if HISTORICAL_CUES.search(flags.masked):
        return Assertion.HISTORICAL
    if flags.hedged_segments:
        return Assertion.HYPOTHETICAL
    return Assertion.FACTUAL


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in taken)


def sentence_facts(index: int, sentence: str, entities: Iterable[str] = ()) -> SentenceFacts:
    entities = tuple(entities)
    masked = mask_entities(sentence, entities)
    mentions: list[TypedMention] = []
    taken: list[tuple[int, int]] = []

    def add(kind: AnswerType, start: int, end: int, text: str) -> None:
        if _overlaps((start, end), taken):
            return
        taken.append((start, end))
        mentions.append(TypedMention(kind, text.strip(), _mention_assertion(masked, start, ())))

    # Order matters: more specific types claim their spans first.
    for kind in (AnswerType.MONEY, AnswerType.DATE, AnswerType.FRAMEWORK, AnswerType.REGULATION):
        for match in PATTERNS[kind].finditer(masked):
            add(kind, match.start(), match.end(), match.group(0))
    for pattern in PERSON_PATTERNS:
        for match in pattern.finditer(masked):
            add(AnswerType.PERSON, match.start(1), match.end(1), match.group(1))
    for name, offset in organisations(masked):
        add(AnswerType.ORGANIZATION, offset, offset + len(name), name)
    for match in PATTERNS[AnswerType.QUANTITY].finditer(masked):
        add(AnswerType.QUANTITY, match.start(), match.end(), match.group(0))
    for match in PATTERNS[AnswerType.REASON].finditer(masked):
        mentions.append(
            TypedMention(AnswerType.REASON, match.group(0), _mention_assertion(masked, match.start(), ()))
        )
    return SentenceFacts(index, sentence, _sentence_assertion(masked, ()), tuple(mentions))


def extract_facts(text: str, entities: Iterable[str] = ()) -> list[SentenceFacts]:
    entities = tuple(entities)
    return [sentence_facts(i, s, entities) for i, s in enumerate(sentences(text))]
