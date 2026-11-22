"""What a question asks for (protocol v3).

``analyse_need`` reads the *structure* of a question, not its category:

* the interrogative form and the noun it governs give the expected answer
  type ("when" → DATE, "how many" → QUANTITY, "which firm" → ORGANIZATION,
  "which decisions" → a list of decisions), using a small taxonomy of
  answer-type nouns in the tradition of question classification (Li & Roth,
  2002). The taxonomy is about kinds of answers, not about this corpus;
* the *focus* is the content words that say which attribute is wanted
  ("budget", "go live", "chair"), after removing the referenced decision,
  interrogatives and function words;
* the *reference* is the decision the question is about ("the decision to
  adopt the payroll platform"), when there is one.

No reference answer, question category or label is read.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from decision_intelligence.embeddings.hashing import STOPWORDS, tokenize
from decision_intelligence.intent.parsers import RuleBasedIntentParser
from decision_intelligence.models import IntentSignal
from decision_intelligence.semantic_signals.lexicon import LEXICON_V3, mask_entities
from decision_intelligence.sufficiency.facts import AnswerType

TYPE_NOUNS: dict[AnswerType, frozenset[str]] = {
    AnswerType.ORGANIZATION: frozenset(
        [
            "company",
            "companies",
            "firm",
            "firms",
            "vendor",
            "vendors",
            "supplier",
            "suppliers",
            "provider",
            "providers",
            "partner",
            "partners",
            "contractor",
            "contractors",
            "organisation",
            "organization",
            "organisations",
            "organizations",
            "agency",
            "consultancy",
            "operator",
            "operators",
        ]
    ),
    AnswerType.REGULATION: frozenset(
        [
            "law",
            "laws",
            "regulation",
            "regulations",
            "act",
            "statute",
            "statutes",
            "legislation",
            "rule",
            "rules",
            "directive",
            "ordinance",
        ]
    ),
    AnswerType.FRAMEWORK: frozenset(
        [
            "framework",
            "frameworks",
            "standard",
            "standards",
            "certification",
            "certifications",
            "attestation",
            "attestations",
        ]
    ),
    AnswerType.MONEY: frozenset(
        [
            "budget",
            "budgets",
            "cost",
            "costs",
            "price",
            "value",
            "amount",
            "spend",
            "spending",
            "fee",
            "fees",
            "funding",
            "saving",
            "savings",
            "money",
            "investment",
        ]
    ),
    AnswerType.DATE: frozenset(["date", "dates", "deadline", "day", "month", "year", "quarter"]),
    AnswerType.PERSON: frozenset(
        [
            "person",
            "people",
            "manager",
            "lead",
            "leader",
            "owner",
            "chair",
            "sponsor",
            "head",
            "coordinator",
            "individual",
        ]
    ),
    AnswerType.QUANTITY: frozenset(["number", "count", "headcount"]),
}
_DECISION_NOUNS = frozenset({"decision", "decisions", "record", "records"})
_WH = frozenset({"what", "which", "who", "whom", "whose", "when", "why", "how", "where", "name"})
_FUNCTION = (
    STOPWORDS
    | _WH
    | frozenset(
        {
            "decision",
            "decisions",
            "one",
            "result",
            "under",
            "behind",
            "after",
            "before",
            "during",
            "each",
            "per",
        }
    )
)
_IRREGULAR = {
    "led": "lead",
    "ran": "run",
    "run": "run",
    "set": "set",
    "took": "take",
    "taken": "take",
    "made": "make",
    "chose": "choose",
    "chosen": "choose",
    "began": "begin",
    "begun": "begin",
}
_REFERENCE = re.compile(r"\bdecision (?:to|on|about) (.+?)(?:\?|$)", re.IGNORECASE)


def lemma(token: str) -> str:
    return _IRREGULAR.get(token, token)


def lemmas(text: str) -> list[str]:
    return [lemma(t) for t in tokenize(text)]


def related(a: str, b: str) -> bool:
    """Two lemmas match when equal or when one is a prefix of the other (≥ 5 characters)."""

    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    return len(short) >= 5 and long.startswith(short)


@dataclass(frozen=True)
class InformationNeed:
    text: str
    kind: str  # attribute | list | reason | unknown
    answer_type: AnswerType
    accepted_types: tuple[AnswerType, ...]
    focus: tuple[str, ...]
    reference: str = ""
    constraints: tuple[IntentSignal, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "answer_type": self.answer_type.value,
            "accepted_types": [t.value for t in self.accepted_types],
            "focus": list(self.focus),
            "reference": self.reference,
            "constraints": [
                f"{c.signal.value}:{c.polarity.value}:{c.requirement.value}" for c in self.constraints
            ],
        }


def _type_of_noun(word: str) -> AnswerType | None:
    for kind, nouns in TYPE_NOUNS.items():
        if word in nouns:
            return kind
    return None


def _answer_type(words: list[str]) -> AnswerType:
    for i, word in enumerate(words):
        if word == "why":
            return AnswerType.REASON
        if word == "when":
            return AnswerType.DATE
        if word in {"who", "whom", "whose"}:
            return AnswerType.PERSON
        if word == "how" and i + 1 < len(words):
            if words[i + 1] == "many":
                return AnswerType.QUANTITY
            if words[i + 1] == "much":
                return AnswerType.MONEY
        if word in {"what", "which", "name"}:
            for following in words[i + 1 : i + 5]:
                if following in _DECISION_NOUNS:
                    return AnswerType.DECISIONS
                kind = _type_of_noun(following)
                if kind is not None:
                    return kind
    for word in words:  # imperative or elliptical forms: the first answer-type noun
        kind = _type_of_noun(word)
        if kind is not None:
            return kind
    return AnswerType.UNKNOWN


def analyse_need(question: str, entities: Iterable[str] = ()) -> InformationNeed:
    entities = tuple(entities)
    words = re.findall(r"[a-z]+", question.lower())
    answer_type = _answer_type(words)
    reference_match = _REFERENCE.search(question)
    reference = reference_match.group(1).strip() if reference_match else ""
    # The reference's end is unknown without the records (titles vary), so only entity names are
    # removed here; the checker removes each record's own title words from the focus.
    excluded = set(lemmas(" ".join(entities)))
    masked = mask_entities(question, entities)
    focus = tuple(
        dict.fromkeys(
            word
            for word in lemmas(masked)
            if word not in excluded and word not in _FUNCTION and len(word) > 2 and word != "entityname"
        )
    )
    if answer_type is AnswerType.DECISIONS:
        kind = "list"
    elif answer_type is AnswerType.REASON:
        kind = "reason"
    elif answer_type is AnswerType.UNKNOWN:
        kind = "unknown"
    else:
        kind = "attribute"
    accepted = (
        (AnswerType.PERSON, AnswerType.ORGANIZATION) if answer_type is AnswerType.PERSON else (answer_type,)
    )
    constraints: tuple[IntentSignal, ...] = ()
    if kind == "list":
        constraints = RuleBasedIntentParser(entities, LEXICON_V3).parse(question).signals
    return InformationNeed(question, kind, answer_type, accepted, focus, reference, constraints)
