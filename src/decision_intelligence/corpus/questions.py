"""Deterministic construction of the 50-question evaluation set.

Questions are built from templates over the experimental corpus, with the
expected evidence derived from the generator's intended signal labels. The set
contains 40 answerable and 10 unanswerable questions, split 10 dev / 40 holdout
with the same proportion of unanswerable questions in each split.

Because references are derived from the same generator as the corpus they are
not independent annotations; ``docs/limitations.md`` discusses the consequences.
"""

from __future__ import annotations

import random
from collections import defaultdict

from decision_intelligence.corpus.generator import FRAMEWORKS, REGULATIONS, VENDORS
from decision_intelligence.models import DecisionDocument, Question, Signal

SIGNAL_QUERY_PHRASES: dict[Signal, list[str]] = {
    Signal.THIRD_PARTY: [
        "depend on an external vendor",
        "rely on an outside supplier or partner",
        "involve a third-party provider",
    ],
    Signal.OPERATIONAL: [
        "were motivated by operational problems",
        "were driven by reliability or efficiency concerns",
        "aim to cut manual toil or outages",
    ],
    Signal.LEGAL: [
        "are required by law or regulation",
        "depend on a legal obligation",
        "were triggered by statutory requirements",
    ],
    Signal.SECURITY: [
        "have security dependencies",
        "depend on controls such as encryption or access management",
        "address a security weakness",
    ],
    Signal.COMPLIANCE: [
        "are relevant for audits or certification",
        "matter for compliance reporting",
        "produce evidence for an audit framework",
    ],
}

SIGNAL_NEGATED_PHRASES: dict[Signal, str] = {
    Signal.THIRD_PARTY: "no external vendor is involved",
    Signal.OPERATIONAL: "operational concerns played no role",
    Signal.LEGAL: "there is no legal requirement",
    Signal.SECURITY: "there are no special security implications",
    Signal.COMPLIANCE: "the work is outside any audit scope",
}

UNANSWERABLE_QUESTIONS = [
    "What did the organisation decide about its carbon offset purchasing programme?",
    "Which catering company was selected for the staff cafeteria contract?",
    "What was decided about sponsoring a local sports team?",
    "Which travel agency handles corporate travel bookings?",
    "What budget was approved for the drone delivery trial?",
    "Which venue was chosen for the annual company retreat?",
    "What policy was adopted for bring-your-own-device stipends?",
    "Which decisions concern the trademark renewal strategy?",
    "What did the board decide about opening a quantum computing research lab?",
    "Which provider was selected to run the employee parking allowance scheme?",
]

# Answerable categories and how many questions each contributes (total 40).
CATEGORY_QUOTAS = {
    "provider_lookup": 8,
    "regulation_lookup": 6,
    "framework_lookup": 6,
    "signal_filter": 12,
    "negated_filter": 5,
    "multi_signal_filter": 3,
}

# Attribute pools: naming another value of the same attribute is an incorrect answer.
_FIELD_POOLS = {"vendor": VENDORS, "regulation": REGULATIONS, "framework": FRAMEWORKS}

_LOOKUP_FIELDS = {
    "provider_lookup": ("vendor", "Which external provider is involved in the decision to {title}?"),
    "regulation_lookup": ("regulation", "Which regulation makes the decision to {title} necessary?"),
    "framework_lookup": ("framework", "Which audit framework is relevant to the decision to {title}?"),
}


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _list_reference(docs: list[DecisionDocument]) -> str:
    return "; ".join(f"{d.title} ({d.id})" for d in docs)


def _build_answerable(documents: list[DecisionDocument], rng: random.Random) -> list[Question]:
    by_subject: dict[str, list[DecisionDocument]] = defaultdict(list)
    for doc in documents:
        by_subject[doc.subject].append(doc)
    subjects = sorted(by_subject)

    all_titles = [d.title for d in documents]

    def titles_except(matches: list[DecisionDocument]) -> tuple[str, ...]:
        expected = {d.title for d in matches}
        return tuple(t for t in all_titles if t not in expected)

    def label(doc: DecisionDocument, signal: Signal) -> str:
        return doc.ground_truth_signals[signal.value]

    drafts: list[tuple[str, str, list[DecisionDocument], tuple[str, ...], tuple[str, ...], str]] = []
    used_docs: set[str] = set()

    for category, (field, template) in _LOOKUP_FIELDS.items():
        pool = [d for d in documents if d.metadata.get(field) and d.id not in used_docs]
        rng.shuffle(pool)
        for doc in pool[: CATEGORY_QUOTAS[category]]:
            used_docs.add(doc.id)
            value = doc.metadata[field]
            drafts.append(
                (
                    category,
                    template.format(title=_lower_first(doc.title)),
                    [doc],
                    (value,),
                    tuple(v for v in _FIELD_POOLS[field] if v != value),
                    f"{value} ({doc.id})",
                )
            )

    def filter_candidates(
        signals: tuple[Signal, ...], wanted: str
    ) -> list[tuple[str, list[DecisionDocument], list[DecisionDocument]]]:
        found = []
        for subject in subjects:
            docs = by_subject[subject]
            matches = [d for d in docs if all(label(d, s) == wanted for s in signals)]
            if 1 <= len(matches) <= 4:
                others = [d for d in docs if d not in matches]
                found.append((subject, matches, others))
        return found

    used_pairs: set[tuple[str, str]] = set()
    signal_cycle = list(Signal)
    for index in range(CATEGORY_QUOTAS["signal_filter"]):
        signal = signal_cycle[index % len(signal_cycle)]
        candidates = [
            c for c in filter_candidates((signal,), "PRESENT") if (c[0], signal.value) not in used_pairs
        ]
        subject, matches, _others = rng.choice(candidates)
        used_pairs.add((subject, signal.value))
        phrase = SIGNAL_QUERY_PHRASES[signal][index % len(SIGNAL_QUERY_PHRASES[signal])]
        drafts.append(
            (
                "signal_filter",
                f"Which decisions about the {subject} {phrase}?",
                matches,
                tuple(d.title for d in matches),
                titles_except(matches),
                _list_reference(matches),
            )
        )

    for index in range(CATEGORY_QUOTAS["negated_filter"]):
        signal = signal_cycle[index % len(signal_cycle)]
        candidates = [
            c
            for c in filter_candidates((signal,), "ABSENT")
            if (c[0], f"not-{signal.value}") not in used_pairs
        ]
        subject, matches, _others = rng.choice(candidates)
        used_pairs.add((subject, f"not-{signal.value}"))
        drafts.append(
            (
                "negated_filter",
                f"Which decisions about the {subject} explicitly state that "
                f"{SIGNAL_NEGATED_PHRASES[signal]}?",
                matches,
                tuple(d.title for d in matches),
                titles_except(matches),
                _list_reference(matches),
            )
        )

    pairs = [
        (Signal.LEGAL, Signal.SECURITY),
        (Signal.THIRD_PARTY, Signal.COMPLIANCE),
        (Signal.OPERATIONAL, Signal.SECURITY),
    ]
    for first, second in pairs[: CATEGORY_QUOTAS["multi_signal_filter"]]:
        subject, matches, _others = rng.choice(filter_candidates((first, second), "PRESENT"))
        drafts.append(
            (
                "multi_signal_filter",
                f"Which decisions about the {subject} {SIGNAL_QUERY_PHRASES[first][0]} "
                f"and also {SIGNAL_QUERY_PHRASES[second][0]}?",
                matches,
                tuple(d.title for d in matches),
                titles_except(matches),
                _list_reference(matches),
            )
        )

    return [
        Question(
            id="",
            split="",
            category=category,
            text=text,
            answerable=True,
            expected_doc_ids=tuple(d.id for d in docs),
            reference_answer=reference,
            key_facts=key_facts,
            forbidden_facts=forbidden,
        )
        for category, text, docs, key_facts, forbidden, reference in drafts
    ]


def build_question_set(documents: list[DecisionDocument], seed: int = 909) -> list[Question]:
    """Build the 50 questions and assign the dev/holdout split (10 / 40)."""

    if len(documents) < 150:
        raise ValueError("the evaluation set is defined over the 150-record experimental corpus")
    rng = random.Random(seed)
    answerable = _build_answerable(documents, rng)
    unanswerable = [
        Question(
            id="",
            split="",
            category="unanswerable",
            text=text,
            answerable=False,
            expected_doc_ids=(),
            reference_answer="ABSTAIN: the corpus contains no decision on this topic.",
        )
        for text in UNANSWERABLE_QUESTIONS
    ]

    # Stratified split: 8 answerable + 2 unanswerable go to dev.
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    dev_answerable = _stratified_pick(answerable, 8)
    dev_ids = {id(q) for q in dev_answerable} | {id(q) for q in unanswerable[:2]}

    ordered = answerable + unanswerable
    rng.shuffle(ordered)
    questions: list[Question] = []
    for number, q in enumerate(ordered, start=1):
        questions.append(
            Question(
                id=f"Q-{number:03d}",
                split="dev" if id(q) in dev_ids else "holdout",
                category=q.category,
                text=q.text,
                answerable=q.answerable,
                expected_doc_ids=q.expected_doc_ids,
                reference_answer=q.reference_answer,
                key_facts=q.key_facts,
                forbidden_facts=q.forbidden_facts,
            )
        )
    return questions


def _stratified_pick(questions: list[Question], count: int) -> list[Question]:
    by_category: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_category[q.category].append(q)
    picked: list[Question] = []
    categories = sorted(by_category)
    while len(picked) < count:
        for category in categories:
            if len(picked) == count:
                break
            remaining = [q for q in by_category[category] if q not in picked]
            if remaining:
                picked.append(remaining[0])
    return picked
