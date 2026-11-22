"""Construction of the protocol v3 evaluation set.

``data/v3/question_specs.json`` holds the phrasings per category and the set
sizes, fixed before any run (8 dev + 20 holdout per category). The builder
draws targets with a seeded generator and derives every expected answer from
the generator's recorded facts (``metadata``) and intended labels
(``ground_truth_signals``) — never from the system:

* ``entity_lookup`` / ``fact_lookup`` — a record that states the attribute;
  the key fact is its value, the forbidden facts are the other values of the
  same kind;
* ``entity_missing`` / ``fact_missing`` — the *same phrasings* about a record
  that does not state the attribute (for entities, also no statement either
  way about the signal); these are in-domain unanswerable questions;
* ``signal_filter`` / ``negated_filter`` / ``multi_condition`` — the records
  of a subject whose intended labels satisfy every condition (1–4 records);
* ``out_of_domain`` — hand-written questions about topics no record covers.

Phrasings are used round-robin within a category, so each appears 2–3 times
with different targets. Within each category the first 8 drafts of a seeded
shuffle form the dev split and the rest the holdout.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from decision_intelligence.corpus.generator import (
    AMOUNTS,
    FRAMEWORKS,
    PEOPLE,
    REGULATIONS,
    USER_COUNTS,
    VENDORS,
)
from decision_intelligence.models import DecisionDocument, Question, Signal

MAX_EXPECTED = 4
LOOKUPS = ("entity_lookup", "fact_lookup")
MISSING = ("entity_missing", "fact_missing")
FILTERS = ("signal_filter", "negated_filter", "multi_condition")
CATEGORIES = (*LOOKUPS, *MISSING, *FILTERS, "out_of_domain")
UNANSWERABLE = (*MISSING, "out_of_domain")
IN_DOMAIN_UNANSWERABLE = MISSING
_POOLS: dict[str, list[str]] = {
    "vendor": VENDORS,
    "regulation": REGULATIONS,
    "framework": FRAMEWORKS,
    "budget": AMOUNTS,
    "go_live": [],
    "lead": PEOPLE,
    "users": USER_COUNTS,
}
# The signal whose explicit statement makes an entity question answerable ("none" would be an answer).
_ENTITY_SIGNAL = {
    "vendor": Signal.THIRD_PARTY,
    "regulation": Signal.LEGAL,
    "framework": Signal.COMPLIANCE,
}


class QuestionSpecError(ValueError):
    pass


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _satisfies(doc: DecisionDocument, conditions: list[list[str]]) -> bool:
    return all(doc.ground_truth_signals[Signal(signal).value] == label for signal, label in conditions)


def _lookup_targets(documents: list[DecisionDocument], field: str, present: bool) -> list[DecisionDocument]:
    if present:
        return [d for d in documents if d.metadata.get(field)]
    signal = _ENTITY_SIGNAL.get(field)
    return [
        d
        for d in documents
        if not d.metadata.get(field) and (signal is None or d.ground_truth_signals[signal.value] == "UNCLEAR")
    ]


def _drafts(
    category: str,
    spec: dict[str, Any],
    templates: list[dict[str, Any]],
    documents: list[DecisionDocument],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    drafts: list[dict[str, Any]] = []
    if category == "out_of_domain":
        texts = list(spec["questions"])
        if len(texts) < count:
            raise QuestionSpecError("not enough out-of-domain questions")
        return [{"text": t, "answerable": False, "expected": []} for t in texts[:count]]
    by_subject: dict[str, list[DecisionDocument]] = defaultdict(list)
    for doc in documents:
        by_subject[doc.subject].append(doc)
    used: set[str] = set()
    order = list(templates)
    rng.shuffle(order)
    index = 0
    attempts = 0
    while len(drafts) < count:
        attempts += 1
        if attempts > 50 * count:
            raise QuestionSpecError(f"cannot draw {count} questions for {category}")
        template = order[index % len(order)]
        index += 1
        if category in LOOKUPS or category in MISSING:
            field = template["field"]
            pool = [
                d
                for d in _lookup_targets(documents, field, category in LOOKUPS)
                if f"{d.id}|{field}" not in used
            ]
            if not pool:
                continue
            doc = rng.choice(pool)
            used.add(f"{doc.id}|{field}")
            text = template["text"].format(title=_lower_first(doc.title))
            if category in MISSING:
                # The target is recorded for auditing the gold standard; the system never reads it.
                drafts.append(
                    {
                        "text": text,
                        "answerable": False,
                        "expected": [],
                        "reference": f"ABSTAIN: {doc.id} states no {field}.",
                    }
                )
                continue
            value = doc.metadata[field]
            drafts.append(
                {
                    "text": text,
                    "answerable": True,
                    "expected": [doc],
                    "key_facts": (value,),
                    "forbidden": tuple(v for v in _POOLS[field] if v != value),
                    "reference": f"{value} ({doc.id})",
                }
            )
            continue
        eligible = []
        for subject in sorted(by_subject):
            matches = [d for d in by_subject[subject] if _satisfies(d, template["all"])]
            if 1 <= len(matches) <= MAX_EXPECTED and f"{subject}|{template['text']}" not in used:
                eligible.append((subject, matches))
        if not eligible:
            continue
        subject, matches = rng.choice(eligible)
        used.add(f"{subject}|{template['text']}")
        titles = {d.title for d in matches}
        drafts.append(
            {
                "text": template["text"].format(subject=subject),
                "answerable": True,
                "expected": matches,
                "key_facts": tuple(d.title for d in matches),
                "forbidden": tuple(d.title for d in documents if d.title not in titles),
                "reference": "; ".join(f"{d.title} ({d.id})" for d in matches),
            }
        )
    return drafts


def build_question_set(
    documents: list[DecisionDocument], specs: dict[str, Any], seed: int = 2929
) -> list[Question]:
    """Build every category's drafts, then assign the dev/holdout split per category."""

    rng = random.Random(seed)
    sizes = specs["sizes"]
    per_category = int(sizes["dev"]) + int(sizes["holdout"])
    categories = specs["categories"]
    if set(categories) != set(CATEGORIES):
        raise QuestionSpecError(f"categories must be exactly {sorted(CATEGORIES)}")
    all_drafts: list[tuple[str, str, dict[str, Any]]] = []
    for category in CATEGORIES:
        spec = categories[category]
        templates = (
            categories[spec["templates_from"]]["templates"]
            if "templates_from" in spec
            else spec.get("templates", [])
        )
        drafts = _drafts(category, spec, templates, documents, per_category, rng)
        rng.shuffle(drafts)
        for position, draft in enumerate(drafts):
            all_drafts.append((category, "dev" if position < int(sizes["dev"]) else "holdout", draft))
    texts = [d["text"] for _, _, d in all_drafts]
    if len(texts) != len(set(texts)):
        raise QuestionSpecError("two drafts produced the same question text")
    order = list(range(len(all_drafts)))
    rng.shuffle(order)
    questions = []
    for number, index in enumerate(order, start=1):
        category, split, draft = all_drafts[index]
        questions.append(
            Question(
                id=f"Q-{number:03d}",
                split=split,
                category=category,
                text=draft["text"],
                answerable=draft["answerable"],
                expected_doc_ids=tuple(d.id for d in draft["expected"]),
                reference_answer=draft.get(
                    "reference", "ABSTAIN: no record contains the requested information."
                ),
                key_facts=draft.get("key_facts", ()),
                forbidden_facts=draft.get("forbidden", ()),
            )
        )
    return questions
