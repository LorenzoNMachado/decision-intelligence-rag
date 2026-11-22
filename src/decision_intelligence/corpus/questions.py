"""Construction of the protocol v2 evaluation set from hand-written specifications.

``data/v2/question_specs.json`` holds one hand-written phrasing per question
(no phrasing is reused) with a category and, for filters, the constraint it
expresses. ``build_question_set`` turns the specifications into questions:

* the subject (filters) or record (lookups, in-domain unanswerables) is drawn
  with a seeded generator from the eligible ones, so the author did not choose
  which records each question targets;
* expected evidence is derived from the generator's intended labels: a filter
  matches the records of the subject that satisfy ``all`` (conjunction) or
  ``any`` (disjunction) of its (signal, label) constraints, and a filter is
  eligible only if 1–4 records match (the top-k is 4);
* the dev/holdout split is drawn per category in proportion (``DEV_FRACTION``,
  at least one per category in each split), so both splits have the same mix.

Because references come from the same generator as the corpus, they are not
independent annotations; ``docs/limitations.md`` discusses the consequences.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from decision_intelligence.corpus.generator import FRAMEWORKS, REGULATIONS, VENDORS
from decision_intelligence.models import DecisionDocument, Question, Signal

DEV_FRACTION = 0.3
MAX_EXPECTED = 4
UNANSWERABLE = ("unanswerable_in_domain", "unanswerable_out_of_domain")
LOOKUPS = ("provider_lookup", "regulation_lookup", "framework_lookup")
FILTERS = ("signal_filter", "negated_filter", "conjunction_filter", "disjunction_filter", "hedged_filter")
CATEGORIES = LOOKUPS + FILTERS + UNANSWERABLE
_FIELD_POOLS = {"vendor": VENDORS, "regulation": REGULATIONS, "framework": FRAMEWORKS}


class QuestionSpecError(ValueError):
    pass


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _list_reference(docs: list[DecisionDocument]) -> str:
    return "; ".join(f"{d.title} ({d.id})" for d in docs)


def _matches(doc: DecisionDocument, spec: dict[str, Any]) -> bool:
    def holds(constraint: list[str]) -> bool:
        signal, label = constraint
        return doc.ground_truth_signals[Signal(signal).value] == label

    if "all" in spec:
        return all(holds(c) for c in spec["all"])
    return any(holds(c) for c in spec["any"])


def _draft(
    spec: dict[str, Any],
    documents: list[DecisionDocument],
    by_subject: dict[str, list[DecisionDocument]],
    rng: random.Random,
    used: set[str],
) -> dict[str, Any]:
    category = spec["category"]
    titles = [d.title for d in documents]
    if category in LOOKUPS:
        field = spec["field"]
        pool = [d for d in documents if d.metadata.get(field) and d.id not in used]
        if not pool:
            raise QuestionSpecError(f"no record left with {field} for: {spec['text']}")
        doc = rng.choice(pool)
        used.add(doc.id)
        value = doc.metadata[field]
        return {
            "text": spec["text"].format(title=_lower_first(doc.title)),
            "answerable": True,
            "expected": [doc],
            "key_facts": (value,),
            "forbidden": tuple(v for v in _FIELD_POOLS[field] if v != value),
            "reference": f"{value} ({doc.id})",
        }
    if category == "unanswerable_in_domain":
        doc = rng.choice([d for d in documents if d.id not in used])
        used.add(doc.id)
        return {
            "text": spec["text"].format(title=_lower_first(doc.title)),
            "answerable": False,
            "expected": [],
        }
    if category == "unanswerable_out_of_domain":
        return {"text": spec["text"], "answerable": False, "expected": []}
    if category not in FILTERS:
        raise QuestionSpecError(f"unknown category {category!r}")
    eligible = []
    for subject in sorted(by_subject):
        key = f"{subject}|{spec['text']}"
        matches = [d for d in by_subject[subject] if _matches(d, spec)]
        if 1 <= len(matches) <= MAX_EXPECTED and key not in used:
            eligible.append((subject, matches))
    if not eligible:
        raise QuestionSpecError(f"no subject with 1-{MAX_EXPECTED} matching records for: {spec['text']}")
    subject, matches = rng.choice(eligible)
    used.add(f"{subject}|{spec['text']}")
    expected_titles = {d.title for d in matches}
    return {
        "text": spec["text"].format(subject=subject),
        "answerable": True,
        "expected": matches,
        "key_facts": tuple(d.title for d in matches),
        "forbidden": tuple(t for t in titles if t not in expected_titles),
        "reference": _list_reference(matches),
    }


def build_question_set(
    documents: list[DecisionDocument], specs: list[dict[str, Any]], seed: int = 1717
) -> list[Question]:
    """Build the questions and assign the stratified dev/holdout split."""

    texts = [s["text"] for s in specs]
    if len(texts) != len(set(texts)):
        raise QuestionSpecError("every specification must have its own phrasing")
    rng = random.Random(seed)
    by_subject: dict[str, list[DecisionDocument]] = defaultdict(list)
    for doc in documents:
        by_subject[doc.subject].append(doc)
    used: set[str] = set()
    drafts = []
    for spec in specs:
        draft = _draft(spec, documents, by_subject, rng, used)
        drafts.append((spec["category"], draft))

    by_category: dict[str, list[int]] = defaultdict(list)
    for index, (category, _) in enumerate(drafts):
        by_category[category].append(index)
    dev: set[int] = set()
    for category in sorted(by_category):
        indices = by_category[category]
        count = min(len(indices) - 1, max(1, round(len(indices) * DEV_FRACTION)))
        dev.update(rng.sample(indices, count))

    order = list(range(len(drafts)))
    rng.shuffle(order)
    questions = []
    for number, index in enumerate(order, start=1):
        category, draft = drafts[index]
        questions.append(
            Question(
                id=f"Q-{number:03d}",
                split="dev" if index in dev else "holdout",
                category=category,
                text=draft["text"],
                answerable=draft["answerable"],
                expected_doc_ids=tuple(d.id for d in draft["expected"]),
                reference_answer=draft.get(
                    "reference", "ABSTAIN: the corpus contains no record with this information."
                ),
                key_facts=draft.get("key_facts", ()),
                forbidden_facts=draft.get("forbidden", ()),
            )
        )
    return questions
