"""Evidence sufficiency (protocol v3): does an admitted record contain what the question asks for?

The documents here are hand-written for the tests and unrelated to any protocol's
corpus. The checker only sees the question text, the retrieval result and the
records: never a reference answer or a question category.
"""

from __future__ import annotations

import inspect

import pytest

from decision_intelligence.models import (
    DecisionDocument,
    QueryIntent,
    RetrievalResult,
    RetrievedDocument,
)
from decision_intelligence.sufficiency import (
    AnswerType,
    Assertion,
    EvidenceSufficiencyChecker,
    Sufficiency,
    analyse_need,
    extract_facts,
)

SUBJECT = "payroll platform"
ENTITIES = (SUBJECT, "People Operations", "VP of People", "records archive")
SCOPES = (SUBJECT, "records archive", "People Operations")


def _doc(doc_id: str, action: str, *sentences: str, subject: str = SUBJECT) -> DecisionDocument:
    return DecisionDocument(
        id=doc_id,
        title=f"{action} {subject}",
        domain="People Operations",
        subject=subject,
        date="2025-01-01",
        status="approved",
        owner_role="VP of People",
        decision=f"The VP of People approved the proposal to {action.lower()} {subject}.",
        context=" ".join(sentences[: len(sentences) // 2]),
        rationale=" ".join(sentences[len(sentences) // 2 :]),
    )


def _retrieval(question: str, *docs: DecisionDocument) -> RetrievalResult:
    return RetrievalResult(
        strategy="vector",
        question=question,
        intent=QueryIntent("none", "NONE"),
        documents=tuple(
            RetrievedDocument(d.id, i, 0.5, 0.5, f"{d.id}#c0", True, "admitted")
            for i, d in enumerate(docs, 1)
        ),
    )


def _check(question: str, *docs: DecisionDocument):  # type: ignore[no-untyped-def]
    checker = EvidenceSufficiencyChecker(entities=ENTITIES, scopes=SCOPES)
    return checker.check(question, _retrieval(question, *docs), {d.id: d for d in docs})


# --- information need -------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "answer_type"),
    [
        ("What budget was approved for the decision to adopt the payroll platform?", AnswerType.MONEY),
        ("How much funding was set aside for the decision to adopt the payroll platform?", AnswerType.MONEY),
        ("When does the decision to adopt the payroll platform take effect?", AnswerType.DATE),
        ("On which date will the decision to adopt the payroll platform go live?", AnswerType.DATE),
        ("Who leads the work on the decision to adopt the payroll platform?", AnswerType.PERSON),
        (
            "How many employees will use the result of the decision to adopt the payroll platform?",
            AnswerType.QUANTITY,
        ),
        ("Which firm delivers the decision to adopt the payroll platform?", AnswerType.ORGANIZATION),
        (
            "Under which statute was the decision to adopt the payroll platform required?",
            AnswerType.REGULATION,
        ),
        (
            "Which certification does the decision to adopt the payroll platform support?",
            AnswerType.FRAMEWORK,
        ),
        ("Why was the decision to adopt the payroll platform taken?", AnswerType.REASON),
        ("Which decisions about the payroll platform depend on an outside vendor?", AnswerType.DECISIONS),
    ],
)
def test_answer_type_is_read_from_question_structure(question: str, answer_type: AnswerType) -> None:
    assert analyse_need(question, entities=ENTITIES).answer_type is answer_type


def test_attribute_questions_resolve_the_referenced_decision() -> None:
    need = analyse_need(
        "What budget was approved for the decision to adopt the payroll platform?", entities=ENTITIES
    )
    assert need.kind == "attribute" and "budget" in need.focus and need.reference.endswith("payroll platform")


# --- typed facts and assertion scope ---------------------------------------------------


def test_fact_extraction_recognises_typed_values() -> None:
    facts = extract_facts(
        "The approved budget is €240,000. Go-live is scheduled for 14 March 2025. "
        "The work is led by Dana Whitfield. About 1,200 employees will use it. "
        "Delivery relies on a service contract with Halvard Systems. "
        "It must follow the Regional Data Protection Act and feeds the ISO/IEC 27001 audit."
    )
    types = {m.type for f in facts for m in f.mentions}
    assert {
        AnswerType.MONEY,
        AnswerType.DATE,
        AnswerType.PERSON,
        AnswerType.QUANTITY,
        AnswerType.ORGANIZATION,
        AnswerType.REGULATION,
        AnswerType.FRAMEWORK,
    } <= types


@pytest.mark.parametrize(
    ("sentence", "assertion"),
    [
        ("The approved budget is €240,000.", Assertion.FACTUAL),
        ("A budget of €50,000 may be requested next year.", Assertion.HYPOTHETICAL),
        ("If the pilot succeeds, a vendor could be engaged.", Assertion.HYPOTHETICAL),
        ("The previous contract was worth €90,000.", Assertion.HISTORICAL),
        ("Until last year the work relied on an outside firm.", Assertion.HISTORICAL),
    ],
)
def test_sentence_assertion_scope(sentence: str, assertion: Assertion) -> None:
    assert extract_facts(sentence)[0].assertion is assertion


# --- attribute questions ----------------------------------------------------------------

BUDGET_Q = "What budget was approved for the decision to adopt the payroll platform?"


def test_fact_present_is_sufficient_and_points_at_the_sentence() -> None:
    doc = _doc("DEC-001", "Adopt the", "Usage doubled last year.", "The approved budget is €240,000.")
    report = _check(BUDGET_Q, doc)
    assert report.status is Sufficiency.SUFFICIENT
    assert report.documents[0].supporting == ("The approved budget is €240,000.",)


def test_fact_absent_is_insufficient_even_for_a_similar_record() -> None:
    doc = _doc(
        "DEC-001", "Adopt the", "Usage doubled last year.", "The steering group compared three options."
    )
    report = _check(BUDGET_Q, doc)
    assert report.status is Sufficiency.INSUFFICIENT
    assert report.documents[0].status is Sufficiency.INSUFFICIENT


def test_value_of_the_right_type_about_another_attribute_is_only_partial() -> None:
    doc = _doc("DEC-001", "Adopt the", "The previous contract was worth €90,000.", "Usage doubled last year.")
    report = _check(BUDGET_Q, doc)
    assert report.status is Sufficiency.INSUFFICIENT
    assert report.documents[0].status is Sufficiency.PARTIAL


def test_hypothetical_value_is_not_evidence() -> None:
    doc = _doc("DEC-001", "Adopt the", "A budget of €50,000 may be requested next year.", "Usage doubled.")
    assert _check(BUDGET_Q, doc).status is Sufficiency.INSUFFICIENT


def test_a_sibling_record_with_the_fact_does_not_answer_for_the_referenced_one() -> None:
    adopt = _doc("DEC-001", "Adopt the", "Usage doubled last year.", "The steering group met twice.")
    pilot = _doc("DEC-002", "Pilot the", "The approved budget is €240,000.", "Usage doubled.")
    report = _check(BUDGET_Q, adopt, pilot)
    assert report.status is Sufficiency.INSUFFICIENT
    assert {d.doc_id: d.status for d in report.documents}["DEC-002"] is Sufficiency.OFF_TOPIC


@pytest.mark.parametrize(
    ("question", "present", "absent"),
    [
        (
            "When will the decision to adopt the payroll platform go live?",
            "Go-live is scheduled for 14 March 2025.",
            "A review is planned for May 2025.",
        ),
        (
            "Who leads the work on the decision to adopt the payroll platform?",
            "The work is led by Dana Whitfield.",
            "Stakeholders from finance were consulted.",
        ),
        (
            "How many employees will use the result of the decision to adopt the payroll platform?",
            "About 1,200 employees will use the new platform.",
            "Usage has roughly doubled since the last review.",
        ),
        (
            "Which company delivers the work for the decision to adopt the payroll platform?",
            "Delivery relies on a service contract with Halvard Systems.",
            "The vendor management office reviewed the business case.",
        ),
        (
            "Why was the decision to adopt the payroll platform taken?",
            "The change is required to meet obligations under the Regional Data Protection Act.",
            "The estimated effort fits within the current planning cycle.",
        ),
    ],
)
def test_dates_people_numbers_organisations_and_reasons(question: str, present: str, absent: str) -> None:
    assert (
        _check(question, _doc("DEC-001", "Adopt the", present, "Usage doubled.")).status
        is Sufficiency.SUFFICIENT
    )
    assert (
        _check(question, _doc("DEC-001", "Adopt the", absent, "Usage doubled.")).status
        is Sufficiency.INSUFFICIENT
    )


# --- list questions with conditions -----------------------------------------------------

MULTI_Q = (
    "Which decisions about the payroll platform depend on an outside vendor and have no legal requirement?"
)


def test_every_condition_must_be_asserted_by_the_record() -> None:
    both = _doc(
        "DEC-001", "Adopt the", "Delivery relies on Halvard Systems.", "There is no legal requirement."
    )
    one = _doc("DEC-002", "Pilot the", "Delivery relies on Halvard Systems.", "Usage doubled.")
    against = _doc(
        "DEC-003", "Expand the", "Delivery relies on Halvard Systems.", "The Data Act makes it mandatory."
    )
    report = _check(MULTI_Q, both, one, against)
    statuses = {d.doc_id: d.status for d in report.documents}
    assert report.status is Sufficiency.SUFFICIENT
    assert statuses == {
        "DEC-001": Sufficiency.SUFFICIENT,
        "DEC-002": Sufficiency.INSUFFICIENT,
        "DEC-003": Sufficiency.INSUFFICIENT,
    }


def test_contradictory_record_is_ambiguous() -> None:
    doc = _doc(
        "DEC-001", "Adopt the", "Delivery relies on Halvard Systems.", "No external vendor is involved."
    )
    report = _check("Which decisions about the payroll platform depend on an outside vendor?", doc)
    assert report.documents[0].status is Sufficiency.AMBIGUOUS
    assert report.status is not Sufficiency.SUFFICIENT


def test_historical_and_contextual_mentions_do_not_satisfy_a_condition() -> None:
    history = _doc(
        "DEC-001",
        "Adopt the",
        "Until last year the work relied on Halvard Systems; it has since been brought in-house.",
        "Usage doubled.",
    )
    context = _doc(
        "DEC-002", "Pilot the", "The vendor management office reviewed the business case.", "Usage doubled."
    )
    positive = _check(
        "Which decisions about the payroll platform depend on an outside vendor?", history, context
    )
    assert positive.status is Sufficiency.INSUFFICIENT
    negative = _check("Which decisions about the payroll platform involve no outside vendor?", history)
    assert negative.status is Sufficiency.SUFFICIENT


def test_records_of_another_subject_are_off_topic() -> None:
    other = _doc(
        "DEC-009", "Adopt the", "No external vendor is involved.", "Usage doubled.", subject="records archive"
    )
    report = _check("Which decisions about the payroll platform involve no outside vendor?", other)
    assert report.documents[0].status is Sufficiency.OFF_TOPIC and report.status is Sufficiency.INSUFFICIENT


def test_nothing_admitted_is_insufficient() -> None:
    checker = EvidenceSufficiencyChecker(entities=ENTITIES, scopes=SCOPES)
    empty = RetrievalResult("vector", BUDGET_Q, QueryIntent("none", "NONE"), ())
    assert checker.check(BUDGET_Q, empty, {}).status is Sufficiency.INSUFFICIENT


def test_checker_never_receives_reference_answers() -> None:
    parameters = set(inspect.signature(EvidenceSufficiencyChecker.check).parameters)
    assert parameters == {"self", "question", "retrieval", "documents"}


def test_focus_words_survive_when_they_also_occur_in_an_entity_name() -> None:
    # Regression: an entity called "budget planning tool" used to remove "budget" from every question.
    need = analyse_need(BUDGET_Q, entities=(*ENTITIES, "budget planning tool"))
    assert "budget" in need.focus


def test_copular_who_question_asks_for_the_named_type() -> None:
    need = analyse_need(
        "Who is the external supplier behind the decision to adopt the payroll platform?", ENTITIES
    )
    assert need.answer_type is AnswerType.ORGANIZATION


def test_an_attribute_word_that_also_occurs_in_the_title_stays_in_the_focus() -> None:
    doc = DecisionDocument(
        id="DEC-001",
        title="Introduce a budget planning tool",
        domain="Finance",
        subject="budget planning tool",
        date="2025-01-01",
        status="approved",
        owner_role="Controller",
        decision="The Controller approved the proposal to introduce a budget planning tool.",
        context="Usage doubled.",
        rationale="The project budget is capped at €1.2 million.",
    )
    question = "What budget was approved for the decision to introduce a budget planning tool?"
    checker = EvidenceSufficiencyChecker(entities=("budget planning tool",), scopes=("budget planning tool",))
    report = checker.check(question, _retrieval(question, doc), {doc.id: doc})
    assert report.status is Sufficiency.SUFFICIENT


def test_a_person_named_in_passing_is_not_taken_for_an_organisation() -> None:
    question = "Which company runs the service for the decision to adopt the payroll platform?"
    consulted = _doc("DEC-001", "Adopt the", "Ines Marchetti from finance was consulted.", "Usage doubled.")
    assert _check(question, consulted).status is Sufficiency.INSUFFICIENT
    provided = _doc(
        "DEC-001", "Adopt the", "The platform itself is provided by Arden Field Systems.", "Usage doubled."
    )
    assert _check(question, provided).status is Sufficiency.SUFFICIENT
