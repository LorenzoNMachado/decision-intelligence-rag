"""Answer state machine, claim validation and correctness grading (audit P1-4, P2-1).

Every LLM here is scripted; no provider is contacted.
"""

from __future__ import annotations

from typing import Any

import pytest

from decision_intelligence.evaluation.judges import LexicalJudge, LLMCorrectnessJudge, grade
from decision_intelligence.evaluation.metrics import compute_metrics
from decision_intelligence.generation.generators import ExtractiveGenerator, LLMGenerator
from decision_intelligence.llm.client import LLMError, ScriptedLLM
from decision_intelligence.models import (
    ABSTAIN_TEXT,
    AnswerStatus,
    Claim,
    DecisionDocument,
    GeneratedAnswer,
    Outcome,
    QueryIntent,
    Question,
    RetrievalResult,
    RetrievedDocument,
)


def _doc(doc_id: str, vendor: str) -> DecisionDocument:
    return DecisionDocument(
        id=doc_id,
        title=f"Pilot the tool {doc_id}",
        domain="Finance",
        subject="tool",
        date="2024-01-01",
        status="approved",
        owner_role="Chief Financial Officer",
        decision=f"The Chief Financial Officer approved the pilot of the tool {doc_id}.",
        context=f"Delivery relies on a service contract with {vendor}.",
        rationale="Expected benefits will be reviewed after 6 months.",
        metadata={"vendor": vendor},
    )


DOCS = {"DEC-001": _doc("DEC-001", "Halvard Systems"), "DEC-002": _doc("DEC-002", "Bluefin Hosting")}
QUESTION = Question(
    id="Q-1",
    split="dev",
    category="provider_lookup",
    text="Which provider delivers the tool DEC-001?",
    answerable=True,
    expected_doc_ids=("DEC-001",),
    reference_answer="Halvard Systems (DEC-001)",
    key_facts=("Halvard Systems",),
    forbidden_facts=("Bluefin Hosting",),
)
UNANSWERABLE = Question(
    id="Q-2",
    split="dev",
    category="unanswerable_in_domain",
    text="What budget was approved for the tool?",
    answerable=False,
    expected_doc_ids=(),
    reference_answer="ABSTAIN",
)


def _retrieval(*admitted: str) -> RetrievalResult:
    documents = tuple(
        RetrievedDocument(
            doc_id=d,
            rank=i,
            similarity=0.5,
            final_score=0.5,
            best_chunk_id=f"{d}#c0",
            admitted=True,
            gate_reason="admitted",
        )
        for i, d in enumerate(admitted, start=1)
    )
    return RetrievalResult("vector", QUESTION.text, QueryIntent("none", "NONE"), documents)


def _generate(payload: dict[str, Any] | LLMError, *admitted: str) -> GeneratedAnswer:
    generator = LLMGenerator(ScriptedLLM(lambda s, u: payload), enforce_quotes=True)
    return generator.generate(QUESTION.text, _retrieval(*admitted), DOCS)


GOOD_CLAIM = {
    "statement": "The tool DEC-001 is delivered by Halvard Systems.",
    "source_ids": ["DEC-001"],
    "quote": "Delivery relies on a service contract with Halvard Systems.",
}


# --- state machine ---------------------------------------------------------------


def test_state_machine_rejects_invalid_transitions() -> None:
    with pytest.raises(ValueError):
        GeneratedAnswer("g", AnswerStatus.ANSWER, "x", outcome=Outcome.MODEL_ABSTAIN)
    with pytest.raises(ValueError):
        GeneratedAnswer("g", AnswerStatus.ANSWER, "x", outcome=Outcome.MODEL_ANSWER)  # no claims
    claim = Claim("s", ("DEC-001",), "q", quote_verified=True)
    with pytest.raises(ValueError):
        GeneratedAnswer(
            "g", AnswerStatus.ABSTAIN, ABSTAIN_TEXT, claims=(claim,), outcome=Outcome.MODEL_ANSWER
        )
    with pytest.raises(ValueError):
        GeneratedAnswer("g", AnswerStatus.ABSTAIN, "The provider is Halvard.", outcome=Outcome.MODEL_ANSWER)
    with pytest.raises(ValueError):
        GeneratedAnswer("g", AnswerStatus.ERROR, "partial text", outcome=Outcome.PROVIDER_ERROR)


def test_valid_answer_is_composed_from_validated_claims_only() -> None:
    answer = _generate(
        {
            "status": "ANSWER",
            "answer": "Halvard Systems, and also Sorrel Payments under DEC-150.",
            "claims": [GOOD_CLAIM],
            "abstain_reason": "",
        },
        "DEC-001",
    )
    assert answer.status is AnswerStatus.ANSWER
    assert "Sorrel" not in answer.text and "DEC-150" not in answer.text
    assert answer.text == "- The tool DEC-001 is delivered by Halvard Systems. [DEC-001]"
    assert answer.discarded_text.startswith("Halvard Systems, and also Sorrel")
    assert "DEC-150" in answer.rejected_citations
    (claim,) = answer.claims
    assert claim.quote_verified and claim.quote_origin == "model" and claim.support == "NOT_CHECKED"


def test_citation_to_unadmitted_record_makes_an_abstention_without_leaking_text() -> None:
    answer = _generate(
        {
            "status": "ANSWER",
            "answer": "The provider is Halvard Systems.",
            "claims": [{**GOOD_CLAIM, "source_ids": ["DEC-999"]}],
            "abstain_reason": "",
        },
        "DEC-001",
    )
    assert answer.status is AnswerStatus.ABSTAIN and answer.outcome is Outcome.MODEL_ANSWER
    assert answer.text == ABSTAIN_TEXT and answer.claims == ()
    assert "Halvard" in answer.discarded_text and "Halvard" not in answer.text
    assert answer.rejected_citations == ("DEC-999",)


@pytest.mark.parametrize(
    "claim",
    [
        {**GOOD_CLAIM, "quote": "an invented sentence that is not in the record"},
        {**GOOD_CLAIM, "quote": "Pilot the tool DEC-001"},  # title/header only, not the record body
        {**GOOD_CLAIM, "quote": "Halvard"},  # too short to be evidence
        {**GOOD_CLAIM, "statement": "DEC-002 is also delivered by Halvard Systems."},  # unadmitted id
    ],
)
def test_unverifiable_claims_are_dropped_under_the_enforce_policy(claim: dict[str, Any]) -> None:
    answer = _generate(
        {"status": "ANSWER", "answer": "x", "claims": [claim], "abstain_reason": ""}, "DEC-001"
    )
    assert answer.status is AnswerStatus.ABSTAIN
    assert answer.abstain_reason == "no claim survived citation validation"


def test_record_policy_keeps_unverified_quotes_flagged_not_verified() -> None:
    generator = LLMGenerator(
        ScriptedLLM(
            lambda s, u: {
                "status": "ANSWER",
                "answer": "x",
                "claims": [{**GOOD_CLAIM, "quote": "an invented sentence that is not in the record"}],
                "abstain_reason": "",
            }
        ),
        enforce_quotes=False,
    )
    answer = generator.generate(QUESTION.text, _retrieval("DEC-001"), DOCS)
    assert answer.status is AnswerStatus.ANSWER and answer.claims[0].quote_verified is False


def test_model_abstention_shows_only_the_system_message() -> None:
    answer = _generate(
        {
            "status": "ABSTAIN",
            "answer": "Probably Halvard Systems.",
            "claims": [],
            "abstain_reason": "unsure",
        },
        "DEC-001",
    )
    assert answer.status is AnswerStatus.ABSTAIN and answer.text == ABSTAIN_TEXT
    assert answer.discarded_text == "Probably Halvard Systems."


@pytest.mark.parametrize(
    "payload",
    [
        LLMError("timeout (ReadTimeout)", kind="timeout"),
        {"status": "ANSWER", "answer": "x", "claims": "not a list", "abstain_reason": ""},
        {"status": "MAYBE", "answer": "x", "claims": [], "abstain_reason": ""},
        {"status": "ANSWER", "answer": 42, "claims": [GOOD_CLAIM], "abstain_reason": ""},
    ],
)
def test_provider_and_schema_failures_are_errors_not_abstentions(payload: dict[str, Any] | LLMError) -> None:
    answer = _generate(payload, "DEC-001")
    assert answer.status is AnswerStatus.ERROR and answer.error_kind in {"timeout", "schema"}
    assert answer.text == "" and answer.claims == ()


def test_malformed_claim_items_are_ignored_not_fatal() -> None:
    answer = _generate(
        {
            "status": "ANSWER",
            "answer": "x",
            "claims": ["junk", {"statement": 3, "source_ids": "DEC-001", "quote": None}, GOOD_CLAIM],
            "abstain_reason": "",
        },
        "DEC-001",
    )
    assert answer.status is AnswerStatus.ANSWER and len(answer.claims) == 1


def test_extractive_quotes_are_marked_as_extracted() -> None:
    answer = ExtractiveGenerator().generate(QUESTION.text, _retrieval("DEC-001"), DOCS)
    assert answer.status is AnswerStatus.ANSWER
    assert all(c.quote_origin == "extracted" for c in answer.claims)


# --- grading ---------------------------------------------------------------------


def test_grading_respects_the_final_state() -> None:
    calls: list[str] = []

    def assess(question: Question, text: str) -> str:
        calls.append(text)
        return "CORRECT"

    assert grade(QUESTION, "ABSTAIN", assess).label == "INCORRECT"
    assert grade(QUESTION, "ERROR", assess).label == "INCORRECT"
    assert grade(UNANSWERABLE, "ABSTAIN", assess).label == "CORRECT"
    assert grade(UNANSWERABLE, "ANSWER", assess).label == "INCORRECT"
    assert grade(UNANSWERABLE, "ERROR", assess).label == "INCORRECT"
    assert calls == []  # the assessor is consulted only for answered answerable questions
    assert grade(QUESTION, "ANSWER", assess, text="Halvard Systems").label == "CORRECT"
    assert calls == ["Halvard Systems"]


def test_llm_judge_never_receives_abstentions() -> None:
    llm = ScriptedLLM(lambda s, u: {"label": "CORRECT", "rationale": "names the provider"})
    judge = LLMCorrectnessJudge(llm)
    assert judge.judge(QUESTION, "ABSTAIN", "The provider is Halvard Systems.").label == "INCORRECT"
    assert llm.calls == []
    assert judge.judge(QUESTION, "ANSWER", "Halvard Systems").label == "CORRECT"
    assert len(llm.calls) == 1


def _row(status: str, text: str, claims: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "config": "C",
        "rep": 1,
        "question_id": QUESTION.id,
        "retrieved": [{"doc_id": "DEC-001", "admitted": True}],
        "intent": {"status": "NONE"},
        "answer": {
            "status": status,
            "text": text,
            "claims": claims or [],
            "rejected_citations": [],
        },
        "latency_ms": 1.0,
        "cost_usd": 0.0,
    }


def test_metrics_ignore_a_judge_label_attached_to_an_abstention() -> None:
    rows = [_row("ABSTAIN", ABSTAIN_TEXT)]
    metrics = compute_metrics(rows, {QUESTION.id: QUESTION}, correctness=lambda r: "CORRECT")
    assert metrics["correctness"] == 0.0 and metrics["false_abstention_rate"] == 1.0


def test_metrics_keep_operational_errors_separate_and_incorrect() -> None:
    rows = [_row("ERROR", "")]
    metrics = compute_metrics(rows, {QUESTION.id: QUESTION})
    assert metrics["operational_failures"] == 1
    assert metrics["correctness"] == 0.0
    assert metrics["false_abstention_rate"] == 0.0


def test_faithfulness_is_not_reported_for_extracted_quotes() -> None:
    claim = {
        "statement": "s",
        "source_ids": ["DEC-001"],
        "quote": "q",
        "quote_verified": True,
        "quote_origin": "extracted",
    }
    metrics = compute_metrics([_row("ANSWER", "Halvard Systems", [claim])], {QUESTION.id: QUESTION})
    assert metrics["faithfulness_quote"] is None and metrics["extracted_quotes"] == 1


def test_lexical_judge_follows_the_same_rule() -> None:
    assert LexicalJudge().judge(QUESTION, "ABSTAIN", "Halvard Systems").label == "INCORRECT"
    assert LexicalJudge().judge(QUESTION, "ANSWER", "Halvard Systems").label == "CORRECT"
