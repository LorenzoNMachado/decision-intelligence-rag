"""Answer-correctness grading.

``grade`` applies the answer state machine before any judge is consulted:

* unanswerable question: ABSTAIN is CORRECT; ANSWER and ERROR are INCORRECT;
* answerable question: ABSTAIN and ERROR are INCORRECT (operational errors are
  also counted separately by the metrics);
* answerable question answered (ANSWER): the assessor decides.

So no judge ever sees an abstention, an error or text that validation rejected.

``LexicalJudge`` is a deterministic assessor that needs no model. It checks that
every key fact of the reference appears in the answer and that no forbidden
fact (a decision title or attribute value that should not be in the answer)
appears. It cannot recognise paraphrases, so it under-credits fluent LLM
answers; reports always label it as a proxy.

``LLMCorrectnessJudge`` compares the answer with the reference using an LLM,
without access to the sources.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from decision_intelligence.generation.prompts import load_prompt
from decision_intelligence.llm.client import LLMClient, LLMError
from decision_intelligence.models import Question

LABELS = ("CORRECT", "PARTIAL", "INCORRECT")


@dataclass(frozen=True)
class Verdict:
    label: str  # CORRECT | PARTIAL | INCORRECT | ERROR (judge failure)
    rationale: str = ""
    cost_usd: float | None = 0.0
    assessed: bool = False  # True when an assessor (not the state rule) produced the label


def grade(
    question: Question,
    status: str,
    assess: Callable[[Question, str], str | Verdict],
    text: str = "",
) -> Verdict:
    """Correctness of a final answer; ``assess`` is called only for answered answerable questions."""

    if not question.answerable:
        return Verdict("CORRECT" if status == "ABSTAIN" else "INCORRECT", f"unanswerable question ({status})")
    if status != "ANSWER":
        return Verdict("INCORRECT", f"no answer ({status})")
    result = assess(question, text)
    if isinstance(result, Verdict):
        return Verdict(result.label, result.rationale, result.cost_usd, assessed=True)
    return Verdict(result, assessed=True)


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


class LexicalJudge:
    name = "lexical-key-facts-v1"

    @staticmethod
    def assess(question: Question, answer_text: str) -> Verdict:
        found = [f for f in question.key_facts if _contains(answer_text, f)]
        forbidden = [f for f in question.forbidden_facts if _contains(answer_text, f)]
        if len(found) == len(question.key_facts) and not forbidden:
            return Verdict("CORRECT", "all key facts, no forbidden facts")
        if found:
            return Verdict(
                "PARTIAL", f"{len(found)}/{len(question.key_facts)} key facts, {len(forbidden)} forbidden"
            )
        return Verdict("INCORRECT", "no key fact found")

    def judge(self, question: Question, status: str, answer_text: str) -> Verdict:
        return grade(question, status, self.assess, answer_text)


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "rationale"],
    "properties": {"label": {"type": "string", "enum": list(LABELS)}, "rationale": {"type": "string"}},
}


class LLMCorrectnessJudge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.prompt = load_prompt("judge_correctness_v2")

    @property
    def name(self) -> str:
        return f"{self.prompt.version}:{self.llm.name}"

    def assess(self, question: Question, answer_text: str) -> Verdict:
        user = f"QUESTION: {question.text}\n\nREFERENCE: {question.reference_answer}\n\nANSWER: {answer_text}"
        try:
            response = self.llm.complete_json(
                self.prompt.system, user, JUDGE_SCHEMA, "correctness", max_tokens=400
            )
        except LLMError as exc:
            return Verdict("ERROR", f"{exc.kind}: {exc}", None)
        label = response.parsed.get("label")
        rationale = response.parsed.get("rationale")
        if label not in LABELS:
            return Verdict("ERROR", "invalid label", response.cost_usd)
        return Verdict(
            str(label), str(rationale)[:500] if isinstance(rationale, str) else "", response.cost_usd
        )

    def judge(self, question: Question, status: str, answer_text: str) -> Verdict:
        return grade(question, status, self.assess, answer_text)
