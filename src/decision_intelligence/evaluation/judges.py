"""Answer-correctness judges.

``LexicalJudge`` is a deterministic proxy that needs no model. It checks that
every key fact of the reference appears in the answer and that no forbidden
fact (a decision title or attribute value that should not be in the answer)
appears. It cannot recognise paraphrases, so it under-credits fluent LLM
answers; reports always label it as a proxy.

``LLMCorrectnessJudge`` compares the answer with the reference using an LLM,
without access to the sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decision_intelligence.generation.prompts import load_prompt
from decision_intelligence.llm.client import LLMClient, LLMError
from decision_intelligence.models import Question

LABELS = ("CORRECT", "PARTIAL", "INCORRECT")


@dataclass(frozen=True)
class Verdict:
    label: str  # CORRECT | PARTIAL | INCORRECT | ERROR
    rationale: str = ""
    cost_usd: float | None = 0.0


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


class LexicalJudge:
    name = "lexical-key-facts-v1"

    def judge(self, question: Question, status: str, answer_text: str) -> Verdict:
        if not question.answerable:
            return Verdict("CORRECT" if status == "ABSTAIN" else "INCORRECT", "unanswerable question")
        if status != "ANSWER":
            return Verdict("INCORRECT", f"no answer ({status})")
        found = [f for f in question.key_facts if _contains(answer_text, f)]
        forbidden = [f for f in question.forbidden_facts if _contains(answer_text, f)]
        if len(found) == len(question.key_facts) and not forbidden:
            return Verdict("CORRECT", "all key facts, no forbidden facts")
        if found:
            return Verdict(
                "PARTIAL", f"{len(found)}/{len(question.key_facts)} key facts, {len(forbidden)} forbidden"
            )
        return Verdict("INCORRECT", "no key fact found")


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "rationale"],
    "properties": {"label": {"type": "string", "enum": list(LABELS)}, "rationale": {"type": "string"}},
}


class LLMCorrectnessJudge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.prompt = load_prompt("judge_correctness_v1")

    @property
    def name(self) -> str:
        return f"{self.prompt.version}:{self.llm.name}"

    def judge(self, question: Question, status: str, answer_text: str) -> Verdict:
        shown = answer_text if status == "ANSWER" else f"[The system abstained] {answer_text}"
        user = f"QUESTION: {question.text}\n\nREFERENCE: {question.reference_answer}\n\nANSWER: {shown}"
        try:
            response = self.llm.complete_json(
                self.prompt.system, user, JUDGE_SCHEMA, "correctness", max_tokens=400
            )
        except LLMError as exc:
            return Verdict("ERROR", f"{exc.kind}: {exc}")
        label = str(response.parsed.get("label", ""))
        if label not in LABELS:
            return Verdict("ERROR", "invalid label", response.cost_usd)
        return Verdict(label, str(response.parsed.get("rationale", ""))[:500], response.cost_usd)
