"""EXPERIMENTAL EXTENSION: LLM verification of individual claims.

This step is an addition to the original experimental design and has not been
validated against human judgements. Literal quote matching only shows that a
quote is traceable to a source; this verifier asks a separate LLM whether the
cited sources actually entail the claim. Its labels feed the citation
faithfulness metric when an LLM judge is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from decision_intelligence.generation.prompts import load_prompt
from decision_intelligence.grounding.evidence import render_document
from decision_intelligence.llm.client import LLMClient, LLMError
from decision_intelligence.models import DecisionDocument

CLAIM_LABELS = ("SUPPORTED", "PARTIAL", "NOT_SUPPORTED", "CONTRADICTED")

CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["label", "rationale"],
    "properties": {"label": {"type": "string", "enum": list(CLAIM_LABELS)}, "rationale": {"type": "string"}},
}


@dataclass(frozen=True)
class ClaimVerdict:
    label: str  # one of CLAIM_LABELS, or "ERROR"
    rationale: str
    cost_usd: float | None = 0.0


class ClaimVerifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.prompt = load_prompt("claim_verification_v1")

    @property
    def name(self) -> str:
        return f"{self.prompt.version}:{self.llm.name}"

    def verify(self, statement: str, sources: list[DecisionDocument]) -> ClaimVerdict:
        user = "CLAIM: " + statement + "\n\nSOURCES:\n" + "\n\n".join(render_document(d) for d in sources)
        try:
            response = self.llm.complete_json(
                self.prompt.system, user, CLAIM_SCHEMA, "claim_verdict", max_tokens=400
            )
        except LLMError as exc:
            return ClaimVerdict(label="ERROR", rationale=f"{exc.kind}: {exc}")
        label = str(response.parsed.get("label", ""))
        if label not in CLAIM_LABELS:
            return ClaimVerdict(label="ERROR", rationale="invalid label", cost_usd=response.cost_usd)
        return ClaimVerdict(
            label=label, rationale=str(response.parsed.get("rationale", ""))[:500], cost_usd=response.cost_usd
        )
