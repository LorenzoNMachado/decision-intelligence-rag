"""Metric definitions over experiment rows.

A *row* is one (configuration, repetition, question) execution as written by
the runner. Definitions (see ``docs/evaluation.md``):

* Recall@k — answerable rows: |expected ∩ top-k retrieved| / |expected|.
* Supporting-source precision — answerable rows that answered with ≥1 citation:
  |cited ∩ expected| / |cited|.
* Expected-source coverage — same rows: |cited ∩ expected| / |expected|.
* Answer correctness — answerable rows labelled CORRECT / all answerable rows.
  Abstentions and errors on answerable questions count as incorrect.
* Citation faithfulness — claims whose quote is literally present in a cited
  source / all claims (``quote``); with an LLM verifier, SUPPORTED / judged claims.
* Correct abstention rate — unanswerable rows that abstained / unanswerable rows.
* False-answer rate — unanswerable rows that answered / unanswerable rows.
* False-abstention rate — answerable rows that abstained / answerable rows.
* Latency p50/p95 — end-to-end pipeline latency in milliseconds.
* Cost per question — mean USD per row (intent + generation calls).

Operational failures (status ERROR, e.g. timeouts) are counted separately and
are neither abstentions nor answers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from decision_intelligence.evaluation.judges import LexicalJudge
from decision_intelligence.models import Question

Row = dict[str, Any]


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def cited_ids(row: Row) -> list[str]:
    seen: dict[str, None] = {}
    for claim in row["answer"]["claims"]:
        for source in claim["source_ids"]:
            seen.setdefault(source, None)
    return list(seen)


def enforce_quotes(row: Row) -> Row:
    """Replay a row as if unverified quotes had been rejected (no new model call)."""

    answer = dict(row["answer"])
    if answer["status"] != "ANSWER":
        return row
    claims = [c for c in answer["claims"] if c["quote_verified"]]
    answer["claims"] = claims
    if not claims:
        answer["status"] = "ABSTAIN"
        answer["abstain_reason"] = "replay: no claim with a verified quote"
    return {**row, "answer": answer}


def compute_metrics(
    rows: list[Row],
    questions: dict[str, Question],
    correctness: Callable[[Row], str] | None = None,
    claim_labels: dict[tuple[str, int], str] | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    lexical = LexicalJudge()
    answerable = [r for r in rows if questions[r["question_id"]].answerable]
    unanswerable = [r for r in rows if not questions[r["question_id"]].answerable]

    recalls = []
    precisions = []
    coverages = []
    irrelevant_cited = 0
    for row in answerable:
        expected = set(questions[row["question_id"]].expected_doc_ids)
        top = [d["doc_id"] for d in row["retrieved"][:top_k]]
        recalls.append(len(expected & set(top)) / len(expected))
        cited = cited_ids(row)
        if row["answer"]["status"] == "ANSWER" and cited:
            hits = len(expected & set(cited))
            precisions.append(hits / len(cited))
            coverages.append(hits / len(expected))
            irrelevant_cited += len(set(cited) - expected)

    def label_of(row: Row) -> str:
        if correctness is not None:
            return correctness(row)
        q = questions[row["question_id"]]
        return lexical.judge(q, row["answer"]["status"], row["answer"]["text"]).label

    labels = [label_of(r) for r in answerable]
    judged = [label for label in labels if label != "ERROR"]

    claims = [c for r in rows if r["answer"]["status"] == "ANSWER" for c in r["answer"]["claims"]]
    llm_claims = list(claim_labels.values()) if claim_labels else []
    llm_claims_judged = [label for label in llm_claims if label != "ERROR"]

    def status_count(subset: list[Row], status: str) -> int:
        return sum(r["answer"]["status"] == status for r in subset)

    latencies = [float(r["latency_ms"]) for r in rows]
    costs = [r["cost_usd"] for r in rows]
    known_costs = [float(c) for c in costs if c is not None]
    repetitions = len({r["rep"] for r in rows}) or 1
    return {
        "rows": len(rows),
        "answerable_rows": len(answerable),
        "unanswerable_rows": len(unanswerable),
        "recall_at_k": _mean(recalls),
        "supporting_source_precision": _mean(precisions),
        "expected_source_coverage": _mean(coverages),
        "correctness": _rate(sum(label == "CORRECT" for label in judged), len(judged)),
        "correctness_partial": _rate(sum(label == "PARTIAL" for label in judged), len(judged)),
        "correctness_judge_errors": len(labels) - len(judged),
        "faithfulness_quote": _rate(sum(bool(c["quote_verified"]) for c in claims), len(claims)),
        "faithfulness_llm": _rate(
            sum(label == "SUPPORTED" for label in llm_claims_judged), len(llm_claims_judged)
        )
        if claim_labels
        else None,
        "claims": len(claims),
        "correct_abstention_rate": _rate(status_count(unanswerable, "ABSTAIN"), len(unanswerable)),
        "false_answer_rate": _rate(status_count(unanswerable, "ANSWER"), len(unanswerable)),
        "false_abstention_rate": _rate(status_count(answerable, "ABSTAIN"), len(answerable)),
        "operational_failures": sum(r["answer"]["status"] == "ERROR" for r in rows),
        "intent_failures": sum(r["intent"]["status"] == "UNAVAILABLE" for r in rows),
        "rejected_citations": sum(len(r["answer"]["rejected_citations"]) for r in rows),
        "irrelevant_sources_cited_per_rep": round(irrelevant_cited / repetitions, 2),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "cost_per_question_usd": round(float(np.mean(known_costs)), 6)
        if costs and len(known_costs) == len(costs)
        else None,
        "cost_unknown_rows": len(costs) - len(known_costs),
    }
