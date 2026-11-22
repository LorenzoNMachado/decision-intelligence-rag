"""Metric definitions over experiment rows.

A *row* is one (configuration, repetition, question) execution as written by
the runner. Definitions (see ``docs/evaluation.md``):

* Recall@k — answerable rows: |expected ∩ top-k retrieved| / |expected|.
* Supporting-source precision — answerable rows that answered with ≥1 citation:
  |cited ∩ expected| / |cited|.
* Expected-source coverage — same rows: |cited ∩ expected| / |expected|.
* Answer correctness — answerable rows labelled CORRECT / answerable rows whose
  label is not a judge failure. The state rule (``judges.grade``) applies first:
  abstentions and errors on answerable questions are INCORRECT whatever a judge
  said, and judges only assess ANSWER rows.
* Citation faithfulness — model-written claims whose quote is literally present
  in a cited source / model-written claims (``quote``); extracted quotes are
  verified by construction and excluded. With an LLM verifier: SUPPORTED /
  judged claims.
* Correct abstention rate — unanswerable rows that abstained / unanswerable rows;
  also split into in-domain (a real record, information not stated) and
  out-of-domain rows.
* Unsupported answer rate — ANSWER rows that cite no expected record (or answer
  an unanswerable question) / ANSWER rows.
* Intervals — 95 % Wilson intervals for proportions; a seeded percentile
  bootstrap over questions for Recall@k.
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

from decision_intelligence.evaluation.judges import LexicalJudge, grade
from decision_intelligence.evaluation.statistics import bootstrap_interval, wilson_interval
from decision_intelligence.models import ABSTAIN_TEXT, Question

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
        answer["text"] = ABSTAIN_TEXT
        answer["abstain_reason"] = "replay: no claim with a verified quote"
    return {**row, "answer": answer}


def _per_question_means(rows: list[Row], value_by_row: dict[int, float]) -> list[float]:
    """One value per question (mean over repetitions), so repetitions never count as observations."""

    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["question_id"], []).append(value_by_row[id(row)])
    return [sum(v) / len(v) for _, v in sorted(grouped.items())]


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
    recall_by_row: dict[int, float] = {}
    precisions = []
    coverages = []
    irrelevant_cited = 0
    for row in answerable:
        expected = set(questions[row["question_id"]].expected_doc_ids)
        top = [d["doc_id"] for d in row["retrieved"][:top_k]]
        recalls.append(len(expected & set(top)) / len(expected))
        recall_by_row[id(row)] = recalls[-1]
        cited = cited_ids(row)
        if row["answer"]["status"] == "ANSWER" and cited:
            hits = len(expected & set(cited))
            precisions.append(hits / len(cited))
            coverages.append(hits / len(expected))
            irrelevant_cited += len(set(cited) - expected)

    def label_of(row: Row) -> str:
        # The state rule comes first: a judge label attached to an abstention or an error
        # is ignored, and the assessor is consulted only for answered answerable rows.
        q = questions[row["question_id"]]
        assess = (lambda _q, _t: correctness(row)) if correctness is not None else lexical.assess
        return grade(q, row["answer"]["status"], assess, row["answer"]["text"]).label

    labels = [label_of(r) for r in answerable]
    judged = [label for label in labels if label != "ERROR"]

    claims = [c for r in rows if r["answer"]["status"] == "ANSWER" for c in r["answer"]["claims"]]
    model_claims = [c for c in claims if c.get("quote_origin", "model") == "model"]
    llm_claims = list(claim_labels.values()) if claim_labels else []
    llm_claims_judged = [label for label in llm_claims if label != "ERROR"]

    def status_count(subset: list[Row], status: str) -> int:
        return sum(r["answer"]["status"] == status for r in subset)

    def is_out_of_domain(row: Row) -> bool:
        return "out_of_domain" in questions[row["question_id"]].category

    in_domain = [r for r in unanswerable if not is_out_of_domain(r)]
    out_of_domain = [r for r in unanswerable if is_out_of_domain(r)]
    answered = [r for r in rows if r["answer"]["status"] == "ANSWER"]
    unsupported = sum(
        1
        for r in answered
        if not questions[r["question_id"]].answerable
        or not set(cited_ids(r)) & set(questions[r["question_id"]].expected_doc_ids)
    )
    correct_count = sum(label == "CORRECT" for label in judged)

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
        # Extracted quotes are copied from the record, so they are verified by construction and
        # are excluded: the rate is reported only over quotes written by a model.
        "faithfulness_quote": _rate(sum(bool(c["quote_verified"]) for c in model_claims), len(model_claims)),
        "extracted_quotes": len(claims) - len(model_claims),
        "faithfulness_llm": _rate(
            sum(label == "SUPPORTED" for label in llm_claims_judged), len(llm_claims_judged)
        )
        if claim_labels
        else None,
        "claims": len(claims),
        "correct_abstention_rate": _rate(status_count(unanswerable, "ABSTAIN"), len(unanswerable)),
        "correct_abstention_in_domain": _rate(status_count(in_domain, "ABSTAIN"), len(in_domain)),
        "correct_abstention_out_of_domain": _rate(status_count(out_of_domain, "ABSTAIN"), len(out_of_domain)),
        "unsupported_answer_rate": _rate(unsupported, len(answered)),
        "answered_rows": len(answered),
        "correctness_ci95": wilson_interval(correct_count, len(judged)),
        "recall_at_k_ci95": bootstrap_interval(_per_question_means(answerable, recall_by_row)),
        "correct_abstention_in_domain_ci95": wilson_interval(
            status_count(in_domain, "ABSTAIN"), len(in_domain)
        ),
        "correct_abstention_out_of_domain_ci95": wilson_interval(
            status_count(out_of_domain, "ABSTAIN"), len(out_of_domain)
        ),
        "false_abstention_ci95": wilson_interval(status_count(answerable, "ABSTAIN"), len(answerable)),
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
