"""Retrieval-only studies: semantic-enrichment ablation and gate calibration.

``retrieval_ablation`` runs the three retrieval strategies over the same
questions and embeddings, with no generator, and reports how the semantic
signals change ranking and evidence-gate admission relative to the vector
baseline.

``calibrate_gate`` and ``calibrate_band`` sweep the evidence-gate threshold
(per scorer) and the topical band on the *dev* split only, so the holdout
never influences them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from decision_intelligence.config import Components, make_retriever
from decision_intelligence.evaluation.statistics import paired_comparison
from decision_intelligence.llm.client import LLMClient
from decision_intelligence.models import Question, RetrievalResult
from decision_intelligence.retrieval.retriever import Retriever


def _retrieval_metrics(results: list[tuple[Question, RetrievalResult]]) -> dict[str, Any]:
    recalls = []
    admitted_precision = []
    expected_all_admitted = []
    non_expected_admitted = 0
    unanswerable_with_admission = 0
    unanswerable = 0
    vetoed = 0
    for question, result in results:
        expected = set(question.expected_doc_ids)
        admitted = [d.doc_id for d in result.admitted()]
        vetoed += sum(d.gate_reason == "signal_contradiction" for d in result.documents)
        if question.answerable:
            recalls.append(len(expected & set(result.top_ids())) / len(expected))
            expected_all_admitted.append(expected <= set(admitted))
            if admitted:
                admitted_precision.append(len(expected & set(admitted)) / len(admitted))
            non_expected_admitted += len(set(admitted) - expected)
        else:
            unanswerable += 1
            unanswerable_with_admission += bool(admitted)
            non_expected_admitted += len(admitted)
    return {
        "questions": len(results),
        "recall_at_k": round(float(np.mean(recalls)), 4) if recalls else None,
        "admitted_precision": round(float(np.mean(admitted_precision)), 4) if admitted_precision else None,
        "expected_all_admitted_rate": round(float(np.mean(expected_all_admitted)), 4)
        if expected_all_admitted
        else None,
        "non_expected_admitted_total": non_expected_admitted,
        "unanswerable_with_any_admission": f"{unanswerable_with_admission}/{unanswerable}",
        "signal_vetoes": vetoed,
    }


def _recalls(results: list[tuple[Question, RetrievalResult]]) -> dict[str, float]:
    return {
        q.id: len(set(q.expected_doc_ids) & set(r.top_ids())) / len(q.expected_doc_ids)
        for q, r in results
        if q.answerable
    }


def retrieval_ablation(
    components: Components, questions: list[Question], strategies: list[str], llm: LLMClient | None = None
) -> dict[str, Any]:
    per_strategy: dict[str, list[tuple[Question, RetrievalResult]]] = {}
    for strategy in strategies:
        retriever = make_retriever(components, strategy, llm)
        per_strategy[strategy] = [(q, retriever.retrieve(q.text)) for q in questions]
    report: dict[str, Any] = {
        "embedder": components.embedder.name,
        "evidence_threshold": components.protocol.threshold_for(components.embedder),
        "strategies": {s: _retrieval_metrics(r) for s, r in per_strategy.items()},
        "changes_vs_vector": {},
        "per_question": [],
    }
    baseline = {q.id: r for q, r in per_strategy.get("vector", [])}
    for strategy, results in per_strategy.items():
        if strategy == "vector" or not baseline:
            continue
        changed = sum(r.top_ids() != baseline[q.id].top_ids() for q, r in results)
        active = sum(r.intent.status == "OK" for _, r in results)
        report["changes_vs_vector"][strategy] = {
            "questions_with_active_signals": active,
            "questions_with_changed_top_k": changed,
            "recall_at_k_paired": paired_comparison(_recalls(per_strategy["vector"]), _recalls(results)),
        }
    for index, question in enumerate(questions):
        entry: dict[str, Any] = {
            "question_id": question.id,
            "category": question.category,
            "expected": list(question.expected_doc_ids),
        }
        for strategy, results in per_strategy.items():
            result = results[index][1]
            entry[strategy] = {
                "intent": [
                    f"{s.signal.value}:{s.polarity.value}:{s.requirement.value}"
                    for s in result.intent.signals
                ],
                "top": [f"{d.doc_id}{'' if d.admitted else '(x)'}" for d in result.documents],
            }
        report["per_question"].append(entry)
    return report


def _median_of_optimal(
    rows: list[dict[str, Any]], key: tuple[str, ...]
) -> tuple[dict[str, Any], list[float]]:
    top = max(tuple(r[k] for k in key) for r in rows)
    optimal = [r for r in rows if tuple(r[k] for k in key) == top]
    return optimal[len(optimal) // 2], [optimal[0]["value"], optimal[-1]["value"]]


def calibrate_gate(
    components: Components, questions: list[Question], strategy: str = "vector"
) -> dict[str, Any]:
    """Evidence-gate threshold for one scorer, on dev questions only (pre-declared in docs/v3/protocol.md).

    The gate decides topical relevance, so it is calibrated to separate answerable
    questions from *out-of-domain* ones. In-domain unanswerable questions name real
    records and are the sufficiency check's job, not the gate's. Objective: Youden's J
    of "some evidence admitted", ties broken by the share of expected records
    admitted; the suggestion is the median of the optimal set.
    """

    if any(q.split != "dev" for q in questions):
        raise ValueError("calibration must use dev questions only")
    retriever = make_retriever(components, strategy)
    top_k = int(components.protocol.data["retrieval"]["top_k"])
    sims = []
    for question in questions:
        if not question.answerable and "out_of_domain" not in question.category:
            continue
        best = retriever.similarities(question.text)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:top_k]
        sims.append((question, [(doc_id, score) for doc_id, (score, _) in ranked]))
    ceiling = max((s for _, top in sims for _, s in top), default=1.0)
    rows = []
    for step in range(0, 101):
        threshold = round(ceiling * step / 100, 4)
        answerable_hit, ood_hit, expected_admitted = [], [], []
        for question, top_docs in sims:
            admitted = {d for d, s in top_docs if s >= threshold}
            if question.answerable:
                answerable_hit.append(bool(admitted))
                expected = set(question.expected_doc_ids)
                expected_admitted.append(len(expected & admitted) / len(expected))
            else:
                ood_hit.append(bool(admitted))
        tpr = float(np.mean(answerable_hit)) if answerable_hit else 0.0
        fpr = float(np.mean(ood_hit)) if ood_hit else 0.0
        rows.append(
            {
                "value": threshold,
                "answerable_with_admission": round(tpr, 3),
                "out_of_domain_with_admission": round(fpr, 3),
                "expected_admitted_rate": round(float(np.mean(expected_admitted)), 3)
                if expected_admitted
                else 0.0,
                "youden_j": round(tpr - fpr, 3),
            }
        )
    suggested, optimal_range = _median_of_optimal(rows, ("youden_j", "expected_admitted_rate"))
    return {
        "scorer": retriever.scorer.name if retriever.scorer is not None else components.embedder.name,
        "dev_questions": len(sims),
        "suggested": suggested,
        "optimal_range": optimal_range,
        "sweep": rows,
    }


BAND_GRID = tuple(round(0.02 * i, 2) for i in range(1, 11))


def calibrate_band(components: Components, questions: list[Question]) -> dict[str, Any]:
    """Topical band of the rules-v3 ranking, on dev questions only (pre-declared).

    Objective: mean Recall@4 of rules-v3 over the answerable dev questions; the
    suggestion is the median of the bands that reach the maximum.
    """

    if any(q.split != "dev" for q in questions):
        raise ValueError("calibration must use dev questions only")
    answerable = [q for q in questions if q.answerable]
    base = make_retriever(components, "rules-v3")
    rows = []
    for band in BAND_GRID:
        retriever = Retriever(
            strategy="rules-v3",
            chunks=components.chunks,
            embedder=components.embedder,
            params=replace(base.params, topical_band=band),
            intent_parser=base.intent_parser,
            signal_index=base.signal_index,
            index=base.index,
        )
        recalls = [
            len(set(q.expected_doc_ids) & set(retriever.retrieve(q.text).top_ids())) / len(q.expected_doc_ids)
            for q in answerable
        ]
        rows.append({"value": band, "recall_at_k": round(float(np.mean(recalls)), 4)})
    suggested, optimal_range = _median_of_optimal(rows, ("recall_at_k",))
    return {
        "dev_questions": len(answerable),
        "suggested": suggested,
        "optimal_range": optimal_range,
        "sweep": rows,
    }
