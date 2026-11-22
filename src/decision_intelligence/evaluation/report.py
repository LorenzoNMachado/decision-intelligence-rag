"""Summaries of a run: metrics per configuration and paired comparisons.

``report_run`` reads ``rows.jsonl`` (and ``judgements.jsonl`` when present),
computes the metrics of ``evaluation.metrics`` per configuration, compares
every pair of configurations question by question with an exact sign test, and
writes ``summary.json``, ``summary.csv`` and ``summary.md``. It never re-runs
the pipeline.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Any

from decision_intelligence.config import Workspace
from decision_intelligence.corpus.io import read_json, read_jsonl, write_json
from decision_intelligence.evaluation import manifest as manifest_mod
from decision_intelligence.evaluation.judges import LexicalJudge
from decision_intelligence.evaluation.metrics import Row, compute_metrics, enforce_quotes
from decision_intelligence.evaluation.statistics import paired_comparison
from decision_intelligence.models import Question

SUMMARY_COLUMNS = [
    "recall_at_k",
    "supporting_source_precision",
    "expected_source_coverage",
    "correctness",
    "correctness_partial",
    "faithfulness_quote",
    "faithfulness_llm",
    "correct_abstention_rate",
    "false_answer_rate",
    "false_abstention_rate",
    "operational_failures",
    "intent_failures",
    "rejected_citations",
    "irrelevant_sources_cited_per_rep",
    "latency_p50_ms",
    "latency_p95_ms",
    "cost_per_question_usd",
]

NOTES = [
    "Correctness uses the lexical key-fact proxy unless an LLM-judge table follows.",
    "Questions are the unit of comparison; deterministic configurations run once, so there is no "
    "repetition variance to report.",
    "Quote faithfulness is reported only for model-written quotes; extractive quotes are copied from "
    "the record and verified by construction (shown as —).",
    "Latency is measured on one machine within one run and includes warm-up effects; it is not "
    "evidence that one configuration is faster than another.",
]


def row_key(row: Row) -> str:
    return f"{row['config']}|{row['rep']}|{row['question_id']}"


def latest_judgements(path: Path) -> tuple[dict[str, str], dict[str, dict[tuple[str, int], str]]]:
    correctness: dict[str, str] = {}
    claims: dict[str, dict[tuple[str, int], str]] = {}
    if not path.exists():
        return correctness, claims
    for j in read_jsonl(path):
        if j["kind"] == "correctness":
            if j["label"] != "ERROR" or j["key"] not in correctness:
                correctness[j["key"]] = j["label"]
        else:
            config = j["key"].split("|", 1)[0]
            bucket = claims.setdefault(config, {})
            if j["label"] != "ERROR" or (j["key"], j["claim"]) not in bucket:
                bucket[(j["key"], j["claim"])] = j["label"]
    return correctness, claims


def _per_question(
    rows: list[Row], questions: dict[str, Question], top_k: int
) -> dict[str, dict[str, dict[str, float]]]:
    """config → metric → question → value, averaged over repetitions."""

    lexical = LexicalJudge()
    sums: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        q = questions[row["question_id"]]
        bucket = sums.setdefault(row["config"], {"recall_at_k": {}, "correct": {}})
        label = lexical.judge(q, row["answer"]["status"], row["answer"]["text"]).label
        bucket["correct"].setdefault(q.id, []).append(1.0 if label == "CORRECT" else 0.0)
        if q.answerable:
            expected = set(q.expected_doc_ids)
            top = {d["doc_id"] for d in row["retrieved"][:top_k]}
            bucket["recall_at_k"].setdefault(q.id, []).append(len(expected & top) / len(expected))
    return {
        config: {
            metric: {q: sum(v) / len(v) for q, v in values.items()} for metric, values in metrics.items()
        }
        for config, metrics in sums.items()
    }


def build_summary(
    name: str,
    rows: list[Row],
    metadata: dict[str, Any],
    questions: dict[str, Question],
    judgements: Path | None = None,
) -> dict[str, Any]:
    llm_correctness, llm_claims = latest_judgements(judgements) if judgements else ({}, {})
    top_k = int(metadata["protocol"]["retrieval"]["top_k"])
    summary: dict[str, Any] = {
        "run": name,
        "protocol": metadata["protocol"].get("protocol_version"),
        "split": metadata["split"],
        "embedder": metadata["embedder"],
        "evidence_threshold": metadata["evidence_threshold"],
        "questions": len({r["question_id"] for r in rows}),
        "configs": {},
        "paired": {},
        "notes": NOTES,
    }
    for config in sorted({r["config"] for r in rows}):
        subset = [r for r in rows if r["config"] == config]
        entry: dict[str, Any] = {
            "retrieval": subset[0]["retrieval_strategy"],
            "generator": subset[0]["generator"],
            "repetitions": len({r["rep"] for r in subset}),
            "lexical_judge": compute_metrics(subset, questions, top_k=top_k),
        }
        if any(row_key(r) in llm_correctness for r in subset):
            entry["llm_judge"] = compute_metrics(
                subset,
                questions,
                correctness=lambda r: llm_correctness.get(row_key(r), "ERROR"),
                claim_labels=llm_claims.get(config, {}),
                top_k=top_k,
            )
        if subset[0]["generator"] == "llm":
            entry["quote_enforce_replay"] = compute_metrics(
                [enforce_quotes(r) for r in subset], questions, top_k=top_k
            )
        summary["configs"][config] = entry
    per_question = _per_question(rows, questions, top_k)
    for first, second in itertools.combinations(sorted(per_question), 2):
        summary["paired"][f"{first}->{second}"] = {
            metric: paired_comparison(per_question[first][metric], per_question[second][metric])
            for metric in ("recall_at_k", "correct")
        }
    return summary


def report_run(workspace: Workspace, name: str) -> dict[str, Any]:
    """Recompute the summary of a run. Holdout runs need the frozen (or amended) evaluation code."""

    directory = workspace.results_dir / name
    rows = read_jsonl(directory / "rows.jsonl")
    metadata = read_json(directory / "metadata.json")
    if metadata["split"] == "holdout":
        recorded = read_json(workspace.manifest_path)["inputs"]["evaluation"]
        if manifest_mod.tracked_inputs(workspace)["evaluation"] != recorded:
            raise manifest_mod.ManifestError(
                "evaluation code or judge prompts changed after freeze; record the correction with "
                "`dirag amend-evaluation --note ...` before reporting a holdout run"
            )
    questions = {q.id: q for q in workspace.load_questions()}
    summary = build_summary(name, rows, metadata, questions, directory / "judgements.jsonl")
    write_json(directory / "summary.json", summary)
    _write_csv(directory / "summary.csv", summary)
    (directory / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["config", "retrieval", "generator", "judge", *SUMMARY_COLUMNS])
        for config, entry in summary["configs"].items():
            for judge in ("lexical_judge", "llm_judge", "quote_enforce_replay"):
                if judge in entry:
                    metrics = entry[judge]
                    writer.writerow(
                        [
                            config,
                            entry["retrieval"],
                            entry["generator"],
                            judge,
                            *[metrics[c] for c in SUMMARY_COLUMNS],
                        ]
                    )


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    configs = summary["configs"]
    lines = [
        f"# Results: {summary['run']} (protocol v{summary['protocol']}, {summary['split']} split)",
        "",
        f"Embedder: `{summary['embedder']}` · evidence threshold: {summary['evidence_threshold']} · "
        f"questions: {summary['questions']}",
        "",
        *[f"* {note}" for note in summary["notes"]],
        "",
        "| Metric | " + " | ".join(configs) + " |",
        "|---|" + "---:|" * len(configs),
        "| Retrieval | " + " | ".join(e["retrieval"] for e in configs.values()) + " |",
        "| Generator | " + " | ".join(e["generator"] for e in configs.values()) + " |",
        "| Repetitions | " + " | ".join(str(e["repetitions"]) for e in configs.values()) + " |",
    ]
    for column in SUMMARY_COLUMNS:
        lines.append(
            f"| {column} | " + " | ".join(_fmt(e["lexical_judge"][column]) for e in configs.values()) + " |"
        )
    if summary["paired"]:
        lines += [
            "",
            "## Paired comparisons (per question, exact two-sided sign test)",
            "",
            "| Pair | Metric | Questions | Second better | First better | Ties | p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for pair, metrics in summary["paired"].items():
            for metric, c in metrics.items():
                lines.append(
                    f"| {pair} | {metric} | {c['questions']} | {c['second_better']} | {c['first_better']} | "
                    f"{c['ties']} | {_fmt(c['sign_test_p'])} |"
                )
    judged = {c: e for c, e in configs.items() if "llm_judge" in e}
    if judged:
        lines += [
            "",
            "## LLM judge",
            "",
            "| Metric | " + " | ".join(judged) + " |",
            "|---|" + "---:|" * len(judged),
        ]
        for column in ("correctness", "correctness_partial", "correctness_judge_errors", "faithfulness_llm"):
            lines.append(
                f"| {column} | " + " | ".join(_fmt(e["llm_judge"][column]) for e in judged.values()) + " |"
            )
    replay = {c: e for c, e in configs.items() if "quote_enforce_replay" in e}
    if replay:
        lines += [
            "",
            "## Quote gate replay (enforce)",
            "",
            "| Metric | " + " | ".join(replay) + " |",
            "|---|" + "---:|" * len(replay),
        ]
        for column in (
            "correctness",
            "false_answer_rate",
            "false_abstention_rate",
            "supporting_source_precision",
        ):
            lines.append(
                f"| {column} | "
                + " | ".join(_fmt(e["quote_enforce_replay"][column]) for e in replay.values())
                + " |"
            )
    return "\n".join(lines) + "\n"
