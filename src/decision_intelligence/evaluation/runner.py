"""Experiment execution, judging and reporting.

``run_experiment`` executes configurations × repetitions × questions and
appends one JSON row per execution to ``rows.jsonl`` (resumable: rows already
present are skipped). Configurations E and F share one intent interpretation
per (question, repetition), so the generator is the only difference between them.

``judge_run`` adds LLM correctness verdicts and claim verifications in
``judgements.jsonl``. ``report_run`` computes the metrics per configuration and
writes ``summary.json``, ``summary.csv`` and ``summary.md``.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_intelligence import __version__
from decision_intelligence.config import (
    ConfigurationError,
    Environment,
    ExperimentConfig,
    Workspace,
    build_llm,
    load_components,
    make_generator,
    make_retriever,
)
from decision_intelligence.corpus.io import read_json, read_jsonl, write_json
from decision_intelligence.embeddings.base import EmbeddingError
from decision_intelligence.evaluation import manifest as manifest_mod
from decision_intelligence.evaluation.judges import LLMCorrectnessJudge
from decision_intelligence.evaluation.metrics import Row, compute_metrics, enforce_quotes
from decision_intelligence.grounding.claim_verification import ClaimVerifier
from decision_intelligence.llm.client import Budget, LLMClient, LLMError, sanitise
from decision_intelligence.models import (
    AnswerStatus,
    GeneratedAnswer,
    Outcome,
    QueryIntent,
    Question,
    RetrievalResult,
    add_costs,
)
from decision_intelligence.pipeline import Pipeline

SPLITS = ("dev", "holdout")


class ProtocolViolation(RuntimeError):
    pass


def row_key(row: Row) -> str:
    return f"{row['config']}|{row['rep']}|{row['question_id']}"


def check_split_allowed(
    split: str, workspace: Workspace, env: Environment, needs_llm: bool = False
) -> dict[str, Any]:
    if split not in SPLITS:
        raise ConfigurationError(f"unknown split {split!r}")
    if split == "dev":
        return {"status": "not required for dev"}
    problems = manifest_mod.verify(workspace, env)
    if problems:
        raise ProtocolViolation("holdout is locked: " + "; ".join(problems))
    if needs_llm and not manifest_mod.llm_frozen(workspace):
        raise ProtocolViolation(
            "holdout is locked for LLM configurations: the manifest has no frozen LLM provider "
            "(run `dirag amend-providers` before the first LLM holdout run)"
        )
    data = read_json(workspace.manifest_path)
    return {
        "status": data["status"],
        "frozen_at": data["frozen_at"],
        "sha256": manifest_mod.sha256_file(workspace.manifest_path),
    }


@dataclass
class RunSummary:
    directory: Path
    new_rows: int
    skipped_rows: int
    spent_usd: float


def run_experiment(
    workspace: Workspace,
    env: Environment,
    split: str,
    config_ids: list[str],
    name: str,
    repetitions: int | None = None,
    llm: LLMClient | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunSummary:
    configs = workspace.load_configs(config_ids)
    manifest_info = check_split_allowed(split, workspace, env, needs_llm=any(c.needs_llm for c in configs))
    components = load_components(workspace, env)
    protocol = components.protocol
    reps = repetitions or protocol.repetitions
    budget = Budget(env.budget_usd)
    if any(c.needs_llm for c in configs) and llm is None:
        llm = build_llm(env, budget=budget)
    questions = [q for q in workspace.load_questions() if q.split == split]

    directory = workspace.results_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    rows_path = directory / "rows.jsonl"
    existing = {row_key(r) for r in read_jsonl(rows_path)} if rows_path.exists() else set()
    metadata = {
        "run": name,
        "split": split,
        "configs": [c.id for c in configs],
        "repetitions": reps,
        "protocol": protocol.data,
        "embedder": components.embedder.name,
        "evidence_threshold": protocol.threshold_for(components.embedder),
        "signal_index": components.signal_index.classifier,
        "providers": env.public(),
        "manifest": manifest_info,
        "package_version": __version__,
        "python": platform.python_version(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_json(directory / "metadata.json", metadata)

    shared_intents: dict[tuple[str, int], QueryIntent] = {}
    new_rows = 0
    skipped = 0
    with rows_path.open("a", encoding="utf-8") as handle:
        for config in configs:
            retriever = make_retriever(components, config.retrieval, llm)
            generator = make_generator(config.generator, llm, protocol)
            pipeline = Pipeline(retriever, generator, components.documents)
            for rep in range(1, reps + 1):
                for question in questions:
                    key = f"{config.id}|{rep}|{question.id}"
                    if key in existing:
                        skipped += 1
                        continue
                    row = _execute(pipeline, config, rep, question, shared_intents)
                    row.update({"run": name, "split": split})
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    new_rows += 1
                    if progress is not None:
                        progress(f"{key} {row['answer']['status']}")
    metadata["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    metadata["spent_usd"] = round(budget.spent_usd, 6)
    write_json(directory / "metadata.json", metadata)
    return RunSummary(
        directory=directory, new_rows=new_rows, skipped_rows=skipped, spent_usd=budget.spent_usd
    )


def _execute(
    pipeline: Pipeline,
    config: ExperimentConfig,
    rep: int,
    question: Question,
    shared_intents: dict[tuple[str, int], QueryIntent],
) -> Row:
    """Run one (configuration, repetition, question).

    A failure anywhere in the pipeline becomes an ERROR row (outcome
    ``PIPELINE_ERROR``) instead of ending the batch: it is counted as an
    operational failure, never as an abstention, and the next row still runs.
    """

    started = time.perf_counter()
    intent: QueryIntent | None = None
    reused = False
    retrieval: RetrievalResult | None = None
    try:
        if config.retrieval == "llm_intent":
            intent = shared_intents.get((question.id, rep))
            reused = intent is not None
            if intent is None:
                intent = pipeline.retriever.intent_parser.parse(question.text)
                shared_intents[(question.id, rep)] = intent
        retrieval = pipeline.retriever.retrieve(question.text, intent=intent)
        answer = pipeline.generator.generate(question.text, retrieval, pipeline.documents)
    except Exception as exc:  # isolate the row; the error is recorded, not swallowed
        kind = exc.kind if isinstance(exc, EmbeddingError | LLMError) else "internal"
        answer = GeneratedAnswer(
            generator=pipeline.generator.name,
            status=AnswerStatus.ERROR,
            text="",
            error=f"{type(exc).__name__}: {sanitise(str(exc))}",
            error_kind=kind,
            outcome=Outcome.PIPELINE_ERROR,
            cost_usd=None,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    effective_intent = retrieval.intent if retrieval is not None else intent
    # A reused intent was computed by an earlier configuration; its latency and cost are
    # still attributed here, because a deployed system with this configuration pays them.
    latency_ms = elapsed_ms + (effective_intent.latency_ms if reused and effective_intent else 0.0)
    intent_cost = effective_intent.cost_usd if effective_intent is not None else 0.0
    return {
        "config": config.id,
        "retrieval_strategy": config.retrieval,
        "generator": config.generator,
        "rep": rep,
        "question_id": question.id,
        "category": question.category,
        "answerable": question.answerable,
        "expected_doc_ids": list(question.expected_doc_ids),
        "intent": effective_intent.to_dict() if effective_intent is not None else {"status": "NOT_RUN"},
        "intent_reused": reused,
        "retrieved": [d.to_dict() for d in retrieval.documents] if retrieval is not None else [],
        "answer": answer.to_dict(),
        "latency_ms": round(latency_ms, 2),
        "cost_usd": _round_cost(add_costs(answer.cost_usd, intent_cost)),
    }


def _round_cost(value: float | None) -> float | None:
    return None if value is None else round(value, 8)


def judge_run(
    workspace: Workspace,
    env: Environment,
    name: str,
    llm: LLMClient | None = None,
    verify_claims: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Path:
    directory = workspace.results_dir / name
    rows = read_jsonl(directory / "rows.jsonl")
    questions = {q.id: q for q in workspace.load_questions()}
    documents = {d.id: d for d in workspace.load_documents()}
    if llm is None:
        llm = build_llm(env, model=env.judge_model or None, budget=Budget(env.budget_usd))
    judge = LLMCorrectnessJudge(llm)
    verifier = ClaimVerifier(llm)
    path = directory / "judgements.jsonl"
    done = set()
    if path.exists():
        done = {(j["key"], j["kind"], j.get("claim")) for j in read_jsonl(path) if j["label"] != "ERROR"}
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            key = row_key(row)
            question = questions[row["question_id"]]
            # Only answered answerable rows are judged; abstentions and errors are graded by the
            # state rule in the metrics and are never shown to the judge.
            needs_judge = question.answerable and row["answer"]["status"] == "ANSWER"
            if needs_judge and (key, "correctness", None) not in done:
                verdict = judge.assess(question, row["answer"]["text"])
                record = {
                    "key": key,
                    "kind": "correctness",
                    "judge": judge.name,
                    "label": verdict.label,
                    "rationale": verdict.rationale,
                    "cost_usd": verdict.cost_usd,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if progress:
                    progress(f"{key} correctness {verdict.label}")
            if not verify_claims or row["answer"]["status"] != "ANSWER" or row["generator"] != "llm":
                continue
            for index, claim in enumerate(row["answer"]["claims"]):
                if (key, "claim", index) in done:
                    continue
                sources = [documents[s] for s in claim["source_ids"] if s in documents]
                claim_verdict = verifier.verify(claim["statement"], sources)
                record = {
                    "key": key,
                    "kind": "claim",
                    "claim": index,
                    "judge": verifier.name,
                    "label": claim_verdict.label,
                    "rationale": claim_verdict.rationale,
                    "cost_usd": claim_verdict.cost_usd,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _latest_judgements(path: Path) -> tuple[dict[str, str], dict[str, dict[tuple[str, int], str]]]:
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


def report_run(workspace: Workspace, name: str) -> dict[str, Any]:
    directory = workspace.results_dir / name
    rows = read_jsonl(directory / "rows.jsonl")
    metadata = read_json(directory / "metadata.json")
    questions = {q.id: q for q in workspace.load_questions()}
    llm_correctness, llm_claims = _latest_judgements(directory / "judgements.jsonl")
    top_k = int(metadata["protocol"]["retrieval"]["top_k"])
    summary: dict[str, Any] = {
        "run": name,
        "split": metadata["split"],
        "embedder": metadata["embedder"],
        "evidence_threshold": metadata["evidence_threshold"],
        "configs": {},
    }
    for config in sorted({r["config"] for r in rows}):
        subset = [r for r in rows if r["config"] == config]
        entry: dict[str, Any] = {
            "retrieval": subset[0]["retrieval_strategy"],
            "generator": subset[0]["generator"],
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
    write_json(directory / "summary.json", summary)
    _write_csv(directory / "summary.csv", summary)
    (directory / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
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
        f"# Results: {summary['run']} ({summary['split']} split)",
        "",
        f"Embedder: `{summary['embedder']}` · evidence threshold: {summary['evidence_threshold']}",
        "",
        "Correctness in this table uses the **lexical key-fact proxy** unless an LLM-judge table follows.",
        "",
        "| Metric | " + " | ".join(configs) + " |",
        "|---|" + "---:|" * len(configs),
        "| Retrieval | " + " | ".join(e["retrieval"] for e in configs.values()) + " |",
        "| Generator | " + " | ".join(e["generator"] for e in configs.values()) + " |",
    ]
    for column in SUMMARY_COLUMNS:
        lines.append(
            f"| {column} | " + " | ".join(_fmt(e["lexical_judge"][column]) for e in configs.values()) + " |"
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
