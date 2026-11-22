"""Experiment execution and LLM judging (protocol v2).

``run_experiment`` executes configurations × repetitions × questions and
appends one JSON row per execution to ``rows.jsonl``.

Safeguards:

* a results directory is never reused by accident: an existing run is only
  continued with ``resume=True``, and only if its split, embedder, threshold,
  pipeline fingerprint and manifest are the ones recorded when it started;
  every continuation is logged as a session in ``metadata.json``;
* LLM intent interpretations are stored in ``intents.jsonl``, so a resumed run
  and configurations E and F reuse exactly the same interpretation per
  (question, repetition) instead of asking the model again;
* the holdout split requires an intact manifest (``manifest.verify``), and LLM
  configurations additionally a frozen LLM provider;
* one failing row becomes an ERROR row and never ends the batch.

``judge_run`` adds LLM correctness verdicts (answered answerable rows only) and
claim verifications in ``judgements.jsonl``. Reports are written by
``evaluation.report``.
"""

from __future__ import annotations

import hashlib
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
from decision_intelligence.evaluation.metrics import Row
from decision_intelligence.evaluation.report import row_key
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
from decision_intelligence.sufficiency import SufficiencyReport

SPLITS = ("dev", "holdout")
WARM_UP_QUESTION = "warm-up query for the vector index"


class ProtocolViolation(RuntimeError):
    pass


def pipeline_fingerprint(workspace: Workspace) -> str:
    inputs = manifest_mod.tracked_inputs(workspace)["pipeline"]
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def check_split_allowed(
    split: str, workspace: Workspace, env: Environment, needs_llm: bool = False
) -> dict[str, Any]:
    if split not in SPLITS:
        raise ConfigurationError(f"unknown split {split!r}")
    if split == "dev":
        return {"status": "not required for dev"}
    check = manifest_mod.verify(workspace, env)
    if not check.ok:
        raise ProtocolViolation("holdout is locked: " + "; ".join(check.problems))
    if needs_llm and not manifest_mod.llm_frozen(workspace):
        raise ProtocolViolation(
            "holdout is locked for LLM configurations: the manifest has no frozen LLM provider "
            "(run `dirag amend-providers` before the first LLM holdout run)"
        )
    data = read_json(workspace.manifest_path)
    return {
        "status": data["status"],
        "frozen_at": data["frozen_at"],
        "sha256": manifest_mod.manifest_sha(workspace),
        "warnings": check.warnings,
    }


@dataclass
class RunSummary:
    directory: Path
    new_rows: int
    skipped_rows: int
    error_rows: int
    spent_usd: float
    unknown_cost_calls: int


_COMPATIBILITY_KEYS = ("split", "embedder", "evidence_threshold", "pipeline_fingerprint", "protocol")


def _load_intents(path: Path) -> dict[tuple[str, int], QueryIntent]:
    if not path.exists():
        return {}
    return {(r["question_id"], int(r["rep"])): QueryIntent.from_record(r["intent"]) for r in read_jsonl(path)}


def run_experiment(
    workspace: Workspace,
    env: Environment,
    split: str,
    config_ids: list[str],
    name: str,
    repetitions: int | None = None,
    llm: LLMClient | None = None,
    progress: Callable[[str], None] | None = None,
    resume: bool = False,
) -> RunSummary:
    configs = workspace.load_configs(config_ids)
    manifest_info = check_split_allowed(split, workspace, env, needs_llm=any(c.needs_llm for c in configs))
    components = load_components(workspace, env)
    protocol = components.protocol
    budget = Budget(env.budget_usd)
    if any(c.needs_llm for c in configs) and llm is None:
        llm = build_llm(env, budget=budget)
    questions = [q for q in workspace.load_questions() if q.split == split]

    directory = workspace.results_dir / name
    rows_path = directory / "rows.jsonl"
    metadata_path = directory / "metadata.json"
    identity = {
        "split": split,
        "embedder": components.embedder.name,
        "evidence_threshold": protocol.threshold_for(components.embedder),
        "pipeline_fingerprint": pipeline_fingerprint(workspace),
        "protocol": protocol.data,
    }
    session: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "configs": [c.id for c in configs],
        "manifest": manifest_info,
        "providers": env.public(),
        "package_version": __version__,
        "python": platform.python_version(),
    }
    if directory.exists() and any(directory.iterdir()):
        if not resume:
            raise ProtocolViolation(
                f"results already exist in {directory.relative_to(workspace.root)}; "
                "pass --resume to continue that run, or choose a new --name"
            )
        metadata = read_json(metadata_path)
        mismatched = [k for k in _COMPATIBILITY_KEYS if metadata.get(k) != identity[k]]
        if (
            split == "holdout"
            and metadata["sessions"][0]["manifest"].get("sha256") != manifest_info["sha256"]
        ):
            mismatched.append("manifest")
        if mismatched:
            raise ProtocolViolation(f"cannot resume {name}: {', '.join(mismatched)} changed since it started")
        metadata["sessions"].append(session)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {"run": name, **identity, "sessions": [session]}
    write_json(metadata_path, metadata)

    existing = {row_key(r) for r in read_jsonl(rows_path)} if rows_path.exists() else set()
    intents_path = directory / "intents.jsonl"
    shared_intents = _load_intents(intents_path)
    new_rows = skipped = errors = 0
    with (
        rows_path.open("a", encoding="utf-8") as rows_handle,
        intents_path.open("a", encoding="utf-8") as intents,
    ):
        for config in configs:
            retriever = make_retriever(components, config.retrieval, llm)
            generator = make_generator(config.generator, llm, protocol)
            pipeline = Pipeline(retriever, generator, components.documents)
            # Untimed warm-up so the first timed row does not pay one-off costs. No model call.
            warm = retriever.retrieve(WARM_UP_QUESTION, intent=QueryIntent("warm-up", "NONE"))
            if config.generator == "extractive":
                generator.generate(WARM_UP_QUESTION, warm, components.documents)
            reps = repetitions or protocol.repetitions_for(config)
            for rep in range(1, reps + 1):
                for question in questions:
                    key = f"{config.id}|{rep}|{question.id}"
                    if key in existing:
                        skipped += 1
                        continue
                    had_intent = (question.id, rep) in shared_intents
                    row = _execute(pipeline, config, rep, question, shared_intents)
                    if (
                        config.retrieval == "llm_intent"
                        and not had_intent
                        and (question.id, rep) in shared_intents
                    ):
                        record = {
                            "question_id": question.id,
                            "rep": rep,
                            "intent": shared_intents[(question.id, rep)].to_record(),
                        }
                        intents.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        intents.flush()
                    row.update({"run": name, "split": split})
                    rows_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    rows_handle.flush()
                    existing.add(key)
                    new_rows += 1
                    errors += row["answer"]["status"] == "ERROR"
                    if progress is not None:
                        progress(f"{key} {row['answer']['status']}")
    session.update(
        {
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "new_rows": new_rows,
            "error_rows": errors,
            "spent_usd": round(budget.spent_usd, 6),
            "spent_confirmed_usd": round(budget.confirmed_usd, 6),
            "unknown_cost_calls": budget.unknown_cost_calls,
        }
    )
    write_json(metadata_path, metadata)
    return RunSummary(directory, new_rows, skipped, errors, budget.spent_usd, budget.unknown_cost_calls)


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
    report: SufficiencyReport | None = None
    try:
        if config.retrieval == "llm_intent":
            intent = shared_intents.get((question.id, rep))
            reused = intent is not None
            if intent is None:
                intent = pipeline.retriever.intent_parser.parse(question.text)
                shared_intents[(question.id, rep)] = intent
        retrieval = pipeline.retriever.retrieve(question.text, intent=intent)
        answer, report = pipeline.answer(question.text, retrieval)
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
    # A reused intent was computed by an earlier configuration or session; its latency and
    # cost are still attributed here, because a deployed system with this configuration pays them.
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
        "sufficiency": report.to_dict() if report is not None else None,
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
    if read_json(directory / "metadata.json")["split"] == "holdout":
        check = manifest_mod.verify(workspace, env)
        if not check.ok:
            raise ProtocolViolation(
                "judging a holdout run needs an intact manifest: " + "; ".join(check.problems)
            )
        if llm is None and not manifest_mod.llm_frozen(workspace):
            raise ProtocolViolation("judging a holdout run needs the judge provider frozen in the manifest")
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
