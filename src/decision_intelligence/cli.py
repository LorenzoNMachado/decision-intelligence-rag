"""Command-line interface: ``dirag <command>``.

Run ``dirag --help`` for the list of commands. Every command works from the
repository root, uses protocol v3 (``data/v3``, ``experiments/v3``) and reads
provider settings from environment variables. Protocols v1 and v2 are
historical and can only be verified (``dirag verify-v1``, ``dirag verify-v2``).

Commands never overwrite an existing results file: give a new ``--out`` or
``--name``, or ``--resume`` a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from decision_intelligence.config import (
    GENERATORS,
    RETRIEVAL_STRATEGIES,
    ConfigurationError,
    Environment,
    Workspace,
    build_llm,
    load_components,
    make_checker,
    make_generator,
    make_retriever,
)
from decision_intelligence.corpus.chunking import chunk_corpus
from decision_intelligence.corpus.generator import SPECS, generate_corpus
from decision_intelligence.corpus.io import load_documents, read_json, save_questions, write_json, write_jsonl
from decision_intelligence.corpus.questions import QuestionSpecError, build_question_set
from decision_intelligence.corpus.validation import validate_corpus
from decision_intelligence.embeddings.base import EmbeddingError
from decision_intelligence.evaluation import manifest as manifest_mod
from decision_intelligence.evaluation.ablation import calibrate_band, calibrate_gate, retrieval_ablation
from decision_intelligence.evaluation.historical import HistoricalError, verify_v1
from decision_intelligence.evaluation.historical_v2 import verify_v2
from decision_intelligence.evaluation.report import report_run
from decision_intelligence.evaluation.runner import (
    ProtocolViolation,
    check_split_allowed,
    judge_run,
    run_experiment,
)
from decision_intelligence.pipeline import Pipeline
from decision_intelligence.semantic_signals.classifier import (
    LLMSignalClassifier,
    RuleBasedSignalClassifier,
    ScopedSignalClassifier,
    entity_names,
)
from decision_intelligence.semantic_signals.index import (
    SignalIndex,
    build_signal_index,
    evaluate_against_ground_truth,
)


class OutputExists(RuntimeError):
    pass


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _new_output(path: Path) -> Path:
    if path.exists():
        raise OutputExists(f"{path} already exists; results are never overwritten (choose another --out)")
    return path


def cmd_corpus_generate(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    out = Path(args.out) if args.out else ws.data_dir
    corpus = ws.load_protocol().data["corpus"]
    documents = generate_corpus(args.size, seed=args.seed if args.seed is not None else int(corpus["seed"]))
    chunks = chunk_corpus(documents)
    write_jsonl(out / "corpus.jsonl", (d.to_dict() for d in documents))
    write_jsonl(out / "chunks.jsonl", (c.to_dict() for c in chunks))
    questions = None
    if args.size == "experimental":
        specs = read_json(ws.question_specs_path)
        questions = build_question_set(documents, specs, seed=int(corpus["question_seed"]))
        save_questions(out / "questions.json", questions)
    entities = entity_names(documents)
    for name, classifier in (
        ("rules-v2", RuleBasedSignalClassifier(entities)),
        ("rules-v3", ScopedSignalClassifier(entities)),
    ):
        index, _ = build_signal_index(documents, classifier)
        index.save(out / f"signal_index.{name}.json")
    report = validate_corpus(documents, chunks, questions)
    _print({"output": str(out), "valid": report.ok, "errors": report.errors, **report.stats})
    return 0 if report.ok else 1


def cmd_corpus_validate(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    questions = ws.load_questions() if ws.questions_path.exists() else None
    report = validate_corpus(ws.load_documents(), ws.load_chunks(), questions)
    _print({"valid": report.ok, "errors": report.errors, **report.stats})
    return 0 if report.ok else 1


def cmd_signals_classify(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    documents = ws.load_documents()
    if args.classifier in ("rules-v2", "rules-v3"):
        rules = RuleBasedSignalClassifier if args.classifier == "rules-v2" else ScopedSignalClassifier
        index, cost = build_signal_index(documents, rules(entity_names(documents)))
        path = ws.signal_index_path(args.classifier)
    else:
        path = ws.data_dir / "signal_index.llm.json"
        previous = SignalIndex.load(path) if path.exists() else None
        index, cost = build_signal_index(documents, LLMSignalClassifier(build_llm(env)), previous)
    index.save(path)
    _print({"path": str(path), "classifier": index.classifier, "errors": len(index.errors), "cost_usd": cost})
    return 0


def cmd_signals_evaluate(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    path = Path(args.index) if args.index else ws.signal_index_path(args.classifier)
    report = evaluate_against_ground_truth(SignalIndex.load(path), load_documents(ws.corpus_path))
    if args.out:
        write_json(_new_output(Path(args.out)), report)
    _print(report)
    return 0


def cmd_ask(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    components = load_components(ws, env)
    needs_llm = args.retrieval == "llm_intent" or args.generator == "llm"
    llm = build_llm(env) if needs_llm else None
    if args.generator == "extractive-v2" and not args.sufficiency:
        raise ConfigurationError("extractive-v2 quotes what the sufficiency check found: add --sufficiency")
    pipeline = Pipeline(
        make_retriever(components, args.retrieval, llm),
        make_generator(args.generator, llm, components.protocol),
        components.documents,
        checker=make_checker(components) if args.sufficiency else None,
    )
    _print(pipeline.run(args.question).to_dict())
    return 0


def cmd_calibrate(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    components = load_components(ws, env)
    questions = [q for q in ws.load_questions() if q.split == "dev"]
    report = {
        "gate": {s: calibrate_gate(components, questions, s) for s in ("vector", "bm25")},
        "topical_band": calibrate_band(components, questions),
    }
    if args.out:
        write_json(_new_output(Path(args.out)), report)
    _print(
        {
            "gate": {
                s: {k: r[k] for k in ("scorer", "suggested", "optimal_range")}
                for s, r in report["gate"].items()
            },
            "topical_band": {k: report["topical_band"][k] for k in ("suggested", "optimal_range")},
        }
    )
    return 0


def cmd_ablation(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    strategies = args.strategies.split(",")
    out = _new_output(Path(args.out) if args.out else ws.results_dir / f"ablation-{args.split}.json")
    check_split_allowed(args.split, ws, env, needs_llm="llm_intent" in strategies)
    components = load_components(ws, env)
    llm = build_llm(env) if "llm_intent" in strategies else None
    questions = [q for q in ws.load_questions() if q.split == args.split]
    report = retrieval_ablation(components, questions, strategies, llm)
    write_json(out, report)
    _print(
        {
            "output": str(out),
            "strategies": report["strategies"],
            "changes_vs_vector": report["changes_vs_vector"],
        }
    )
    return 0


def cmd_run(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    summary = run_experiment(
        ws,
        env,
        split=args.split,
        config_ids=args.configs.split(","),
        name=args.name,
        repetitions=args.reps,
        progress=(lambda m: print(m, file=sys.stderr)) if args.verbose else None,
        resume=args.resume,
    )
    report = report_run(ws, args.name)
    _print(
        {
            "directory": str(summary.directory),
            "new_rows": summary.new_rows,
            "skipped_rows": summary.skipped_rows,
            "error_rows": summary.error_rows,
            "spent_usd": round(summary.spent_usd, 6),
            "unknown_cost_calls": summary.unknown_cost_calls,
            "configs": list(report["configs"]),
        }
    )
    return 0


def cmd_judge(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    path = judge_run(
        ws,
        env,
        args.name,
        verify_claims=not args.no_claims,
        progress=(lambda m: print(m, file=sys.stderr)) if args.verbose else None,
    )
    report_run(ws, args.name)
    _print({"judgements": str(path)})
    return 0


def cmd_report(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    summary = report_run(ws, args.name)
    print((ws.results_dir / args.name / "summary.md").read_text(encoding="utf-8"))
    return 0 if summary["configs"] else 1


def cmd_freeze(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    manifest = manifest_mod.freeze(ws, env, note=args.note, force=args.force)
    groups = manifest.data["inputs"]
    _print({"manifest": str(ws.manifest_path), "inputs": {g: len(v) for g, v in groups.items()}})
    return 0


def cmd_amend_providers(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    manifest = manifest_mod.amend_llm_providers(ws, env, note=args.note)
    _print({"providers": manifest.data["providers"], "amendments": manifest.data["amendments"]})
    return 0


def cmd_amend_evaluation(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    manifest = manifest_mod.amend_evaluation(ws, env, note=args.note)
    _print({"amendments": manifest.data["amendments"]})
    return 0


def cmd_verify(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    check = manifest_mod.verify(ws, env)
    _print({"ok": check.ok, "problems": check.problems, "warnings": check.warnings})
    return 0 if check.ok else 1


def cmd_verify_v1(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    report = verify_v1(ws.root)
    _print(report)
    return 0 if report["ok"] else 1


def cmd_verify_v2(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    report = verify_v2(ws.root)
    _print(report)
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dirag", description="Decision Intelligence RAG research toolkit")
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "corpus-generate", help="generate corpus, chunks, questions and rule-based signal index"
    )
    p.add_argument("--size", choices=sorted(SPECS), default="experimental")
    p.add_argument("--seed", type=int, help="corpus seed (default: from protocol.json)")
    p.add_argument("--out", help="output directory (default: data/v3)")
    p.set_defaults(func=cmd_corpus_generate)

    p = sub.add_parser("corpus-validate", help="check corpus, chunk and question integrity")
    p.set_defaults(func=cmd_corpus_validate)

    p = sub.add_parser("signals-classify", help="classify the five semantic signals for every document")
    p.add_argument("--classifier", choices=["rules-v2", "rules-v3", "llm"], default="rules-v3")
    p.set_defaults(func=cmd_signals_classify)

    p = sub.add_parser("signals-evaluate", help="compare a signal index with the generator's intended labels")
    p.add_argument("--classifier", choices=["rules-v2", "rules-v3"], default="rules-v3")
    p.add_argument("--index")
    p.add_argument("--out")
    p.set_defaults(func=cmd_signals_evaluate)

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    p.add_argument("--retrieval", choices=list(RETRIEVAL_STRATEGIES), default="vector")
    p.add_argument("--generator", choices=list(GENERATORS), default="extractive-v1")
    p.add_argument(
        "--sufficiency", action="store_true", help="check that the evidence contains what is asked (v3)"
    )
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("calibrate", help="sweep the evidence-gate threshold on the dev split")
    p.add_argument("--out")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("ablation", help="retrieval-only comparison of the three strategies")
    p.add_argument("--split", choices=["dev", "holdout"], default="dev")
    p.add_argument("--strategies", default="vector,bm25,rules-v2,rules-v3-flat,rules-v3")
    p.add_argument("--out", help="output file (default: experiments/v3/results/ablation-<split>.json)")
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser("run", help="run configurations on a split")
    p.add_argument("--split", choices=["dev", "holdout"], default="dev")
    p.add_argument("--configs", default="A,B")
    p.add_argument("--name", required=True)
    p.add_argument("--reps", type=int)
    p.add_argument("--resume", action="store_true", help="continue an existing run with the same setup")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("judge", help="LLM correctness judge and claim verification for a run")
    p.add_argument("--name", required=True)
    p.add_argument("--no-claims", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("report", help="recompute metrics and print the summary table")
    p.add_argument("--name", required=True)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("freeze", help="freeze the protocol before using the holdout split")
    p.add_argument("--note", default="")
    p.add_argument("--force", action="store_true", help="replace a manifest that no holdout run has used")
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("amend-providers", help="record the LLM provider in a manifest frozen without one")
    p.add_argument("--note", required=True)
    p.set_defaults(func=cmd_amend_providers)

    p = sub.add_parser(
        "amend-evaluation", help="accept corrected evaluation code (logged; pipeline must be intact)"
    )
    p.add_argument("--note", required=True)
    p.set_defaults(func=cmd_amend_evaluation)

    p = sub.add_parser("verify-manifest", help="check that nothing changed since the freeze (protocol v3)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("verify-v1", help="check that the historical protocol v1 artefacts are intact")
    p.set_defaults(func=cmd_verify_v1)

    p = sub.add_parser("verify-v2", help="check that the historical protocol v2 artefacts are intact")
    p.set_defaults(func=cmd_verify_v2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Workspace(Path(args.root).resolve())
    try:
        env = Environment.from_env()
        code: int = args.func(args, workspace, env)
    except (
        ConfigurationError,
        ProtocolViolation,
        manifest_mod.ManifestError,
        HistoricalError,
        QuestionSpecError,
        EmbeddingError,
        OutputExists,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
