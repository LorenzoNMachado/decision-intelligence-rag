"""Command-line interface: ``dirag <command>``.

Run ``dirag --help`` for the list of commands. Every command works from the
repository root and reads provider settings from environment variables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from decision_intelligence.config import (
    ConfigurationError,
    Environment,
    Workspace,
    build_llm,
    load_components,
    make_generator,
    make_retriever,
)
from decision_intelligence.corpus.chunking import chunk_corpus
from decision_intelligence.corpus.generator import SPECS, generate_corpus
from decision_intelligence.corpus.io import load_documents, save_questions, write_json, write_jsonl
from decision_intelligence.corpus.questions import build_question_set
from decision_intelligence.corpus.validation import validate_corpus
from decision_intelligence.evaluation import manifest as manifest_mod
from decision_intelligence.evaluation.ablation import calibrate_threshold, retrieval_ablation
from decision_intelligence.evaluation.runner import ProtocolViolation, judge_run, report_run, run_experiment
from decision_intelligence.llm.client import Budget
from decision_intelligence.pipeline import Pipeline
from decision_intelligence.semantic_signals.classifier import LLMSignalClassifier, RuleBasedSignalClassifier
from decision_intelligence.semantic_signals.index import (
    SignalIndex,
    build_signal_index,
    evaluate_against_ground_truth,
)


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_corpus_generate(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    out = Path(args.out) if args.out else ws.data_dir
    documents = generate_corpus(args.size, seed=args.seed)
    chunks = chunk_corpus(documents)
    write_jsonl(out / "corpus.jsonl", (d.to_dict() for d in documents))
    write_jsonl(out / "chunks.jsonl", (c.to_dict() for c in chunks))
    questions = None
    if args.size == "experimental":
        questions = build_question_set(documents)
        save_questions(out / "questions.json", questions)
    index, _ = build_signal_index(documents, RuleBasedSignalClassifier())
    index.save(out / "signal_index.rules.json")
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
    if args.classifier == "rules":
        index, cost = build_signal_index(documents, RuleBasedSignalClassifier())
        path = ws.signal_index_path
    else:
        path = ws.data_dir / "signal_index.llm.json"
        previous = SignalIndex.load(path) if path.exists() else None
        index, cost = build_signal_index(
            documents, LLMSignalClassifier(build_llm(env, budget=Budget(env.budget_usd))), previous
        )
    index.save(path)
    _print({"path": str(path), "classifier": index.classifier, "errors": len(index.errors), "cost_usd": cost})
    return 0


def cmd_signals_evaluate(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    path = Path(args.index) if args.index else ws.signal_index_path
    report = evaluate_against_ground_truth(SignalIndex.load(path), load_documents(ws.corpus_path))
    if args.out:
        write_json(Path(args.out), report)
    _print(report)
    return 0


def cmd_ask(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    components = load_components(ws, env)
    needs_llm = args.retrieval == "llm_intent" or args.generator == "llm"
    llm = build_llm(env, budget=Budget(env.budget_usd)) if needs_llm else None
    pipeline = Pipeline(
        make_retriever(components, args.retrieval, llm),
        make_generator(args.generator, llm, components.protocol),
        components.documents,
    )
    _print(pipeline.run(args.question).to_dict())
    return 0


def cmd_calibrate(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    components = load_components(ws, env)
    questions = [q for q in ws.load_questions() if q.split == "dev"]
    report = calibrate_threshold(components, questions)
    if args.out:
        write_json(Path(args.out), report)
    _print({"embedder": report["embedder"], "suggested": report["suggested"]})
    return 0


def cmd_ablation(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    from decision_intelligence.evaluation.runner import check_split_allowed

    strategies = args.strategies.split(",")
    check_split_allowed(args.split, ws, env, needs_llm="llm_intent" in strategies)
    components = load_components(ws, env)
    llm = build_llm(env, budget=Budget(env.budget_usd)) if "llm_intent" in strategies else None
    questions = [q for q in ws.load_questions() if q.split == args.split]
    report = retrieval_ablation(components, questions, strategies, llm)
    out = Path(args.out) if args.out else ws.results_dir / f"ablation-{args.split}.json"
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
    )
    report = report_run(ws, args.name)
    _print(
        {
            "directory": str(summary.directory),
            "new_rows": summary.new_rows,
            "skipped_rows": summary.skipped_rows,
            "spent_usd": round(summary.spent_usd, 6),
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
    if ws.manifest_path.exists() and not args.force:
        print("manifest already exists; refusing to overwrite (use --force only before any holdout run)")
        return 1
    manifest = manifest_mod.freeze(ws, env, note=args.note)
    _print({"manifest": str(ws.manifest_path), "inputs": len(manifest.data["inputs"])})
    return 0


def cmd_amend_providers(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    try:
        manifest = manifest_mod.amend_llm_providers(ws, env, note=args.note)
    except manifest_mod.ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print({"providers": manifest.data["providers"], "amendments": manifest.data["amendments"]})
    return 0


def cmd_verify(args: argparse.Namespace, ws: Workspace, env: Environment) -> int:
    problems = manifest_mod.verify(ws, env)
    _print({"ok": not problems, "problems": problems})
    return 0 if not problems else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dirag", description="Decision Intelligence RAG research toolkit")
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "corpus-generate", help="generate corpus, chunks, questions and rule-based signal index"
    )
    p.add_argument("--size", choices=sorted(SPECS), default="experimental")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", help="output directory (default: data/synthetic)")
    p.set_defaults(func=cmd_corpus_generate)

    p = sub.add_parser("corpus-validate", help="check corpus, chunk and question integrity")
    p.set_defaults(func=cmd_corpus_validate)

    p = sub.add_parser("signals-classify", help="classify the five semantic signals for every document")
    p.add_argument("--classifier", choices=["rules", "llm"], default="rules")
    p.set_defaults(func=cmd_signals_classify)

    p = sub.add_parser("signals-evaluate", help="compare a signal index with the generator's intended labels")
    p.add_argument("--index")
    p.add_argument("--out")
    p.set_defaults(func=cmd_signals_evaluate)

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    p.add_argument("--retrieval", choices=["vector", "rules", "llm_intent"], default="vector")
    p.add_argument("--generator", choices=["extractive", "llm"], default="extractive")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("calibrate", help="sweep the evidence-gate threshold on the dev split")
    p.add_argument("--out")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("ablation", help="retrieval-only comparison of the three strategies")
    p.add_argument("--split", choices=["dev", "holdout"], default="dev")
    p.add_argument("--strategies", default="vector,rules")
    p.add_argument("--out")
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser("run", help="run configurations on a split")
    p.add_argument("--split", choices=["dev", "holdout"], default="dev")
    p.add_argument("--configs", default="A,B")
    p.add_argument("--name", required=True)
    p.add_argument("--reps", type=int)
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
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("amend-providers", help="record the LLM provider in a manifest frozen without one")
    p.add_argument("--note", required=True)
    p.set_defaults(func=cmd_amend_providers)

    p = sub.add_parser("verify-manifest", help="check that nothing changed since the freeze")
    p.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Workspace(Path(args.root).resolve())
    env = Environment.from_env()
    try:
        code: int = args.func(args, workspace, env)
    except (ConfigurationError, ProtocolViolation) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
