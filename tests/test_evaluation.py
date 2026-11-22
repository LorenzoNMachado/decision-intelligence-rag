from __future__ import annotations

import json
from typing import Any

import pytest

from decision_intelligence.cli import main
from decision_intelligence.config import Environment, Workspace
from decision_intelligence.corpus.io import read_jsonl
from decision_intelligence.evaluation import manifest
from decision_intelligence.evaluation.judges import LexicalJudge
from decision_intelligence.evaluation.metrics import compute_metrics, enforce_quotes
from decision_intelligence.evaluation.runner import ProtocolViolation, judge_run, report_run, run_experiment
from decision_intelligence.llm.client import ScriptedLLM
from decision_intelligence.models import Question


def _question(answerable: bool = True, **kw: Any) -> Question:
    base: dict[str, Any] = {
        "id": "Q-1",
        "split": "dev",
        "category": "signal_filter",
        "text": "q",
        "answerable": answerable,
        "expected_doc_ids": ("DEC-001", "DEC-002") if answerable else (),
        "reference_answer": "r",
        "key_facts": ("Alpha", "Beta") if answerable else (),
        "forbidden_facts": ("Gamma",),
    }
    base.update(kw)
    return Question(**base)


def _row(
    status: str, cited: list[str], retrieved: list[str], text: str = "", verified: bool = True, rep: int = 1
) -> dict[str, Any]:
    claims = [{"statement": "s", "source_ids": [c], "quote": "q", "quote_verified": verified} for c in cited]
    return {
        "config": "X",
        "rep": rep,
        "question_id": "Q-1",
        "latency_ms": 10.0,
        "cost_usd": 0.001,
        "retrieved": [{"doc_id": d} for d in retrieved],
        "intent": {"status": "NONE"},
        "answer": {"status": status, "text": text, "claims": claims, "rejected_citations": []},
    }


def test_lexical_judge() -> None:
    judge = LexicalJudge()
    q = _question()
    assert judge.judge(q, "ANSWER", "alpha and BETA").label == "CORRECT"
    assert judge.judge(q, "ANSWER", "Alpha, Beta and Gamma").label == "PARTIAL"
    assert judge.judge(q, "ANSWER", "Delta").label == "INCORRECT"
    assert judge.judge(q, "ABSTAIN", "Alpha Beta").label == "INCORRECT"
    unanswerable = _question(answerable=False)
    assert judge.judge(unanswerable, "ABSTAIN", "").label == "CORRECT"
    assert judge.judge(unanswerable, "ANSWER", "x").label == "INCORRECT"


def test_metrics_definitions() -> None:
    q = _question()
    rows = [
        _row(
            "ANSWER",
            ["DEC-001", "DEC-009"],
            ["DEC-001", "DEC-003", "DEC-004", "DEC-005"],
            "Alpha Beta",
            verified=False,
        )
    ]
    m = compute_metrics(rows, {"Q-1": q})
    assert m["recall_at_k"] == 0.5
    assert m["supporting_source_precision"] == 0.5
    assert m["expected_source_coverage"] == 0.5
    assert m["correctness"] == 1.0
    assert m["faithfulness_quote"] == 0.0
    assert m["irrelevant_sources_cited_per_rep"] == 1

    replay = compute_metrics([enforce_quotes(r) for r in rows], {"Q-1": q})
    assert replay["false_abstention_rate"] == 1.0


def test_abstention_metrics_and_operational_failures() -> None:
    q = _question(answerable=False)
    rows = [
        _row("ABSTAIN", [], [], rep=1),
        _row("ANSWER", ["DEC-001"], [], "x", rep=2),
        _row("ERROR", [], [], rep=3),
    ]
    m = compute_metrics(rows, {"Q-1": q})
    assert m["correct_abstention_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["false_answer_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["operational_failures"] == 1


def test_offline_run_report_and_resume(workspace: Workspace, offline_env: Environment) -> None:
    summary = run_experiment(workspace, offline_env, "dev", ["A", "B"], name="t", repetitions=2)
    assert summary.new_rows == 2 * 2 * 10
    again = run_experiment(workspace, offline_env, "dev", ["A", "B"], name="t", repetitions=2)
    assert again.new_rows == 0 and again.skipped_rows == 40
    report = report_run(workspace, "t")
    assert set(report["configs"]) == {"A", "B"}
    for name in ("summary.json", "summary.csv", "summary.md", "metadata.json"):
        assert (workspace.results_dir / "t" / name).exists()
    metadata = json.loads((workspace.results_dir / "t" / "metadata.json").read_text())
    assert "api_key" not in json.dumps(metadata).replace("llm_api_key_configured", "")


def test_holdout_requires_frozen_and_untampered_manifest(
    workspace: Workspace, offline_env: Environment
) -> None:
    with pytest.raises(ProtocolViolation):
        run_experiment(workspace, offline_env, "holdout", ["A"], name="h", repetitions=1)
    manifest.freeze(workspace, offline_env)
    assert manifest.verify(workspace, offline_env) == []
    run_experiment(workspace, offline_env, "holdout", ["A"], name="h", repetitions=1)

    data = json.loads(workspace.protocol_path.read_text())
    data["retrieval"]["top_k"] = 5
    workspace.protocol_path.write_text(json.dumps(data))
    problems = manifest.verify(workspace, offline_env)
    assert any("protocol.json" in p for p in problems)
    with pytest.raises(ProtocolViolation):
        run_experiment(workspace, offline_env, "holdout", ["A"], name="h2", repetitions=1)


def test_llm_configurations_need_a_frozen_provider(workspace: Workspace, offline_env: Environment) -> None:
    manifest.freeze(workspace, offline_env)
    with pytest.raises(ProtocolViolation):
        run_experiment(
            workspace,
            offline_env,
            "holdout",
            ["C"],
            name="h",
            repetitions=1,
            llm=ScriptedLLM(lambda s, u: {}),
        )


def test_scripted_llm_run_and_judge(workspace: Workspace, offline_env: Environment) -> None:
    documents = {d.id: d for d in workspace.load_documents()}

    def responder(system: str, user: str) -> dict[str, Any]:
        if "REFERENCE:" in user:
            return {"label": "PARTIAL", "rationale": "scripted"}
        if "QUESTION:" in user and "EVIDENCE:" in user:
            doc_id = user.split("EVIDENCE:\n[", 1)[1].split("]", 1)[0]
            return {
                "status": "ANSWER",
                "answer": documents[doc_id].title,
                "claims": [{"statement": "s", "source_ids": [doc_id], "quote": documents[doc_id].decision}],
                "abstain_reason": "",
            }
        if "QUESTION:" in user:  # intent
            return {"signals": [], "notes": ""}
        return {"label": "SUPPORTED", "rationale": "scripted"}

    llm = ScriptedLLM(responder)
    run_experiment(workspace, offline_env, "dev", ["C", "E", "F"], name="s", repetitions=1, llm=llm)
    rows = read_jsonl(workspace.results_dir / "s" / "rows.jsonl")
    e_rows = {r["question_id"]: r for r in rows if r["config"] == "E"}
    f_rows = {r["question_id"]: r for r in rows if r["config"] == "F"}
    assert all(f_rows[q]["intent_reused"] and not e_rows[q]["intent_reused"] for q in e_rows)
    judge_run(workspace, offline_env, "s", llm=llm)
    report = report_run(workspace, "s")
    assert report["configs"]["C"]["llm_judge"]["correctness_partial"] == 1.0
    assert report["configs"]["C"]["llm_judge"]["faithfulness_llm"] == 1.0
    assert "quote_enforce_replay" in report["configs"]["C"]


def test_cli_offline_commands(workspace: Workspace, capsys: pytest.CaptureFixture[str]) -> None:
    root = str(workspace.root)
    assert main(["--root", root, "corpus-validate"]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "--root",
                root,
                "ask",
                "Which decisions about the feature flag service have security dependencies?",
                "--retrieval",
                "rules",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["strategy"] == "rules" and output["answer"]["status"] in {"ANSWER", "ABSTAIN"}
    assert main(["--root", root, "run", "--split", "holdout", "--configs", "A", "--name", "x"]) == 2
    assert main(["--root", root, "ask", "q", "--generator", "llm"]) == 2  # no credentials configured
