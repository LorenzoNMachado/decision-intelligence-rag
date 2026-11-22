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
from decision_intelligence.evaluation.report import report_run
from decision_intelligence.evaluation.runner import ProtocolViolation, judge_run, run_experiment
from decision_intelligence.llm.client import LLMError, ScriptedLLM
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


def _dev_count(workspace: Workspace) -> int:
    return sum(q.split == "dev" for q in workspace.load_questions())


def test_offline_run_report_and_resume(workspace: Workspace, offline_env: Environment) -> None:
    dev = _dev_count(workspace)
    summary = run_experiment(workspace, offline_env, "dev", ["A", "B"], name="t", repetitions=2)
    assert summary.new_rows == 2 * 2 * dev
    with pytest.raises(ProtocolViolation, match="--resume"):
        run_experiment(workspace, offline_env, "dev", ["A", "B"], name="t", repetitions=2)
    again = run_experiment(workspace, offline_env, "dev", ["A", "B"], name="t", repetitions=2, resume=True)
    assert again.new_rows == 0 and again.skipped_rows == 4 * dev
    report = report_run(workspace, "t")
    assert set(report["configs"]) == {"A", "B"}
    assert report["paired"]["A->B"]["recall_at_k"]["questions"] > 0
    for name in ("summary.json", "summary.csv", "summary.md", "metadata.json"):
        assert (workspace.results_dir / "t" / name).exists()
    metadata = json.loads((workspace.results_dir / "t" / "metadata.json").read_text())
    assert len(metadata["sessions"]) == 2
    assert "api_key" not in json.dumps(metadata).replace("llm_api_key_configured", "")


def test_deterministic_configurations_run_once_by_default(
    workspace: Workspace, offline_env: Environment
) -> None:
    summary = run_experiment(workspace, offline_env, "dev", ["A"], name="once")
    assert summary.new_rows == _dev_count(workspace)


def test_resume_refuses_a_changed_pipeline(workspace: Workspace, offline_env: Environment) -> None:
    run_experiment(workspace, offline_env, "dev", ["A"], name="r", repetitions=1)
    data = json.loads(workspace.protocol_path.read_text())
    data["retrieval"]["candidate_pool"] = 21
    workspace.protocol_path.write_text(json.dumps(data))
    with pytest.raises(ProtocolViolation, match="pipeline_fingerprint"):
        run_experiment(workspace, offline_env, "dev", ["B"], name="r", repetitions=1, resume=True)


def test_holdout_requires_frozen_and_untampered_manifest(
    workspace: Workspace, offline_env: Environment
) -> None:
    with pytest.raises(ProtocolViolation):
        run_experiment(workspace, offline_env, "holdout", ["A"], name="h", repetitions=1)
    manifest.freeze(workspace, offline_env)
    assert manifest.verify(workspace, offline_env).ok
    run_experiment(workspace, offline_env, "holdout", ["A"], name="h", repetitions=1)

    data = json.loads(workspace.protocol_path.read_text())
    data["retrieval"]["top_k"] = 5
    workspace.protocol_path.write_text(json.dumps(data))
    check = manifest.verify(workspace, offline_env)
    assert any("protocol.json" in p for p in check.problems)
    with pytest.raises(ProtocolViolation):
        run_experiment(workspace, offline_env, "holdout", ["A"], name="h2", repetitions=1)


def test_manifest_records_environment_and_flags_dependency_changes(
    workspace: Workspace, offline_env: Environment
) -> None:
    frozen = manifest.freeze(workspace, offline_env)
    assert set(frozen.data["inputs"]) == {"pipeline", "evaluation"}
    assert frozen.data["environment"]["uv_lock_sha256"] and frozen.data["environment"]["packages"]["numpy"]
    data = json.loads(workspace.manifest_path.read_text())
    data["environment"]["python"] = "0.0.0"
    workspace.manifest_path.write_text(json.dumps(data))
    check = manifest.verify(workspace, offline_env)
    assert check.ok and any("python" in w for w in check.warnings)
    (workspace.root / "uv.lock").write_text("changed")
    assert "environment: uv.lock changed after freeze" in manifest.verify(workspace, offline_env).problems


def test_freeze_never_replaces_a_manifest_used_by_a_holdout_run(
    workspace: Workspace, offline_env: Environment
) -> None:
    manifest.freeze(workspace, offline_env)
    with pytest.raises(manifest.ManifestError, match="already exists"):
        manifest.freeze(workspace, offline_env)
    manifest.freeze(workspace, offline_env, force=True)  # allowed: no holdout result yet
    run_experiment(workspace, offline_env, "holdout", ["A"], name="h", repetitions=1)
    with pytest.raises(manifest.ManifestError, match="holdout results"):
        manifest.freeze(workspace, offline_env, force=True)


def test_evaluation_code_changes_need_a_logged_amendment(
    workspace: Workspace, offline_env: Environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest.freeze(workspace, offline_env)
    run_experiment(workspace, offline_env, "holdout", ["A"], name="h", repetitions=1)
    real = manifest.source_hashes()
    monkeypatch.setattr(manifest, "source_hashes", lambda: {**real, "evaluation": "f" * 64})
    with pytest.raises(manifest.ManifestError, match="amend-evaluation"):
        report_run(workspace, "h")
    with pytest.raises(manifest.ManifestError, match="note"):
        manifest.amend_evaluation(workspace, offline_env, note=" ")
    amended = manifest.amend_evaluation(workspace, offline_env, note="fix a metric denominator")
    entry = amended.data["amendments"][-1]
    assert entry["field"] == "inputs.evaluation" and entry["previous"] != entry["new"]
    assert report_run(workspace, "h")["configs"]
    monkeypatch.setattr(manifest, "source_hashes", lambda: {"pipeline": "0" * 64, "evaluation": "e" * 64})
    with pytest.raises(manifest.ManifestError, match="new protocol version"):
        manifest.amend_evaluation(workspace, offline_env, note="also change the pipeline")


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


def test_llm_intents_are_persisted_and_reused_after_resume(
    workspace: Workspace, offline_env: Environment
) -> None:
    calls = {"intent": 0}

    def first(system: str, user: str) -> dict[str, Any]:
        calls["intent"] += 1
        return {"signals": [], "notes": f"call {calls['intent']}"}

    run_experiment(workspace, offline_env, "dev", ["F"], name="i", repetitions=1, llm=ScriptedLLM(first))
    assert calls["intent"] == _dev_count(workspace)

    def never(system: str, user: str) -> LLMError:
        return LLMError("must not be called for intents", kind="network")

    run_experiment(
        workspace, offline_env, "dev", ["F"], name="i", repetitions=2, llm=ScriptedLLM(never), resume=True
    )
    rows = read_jsonl(workspace.results_dir / "i" / "rows.jsonl")
    second = [r for r in rows if r["rep"] == 2]
    assert second and all(r["intent"]["status"] == "UNAVAILABLE" for r in second)  # rep 2 was new
    stored = read_jsonl(workspace.results_dir / "i" / "intents.jsonl")
    assert {(r["question_id"], r["rep"]) for r in stored} >= {(r["question_id"], 1) for r in rows}


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
    assert (workspace.results_dir / "s" / "intents.jsonl").exists()
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
    assert main(["--root", root, "ablation", "--split", "dev"]) == 0
    capsys.readouterr()
    assert main(["--root", root, "ablation", "--split", "dev"]) == 2  # never overwrites a result
    assert "never overwritten" in capsys.readouterr().err


def test_missing_llm_provider_is_reported_clearly_without_side_effects(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    root = str(workspace.root)
    # A local run exists, so `judge` reaches the provider check rather than a missing-file error.
    assert (
        main(["--root", root, "run", "--split", "dev", "--configs", "A", "--name", "local", "--reps", "1"])
        == 0
    )
    capsys.readouterr()
    for argv in (
        ["ask", "q", "--generator", "llm"],
        ["ask", "q", "--retrieval", "llm_intent"],
        ["run", "--split", "dev", "--configs", "C", "--name", "no-llm"],
        ["ablation", "--split", "dev", "--strategies", "vector,llm_intent"],
        ["judge", "--name", "local"],
    ):
        code = main(["--root", root, *argv])
        err = capsys.readouterr().err
        assert code == 2, argv
        assert "DIRAG_LLM_BASE_URL" in err and ".env.example" in err, argv
    # No partial results are written for the LLM run, and local configurations still work.
    assert not (workspace.results_dir / "no-llm" / "rows.jsonl").exists()
    assert not (workspace.results_dir / "local" / "judgements.jsonl").exists()
    assert (
        main(
            [
                "--root",
                root,
                "ask",
                "Which decisions about the feature flag service have security dependencies?",
            ]
        )
        == 0
    )
