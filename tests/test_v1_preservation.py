"""Protocol v1 is historical: its artefacts must stay byte-identical to the v1 commit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_intelligence.evaluation.historical import V1_COMMIT, V1_INTEGRITY, verify_v1

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v1_artefacts_match_the_v1_commit() -> None:
    report = verify_v1(REPO_ROOT)
    assert report["problems"] == []
    assert report["ok"] and report["artefacts_checked"] == 26
    if not report["git_commit_checked"]:
        pytest.skip(f"git history without {V1_COMMIT[:7]}: only recorded SHA-256 values were checked")


def test_v1_verification_detects_a_changed_artefact(tmp_path: Path) -> None:
    (tmp_path / "experiments" / "v1").mkdir(parents=True)
    (tmp_path / "result.json").write_text("{}")
    record = {"artefacts": {"result.json": "0" * 64, "gone.json": "0" * 64}}
    (tmp_path / V1_INTEGRITY).write_text(json.dumps(record))
    report = verify_v1(tmp_path)
    assert not report["ok"]
    assert "v1 artefact changed: result.json" in report["problems"]
    assert "missing v1 artefact: gone.json" in report["problems"]
