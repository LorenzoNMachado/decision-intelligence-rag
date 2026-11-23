"""Protocol v2 is historical: its artefacts must stay byte-identical to the v2 commit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_intelligence.evaluation.historical_v2 import V2_COMMIT, V2_INTEGRITY, verify_v2

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v2_artefacts_and_manifest_inputs_match_the_v2_commit() -> None:
    report = verify_v2(REPO_ROOT)
    assert report["problems"] == []
    assert report["ok"] and report["artefacts_checked"] == 30
    if not report["git_commit_checked"]:
        pytest.skip(f"git history without {V2_COMMIT[:7]}: only recorded SHA-256 values were checked")


def test_v2_verification_detects_a_changed_artefact(tmp_path: Path) -> None:
    (tmp_path / V2_INTEGRITY).parent.mkdir(parents=True)
    (tmp_path / "rows.jsonl").write_text("{}")
    (tmp_path / V2_INTEGRITY).write_text(json.dumps({"artefacts": {"rows.jsonl": "0" * 64}}))
    report = verify_v2(tmp_path)
    assert not report["ok"] and "v2 artefact changed: rows.jsonl" in report["problems"]
