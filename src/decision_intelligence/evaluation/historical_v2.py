"""Verification of the historical protocol v2.

Protocol v2 was frozen and its holdout run on 2026-11-22; it was then
published and its holdout inspected while protocol v3 was designed, so it is
kept as a historical record only. Its artefacts (``data/v2``,
``experiments/v2``) stay at their paths and are never regenerated with later
code. Because the current source differs from v2's, v2 is verified against
the git objects of ``V2_COMMIT`` (the published ``main`` that contains it):

* every v2 artefact must equal its blob in that commit and the SHA-256 in
  ``experiments/historical/v2-integrity.json``;
* every input hash of the v2 manifest — data, prompts, the pipeline and
  evaluation source groups, and ``uv.lock`` — is recomputed from that commit's
  tree with the v2 hashing rules, frozen here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from decision_intelligence.corpus.io import read_json
from decision_intelligence.evaluation.historical import HistoricalError, _git

V2_COMMIT = "9153e5d371cfaaaa5bb59bda66a3cf6fe4a32345"
V2_MANIFEST = "experiments/v2/manifest.json"
V2_INTEGRITY = "experiments/historical/v2-integrity.json"
V2_ARTEFACT_ROOTS = ("data/v2", "experiments/v2")
# The v2 hashing rules (evaluation.manifest at v2), frozen.
_PACKAGE = "src/decision_intelligence/"
_V2_EVALUATION_FILES = frozenset(
    {
        "evaluation/metrics.py",
        "evaluation/judges.py",
        "evaluation/report.py",
        "evaluation/statistics.py",
        "grounding/claim_verification.py",
        "generation/prompts/judge_correctness_v2.md",
        "generation/prompts/claim_verification_v1.md",
    }
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _show(root: Path, path: str) -> bytes:
    return _git(root, "show", f"{V2_COMMIT}:{path}")


def _files(root: Path, prefix: str) -> list[str]:
    out = _git(root, "ls-tree", "-r", "--name-only", V2_COMMIT, "--", prefix).decode()
    return [line for line in out.splitlines() if line]


def _available(root: Path) -> bool:
    try:
        _git(root, "cat-file", "-e", f"{V2_COMMIT}^{{commit}}")
    except HistoricalError:
        return False
    return True


def source_hashes_at_commit(root: Path) -> dict[str, str]:
    relative = sorted(
        PurePosixPath(p[len(_PACKAGE) :])
        for p in _files(root, _PACKAGE)
        if p.endswith((".py", ".md")) and "__pycache__" not in p
    )
    digests = {"pipeline": hashlib.sha256(), "evaluation": hashlib.sha256()}
    for rel in relative:
        group = "evaluation" if str(rel) in _V2_EVALUATION_FILES else "pipeline"
        digests[group].update(str(rel).encode())
        digests[group].update(b"\0")
        digests[group].update(_show(root, _PACKAGE + str(rel)).replace(b"\r\n", b"\n"))
    return {group: digest.hexdigest() for group, digest in digests.items()}


def _input_at_commit(root: Path, key: str, sources: dict[str, str]) -> str:
    if key.startswith("source:"):
        return sources[key.split(":", 1)[1]]
    if key.startswith("prompt:"):
        text = _show(root, f"{_PACKAGE}generation/prompts/{key.split(':', 1)[1]}.md").decode("utf-8")
        return _sha256(text.encode("utf-8"))
    return _sha256(_show(root, key))


def artefact_paths(root: Path) -> list[str]:
    return sorted(p for prefix in V2_ARTEFACT_ROOTS for p in _files(root, prefix))


def build_integrity_record(root: Path) -> dict[str, Any]:
    return {
        "protocol": "v2",
        "status": "historical",
        "commit": V2_COMMIT,
        "note": (
            "Protocol v2 holdout was run once and later inspected while protocol v3 was designed; it is "
            "no longer blind. These artefacts are a historical record and must not be regenerated."
        ),
        "artefacts": {p: _sha256(_show(root, p)) for p in artefact_paths(root)},
    }


def verify_v2(root: Path) -> dict[str, Any]:
    problems: list[str] = []
    integrity = root / V2_INTEGRITY
    if not integrity.exists():
        return {"ok": False, "problems": [f"missing {V2_INTEGRITY}"]}
    recorded: dict[str, str] = read_json(integrity)["artefacts"]
    for rel, expected in sorted(recorded.items()):
        path = root / rel
        if not path.exists():
            problems.append(f"missing v2 artefact: {rel}")
        elif _sha256(path.read_bytes()) != expected:
            problems.append(f"v2 artefact changed: {rel}")
    checked = _available(root)
    if checked:
        listed = set(artefact_paths(root))
        if listed != set(recorded):
            problems.append("integrity record does not list exactly the artefacts of the v2 commit")
        for rel in sorted(listed & set(recorded)):
            if _sha256(_show(root, rel)) != recorded[rel]:
                problems.append(f"integrity record disagrees with the v2 commit: {rel}")
        manifest = json.loads(_show(root, V2_MANIFEST))
        sources = source_hashes_at_commit(root)
        for group in ("pipeline", "evaluation"):
            for key, expected in sorted(manifest["inputs"][group].items()):
                if _input_at_commit(root, key, sources) != expected:
                    problems.append(f"v2 commit does not reproduce manifest input: {group}/{key}")
        if _sha256(_show(root, "uv.lock")) != manifest["environment"]["uv_lock_sha256"]:
            problems.append("v2 commit does not reproduce the frozen uv.lock")
    return {
        "ok": not problems,
        "problems": problems,
        "commit": V2_COMMIT,
        "artefacts_checked": len(recorded),
        "git_commit_checked": checked,
    }
