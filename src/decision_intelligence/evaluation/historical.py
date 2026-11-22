"""Verification of the historical protocol v1.

Protocol v1 was frozen and its holdout split was run on 2026-11-22. The split
was then inspected during an audit, so it is no longer blind, and v1 is kept
only as a historical record. Its artefacts stay at their original paths and are
never rewritten.

The current source code differs from the code that produced v1, so v1 cannot
be checked by re-hashing the working tree. Instead it is checked against the
git objects of the commit that published it (``V1_COMMIT``):

* every v1 artefact (data, configurations, manifest, results) must be identical
  to its blob in that commit and to the SHA-256 recorded in
  ``experiments/v1/integrity.json``;
* every input hash in the v1 manifest (including prompts and package source)
  is recomputed from that commit, which proves the commit really contains the
  frozen protocol.

The current code must not be used to recompute v1 results. To reproduce them,
check out ``V1_COMMIT`` and run its own ``dirag verify-manifest``.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from decision_intelligence.corpus.io import read_json

V1_COMMIT = "30bce655ef7fdd0095c24864aea3b85dac3ebaf7"
V1_TAG = "protocol-v1"
V1_MANIFEST = "experiments/manifest.json"
V1_INTEGRITY = "experiments/v1/integrity.json"
V1_ARTEFACT_ROOTS = ("data/synthetic", "experiments/configurations", "experiments/results")
_PACKAGE_PREFIX = "src/decision_intelligence/"


class HistoricalError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HistoricalError(f"git {' '.join(args[:2])} failed: {type(exc).__name__}") from exc
    return result.stdout


def _commit_available(root: Path) -> bool:
    try:
        _git(root, "cat-file", "-e", f"{V1_COMMIT}^{{commit}}")
    except HistoricalError:
        return False
    return True


def _blob(root: Path, path: str) -> bytes:
    return _git(root, "show", f"{V1_COMMIT}:{path}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _commit_files(root: Path, prefix: str) -> list[str]:
    out = _git(root, "ls-tree", "-r", "--name-only", V1_COMMIT, "--", prefix).decode()
    return [line for line in out.splitlines() if line]


def source_hash_at_commit(root: Path) -> str:
    """Recompute v1's ``source:decision_intelligence`` hash from the commit's tree.

    Mirrors the v1 ``source_tree_hash``: ``.py`` and ``.md`` files, ordered as
    ``sorted(Path.rglob)`` orders them, CRLF normalised.
    """

    files = [
        p
        for p in _commit_files(root, _PACKAGE_PREFIX)
        if p.endswith((".py", ".md")) and "__pycache__" not in p
    ]
    relative = sorted(PurePosixPath(p[len(_PACKAGE_PREFIX) :]) for p in files)
    digest = hashlib.sha256()
    for rel in relative:
        digest.update(str(rel).encode())
        digest.update(b"\0")
        digest.update(_blob(root, _PACKAGE_PREFIX + str(rel)).replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _input_hash_at_commit(root: Path, key: str) -> str:
    if key == "source:decision_intelligence":
        return source_hash_at_commit(root)
    if key.startswith("prompt:"):
        name = key.split(":", 1)[1]
        text = _blob(root, f"{_PACKAGE_PREFIX}generation/prompts/{name}.md").decode("utf-8")
        return _sha256(text.encode("utf-8"))
    return _sha256(_blob(root, key))


def artefact_paths(root: Path) -> list[str]:
    paths = [V1_MANIFEST]
    for prefix in V1_ARTEFACT_ROOTS:
        paths += _commit_files(root, prefix)
    return sorted(paths)


def build_integrity_record(root: Path) -> dict[str, Any]:
    """The content of ``experiments/v1/integrity.json`` (computed from the v1 commit)."""

    return {
        "protocol": "v1",
        "status": "historical",
        "commit": V1_COMMIT,
        "tag": V1_TAG,
        "note": (
            "Protocol v1 holdout was run once, then inspected during an audit; it is no longer "
            "blind. These artefacts are a historical record and must not be regenerated with "
            "later code."
        ),
        "artefacts": {p: _sha256(_blob(root, p)) for p in artefact_paths(root)},
    }


def verify_v1(root: Path) -> dict[str, Any]:
    """Check that v1 is intact. Returns ``{"ok": bool, "problems": [...], ...}``."""

    problems: list[str] = []
    integrity_path = root / V1_INTEGRITY
    if not integrity_path.exists():
        return {"ok": False, "problems": [f"missing {V1_INTEGRITY}"]}
    recorded: dict[str, str] = read_json(integrity_path)["artefacts"]
    for rel, expected in sorted(recorded.items()):
        path = root / rel
        if not path.exists():
            problems.append(f"missing v1 artefact: {rel}")
        elif _sha256(path.read_bytes()) != expected:
            problems.append(f"v1 artefact changed: {rel}")
    git_checked = _commit_available(root)
    if git_checked:
        listed = set(artefact_paths(root))
        if listed != set(recorded):
            problems.append("integrity record does not list exactly the artefacts of the v1 commit")
        for rel in sorted(listed & set(recorded)):
            if _sha256(_blob(root, rel)) != recorded[rel]:
                problems.append(f"integrity record disagrees with the v1 commit: {rel}")
        manifest = read_json(root / V1_MANIFEST)
        for key, expected in sorted(manifest["inputs"].items()):
            if _input_hash_at_commit(root, key) != expected:
                problems.append(f"v1 commit does not reproduce manifest input: {key}")
    return {
        "ok": not problems,
        "problems": problems,
        "commit": V1_COMMIT,
        "artefacts_checked": len(recorded),
        "git_commit_checked": git_checked,
    }
