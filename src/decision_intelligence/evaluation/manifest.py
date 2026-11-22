"""Freezing the experimental protocol before the holdout split is used.

``freeze`` records the SHA-256 of every input that can change a result: the
corpus, chunks, questions, signal index, protocol, configurations, prompts and
the package source, plus the provider model names in use. ``verify`` recomputes
them; the runner refuses to touch the holdout split unless the manifest is
frozen and every hash still matches.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_intelligence.config import Environment, Workspace
from decision_intelligence.corpus.io import read_json, write_json
from decision_intelligence.generation.prompts import load_prompt, prompt_names

PACKAGE_DIR = Path(__file__).resolve().parents[1]


class ManifestError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_tree_hash(package_dir: Path = PACKAGE_DIR) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".md"} and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(package_dir)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def tracked_inputs(workspace: Workspace) -> dict[str, str]:
    files = [
        workspace.corpus_path,
        workspace.chunks_path,
        workspace.questions_path,
        workspace.signal_index_path,
        workspace.protocol_path,
        *sorted(workspace.config_dir.glob("[A-Z].json")),
    ]
    hashes = {str(p.relative_to(workspace.root)): sha256_file(p) for p in files}
    for name in prompt_names():
        hashes[f"prompt:{name}"] = load_prompt(name).sha256
    hashes["source:decision_intelligence"] = source_tree_hash()
    return hashes


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def provider_fingerprint(env: Environment) -> dict[str, Any]:
    """Provider identity without secrets. The LLM block is ``None`` when no LLM is configured."""

    public = env.public()
    llm = None
    if env.llm_available:
        llm = {
            k: public[k]
            for k in ("llm_model", "judge_model", "llm_temperature", "llm_seed", "llm_response_format")
        }
    return {
        "embedding": {"backend": public["embedding_backend"], "model": public["embedding_model"]},
        "llm": llm,
    }


@dataclass
class Manifest:
    data: dict[str, Any]

    @property
    def frozen(self) -> bool:
        return self.data.get("status") == "frozen"


def freeze(workspace: Workspace, env: Environment, note: str = "") -> Manifest:
    data: dict[str, Any] = {
        "status": "frozen",
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(workspace.root),
        "note": note,
        "inputs": tracked_inputs(workspace),
        "providers": provider_fingerprint(env),
        "amendments": [],
    }
    write_json(workspace.manifest_path, data)
    return Manifest(data)


def verify(workspace: Workspace, env: Environment) -> list[str]:
    """Return a list of mismatches (empty when the protocol is intact)."""

    if not workspace.manifest_path.exists():
        return ["no manifest: run `dirag freeze` after finishing development"]
    manifest = Manifest(read_json(workspace.manifest_path))
    problems = []
    if not manifest.frozen:
        problems.append("manifest status is not 'frozen'")
    current = tracked_inputs(workspace)
    for key, expected in manifest.data["inputs"].items():
        if current.get(key) != expected:
            problems.append(f"changed after freeze: {key}")
    for key in sorted(set(current) - set(manifest.data["inputs"])):
        problems.append(f"new input not in manifest: {key}")
    current_providers = provider_fingerprint(env)
    recorded = manifest.data["providers"]
    if current_providers["embedding"] != recorded["embedding"]:
        problems.append("embedding provider differs from manifest")
    if recorded.get("llm") is not None and current_providers["llm"] != recorded["llm"]:
        problems.append("LLM provider settings differ from manifest")
    return problems


def llm_frozen(workspace: Workspace) -> bool:
    return (
        workspace.manifest_path.exists()
        and read_json(workspace.manifest_path)["providers"].get("llm") is not None
    )


def amend_llm_providers(workspace: Workspace, env: Environment, note: str) -> Manifest:
    """Record the LLM provider in a manifest frozen without one (logged amendment).

    Allowed only once and only while the LLM block is empty: the frozen inputs
    (data, prompts, parameters, code) must still match, and an already
    recorded provider can never be changed.
    """

    problems = verify(workspace, env)
    if problems:
        raise ManifestError("cannot amend a manifest whose inputs changed: " + "; ".join(problems))
    data = read_json(workspace.manifest_path)
    if data["providers"].get("llm") is not None:
        raise ManifestError("the LLM provider is already frozen and cannot be changed")
    fingerprint = provider_fingerprint(env)
    if fingerprint["llm"] is None:
        raise ManifestError("no LLM provider configured in the environment")
    data["providers"]["llm"] = fingerprint["llm"]
    data["amendments"].append(
        {"at": datetime.now(UTC).isoformat(timespec="seconds"), "field": "providers.llm", "note": note}
    )
    write_json(workspace.manifest_path, data)
    return Manifest(data)
