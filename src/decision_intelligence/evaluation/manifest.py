"""Freezing the experimental protocol before the holdout split is used (protocol v2).

``freeze`` records SHA-256 hashes in two groups:

* ``pipeline`` — everything that can change what the system outputs: corpus,
  chunks, questions and their specifications, signal index, protocol,
  configurations, the prompts the system uses, and the package source except
  the evaluation files;
* ``evaluation`` — what turns outputs into metrics: the evaluation modules
  (``EVALUATION_MODULES``) and the judge and claim-verifier prompts.

It also records the environment (Python, key package versions, the SHA-256 of
``uv.lock``) and the provider identity (backend, model and endpoint, never a
credential).

``verify`` recomputes everything. The runner refuses the holdout split unless
both groups, the lock file and the providers match; differences in Python or
package versions are reported as warnings because they do not by themselves
change the deterministic configurations. Evaluation code may be corrected
after a holdout run only through ``amend_evaluation``, which logs the old and
new hashes; the pipeline group can never be amended.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from decision_intelligence.config import Environment, Workspace
from decision_intelligence.corpus.io import read_json, write_json
from decision_intelligence.embeddings.openai_compatible import endpoint_identity
from decision_intelligence.generation.prompts import load_prompt, prompt_names

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
EVALUATION_MODULES = (
    "evaluation/metrics.py",
    "evaluation/judges.py",
    "evaluation/report.py",
    "evaluation/statistics.py",
    "grounding/claim_verification.py",
)
EVALUATION_PROMPTS = ("judge_correctness_v2", "claim_verification_v1")
TRACKED_PACKAGES = ("numpy", "httpx")
GROUPS = ("pipeline", "evaluation")


class ManifestError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hash(files: list[Path], package_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(package_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def source_hashes(package_dir: Path = PACKAGE_DIR) -> dict[str, str]:
    files = [
        p
        for p in package_dir.rglob("*")
        if p.is_file() and p.suffix in {".py", ".md"} and "__pycache__" not in p.parts
    ]
    evaluation_files = {package_dir / m for m in EVALUATION_MODULES} | {
        package_dir / "generation" / "prompts" / f"{name}.md" for name in EVALUATION_PROMPTS
    }
    return {
        "pipeline": _source_hash([p for p in files if p not in evaluation_files], package_dir),
        "evaluation": _source_hash([p for p in files if p in evaluation_files], package_dir),
    }


def tracked_inputs(workspace: Workspace) -> dict[str, dict[str, str]]:
    data_files = [
        workspace.corpus_path,
        workspace.chunks_path,
        workspace.questions_path,
        workspace.question_specs_path,
        workspace.signal_index_path,
        workspace.protocol_path,
        *sorted(workspace.config_dir.glob("[A-Z].json")),
    ]
    pipeline = {p.relative_to(workspace.root).as_posix(): sha256_file(p) for p in data_files}
    evaluation: dict[str, str] = {}
    for name in prompt_names():
        target = evaluation if name in EVALUATION_PROMPTS else pipeline
        target[f"prompt:{name}"] = load_prompt(name).sha256
    sources = source_hashes()
    pipeline["source:pipeline"] = sources["pipeline"]
    evaluation["source:evaluation"] = sources["evaluation"]
    return {"pipeline": pipeline, "evaluation": evaluation}


def environment_record(workspace: Workspace) -> dict[str, Any]:
    lock = workspace.root / "uv.lock"
    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system().lower(),
        "packages": packages,
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
    }


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
        llm["endpoint"] = endpoint_identity(env.llm_base_url)
    embedding = {"backend": public["embedding_backend"], "model": public["embedding_model"]}
    if env.embedding_backend != "hashing":
        embedding["endpoint"] = public["embedding_endpoint"]
    return {"embedding": embedding, "llm": llm}


@dataclass
class Manifest:
    data: dict[str, Any]

    @property
    def frozen(self) -> bool:
        return self.data.get("status") == "frozen"


def holdout_results(workspace: Workspace) -> list[str]:
    """Holdout result files and directories present in the protocol's results directory."""

    if not workspace.results_dir.exists():
        return []
    found = []
    for metadata_path in sorted(workspace.results_dir.glob("*/metadata.json")):
        if read_json(metadata_path).get("split") == "holdout":
            found.append(metadata_path.parent.name)
    found += [p.name for p in sorted(workspace.results_dir.glob("*holdout*.json"))]
    return found


def freeze(workspace: Workspace, env: Environment, note: str = "", force: bool = False) -> Manifest:
    """Write the manifest. Never replaces a manifest that holdout results were produced under."""

    if workspace.manifest_path.exists():
        used = holdout_results(workspace)
        if used:
            raise ManifestError(
                "refusing to replace a manifest that holdout results were produced under "
                f"({', '.join(used)}); start a new protocol version instead"
            )
        if not force:
            raise ManifestError("manifest already exists; use --force only before any holdout run")
    data: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "protocol": workspace.protocol,
        "status": "frozen",
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(workspace.root),
        "note": note,
        "inputs": tracked_inputs(workspace),
        "environment": environment_record(workspace),
        "providers": provider_fingerprint(env),
        "amendments": [],
    }
    write_json(workspace.manifest_path, data)
    return Manifest(data)


@dataclass(frozen=True)
class Verification:
    problems: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def verify(workspace: Workspace, env: Environment, groups: tuple[str, ...] = GROUPS) -> Verification:
    """Compare the workspace with the manifest. ``problems`` block the holdout; ``warnings`` do not."""

    if not workspace.manifest_path.exists():
        return Verification(["no manifest: run `dirag freeze` after finishing development"], [])
    manifest = Manifest(read_json(workspace.manifest_path))
    if manifest.data.get("schema") != SCHEMA_VERSION:
        return Verification([f"manifest schema {manifest.data.get('schema')} is not {SCHEMA_VERSION}"], [])
    problems: list[str] = []
    warnings: list[str] = []
    if not manifest.frozen:
        problems.append("manifest status is not 'frozen'")
    current = tracked_inputs(workspace)
    for group in groups:
        recorded: dict[str, str] = manifest.data["inputs"][group]
        for key, expected in sorted(recorded.items()):
            if current[group].get(key) != expected:
                problems.append(f"{group}: changed after freeze: {key}")
        for key in sorted(set(current[group]) - set(recorded)):
            problems.append(f"{group}: new input not in manifest: {key}")
    environment = environment_record(workspace)
    recorded_env = manifest.data["environment"]
    if environment["uv_lock_sha256"] != recorded_env["uv_lock_sha256"]:
        problems.append("environment: uv.lock changed after freeze")
    for key in ("python", "implementation"):
        if environment[key] != recorded_env[key]:
            warnings.append(f"environment: {key} {environment[key]} differs from frozen {recorded_env[key]}")
    for name, version in recorded_env["packages"].items():
        if environment["packages"].get(name) != version:
            warnings.append(f"environment: {name} {environment['packages'].get(name)} differs from {version}")
    current_providers = provider_fingerprint(env)
    recorded_providers = manifest.data["providers"]
    if current_providers["embedding"] != recorded_providers["embedding"]:
        problems.append("embedding provider differs from manifest")
    if recorded_providers.get("llm") is not None and current_providers["llm"] != recorded_providers["llm"]:
        problems.append("LLM provider settings differ from manifest")
    return Verification(problems, warnings)


def llm_frozen(workspace: Workspace) -> bool:
    return (
        workspace.manifest_path.exists()
        and read_json(workspace.manifest_path)["providers"].get("llm") is not None
    )


def manifest_sha(workspace: Workspace) -> str | None:
    return sha256_file(workspace.manifest_path) if workspace.manifest_path.exists() else None


def amend_llm_providers(workspace: Workspace, env: Environment, note: str) -> Manifest:
    """Record the LLM provider in a manifest frozen without one (logged amendment, once)."""

    check = verify(workspace, env)
    if not check.ok:
        raise ManifestError("cannot amend a manifest whose inputs changed: " + "; ".join(check.problems))
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


def amend_evaluation(workspace: Workspace, env: Environment, note: str) -> Manifest:
    """Accept corrected evaluation code after freeze; the pipeline group must be intact."""

    if not note.strip():
        raise ManifestError("an evaluation amendment needs a note explaining the correction")
    check = verify(workspace, env, groups=("pipeline",))
    if not check.ok:
        raise ManifestError(
            "the pipeline changed, which requires a new protocol version: " + "; ".join(check.problems)
        )
    data = read_json(workspace.manifest_path)
    current = tracked_inputs(workspace)["evaluation"]
    if current == data["inputs"]["evaluation"]:
        raise ManifestError("the evaluation inputs are unchanged; nothing to amend")
    data["amendments"].append(
        {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "field": "inputs.evaluation",
            "note": note,
            "previous": data["inputs"]["evaluation"],
            "new": current,
        }
    )
    data["inputs"]["evaluation"] = current
    write_json(workspace.manifest_path, data)
    return Manifest(data)
