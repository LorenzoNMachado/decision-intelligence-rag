"""Protocol, configuration and component factory.

Scientific parameters live in ``experiments/v2/configurations/protocol.json``
and the six configurations in ``experiments/v2/configurations/{A..F}.json``.
Provider settings (endpoints, model names, API keys) come from environment
variables, documented in ``.env.example``. Secrets are read only here and are
never written to results or manifests.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decision_intelligence.corpus.io import load_chunks, load_documents, load_questions, read_json
from decision_intelligence.embeddings.base import Embedder
from decision_intelligence.embeddings.hashing import HashingEmbedder
from decision_intelligence.embeddings.openai_compatible import OpenAICompatibleEmbedder, endpoint_identity
from decision_intelligence.generation.generators import ExtractiveGenerator, Generator, LLMGenerator
from decision_intelligence.intent.parsers import (
    IntentParser,
    LLMIntentParser,
    NoIntentParser,
    RuleBasedIntentParser,
)
from decision_intelligence.llm.client import Budget, LLMClient, OpenAICompatibleLLM, Pricing
from decision_intelligence.models import Chunk, DecisionDocument, Question
from decision_intelligence.retrieval.retriever import RetrievalParams, Retriever
from decision_intelligence.semantic_signals.classifier import entity_names
from decision_intelligence.semantic_signals.index import SignalIndex

RETRIEVAL_STRATEGIES = ("vector", "rules", "llm_intent")
GENERATORS = ("extractive", "llm")


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    retrieval: str
    generator: str
    description: str

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        data = read_json(path)
        config = cls(
            id=data["id"],
            retrieval=data["retrieval"],
            generator=data["generator"],
            description=data["description"],
        )
        if config.retrieval not in RETRIEVAL_STRATEGIES or config.generator not in GENERATORS:
            raise ConfigurationError(f"{path}: unknown retrieval or generator")
        return config

    @property
    def needs_llm(self) -> bool:
        return self.retrieval == "llm_intent" or self.generator == "llm"


@dataclass(frozen=True)
class ExperimentProtocol:
    data: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> ExperimentProtocol:
        return cls(read_json(path))

    def threshold_for(self, embedder: Embedder) -> float:
        thresholds: dict[str, float] = self.data["evidence_threshold"]
        backend = embedder.name.split(":", 1)[0]
        if embedder.name in thresholds:
            return float(thresholds[embedder.name])
        if backend in thresholds:
            return float(thresholds[backend])
        return float(thresholds["default"])

    def retrieval_params(self, embedder: Embedder) -> RetrievalParams:
        r = self.data["retrieval"]
        return RetrievalParams(
            top_k=int(r["top_k"]),
            candidate_pool=int(r["candidate_pool"]),
            evidence_threshold=self.threshold_for(embedder),
            required_bonus=float(r["required_bonus"]),
            preferred_bonus=float(r["preferred_bonus"]),
            contradiction_penalty=float(r["contradiction_penalty"]),
            veto_contradictions=bool(r["veto_contradictions"]),
        )

    @property
    def repetitions(self) -> int:
        return int(self.data["repetitions"])

    def repetitions_for(self, config: ExperimentConfig) -> int:
        """LLM configurations repeat (providers are not deterministic); deterministic ones run once.

        Repeating a deterministic configuration produces identical rows, which are not
        independent observations, so protocol v2 runs them ``deterministic_repetitions`` times.
        """

        if config.needs_llm:
            return self.repetitions
        return int(self.data.get("deterministic_repetitions", self.repetitions))


def _pricing(prompt: float | None, completion: float | None) -> Pricing:
    try:
        return Pricing(prompt, completion)
    except ValueError as exc:
        raise ConfigurationError(f"DIRAG_PRICE_*: {exc}") from exc


def _budget_limit(value: float | None) -> float | None:
    try:
        Budget(value)
    except ValueError as exc:
        raise ConfigurationError(f"DIRAG_BUDGET_USD: {exc}") from exc
    return value


@dataclass
class Environment:
    """Provider settings read from environment variables."""

    embedding_backend: str = "hashing"
    embedding_model: str = ""
    embedding_base_url: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    judge_model: str = ""
    llm_temperature: float | None = 0.0
    llm_seed: int | None = None
    llm_response_format: str = "json_schema"
    llm_timeout_s: float = 60.0
    pricing: Pricing = field(default_factory=Pricing)
    budget_usd: float | None = 1.0  # None: no limit (explicit DIRAG_BUDGET_USD=none)
    _llm_api_key: str | None = field(default=None, repr=False)
    _embedding_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> Environment:
        def opt_float(name: str, default: float | None) -> float | None:
            raw = os.environ.get(name)
            if raw is None or raw.strip() == "":
                return default
            if raw.strip().lower() == "none":
                return None
            try:
                return float(raw)
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be a number or 'none'") from exc

        seed = os.environ.get("DIRAG_LLM_SEED")
        if seed and not seed.strip().lstrip("-").isdigit():
            raise ConfigurationError("DIRAG_LLM_SEED must be an integer")
        timeout = opt_float("DIRAG_LLM_TIMEOUT_S", 60.0)
        if timeout is None or timeout <= 0:
            raise ConfigurationError("DIRAG_LLM_TIMEOUT_S must be a positive number")
        return cls(
            embedding_backend=os.environ.get("DIRAG_EMBEDDING_BACKEND", "hashing"),
            embedding_model=os.environ.get("DIRAG_EMBEDDING_MODEL", ""),
            embedding_base_url=os.environ.get(
                "DIRAG_EMBEDDING_BASE_URL", os.environ.get("DIRAG_LLM_BASE_URL", "")
            ),
            llm_base_url=os.environ.get("DIRAG_LLM_BASE_URL", ""),
            llm_model=os.environ.get("DIRAG_LLM_MODEL", ""),
            judge_model=os.environ.get("DIRAG_JUDGE_MODEL", ""),
            llm_temperature=opt_float("DIRAG_LLM_TEMPERATURE", 0.0),
            llm_seed=int(seed) if seed else None,
            llm_response_format=os.environ.get("DIRAG_LLM_RESPONSE_FORMAT", "json_schema"),
            llm_timeout_s=timeout,
            pricing=_pricing(
                opt_float("DIRAG_PRICE_PROMPT_PER_MTOK", None),
                opt_float("DIRAG_PRICE_COMPLETION_PER_MTOK", None),
            ),
            budget_usd=_budget_limit(opt_float("DIRAG_BUDGET_USD", 1.0)),
            _llm_api_key=os.environ.get("DIRAG_LLM_API_KEY") or None,
            _embedding_api_key=os.environ.get("DIRAG_EMBEDDING_API_KEY")
            or os.environ.get("DIRAG_LLM_API_KEY")
            or None,
        )

    @property
    def llm_available(self) -> bool:
        return bool(self._llm_api_key and self.llm_base_url and self.llm_model)

    def public(self) -> dict[str, Any]:
        """Settings safe to record in results (no secrets)."""

        return {
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "embedding_endpoint": endpoint_identity(self.embedding_base_url)
            if self.embedding_backend != "hashing" and self.embedding_base_url
            else "",
            "llm_endpoint": endpoint_identity(self.llm_base_url) if self.llm_base_url else "",
            "llm_model": self.llm_model,
            "judge_model": self.judge_model,
            "llm_temperature": self.llm_temperature,
            "llm_seed": self.llm_seed,
            "llm_response_format": self.llm_response_format,
            "llm_api_key_configured": self._llm_api_key is not None,
            "pricing_configured": self.pricing.configured,
            "budget_usd": self.budget_usd,
        }


PROTOCOL = "v2"


@dataclass
class Workspace:
    """Paths of the artefacts of the current protocol (v2).

    Protocol v1 is historical: its artefacts live at their original paths
    (``data/synthetic``, ``experiments/{configurations,results,manifest.json}``) and
    are only verified (``evaluation.historical``), never run with this code.
    """

    root: Path = field(default_factory=Path.cwd)
    protocol: str = PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != PROTOCOL:
            raise ConfigurationError(
                f"protocol {self.protocol!r} is historical; check out its commit to run it "
                "(see experiments/v1/README.md)"
            )

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / self.protocol

    @property
    def corpus_path(self) -> Path:
        return self.data_dir / "corpus.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def questions_path(self) -> Path:
        return self.data_dir / "questions.json"

    @property
    def question_specs_path(self) -> Path:
        return self.data_dir / "question_specs.json"

    @property
    def signal_index_path(self) -> Path:
        return self.data_dir / "signal_index.rules.json"

    @property
    def experiment_dir(self) -> Path:
        return self.root / "experiments" / self.protocol

    @property
    def config_dir(self) -> Path:
        return self.experiment_dir / "configurations"

    @property
    def protocol_path(self) -> Path:
        return self.config_dir / "protocol.json"

    @property
    def results_dir(self) -> Path:
        return self.experiment_dir / "results"

    @property
    def manifest_path(self) -> Path:
        return self.experiment_dir / "manifest.json"

    @property
    def cache_dir(self) -> Path:
        return self.root / ".cache"

    def load_documents(self) -> list[DecisionDocument]:
        return load_documents(self.corpus_path)

    def load_chunks(self) -> list[Chunk]:
        return load_chunks(self.chunks_path)

    def load_questions(self) -> list[Question]:
        return load_questions(self.questions_path)

    def load_protocol(self) -> ExperimentProtocol:
        return ExperimentProtocol.load(self.protocol_path)

    def load_configs(self, ids: list[str] | None = None) -> list[ExperimentConfig]:
        configs = [ExperimentConfig.load(p) for p in sorted(self.config_dir.glob("[A-Z].json"))]
        if ids:
            wanted = set(ids)
            configs = [c for c in configs if c.id in wanted]
            if {c.id for c in configs} != wanted:
                raise ConfigurationError(
                    f"unknown configuration ids: {sorted(wanted - {c.id for c in configs})}"
                )
        return configs


def build_embedder(env: Environment, workspace: Workspace) -> Embedder:
    if env.embedding_backend == "hashing":
        return HashingEmbedder()
    if env.embedding_backend == "openai-compatible":
        if not env.embedding_model or not env.embedding_base_url:
            raise ConfigurationError("DIRAG_EMBEDDING_MODEL and DIRAG_EMBEDDING_BASE_URL are required")
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", env.embedding_model)
        return OpenAICompatibleEmbedder(
            model=env.embedding_model,
            base_url=env.embedding_base_url,
            api_key=env._embedding_api_key,
            cache_path=workspace.cache_dir / f"embeddings-{safe}.json",
        )
    raise ConfigurationError(f"unknown embedding backend {env.embedding_backend!r}")


def build_llm(env: Environment, model: str | None = None, budget: Budget | None = None) -> LLMClient:
    if not env.llm_available:
        raise ConfigurationError(
            "LLM configurations need DIRAG_LLM_BASE_URL, DIRAG_LLM_MODEL and DIRAG_LLM_API_KEY "
            "(see .env.example)"
        )
    return OpenAICompatibleLLM(
        model=model or env.llm_model,
        base_url=env.llm_base_url,
        api_key=env._llm_api_key,
        temperature=env.llm_temperature,
        seed=env.llm_seed,
        response_format=env.llm_response_format,
        timeout_s=env.llm_timeout_s,
        pricing=env.pricing,
        budget=budget if budget is not None else Budget(env.budget_usd),
    )


@dataclass
class Components:
    documents: dict[str, DecisionDocument]
    chunks: list[Chunk]
    embedder: Embedder
    protocol: ExperimentProtocol
    signal_index: SignalIndex
    entities: tuple[str, ...] = ()
    retrievers: dict[str, Retriever] = field(default_factory=dict)


def load_components(workspace: Workspace, env: Environment) -> Components:
    documents = workspace.load_documents()
    return Components(
        documents={d.id: d for d in documents},
        chunks=workspace.load_chunks(),
        embedder=build_embedder(env, workspace),
        protocol=workspace.load_protocol(),
        signal_index=SignalIndex.load(workspace.signal_index_path),
        entities=tuple(entity_names(documents)),
    )


def make_intent_parser(strategy: str, llm: LLMClient | None, entities: tuple[str, ...] = ()) -> IntentParser:
    if strategy == "vector":
        return NoIntentParser()
    if strategy == "rules":
        return RuleBasedIntentParser(entities)
    if strategy == "llm_intent":
        if llm is None:
            raise ConfigurationError("strategy llm_intent requires an LLM client")
        return LLMIntentParser(llm)
    raise ConfigurationError(f"unknown retrieval strategy {strategy!r}")


def make_retriever(components: Components, strategy: str, llm: LLMClient | None = None) -> Retriever:
    """Build a retriever, reusing the chunk embeddings across strategies."""

    params = components.protocol.retrieval_params(components.embedder)
    existing = next(iter(components.retrievers.values()), None)
    retriever = Retriever(
        strategy=strategy,
        chunks=components.chunks,
        embedder=components.embedder,
        params=params,
        intent_parser=make_intent_parser(strategy, llm, components.entities),
        signal_index=None if strategy == "vector" else components.signal_index,
        index=existing.index if existing is not None else None,
    )
    components.retrievers.setdefault(strategy, retriever)
    return retriever


def make_generator(kind: str, llm: LLMClient | None, protocol: ExperimentProtocol) -> Generator:
    if kind == "extractive":
        return ExtractiveGenerator()
    if kind == "llm":
        if llm is None:
            raise ConfigurationError("the LLM generator requires an LLM client")
        g = protocol.data["generation"]
        return LLMGenerator(
            llm, enforce_quotes=g["quote_policy"] == "enforce", max_tokens=int(g["max_tokens"])
        )
    raise ConfigurationError(f"unknown generator {kind!r}")
