# Architecture

## Overview

```
question
   │
   ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│ Intent parser        │   │ Signal index (offline)       │
│  S1 none             │   │  document × 5 signals →      │
│  S2 rules (lexicon)  │   │  PRESENT / ABSENT / UNCLEAR / │
│  S3 LLM + validation │   │  ERROR                        │
└─────────┬────────────┘   └──────────────┬───────────────┘
          │ constraints                    │ labels
          ▼                                ▼
┌──────────────────────────────────────────────────────────┐
│ Retriever (shared by all strategies)                     │
│  embed question → cosine vs 450 chunks → best chunk per  │
│  document → candidate pool (20) → signal adjustment →    │
│  top-k (4) → evidence gate (threshold + signal veto)     │
└─────────────────────────┬────────────────────────────────┘
                          │ admitted documents only
                          ▼
┌──────────────────────────────────────────────────────────┐
│ Generator                                                │
│  extractive: best sentence per document, quoted          │
│  LLM: structured JSON, per-claim citations and quotes    │
└─────────────────────────┬────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────┐
│ Grounding                                                │
│  drop citations outside the admitted set · record leaked │
│  ids · verify quotes literally · ANSWER with no grounded │
│  claim → ABSTAIN · (experimental) LLM claim verification │
└─────────────────────────┬────────────────────────────────┘
                          ▼
                answer + claims + evidence log
```

## Packages

| Package | Responsibility |
|---|---|
| `corpus` | Synthetic record generator, deterministic chunking, question-set builder, integrity validation, JSONL I/O |
| `embeddings` | `Embedder` protocol, exact cosine `VectorIndex`, offline `HashingEmbedder`, cached `OpenAICompatibleEmbedder` |
| `semantic_signals` | Cue lexicon, rule-based and LLM document classifiers, persisted `SignalIndex`, accuracy against generator labels |
| `intent` | `IntentParser` protocol with the none / rules / LLM parsers |
| `retrieval` | `Retriever`: similarity, signal adjustment, ranking and evidence gate |
| `generation` | Extractive and LLM generators; versioned prompts in `generation/prompts/*.md` |
| `grounding` | Evidence rendering, citation and quote validation, experimental claim verifier |
| `llm` | OpenAI-compatible chat client with classified errors, retries, cost and budget; `ScriptedLLM` for tests |
| `evaluation` | Metrics, judges, experiment runner, manifest (freeze/verify), ablation and calibration |
| `config` | Protocol file, environment variables, component factory |
| `cli` | `dirag` command-line entry point |

`app/streamlit_app.py` is a thin interface over the same factory functions used by the CLI.

## Design decisions

**One retriever, three intent sources.** The strategies differ only in the intent
parser. With no active constraint the signal-aware path reduces exactly to the
vector baseline (a unit test asserts identical scores), so any difference
between strategies is caused by the semantic signals and nothing else.

**Neutral uncertainty.** UNCLEAR and ERROR labels neither add nor subtract
score. A classifier failure therefore degrades a strategy towards the baseline
instead of silently excluding documents.

**Gate on similarity, not on the adjusted score.** The evidence gate uses the raw
cosine similarity so that bonuses cannot admit an irrelevant document. Signal
contradiction of a REQUIRED constraint is an additional, optional veto.

**The LLM sees only admitted evidence.** When the gate admits nothing, the LLM
generator abstains without a model call. Citations outside the admitted set are
removed and logged; an ANSWER without a surviving claim becomes ABSTAIN.

**Quote policy recorded, enforcement replayed.** Experiments run with
`quote_policy: "record"`: every claim carries `quote_verified`. The effect of
enforcing quotes is computed afterwards from the same rows (`quote_enforce_replay`),
so the ablation costs no additional model calls.

**OpenAI-compatible interfaces.** Both the embedder and the chat client target
the de-facto standard `/embeddings` and `/chat/completions` endpoints, so hosted
gateways, vendor endpoints and local servers can be swapped through environment
variables without code changes.

**Secrets stay in the environment.** API keys are read once in `config.py`,
never stored in result files or manifests, and redacted from provider error
messages before they are recorded.
