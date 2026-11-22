# Results: dev-ab (protocol v2, dev split)

Embedder: `hashing-v1:d2048:uni+bi` · evidence threshold: 0.18 · questions: 20

* Correctness uses the lexical key-fact proxy unless an LLM-judge table follows.
* Questions are the unit of comparison; deterministic configurations run once, so there is no repetition variance to report.
* Quote faithfulness is reported only for model-written quotes; extractive quotes are copied from the record and verified by construction (shown as —).
* Latency is measured on one machine within one run and includes warm-up effects; it is not evidence that one configuration is faster than another.

| Metric | A | B |
|---|---:|---:|
| Retrieval | vector | rules |
| Generator | extractive | extractive |
| Repetitions | 1 | 1 |
| recall_at_k | 0.797 | 0.812 |
| supporting_source_precision | 0.359 | 0.370 |
| expected_source_coverage | 0.797 | 0.812 |
| correctness | 0.000 | 0.000 |
| correctness_partial | 0.625 | 0.625 |
| faithfulness_quote | — | — |
| faithfulness_llm | — | — |
| correct_abstention_rate | 0.500 | 0.500 |
| false_answer_rate | 0.500 | 0.500 |
| false_abstention_rate | 0.000 | 0.000 |
| operational_failures | 0 | 0 |
| intent_failures | 0 | 0 |
| rejected_citations | 0 | 0 |
| irrelevant_sources_cited_per_rep | 41.0 | 40.0 |
| latency_p50_ms | 1.300 | 0.900 |
| latency_p95_ms | 2.300 | 1.500 |
| cost_per_question_usd | 0.000 | 0.000 |

## Paired comparisons (per question, exact two-sided sign test)

| Pair | Metric | Questions | Second better | First better | Ties | p |
|---|---|---:|---:|---:|---:|---:|
| A->B | recall_at_k | 16 | 1 | 1 | 14 | 1.000 |
| A->B | correct | 20 | 0 | 0 | 20 | — |
