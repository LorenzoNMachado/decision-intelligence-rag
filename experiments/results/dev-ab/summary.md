# Results: dev-ab (dev split)

Embedder: `hashing-v1:d2048:uni+bi` · evidence threshold: 0.18

Correctness in this table uses the **lexical key-fact proxy** unless an LLM-judge table follows.

| Metric | A | B |
|---|---:|---:|
| Retrieval | vector | rules |
| Generator | extractive | extractive |
| recall_at_k | 1.000 | 1.000 |
| supporting_source_precision | 0.375 | 0.479 |
| expected_source_coverage | 1.000 | 1.000 |
| correctness | 0.125 | 0.250 |
| correctness_partial | 0.750 | 0.625 |
| faithfulness_quote | 1.000 | 1.000 |
| faithfulness_llm | — | — |
| correct_abstention_rate | 1.000 | 1.000 |
| false_answer_rate | 0.000 | 0.000 |
| false_abstention_rate | 0.000 | 0.000 |
| operational_failures | 0 | 0 |
| intent_failures | 0 | 0 |
| rejected_citations | 0 | 0 |
| irrelevant_sources_cited_per_rep | 20.0 | 14.0 |
| latency_p50_ms | 3.000 | 3.000 |
| latency_p95_ms | 5.900 | 4.800 |
| cost_per_question_usd | 0.000 | 0.000 |
