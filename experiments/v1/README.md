# Protocol v1 (historical)

Protocol v1 was frozen on 2026-11-22 and its holdout was run for configurations
A and B. An independent audit then inspected the holdout and found defects in
the system and the design. v1 is therefore kept only as a historical record:

* its holdout is no longer blind and must not be reused as a test set;
* its results must not be recomputed with later code, whose parser, classifier,
  generator and metrics differ.

## Where it lives

All v1 artefacts remain at their original paths, byte-identical to commit
`30bce655ef7fdd0095c24864aea3b85dac3ebaf7` (local tag `protocol-v1`):
`data/synthetic/`, `experiments/configurations/`, `experiments/results/` and
`experiments/manifest.json`. [`integrity.json`](integrity.json) lists the
SHA-256 of all 26 artefacts.

`dirag verify-v1` checks the working tree against those hashes and against the
git blobs of the v1 commit. It also recomputes every input hash of the v1
manifest (data, prompts and package source) from that commit's tree, which
proves that the commit contains exactly the frozen protocol.

To reproduce v1 itself, use its own code:

```bash
git worktree add ../dirag-v1 30bce655ef7fdd0095c24864aea3b85dac3ebaf7
cd ../dirag-v1 && uv sync --extra app && uv run dirag verify-manifest
```

## What the audit found (2026-11-22)

* A vs B on the holdout: Recall@4 0.932 vs 0.948; per question 5 better and 2
  worse, sign test p ≈ 0.453. The three repetitions were identical
  deterministic runs, not independent observations.
* Both regressions came from rule-parser defects. In Q-038 "outside" was read
  as a negation; in Q-012 "supplier" inside a subject name was read as a
  constraint. The original README reported one negated question.
* The rule lexicon labelled all 33 generator templates as intended (93.9 %
  macro accuracy by construction), and every question paraphrase contained a
  lexicon cue.
* With five records per subject and top-4 retrieval, a random choice of four
  of a subject's records already reaches Recall@4 = 0.80.
* The 8/8 correct abstentions were all on out-of-domain questions; four of six
  in-domain probes were answered.
* Extractive citation faithfulness (1.000) held by construction, and the
  latency difference reflected warm-up order.
* Latent defects for the LLM configurations: an LLM judge could grade an
  abstention as correct, model prose was not validated, malformed provider
  responses crashed runs, and the spending limit did not trip with the default
  zero prices.

Protocol v2 addresses these points; see `docs/methodology.md`.
