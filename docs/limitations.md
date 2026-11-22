# Limitations

These limitations bound what the reported numbers can support. They are
listed so that readers can weigh the results, not as a to-do list.

## Data

* **Synthetic corpus.** Records are generated from phrase pools, and real
  decision records are longer, noisier and less regular. Results here say
  nothing about performance on organisational documents.
* **References are not independent annotations.** Questions, expected evidence
  and references are derived from the same generator as the corpus. This
  guarantees that every reference is answerable from the corpus (a check that
  `corpus-validate` enforces), but it also means that the questions follow the
  same vocabulary as the documents.
* **Unanswerable questions are out of domain.** All ten unanswerable questions
  concern topics that no record touches (catering, parking, sponsorship…). They
  test whether the gate rejects unrelated evidence, not whether a system resists
  near-miss questions about subjects that *are* in the corpus. The perfect
  abstention scores of A and B on the holdout should be read in that light.
* **Small evaluation set.** There are 50 questions, of which 10 are unanswerable
  and only 2 are in the dev split. Differences of a few percentage points between
  configurations are within what chance alone can produce, and no significance
  tests are reported.

## Signals

* **Lexicon and generator have the same author.** The rule lexicon was written
  without reading the generator's phrase pools line by line, but both encode the
  same intuitions about each signal. Its measured accuracy (93.9 % macro, see
  `experiments/results/signal-classifier-rules.json`) is therefore an optimistic
  estimate of what the same rules would achieve on real text.
* **Known error sources are kept.** Job titles ("Chief Information Security
  Officer"), department names ("People Operations", "Legal Operations") and
  domain words trigger false PRESENT labels. They were not tuned away, because
  doing so against the generator's labels would overfit the evaluation.

## Embeddings

* **The offline embedder is lexical.** `hashing-v1` measures word and bigram
  overlap, not meaning. Results obtained with it, including all results
  currently published in this repository, are lexical-retrieval results. A dense
  embedding model can be used through `DIRAG_EMBEDDING_BACKEND=openai-compatible`,
  but needs its own dev calibration and its own freeze.
* **Threshold calibrated on 10 questions.** The 0.18 threshold comes from a dev
  split with two unanswerable questions. The optimal range on dev (0.11–0.25)
  says little about how close holdout questions lie to the boundary.

## Judging

* **Lexical correctness is a proxy.** It requires literal key facts and forbids
  literal mentions of non-expected records. Paraphrases are under-credited and
  the extractive generator's multi-record answers are penalised by design.
* **The LLM judge and claim verifier are single automated judges.** Neither has
  been calibrated against human labels in this repository. The claim verifier
  is an experimental extension and is reported separately.

## LLM configurations

* **Not yet executed.** Configurations C–F, Strategy 3 and the LLM judge require
  provider credentials. They are implemented and tested with scripted models,
  but this repository does not yet contain any result measured with a real LLM.
* **Provider determinism.** Temperature 0 and a seed are requested when
  supported, but hosted models do not guarantee identical outputs, which is why
  each configuration runs three repetitions.

## Scope

* This is an independent implementation of concepts studied during applied
  research in Computer Science at PUCPR. It does not reproduce, and makes no
  claim about, the results of that earlier study.
* Latency is measured on the machine that ran the experiment. It is comparable
  between configurations of the same run, not across machines.
* Cost per question excludes judge and claim-verification calls, which are
  evaluation overhead rather than system cost.
