"""Interactive demo: ask a question, pick retrieval and generator, inspect the evidence.

Run with ``uv run --extra app streamlit run app/streamlit_app.py``. Without
provider credentials the app runs fully offline (hashing embedder, vector and
rule-based retrieval, extractive generator); LLM options appear only when the
``DIRAG_LLM_*`` variables are configured.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import streamlit as st

from decision_intelligence.config import (
    Components,
    ConfigurationError,
    Environment,
    Workspace,
    build_llm,
    load_components,
    make_generator,
    make_retriever,
)
from decision_intelligence.corpus.io import read_json
from decision_intelligence.llm.client import LLMClient
from decision_intelligence.models import Claim, GeneratedAnswer
from decision_intelligence.pipeline import Pipeline, PipelineResult

ROOT = Path(__file__).resolve().parents[1]

RETRIEVAL_LABELS = {
    "vector": "Strategy 1 · Vector baseline",
    "rules": "Strategy 2 · Rule-based semantic enrichment",
    "llm_intent": "Strategy 3 · LLM semantic intent",
}
GENERATOR_LABELS = {"extractive": "Deterministic extractive", "llm": "LLM with validated citations"}
CONFIGS = {
    "A": ("vector", "extractive"),
    "B": ("rules", "extractive"),
    "C": ("vector", "llm"),
    "D": ("rules", "llm"),
    "E": ("llm_intent", "llm"),
    "F": ("llm_intent", "extractive"),
}
EXAMPLES = [
    "Which decisions about the feature flag service have security dependencies?",
    "Which decisions about the payroll processing system depend on an external vendor?",
    "Which decisions about the contract repository state that there is no legal requirement?",
    "What budget was approved for the decision to adopt a customer data warehouse?",
    "Which catering company was selected for the staff cafeteria contract?",
]
STATE_HELP = {
    "ANSWER": "Every statement below cites a record admitted by the evidence gate.",
    "ABSTAIN": "The system declined to answer: no admitted evidence, or no claim survived validation.",
    "ERROR": "An operational failure (provider, network, schema). This is not an abstention.",
}
RESULTS = [
    (
        "Protocol v2 · holdout · A and B",
        ROOT / "experiments" / "v2" / "results" / "holdout-ab" / "summary.json",
        "Current protocol. Exploratory: the holdout was unseen by the system but written by its author.",
    ),
    (
        "Protocol v1 · holdout · A and B (historical)",
        ROOT / "experiments" / "results" / "holdout-ab" / "summary.json",
        "Superseded. Its holdout was inspected during an audit and its parser had known defects; "
        "kept only as a historical record.",
    ),
]
SHOWN_METRICS = [
    ("recall_at_k", "Recall@4"),
    ("supporting_source_precision", "Supporting-source precision"),
    ("correctness", "Correctness (lexical proxy, strict)"),
    ("correct_abstention_rate", "Correct abstention (unanswerable)"),
    ("false_answer_rate", "False answers (unanswerable)"),
]

CSS = """
<style>
  .block-container {max-width: 1200px; padding-top: 2rem;}
  h1, h2, h3 {font-family: Georgia, 'Times New Roman', serif; font-weight: 500;}
  .status {display:inline-block; padding:2px 10px; border-radius:4px; font-size:0.85rem;
           font-weight:600; letter-spacing:0.04em;}
  .status-ANSWER {background:#e3efe6; color:#1d5b33;}
  .status-ABSTAIN {background:#f6ecd9; color:#7a5310;}
  .status-ERROR {background:#f6dede; color:#8a1f1f;}
  .muted {color:#6b7280; font-size:0.85rem;}
</style>
"""

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def plain(text: str) -> str:
    """Escape Markdown and HTML so model or record text is shown literally, never rendered."""

    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def format_cost(cost: float | None) -> str:
    return "cost unknown" if cost is None else f"${cost:.5f}"


def claim_badges(claim: Claim) -> str:
    if claim.quote_origin == "extracted":
        quote = "quote copied from the record"
    elif claim.quote_verified:
        quote = "quote found verbatim in the cited record"
    else:
        quote = "quote NOT found in the cited record"
    support = "support not checked" if claim.support == "NOT_CHECKED" else f"support: {claim.support}"
    return f"citation valid (admitted record) · {quote} · {support}"


@st.cache_resource(show_spinner="Embedding the corpus…")
def get_components() -> Components:
    return load_components(Workspace(ROOT), Environment.from_env())


@st.cache_resource
def get_llm() -> LLMClient | None:
    env = Environment.from_env()
    if not env.llm_available:
        return None
    return build_llm(env)


def run(question: str, retrieval: str, generator: str) -> PipelineResult:
    components = get_components()
    llm = get_llm()
    pipeline = Pipeline(
        make_retriever(components, retrieval, llm),
        make_generator(generator, llm, components.protocol),
        components.documents,
    )
    return pipeline.run(question)


def retrieved_table(result: PipelineResult, components: Components) -> list[dict[str, Any]]:
    rows = []
    for d in result.retrieval.documents:
        doc = components.documents[d.doc_id]
        rows.append(
            {
                "rank": d.rank,
                "id": d.doc_id,
                "title": doc.title,
                "similarity": round(d.similarity, 3),
                "final score": round(d.final_score, 3),
                "admitted": "yes" if d.admitted else "no",
                "gate": d.gate_reason,
                "matched signals": ", ".join(d.matched_signals),
                "contradicted": ", ".join(d.contradicted_signals),
            }
        )
    return rows


def show_answer(result: PipelineResult) -> None:
    answer: GeneratedAnswer = result.answer
    status = answer.status.value
    st.markdown(
        f'<span class="status status-{status}">{status}</span> '
        f'<span class="muted">{plain(answer.generator)} · {result.latency_ms:.0f} ms'
        f" · {format_cost(result.cost_usd)}</span>",
        unsafe_allow_html=True,
    )
    st.caption(STATE_HELP[status])
    if status == "ABSTAIN":
        st.markdown(plain(answer.text))
        st.caption(f"Reason: {answer.abstain_reason}")
    elif status == "ERROR":
        st.error(f"Operational failure ({answer.error_kind}): {answer.error}")
    else:
        for claim in answer.claims:
            with st.expander(f"{', '.join(claim.source_ids)} — {claim.statement[:90]}", expanded=True):
                st.markdown(plain(claim.statement))
                st.markdown("> " + plain(claim.quote))
                st.caption(claim_badges(claim))
        st.caption("A valid citation and a matching quote show traceability, not that the statement is true.")
    if answer.rejected_citations:
        st.warning("Rejected citations (not in admitted evidence): " + ", ".join(answer.rejected_citations))
    if answer.discarded_text:
        with st.expander("Model output that failed validation (not an answer)"):
            st.text(answer.discarded_text)


def show_results() -> None:
    st.caption(
        "Measured results only. LLM configurations (C–F) have not been executed, so no LLM result is shown. "
        "Differences between configurations must be read with the paired sign test below them: "
        "one deterministic run per configuration, a few dozen questions, synthetic data."
    )
    for title, path, context in RESULTS:
        st.markdown(f"#### {title}")
        st.caption(context)
        if not path.exists():
            st.info("Not available in this checkout.")
            continue
        summary = read_json(path)
        configs = summary["configs"]
        st.dataframe(
            [
                {"metric": label, **{c: e["lexical_judge"].get(key) for c, e in configs.items()}}
                for key, label in SHOWN_METRICS
            ],
            hide_index=True,
        )
        for pair, metrics in summary.get("paired", {}).items():
            c = metrics["recall_at_k"]
            st.caption(
                f"{pair}, Recall@4 per question: second better {c['second_better']}, first better "
                f"{c['first_better']}, ties {c['ties']} (sign test p = {c['sign_test_p']})."
            )


def main() -> None:
    st.set_page_config(page_title="Decision Intelligence RAG", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Decision Intelligence RAG")
    st.caption(
        "Evidence-grounded retrieval and answer generation over a synthetic decision corpus (protocol v2)."
    )

    try:
        components = get_components()
    except (FileNotFoundError, ConfigurationError) as exc:
        st.error(f"Could not load the corpus: {exc}. Run `uv run dirag corpus-generate` first.")
        return
    llm_ready = get_llm() is not None
    available_retrieval = [k for k in RETRIEVAL_LABELS if llm_ready or k != "llm_intent"]
    available_generators = [k for k in GENERATOR_LABELS if llm_ready or k != "llm"]

    with st.sidebar:
        st.subheader("Setup")
        st.markdown(
            f"**Mode:** {'LLM provider configured' if llm_ready else 'offline demo (no API credentials)'}  \n"
            f"**Embedder:** `{components.embedder.name}`  \n"
            f"**Corpus:** {len(components.documents)} records · {len(components.chunks)} chunks  \n"
            f"**Evidence gate:** cosine ≥ {components.protocol.threshold_for(components.embedder)}"
        )
        if not llm_ready:
            st.info("Strategy 3 and the LLM generator need `DIRAG_LLM_*` variables (see `.env.example`).")
        st.caption("All records are synthetic and fictional.")

    ask_tab, compare_tab, results_tab, corpus_tab = st.tabs(
        ["Ask", "Compare configurations", "Measured results", "Corpus"]
    )

    with ask_tab:
        example = st.selectbox("Example questions", ["—", *EXAMPLES])
        question = st.text_input("Question", value="" if example == "—" else example, max_chars=500)
        left, right = st.columns(2)
        retrieval = left.radio(
            "Retrieval", available_retrieval, format_func=RETRIEVAL_LABELS.__getitem__, horizontal=False
        )
        generator = right.radio("Generator", available_generators, format_func=GENERATOR_LABELS.__getitem__)
        if st.button("Ask", type="primary", disabled=not question.strip()):
            result = run(question, retrieval, generator)
            st.subheader("Answer")
            show_answer(result)
            st.subheader("Retrieved documents")
            intent = result.retrieval.intent
            constraints = ", ".join(
                f"{s.signal.value} = {s.polarity.value} ({s.requirement.value.lower()})"
                for s in intent.signals
            )
            st.caption(
                f"Intent parser `{intent.parser}` · status {intent.status}"
                + (f" · {constraints}" if constraints else "")
                + (f" · {intent.notes}" if intent.notes else "")
            )
            st.dataframe(retrieved_table(result, components), hide_index=True, use_container_width=True)

    with compare_tab:
        runnable = [
            c for c, (r, g) in CONFIGS.items() if r in available_retrieval and g in available_generators
        ]
        chosen = st.multiselect("Configurations", runnable, default=runnable[:2])
        compare_question = st.text_input(
            "Question to compare", value=EXAMPLES[0], key="compare", max_chars=500
        )
        if st.button("Compare", disabled=not chosen):
            columns = st.columns(len(chosen))
            for column, config in zip(columns, chosen, strict=True):
                retrieval_kind, generator_kind = CONFIGS[config]
                with column:
                    st.markdown(f"### {config}")
                    st.caption(f"{RETRIEVAL_LABELS[retrieval_kind]} + {GENERATOR_LABELS[generator_kind]}")
                    result = run(compare_question, retrieval_kind, generator_kind)
                    show_answer(result)
                    st.caption(
                        "Top documents: "
                        + ", ".join(
                            f"{d.doc_id}{'' if d.admitted else ' (gated)'}"
                            for d in result.retrieval.documents
                        )
                    )

    with results_tab:
        show_results()

    with corpus_tab:
        st.caption(
            "Synthetic records generated by `decision_intelligence.corpus.generator`. "
            "All names are fictional."
        )
        doc_id = st.selectbox("Record", sorted(components.documents))
        doc = components.documents[doc_id]
        st.markdown(f"#### {plain(doc.title)}")
        st.caption(f"{doc.id} · {doc.domain} · {doc.status} · {doc.date} · {doc.owner_role}")
        st.markdown(plain(doc.decision))
        st.markdown(f"**Context.** {plain(doc.context)}")
        st.markdown(f"**Rationale.** {plain(doc.rationale)}")
        labels = components.signal_index.labels.get(doc.id, {})
        st.dataframe(
            [
                {
                    "signal": s.value,
                    "rule-based label": lab.value,
                    "generator's intended label": doc.ground_truth_signals[s.value],
                }
                for s, lab in labels.items()
            ],
            hide_index=True,
        )


main()
