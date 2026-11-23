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
    make_checker,
    make_generator,
    make_retriever,
)
from decision_intelligence.corpus.io import read_json
from decision_intelligence.llm.client import LLMClient
from decision_intelligence.models import Claim, GeneratedAnswer
from decision_intelligence.pipeline import Pipeline, PipelineResult

ROOT = Path(__file__).resolve().parents[1]

RETRIEVAL_LABELS = {
    "vector": "Vector baseline (hashing embedder)",
    "bm25": "BM25 lexical baseline",
    "rules-v2": "Rule-based signals, v2 (historical behaviour)",
    "rules-v3": "Rule-based signals, v3 (scoped labels, topical ranking)",
    "llm_intent": "LLM semantic intent (needs a provider)",
}
GENERATOR_LABELS = {
    "extractive-v1": "Extractive v1 (largest word overlap)",
    "extractive-v2": "Extractive v2 (quotes the sentence the sufficiency check found)",
    "llm": "LLM with validated citations (needs a provider)",
}
# (retrieval, sufficiency check, generator) of the protocol v3 configurations.
CONFIGS = {
    "A": ("vector", False, "extractive-v1"),
    "B": ("rules-v2", False, "extractive-v1"),
    "C": ("rules-v3", False, "extractive-v1"),
    "D": ("rules-v2", True, "extractive-v1"),
    "E": ("rules-v3", True, "extractive-v1"),
    "F": ("rules-v3", True, "extractive-v2"),
    "G": ("bm25", False, "extractive-v1"),
    "H": ("vector", True, "extractive-v2"),
    "L": ("rules-v3", True, "llm"),
}
EXAMPLES = [
    "Which decisions about the identity directory have security implications?",
    "Which decisions about the travel booking process involve no external vendor?",
    "What budget was approved for the decision to introduce a patient scheduling system?",
    "Who leads the work on the decision to trial a route optimisation engine?",
    "Which architect designed the new headquarters atrium?",
]
STATE_HELP = {
    "ANSWER": "Every statement below cites an admitted record (judged sufficient, when the check is on).",
    "ABSTAIN": "The system declined: no admitted or no sufficient evidence, or no claim survived validation.",
    "ERROR": "An operational failure (provider, network, schema). This is not an abstention.",
}
DEMO_NOTE = (
    "Live demonstration on the synthetic corpus — not a measured result. "
    "Measured results are in the 'Measured results' tab."
)
RESULTS = [
    (
        "Protocol v3 · holdout · all offline configurations",
        ROOT / "experiments" / "v3" / "results" / "holdout" / "summary.json",
        "Current protocol. Exploratory: the holdout was unseen by the system but written by its author.",
    ),
    (
        "Protocol v2 · holdout · A and B (historical)",
        ROOT / "experiments" / "v2" / "results" / "holdout-ab" / "summary.json",
        "Superseded; different corpus, questions and code. Not comparable with v3.",
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
    ("correct_abstention_in_domain", "Correct abstention, in-domain"),
    ("correct_abstention_out_of_domain", "Correct abstention, out-of-domain"),
    ("false_abstention_rate", "False abstention (answerable)"),
    ("unsupported_answer_rate", "Unsupported answers"),
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


def format_p(p: float | None) -> str:
    if p is None:
        return "n/a, all tied"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


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


def run(question: str, retrieval: str, generator: str, sufficiency: bool) -> PipelineResult:
    components = get_components()
    llm = get_llm()
    pipeline = Pipeline(
        make_retriever(components, retrieval, llm),
        make_generator(generator, llm, components.protocol),
        components.documents,
        checker=make_checker(components) if sufficiency else None,
    )
    return pipeline.run(question)


def retrieved_table(result: PipelineResult, components: Components) -> list[dict[str, Any]]:
    checked = {d.doc_id: d for d in result.sufficiency.documents} if result.sufficiency is not None else {}
    rows = []
    for d in result.retrieval.documents:
        doc = components.documents[d.doc_id]
        row: dict[str, Any] = {
            "rank": d.rank,
            "id": d.doc_id,
            "title": doc.title,
            "similarity": round(d.similarity, 3),
            "final score": round(d.final_score, 3),
            "on topic": "yes" if d.topical else "no",
            "admitted": "yes" if d.admitted else "no",
            "gate": d.gate_reason,
            "matched signals": ", ".join(d.matched_signals),
            "contradicted": ", ".join(d.contradicted_signals),
        }
        if result.sufficiency is not None:
            item = checked.get(d.doc_id)
            row["sufficiency"] = item.status.value if item else "not checked (not admitted)"
            row["why"] = item.reason if item else ""
        rows.append(row)
    return rows


def show_sufficiency(result: PipelineResult) -> None:
    report = result.sufficiency
    if report is None:
        st.caption("Evidence sufficiency check: off (v2 behaviour — the gate alone decides).")
        return
    need = report.need
    constraints = ", ".join(c.split(":")[0] + "=" + c.split(":")[1] for c in need.to_dict()["constraints"])
    st.caption(
        f"Evidence sufficiency (v3 mechanism): **{report.status.value}** — {report.reason}. "
        f"Asked for: {need.answer_type.value.lower()}"
        + (f" · focus: {', '.join(need.focus)}" if need.focus else "")
        + (f" · conditions: {constraints}" if constraints else "")
    )
    for item in report.sufficient():
        for sentence in item.supporting:
            st.markdown(
                f"> {plain(sentence)}  \n<span class='muted'>{item.doc_id}</span>", unsafe_allow_html=True
            )


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
        "Measured results only. Configurations that need an LLM (v3: L) have not been executed, so no LLM "
        "result is shown. Read differences with the paired comparisons below them: one deterministic run "
        "per configuration, synthetic data, and different corpora and questions in each protocol."
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
                {
                    "metric": label,
                    **{
                        c: "—" if e["lexical_judge"].get(key) is None else e["lexical_judge"][key]
                        for c, e in configs.items()
                    },
                }
                for key, label in SHOWN_METRICS
            ],
            hide_index=True,
        )
        shown = [
            p for p in summary.get("paired", {}) if p in ("A->B", "B->C", "C->E", "D->E", "E->F", "A->G")
        ]
        for pair in shown or list(summary.get("paired", {}))[:1]:
            c = summary["paired"][pair]["recall_at_k"]
            st.caption(
                f"{pair}, Recall@4 per question: second better {c['second_better']}, first better "
                f"{c['first_better']}, ties {c['ties']} (sign test p {format_p(c['sign_test_p'])})."
            )


def main() -> None:
    st.set_page_config(page_title="Decision Intelligence RAG", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Decision Intelligence RAG")
    st.caption(
        "Evidence-grounded retrieval and answer generation over a synthetic decision corpus (protocol v3)."
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
            st.info("LLM intent and the LLM generator need `DIRAG_LLM_*` variables (see `.env.example`).")
        st.caption("All records are synthetic and fictional.")

    ask_tab, compare_tab, results_tab, corpus_tab = st.tabs(
        ["Ask", "Compare configurations", "Measured results", "Corpus"]
    )

    with ask_tab:
        st.caption(DEMO_NOTE)
        example = st.selectbox("Example questions", ["—", *EXAMPLES])
        question = st.text_input("Question", value="" if example == "—" else example, max_chars=500)
        left, right = st.columns(2)
        retrieval = left.radio(
            "Retrieval",
            available_retrieval,
            index=available_retrieval.index("rules-v3"),
            format_func=RETRIEVAL_LABELS.__getitem__,
        )
        generator = right.radio(
            "Generator",
            available_generators,
            index=available_generators.index("extractive-v2"),
            format_func=GENERATOR_LABELS.__getitem__,
        )
        sufficiency = (
            st.checkbox(
                "Check evidence sufficiency before answering (v3)",
                value=True,
                disabled=generator == "extractive-v2",
            )
            or generator == "extractive-v2"
        )
        if st.button("Ask", type="primary", disabled=not question.strip()):
            result = run(question, retrieval, generator, sufficiency)
            st.subheader("Answer")
            show_answer(result)
            show_sufficiency(result)
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
        st.caption(DEMO_NOTE)
        runnable = [
            c for c, (r, _, g) in CONFIGS.items() if r in available_retrieval and g in available_generators
        ]
        chosen = st.multiselect("Configurations", runnable, default=["A", "F"])
        compare_question = st.text_input(
            "Question to compare", value=EXAMPLES[2], key="compare", max_chars=500
        )
        if st.button("Compare", disabled=not chosen):
            columns = st.columns(len(chosen))
            for column, config in zip(columns, chosen, strict=True):
                retrieval_kind, check, generator_kind = CONFIGS[config]
                with column:
                    st.markdown(f"### {config}")
                    st.caption(
                        f"{RETRIEVAL_LABELS[retrieval_kind]} · sufficiency {'on' if check else 'off'} · "
                        f"{GENERATOR_LABELS[generator_kind]}"
                    )
                    result = run(compare_question, retrieval_kind, generator_kind, check)
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
        v2 = components.signal_indices["rules-v2"].labels.get(doc.id, {})
        v3 = components.signal_indices["rules-v3"].labels.get(doc.id, {})
        st.dataframe(
            [
                {
                    "signal": s.value,
                    "rules-v2 label": v2[s].value if s in v2 else "",
                    "rules-v3 label": lab.value,
                    "generator's intended label": doc.ground_truth_signals[s.value],
                }
                for s, lab in v3.items()
            ],
            hide_index=True,
        )


main()
