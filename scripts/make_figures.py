"""Generate the documentation figures from the measured protocol v2 results.

    uv run python scripts/make_figures.py          # writes docs/assets/*.svg
    uv run python scripts/make_figures.py --check  # fails if the committed figures are stale

Every number is computed from ``experiments/v2/results/holdout-ab/rows.jsonl``
and ``data/v2/questions.json`` with the project's own metric code, and the
headline metrics are cross-checked against ``summary.json`` before anything is
drawn. The figures are plain SVG (no plotting dependency, so the frozen
``uv.lock`` is untouched), deterministic byte for byte, full-scale (0-1) and
labelled with their denominators. They adapt to light and dark themes through
``prefers-color-scheme``. Nothing here reruns the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "v2" / "results" / "holdout-ab"
QUESTIONS = ROOT / "data" / "v2" / "questions.json"
ASSETS = ROOT / "docs" / "assets"

SERIES = (("A", "A · vector baseline"), ("B", "B · rule-based enrichment"))
CATEGORY_LABELS = {
    "provider_lookup": "Provider lookup",
    "regulation_lookup": "Regulation lookup",
    "framework_lookup": "Framework lookup",
    "signal_filter": "Signal filter",
    "conjunction_filter": "Conjunction filter",
    "hedged_filter": "Hedged filter",
    "disjunction_filter": "Disjunction filter",
    "negated_filter": "Negated filter",
}

STYLE = """
.surface{fill:#fcfcfb}.frame{fill:none;stroke:rgba(11,11,11,0.10)}
.ink{fill:#0b0b0b}.ink2{fill:#52514e}.muted{fill:#898781}
.grid{stroke:#e1e0d9;stroke-width:1}.base{stroke:#c3c2b7;stroke-width:1}
.sA{fill:#2a78d6}.sB{fill:#eb6834}
text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.t{font-size:15px;font-weight:600}.s{font-size:12px}.l{font-size:12px}.v{font-size:12px;font-weight:600}
@media (prefers-color-scheme: dark){
.surface{fill:#1a1a19}.frame{stroke:rgba(255,255,255,0.10)}
.ink{fill:#f0efec}.ink2{fill:#c3c2b7}.muted{fill:#898781}
.grid{stroke:#2c2c2a}.base{stroke:#383835}
.sA{fill:#3987e5}.sB{fill:#d95926}}
"""

BAR = 22  # bar thickness (<= 24px)
GAP = 2  # surface gap between touching bars
RADIUS = 4


@dataclass(frozen=True)
class Figures:
    overview: list[tuple[str, str, dict[str, float]]]  # (label, denominator, values)
    categories: list[tuple[str, int, dict[str, float]]]
    abstention: list[tuple[str, int, dict[str, int]]]  # (label, n, abstained count)


def _num(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def compute() -> Figures:
    sys.path.insert(0, str(ROOT / "src"))
    from decision_intelligence.corpus.io import load_questions, read_json, read_jsonl
    from decision_intelligence.evaluation.metrics import compute_metrics

    rows = read_jsonl(RUN / "rows.jsonl")
    questions = {q.id: q for q in load_questions(QUESTIONS)}
    summary = read_json(RUN / "summary.json")
    metrics = {c: compute_metrics([r for r in rows if r["config"] == c], questions) for c, _ in SERIES}
    for config, _ in SERIES:
        recorded = summary["configs"][config]["lexical_judge"]
        for key in ("recall_at_k", "supporting_source_precision", "correctness", "correctness_partial"):
            if metrics[config][key] != recorded[key]:
                raise SystemExit(
                    f"{config} {key}: recomputed {metrics[config][key]} != summary {recorded[key]}"
                )

    answerable = sum(q.answerable for q in questions.values() if q.split == "holdout")
    answered = {
        c: sum(
            1
            for r in rows
            if r["config"] == c
            and r["answer"]["status"] == "ANSWER"
            and questions[r["question_id"]].answerable
        )
        for c, _ in SERIES
    }
    if len(set(answered.values())) != 1:
        raise SystemExit("the precision denominator differs between configurations")
    overview = [
        ("Recall@4", f"{answerable} answerable", {c: metrics[c]["recall_at_k"] for c, _ in SERIES}),
        (
            "Supporting-source precision",
            f"{answered['A']} answered",
            {c: metrics[c]["supporting_source_precision"] for c, _ in SERIES},
        ),
        ("Strictly correct", f"{answerable} answerable", {c: metrics[c]["correctness"] for c, _ in SERIES}),
        (
            "Partially correct",
            f"{answerable} answerable",
            {c: metrics[c]["correctness_partial"] for c, _ in SERIES},
        ),
    ]

    recalls: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    abstained: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        q = questions[row["question_id"]]
        group = q.category if not q.answerable else "answerable"
        totals[group].add(q.id)
        abstained[group][row["config"]] += row["answer"]["status"] == "ABSTAIN"
        if q.answerable:
            expected = set(q.expected_doc_ids)
            top = {d["doc_id"] for d in row["retrieved"][:4]}
            recalls[q.category][row["config"]].append(len(expected & top) / len(expected))
    categories = [
        (CATEGORY_LABELS[c], len(recalls[c]["A"]), {k: sum(v) / len(v) for k, v in recalls[c].items()})
        for c in CATEGORY_LABELS
        if c in recalls
    ]
    abstention = [
        (
            "Unanswerable, in-domain",
            len(totals["unanswerable_in_domain"]),
            dict(abstained["unanswerable_in_domain"]),
        ),
        (
            "Unanswerable, out-of-domain",
            len(totals["unanswerable_out_of_domain"]),
            dict(abstained["unanswerable_out_of_domain"]),
        ),
        ("Answerable (false abstention)", len(totals["answerable"]), dict(abstained["answerable"])),
    ]
    return Figures(overview, categories, abstention)


def _open(width: int, height: int, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(desc)}</desc>',
        f"<style>{STYLE.strip()}</style>",
        f'<rect class="surface" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8"/>',
        f'<rect class="frame" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8"/>',
    ]


def _header(parts: list[str], title: str, subtitle: str) -> None:
    parts.append(f'<text class="ink t" x="24" y="32">{escape(title)}</text>')
    parts.append(f'<text class="ink2 s" x="24" y="52">{escape(subtitle)}</text>')
    x = 24
    for config, label in SERIES:
        parts.append(f'<rect class="s{config}" x="{x}" y="66" width="10" height="10" rx="2"/>')
        parts.append(f'<text class="ink2 l" x="{x + 16}" y="75">{escape(label)}</text>')
        x += 16 + 7 * len(label) + 24


def _bar(x: float, y: float, length: float, thickness: float, cls: str) -> str:
    """Horizontal bar: square at the baseline, 4px rounded data end."""

    if length <= 0:
        return ""
    r = min(RADIUS, length, thickness / 2)
    end, low = x + length, y + thickness
    path = (
        f"M{x:.1f},{y:.1f} L{end - r:.1f},{y:.1f} Q{end:.1f},{y:.1f} {end:.1f},{y + r:.1f} "
        f"L{end:.1f},{low - r:.1f} Q{end:.1f},{low:.1f} {end - r:.1f},{low:.1f} L{x:.1f},{low:.1f} Z"
    )
    return f'<path class="{cls}" d="{path}"/>'


def _bars_chart(
    title: str,
    subtitle: str,
    desc: str,
    groups: list[tuple[str, str, dict[str, float], dict[str, str]]],
) -> str:
    """Grouped horizontal bars on a full 0-1 scale. Each group: (label, note, values, value labels)."""

    width = 760
    left, right, top = 220, 70, 100
    row = 2 * BAR + GAP + 16
    bottom = top + row * len(groups) - 10
    height = int(bottom) + 40
    plot = width - left - right
    parts = _open(width, height, title, desc)
    _header(parts, title, subtitle)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + tick * plot
        cls = "base" if tick == 0 else "grid"
        parts.append(f'<line class="{cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{bottom:.1f}"/>')
        tick_y = bottom + 18
        parts.append(
            f'<text class="muted l" x="{x:.1f}" y="{tick_y:.1f}" text-anchor="middle">{_num(tick)}</text>'
        )
    for index, (label, note, values, value_labels) in enumerate(groups):
        y = top + index * row
        anchor = f'x="{left - 12}" text-anchor="end"'
        parts.append(f'<text class="ink2 l" {anchor} y="{y + BAR + 2:.1f}">{escape(label)}</text>')
        parts.append(f'<text class="muted l" {anchor} y="{y + BAR + 17:.1f}">{escape(note)}</text>')
        for offset, (config, _) in enumerate(SERIES):
            by = y + offset * (BAR + GAP)
            length = values[config] * plot
            parts.append(_bar(left, by, length, BAR, f"s{config}"))
            parts.append(
                f'<text class="ink v" x="{left + length + 6:.1f}" y="{by + BAR / 2 + 4:.1f}">'
                f"{escape(value_labels[config])}</text>"
            )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def overview_svg(figures: Figures) -> str:
    groups = [
        (label, f"n = {note}", values, {c: _num(v) for c, v in values.items()})
        for label, note, values in figures.overview
    ]
    return _bars_chart(
        "Protocol v2 holdout: A vs B",
        "Offline embedder, extractive generator, lexical correctness proxy · one deterministic run each",
        "Recall@4, supporting-source precision, strict and partial correctness for configurations A and B "
        "on the 40 answerable holdout questions. "
        + "; ".join(f"{label}: A {_num(v['A'])}, B {_num(v['B'])}" for label, _, v in figures.overview),
        groups,
    )


def categories_svg(figures: Figures) -> str:
    groups = [
        (label, f"n = {n}", values, {c: _num(v) for c, v in values.items()})
        for label, n, values in figures.categories
    ]
    return _bars_chart(
        "Recall@4 by question category, A vs B",
        "Holdout, answerable questions · n is small in every category",
        "Holdout Recall@4 per category: "
        + "; ".join(
            f"{label} (n={n}): A {_num(v['A'])}, B {_num(v['B'])}" for label, n, v in figures.categories
        ),
        groups,
    )


def abstention_svg(figures: Figures) -> str:
    groups = []
    for label, n, counts in figures.abstention:
        values = {c: counts.get(c, 0) / n for c, _ in SERIES}
        groups.append(
            (label, f"n = {n} questions", values, {c: f"{counts.get(c, 0)}/{n}" for c, _ in SERIES})
        )
    return _bars_chart(
        "Share of questions where the system abstained",
        "Ideal: 1 for unanswerable questions, 0 for answerable ones · evidence-gate threshold 0.18",
        "Abstention counts on the holdout: "
        + "; ".join(
            f"{label}: A {c.get('A', 0)}/{n}, B {c.get('B', 0)}/{n}" for label, n, c in figures.abstention
        ),
        groups,
    )


def render() -> dict[str, str]:
    figures = compute()
    return {
        "results-overview.svg": overview_svg(figures),
        "recall-by-category.svg": categories_svg(figures),
        "abstention.svg": abstention_svg(figures),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="verify the committed figures instead of writing"
    )
    args = parser.parse_args(argv)
    stale = []
    for name, content in render().items():
        path = ASSETS / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(name)
        else:
            ASSETS.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(path.relative_to(ROOT))
    if stale:
        print("stale figures: " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
