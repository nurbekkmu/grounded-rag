"""
Generate the README charts as SVGs from the same data the tables use.

The charts are artifacts of the eval harness, not drawings: retrieval
numbers come straight from eval/ablation.json; the book-only contextual
comparison and the latency P50s are inlined below with pointers to
their reports (eval/book_contextual_comparison.md and the
eval/measure_latency.py runs — the raw log lives in gitignored data/).

Each chart is written twice, stepped for a light and a dark surface,
and the README swaps them with a <picture> tag so GitHub's dark mode
gets the dark variant.

Usage:  python eval/make_charts.py
Deps:   none (hand-rolled SVG)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# Ink and series colors are stepped per surface, not auto-inverted.
PALETTES = {
    "light": {
        "surface": "#fcfcfb", "border": "rgba(11,11,11,0.10)",
        "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19", "border": "rgba(255,255,255,0.10)",
        "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#199e70"],
    },
}

WIDTH = 720
BAR = 18        # bar thickness
PAIR_GAP = 3    # between the bars of one group
GROUP_GAP = 18  # between groups


def text(x, y, s, size, fill, weight="normal", anchor="start"):
    return (f'<text x="{x:.0f}" y="{y:.0f}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{s}</text>')


def hbar(x, y, w, h, color, r=4):
    """Horizontal bar: square at the baseline, rounded at the data end."""
    r = min(r, w, h / 2)
    return (f'<path d="M{x:.1f},{y:.1f} h{w - r:.1f} '
            f'a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z" fill="{color}"/>')


def grouped_hbars(title, subtitle, groups, series, xmax, ticks, fmt,
                  pal, label_w):
    """One horizontal grouped-bar chart as an SVG string.

    groups: list of (label, [value per series]); a single series skips
    the legend (the subtitle already names what is plotted).
    """
    n_series = len(series)
    group_h = n_series * BAR + (n_series - 1) * PAIR_GAP
    plot_left = 20 + label_w
    plot_right = WIDTH - 80
    scale = (plot_right - plot_left) / xmax

    parts = []
    parts.append(text(20, 36, title, 15, pal["ink"], weight="600"))
    parts.append(text(20, 56, subtitle, 12, pal["muted"]))

    y = 74
    if n_series > 1:
        x = 20
        for i, name in enumerate(series):
            parts.append(f'<rect x="{x}" y="{y}" width="10" height="10" '
                         f'rx="2" fill="{pal["series"][i]}"/>')
            parts.append(text(x + 15, y + 9, name, 12, pal["secondary"]))
            x += 15 + 8 * len(name) + 24
        y += 26

    plot_top = y
    plot_bottom = plot_top + (len(groups) * group_h
                              + (len(groups) - 1) * GROUP_GAP)

    for tick in ticks:
        tx = plot_left + tick * scale
        color = pal["axis"] if tick == 0 else pal["grid"]
        parts.append(f'<line x1="{tx:.1f}" y1="{plot_top}" x2="{tx:.1f}" '
                     f'y2="{plot_bottom}" stroke="{color}" stroke-width="1"/>')
        parts.append(text(tx, plot_bottom + 18, fmt(tick, True), 11,
                          pal["muted"], anchor="middle"))

    gy = plot_top
    for label, values in groups:
        parts.append(text(plot_left - 10, gy + group_h / 2 + 4, label, 13,
                          pal["secondary"], anchor="end"))
        by = gy
        for i, value in enumerate(values):
            w = value * scale
            parts.append(hbar(plot_left, by, w, BAR, pal["series"][i]))
            parts.append(text(plot_left + w + 6, by + BAR / 2 + 4,
                              fmt(value, False), 12, pal["secondary"]))
            by += BAR + PAIR_GAP
        gy += group_h + GROUP_GAP

    height = plot_bottom + 32
    body = "\n".join(parts)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" '
            f'aria-label="{title}">\n'
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" '
            f'height="{height - 1}" rx="8" fill="{pal["surface"]}" '
            f'stroke="{pal["border"]}"/>\n{body}\n</svg>\n')


def metric(rows, index, mode, rerank, name):
    for row in rows:
        if (row["index"] == index and row["mode"] == mode
                and row["rerank"] == rerank):
            return row["metrics"][name]
    raise KeyError((index, mode, rerank, name))


def main():
    rows = json.loads((ROOT / "eval" / "ablation.json").read_text())

    def mrr_fmt(v, is_tick):
        return f"{v:.1f}" if is_tick else f"{v:.3f}"

    def pct_fmt(v, is_tick):
        return f"{v:.0f}%"

    def sec_fmt(v, is_tick):
        return f"{v:.0f} s" if is_tick else f"{v:.1f} s"

    charts = {}

    charts["retrieval-ablation"] = dict(
        title="Retrieval quality by configuration",
        subtitle="MRR on the hardened golden set (n=15), baseline index"
                 " — higher is better",
        groups=[(mode, [metric(rows, "baseline", mode, False, "mrr"),
                        metric(rows, "baseline", mode, True, "mrr")])
                for mode in ("bm25", "vector", "hybrid")],
        series=["no rerank", "+ cross-encoder rerank"],
        xmax=0.8, ticks=[0, 0.2, 0.4, 0.6, 0.8], fmt=mrr_fmt, label_w=70)

    # Book-only controlled pair: eval/book_contextual_comparison.md.
    charts["contextual-retrieval"] = dict(
        title="Contextual retrieval, measured on the vector arm",
        subtitle="failure-rate@20 — lower is better",
        groups=[
            ("book-only controlled (n=12)", [17, 8]),
            ("full corpus (n=15)",
             [metric(rows, "baseline", "vector", False,
                     "failure_rate@20") * 100,
              metric(rows, "contextual", "vector", False,
                     "failure_rate@20") * 100]),
        ],
        series=["baseline index", "contextual index"],
        xmax=25, ticks=[0, 5, 10, 15, 20, 25], fmt=pct_fmt, label_w=190)

    # P50s from the eval/measure_latency.py flight-recorder runs
    # (summarized in the README's latency table; raw log is gitignored).
    charts["latency-budget"] = dict(
        title="Where the ~20 seconds go",
        subtitle="P50 per stage, CPU laptop, warm caches — the P95"
                 " verification tail reaches 90 s",
        groups=[("hybrid retrieval", [1.4]),
                ("cross-encoder rerank", [6.2]),
                ("generation (Gemini Flash)", [4.3]),
                ("NLI verification", [7.8])],
        series=["seconds"],
        xmax=8, ticks=[0, 2, 4, 6, 8], fmt=sec_fmt, label_w=170)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in charts.items():
        for mode, pal in PALETTES.items():
            svg = grouped_hbars(pal=pal, **spec)
            path = OUT / f"{name}-{mode}.svg"
            path.write_text(svg, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
