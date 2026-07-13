"""
Ablation harness (spec §7.3): run the retrieval eval across the config
matrix and generate the README ablation table.

The harness drives the same evaluate() the CI gate uses — the ablation
study IS the production path under different configs, never forked
scripts. Configs whose infrastructure isn't ready (Weaviate down,
contextual index not yet built) are skipped and reported as pending, so
the table grows automatically as the build progresses.

Matrix: index (baseline / contextual if built) x mode (bm25 / vector /
hybrid) x rerank (off / on).

Usage:  python eval/ablate.py [--out eval/ablation.md] [--top-n 75]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from run_eval import evaluate  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

GOLDEN = "eval/golden_set.jsonl"
INDEXES = [("baseline", "data/index/baseline"),
           ("contextual", "data/index/contextual")]
MODES = ["bm25", "vector", "hybrid"]


def run_matrix(top_n: int) -> list:
    results = []
    for index_name, index_dir in INDEXES:
        if not os.path.exists(os.path.join(index_dir, "manifest.json")):
            results.append({"index": index_name, "status": "pending",
                            "reason": "index not built"})
            continue
        for mode in MODES:
            for use_rerank in (False, True):
                label = {"index": index_name, "mode": mode,
                         "rerank": use_rerank}
                try:
                    r = evaluate(GOLDEN, mode, use_rerank,
                                 top_n=top_n, index_dir=index_dir)
                    results.append({**label, "status": "ok",
                                    "metrics": r["overall"],
                                    "unverified": r["unverified"],
                                    "total_golden": r["total_golden"]})
                    m = r["overall"]
                    print(f"  ok      {index_name:10s} {mode:6s} "
                          f"rerank={use_rerank!s:5s}  "
                          f"failure@20={m['failure_rate@20']:.0%}  "
                          f"recall@5={m['recall@5']:.0%}  "
                          f"mrr={m['mrr']:.3f}")
                except Exception as e:
                    results.append({**label, "status": "pending",
                                    "reason": f"{type(e).__name__}: "
                                              f"{str(e)[:80]}"})
                    print(f"  pending {index_name:10s} {mode:6s} "
                          f"rerank={use_rerank!s:5s}  ({type(e).__name__})")
    return results


def to_markdown(results: list) -> str:
    ok = [r for r in results if r["status"] == "ok"]
    unverified = ok[0]["unverified"] if ok else "?"
    total = ok[0]["total_golden"] if ok else "?"
    lines = [
        "# Retrieval ablation",
        "",
        f"Golden set: {total} questions ({unverified} still unverified — "
        "numbers provisional until manual verification is complete).",
        "Metrics follow Anthropic's contextual-retrieval definitions; "
        "failure-rate@20 is directly comparable to their published "
        "baseline 5.7% → 1.9% progression.",
        "",
        "| index | mode | rerank | failure@20 | recall@5 | recall@20 | MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["status"] == "ok":
            m = r["metrics"]
            lines.append(
                f"| {r['index']} | {r['mode']} | "
                f"{'yes' if r['rerank'] else 'no'} | "
                f"{m['failure_rate@20']:.0%} | {m['recall@5']:.0%} | "
                f"{m['recall@20']:.0%} | {m['mrr']:.3f} |")
        elif "mode" in r:
            lines.append(f"| {r['index']} | {r['mode']} | "
                         f"{'yes' if r['rerank'] else 'no'} | "
                         f"_pending_ | | | |")
        else:
            lines.append(f"| {r['index']} | — | — | _pending: "
                         f"{r['reason']}_ | | | |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/ablation.md")
    ap.add_argument("--top-n", type=int, default=75)
    a = ap.parse_args()

    results = run_matrix(a.top_n)
    md = to_markdown(results)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    with open(a.out.replace(".md", ".json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
