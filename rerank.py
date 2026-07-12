"""
Cross-encoder reranking of retrieved candidates (spec §5.6).

The retriever's bi-encoder embeds query and document separately — fast
enough to search the whole corpus, but it never sees them together. The
cross-encoder scores (query, chunk) as one input, far more accurate and
far too slow for the full corpus — so it runs only on the short list:
wide cheap net first (hybrid retrieval, ~75 candidates), expensive
scoring second, keep the top 5-10 for generation.

Default model is ms-marco-MiniLM-L-6-v2: 22M params, seconds per query
on CPU. Known trade-off: its 512-token window truncates the tail of
long (query + chunk) pairs; relevance signal concentrates early, so
this is standard practice — but it is a documented limitation, and
--model swaps in a larger reranker (e.g. BAAI/bge-reranker-base, same
window but stronger; bge-reranker-v2-m3 for an 8k window at ~30x the
latency) for the ablation table.

Scoring uses the raw chunk text — the generated context prefix is
retrieval fuel for the indexes, and the reranker judges the evidence
the generator will actually see.

Usage:  python rerank.py "your question"
            [--mode hybrid] [--top-n 75] [--keep 8]
            [--model cross-encoder/ms-marco-MiniLM-L-6-v2]
Deps:   pip install sentence-transformers
"""

import argparse
import sys

from sentence_transformers import CrossEncoder

from retrieve import CHUNKS_DEFAULT, retrieve

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_DEFAULT = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(query: str, candidates: list, model_name: str = MODEL_DEFAULT,
           keep: int = 8) -> list:
    """Rescore retrieve() candidates; returns top `keep` by cross-encoder
    score, each with rerank_score and rerank_rank added (fused rank kept
    in ranks/score for tracing)."""
    model = CrossEncoder(model_name)
    pairs = [(query, c["chunk"]["text"]) for c in candidates]
    scores = model.predict(pairs, batch_size=16)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: -c["rerank_score"])[:keep]
    for i, c in enumerate(ranked, 1):
        c["rerank_rank"] = i
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"],
                    default="hybrid")
    ap.add_argument("--top-n", type=int, default=75)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--index", default="data/index/baseline")
    ap.add_argument("--chunks", nargs="+", default=CHUNKS_DEFAULT)
    a = ap.parse_args()

    candidates = retrieve(a.query, a.index, a.chunks, a.mode,
                          a.top_n, k=a.top_n)   # rerank sees the full net
    results = rerank(a.query, candidates, a.model, a.keep)
    for r in results:
        ch = r["chunk"]
        where = (f"pp.{ch['page_start']}-{ch['page_end']}"
                 if ch["source"] == "book" else ch["url"])
        fused = ", ".join(f"{arm}#{rk}" for arm, rk in r["ranks"].items())
        print(f"#{r['rerank_rank']}  ce={r['rerank_score']:6.2f}  "
              f"(was {fused})  {r['chunk_id']}")
        print(f"     {ch['section_path'][:70]}  ({where})")
        print(f"     {ch['text'][:140].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    main()
