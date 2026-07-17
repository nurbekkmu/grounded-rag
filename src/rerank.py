"""
Cross-encoder reranking of retrieved candidates (spec §5.6).

The retriever's bi-encoder embeds query and document separately — fast
enough to search the whole corpus, but it never sees them together. The
cross-encoder reads (query, chunk) as one input, which is far more
accurate and far too slow for the full corpus — so it runs only on the
short list: wide cheap net first (hybrid retrieval, ~75 candidates),
expensive scoring second, keep the top 5-10 for generation.

Default model is ms-marco-MiniLM-L-6-v2: 22M params, seconds per query
on CPU. Known trade-off: its 512-token window truncates the tail of
long (query + chunk) pairs; relevance signal concentrates early, so
this is standard practice — but it is a documented limitation, and
--model swaps in a larger reranker for the ablation table (measured:
BAAI/bge-reranker-base matched MiniLM at 12x the latency).

Scoring uses the raw chunk text — the generated context prefix is
retrieval fuel for the indexes, and the reranker judges the evidence
the generator will actually see.

Usage:  python src/rerank.py "your question"
            [--mode hybrid] [--top-n 75] [--keep 8]
            [--model cross-encoder/ms-marco-MiniLM-L-6-v2]
Deps:   pip install sentence-transformers
"""

import argparse
import sys

from sentence_transformers import CrossEncoder

from config import cfg as _cfg
from retrieve import add_retrieval_args, retrieve, source_of

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_DEFAULT = _cfg("rerank.model")

# Loaded models are kept here so each one loads only once per process.
_MODELS = {}


def _model(name):
    if name not in _MODELS:
        _MODELS[name] = CrossEncoder(name)
    return _MODELS[name]


def rerank(query, candidates, model_name=MODEL_DEFAULT, keep=8):
    """Rescore retrieve() candidates with a cross-encoder.

    Returns the top `keep` candidates by cross-encoder score. Each one
    gains rerank_score and rerank_rank; the fused ranks from retrieval
    stay attached for tracing.
    """
    model = _model(model_name)

    pairs = []
    for candidate in candidates:
        pairs.append((query, candidate["chunk"]["text"]))
    scores = model.predict(pairs, batch_size=16)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates,
                    key=lambda c: c["rerank_score"], reverse=True)
    ranked = ranked[:keep]
    for position, candidate in enumerate(ranked, start=1):
        candidate["rerank_rank"] = position
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    add_retrieval_args(ap, keep=True)
    a = ap.parse_args()

    # The reranker needs the whole net, so retrieval keeps all top-n.
    candidates = retrieve(a.query, a.index, a.chunks, a.mode,
                          a.top_n, k=a.top_n)
    results = rerank(a.query, candidates, a.model, a.keep)

    for r in results:
        chunk = r["chunk"]
        fused = ", ".join(f"{arm}#{rank}"
                          for arm, rank in r["ranks"].items())
        snippet = chunk["text"][:140].replace("\n", " ")
        print(f"#{r['rerank_rank']}  ce={r['rerank_score']:6.2f}  "
              f"(was {fused})  {r['chunk_id']}")
        print(f"     {chunk['section_path'][:70]}  ({source_of(chunk)})")
        print(f"     {snippet}")
        print()


if __name__ == "__main__":
    main()
