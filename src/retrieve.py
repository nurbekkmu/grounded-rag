"""
Hybrid retrieval: vector search (Weaviate) + BM25, fused with RRF.

Why both arms (spec §5.5): embeddings understand meaning but blur exact
identifiers; BM25 nails exact terms but misses paraphrase. Reciprocal
Rank Fusion combines the two ranked lists by rank position only —
score = sum over arms of 1 / (RRF_K + rank) — which sidesteps
normalizing BM25's unbounded scores against cosine similarities.

Query embedding uses Nomic's "search_query: " prefix (documents were
indexed with "search_document: "); mixing these up degrades retrieval
silently, so both live here and in embed.py as constants.

--mode vector|bm25|hybrid is the retrieval ablation toggle. Per-arm
ranks are kept on every candidate for tracing and later evaluation.

Usage:  python retrieve.py "your question"
            [--mode hybrid] [--top-n 75] [--k 10]
            [--index data/index/baseline]
            [--chunks data/processed/book_chunks.jsonl
                      data/processed/blog_chunks.jsonl]
Deps:   pip install rank-bm25 weaviate-client sentence-transformers numpy
"""

import argparse
import json
import pickle
import re
import sys
from os.path import join

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console default

from index import collection_for

RRF_K = 60
QUERY_PREFIX = "search_query: "

CHUNKS_DEFAULT = ["data/processed/book_chunks.jsonl",
                  "data/processed/blog_chunks.jsonl"]


def load_chunks(paths) -> dict:
    store = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            c = json.loads(line)
            store[c["chunk_id"]] = c
    return store


def bm25_search(index_dir: str, query: str, top_n: int) -> list:
    """Ranked chunk_ids from the lexical arm."""
    p = pickle.load(open(join(index_dir, "bm25.pkl"), "rb"))
    terms = re.compile(p["term_regex"]).findall(query.lower())
    scores = p["bm25"].get_scores(terms)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_n]
    return [p["chunk_ids"][i] for i in order if scores[i] > 0]


_EMBEDDER = []


def vector_search(query: str, top_n: int, collection: str) -> list:
    """Ranked chunk_ids from the semantic arm (Weaviate near-vector)."""
    import weaviate
    from sentence_transformers import SentenceTransformer

    if not _EMBEDDER:
        _EMBEDDER.append(SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",
                                             trust_remote_code=True))
    qvec = _EMBEDDER[0].encode([QUERY_PREFIX + query],
                               normalize_embeddings=True)[0]
    client = weaviate.connect_to_local()
    try:
        col = client.collections.get(collection)
        res = col.query.near_vector(near_vector=qvec.tolist(), limit=top_n)
        return [o.properties["chunk_id"] for o in res.objects]
    finally:
        client.close()


def rrf_fuse(rankings: dict) -> list:
    """rankings: arm name -> ranked chunk_id list. Returns fused candidates
    [{chunk_id, score, ranks: {arm: rank}}] sorted by fused score."""
    fused = {}
    for arm, ranked in rankings.items():
        for rank, cid in enumerate(ranked, start=1):
            entry = fused.setdefault(cid, {"chunk_id": cid, "score": 0.0,
                                           "ranks": {}})
            entry["score"] += 1.0 / (RRF_K + rank)
            entry["ranks"][arm] = rank
    return sorted(fused.values(), key=lambda e: -e["score"])


def retrieve(query: str, index_dir: str, chunk_paths=None,
             mode: str = "hybrid", top_n: int = 75, k: int = 10) -> list:
    rankings = {}
    if mode in ("bm25", "hybrid"):
        rankings["bm25"] = bm25_search(index_dir, query, top_n)
    if mode in ("vector", "hybrid"):
        rankings["vector"] = vector_search(query, top_n,
                                           collection_for(index_dir))
    candidates = rrf_fuse(rankings)[:k]

    store = load_chunks(chunk_paths or CHUNKS_DEFAULT)
    for c in candidates:
        c["chunk"] = store[c["chunk_id"]]
    return candidates


def main():
    from config import cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"],
                    default=cfg("retrieval.mode"))
    ap.add_argument("--top-n", type=int, default=cfg("retrieval.top_n"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--index", default=cfg("retrieval.index"))
    ap.add_argument("--chunks", nargs="+", default=CHUNKS_DEFAULT)
    a = ap.parse_args()

    results = retrieve(a.query, a.index, a.chunks, a.mode, a.top_n, a.k)
    for r in results:
        ch = r["chunk"]
        where = (f"pp.{ch['page_start']}-{ch['page_end']}"
                 if ch["source"] == "book" else ch["url"])
        ranks = ", ".join(f"{arm}#{rk}" for arm, rk in r["ranks"].items())
        print(f"{r['score']:.4f}  {r['chunk_id']:32s} [{ranks}]")
        print(f"        {ch['section_path'][:70]}  ({where})")
        print(f"        {ch['text'][:140].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    main()
