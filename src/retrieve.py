"""
Hybrid retrieval: vector search (Weaviate) + BM25, fused with RRF.

Why both arms (spec §5.5): embeddings understand meaning but blur exact
identifiers; BM25 nails exact terms but misses paraphrase. Reciprocal
Rank Fusion combines the two ranked lists using only rank positions —
score = sum over arms of 1 / (RRF_K + rank) — which sidesteps
normalizing BM25's unbounded scores against cosine similarities.

Query embedding uses Nomic's "search_query: " prefix (documents were
indexed with "search_document: "); mixing these up degrades retrieval
silently, so both live here and in embed.py as constants.

--mode vector|bm25|hybrid is the retrieval ablation toggle. Per-arm
ranks are kept on every candidate for tracing and later evaluation.

Usage:  python src/retrieve.py "your question"
            [--mode hybrid] [--top-n 75] [--k 10]
            [--index data/index/baseline]
Deps:   pip install rank-bm25 weaviate-client sentence-transformers numpy
"""

import argparse
import json
import pickle
import re
import sys
from os.path import join

from index import collection_for

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

RRF_K = 60
QUERY_PREFIX = "search_query: "

CHUNKS_DEFAULT = ["data/processed/book_chunks.jsonl",
                  "data/processed/blog_chunks.jsonl"]


def load_chunks(paths):
    """Read chunk files and return a dict: chunk_id -> full chunk record."""
    store = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                store[chunk["chunk_id"]] = chunk
    return store


def bm25_search(index_dir, query, top_n):
    """The lexical arm: return chunk_ids ranked by BM25 keyword score."""
    with open(join(index_dir, "bm25.pkl"), "rb") as f:
        index = pickle.load(f)

    term_pattern = re.compile(index["term_regex"])
    query_terms = term_pattern.findall(query.lower())
    scores = index["bm25"].get_scores(query_terms)

    # Sort chunk positions from highest score to lowest.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    ranked_ids = []
    for position in order[:top_n]:
        if scores[position] > 0:            # a zero score means "no match"
            ranked_ids.append(index["chunk_ids"][position])
    return ranked_ids


# The embedding model is loaded once and reused across queries.
_EMBEDDER = []


def vector_search(query, top_n, collection):
    """The semantic arm: return chunk_ids ranked by embedding similarity."""
    import weaviate
    from sentence_transformers import SentenceTransformer

    if not _EMBEDDER:
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",
                                    trust_remote_code=True)
        _EMBEDDER.append(model)

    query_vector = _EMBEDDER[0].encode([QUERY_PREFIX + query],
                                       normalize_embeddings=True)[0]

    client = weaviate.connect_to_local()
    try:
        col = client.collections.get(collection)
        result = col.query.near_vector(near_vector=query_vector.tolist(),
                                       limit=top_n)
        ranked_ids = []
        for obj in result.objects:
            ranked_ids.append(obj.properties["chunk_id"])
        return ranked_ids
    finally:
        client.close()


def rrf_fuse(rankings):
    """Merge ranked lists from several arms into one list.

    rankings: dict of arm name -> ranked chunk_id list.
    Each appearance earns 1 / (RRF_K + rank); scores add up across arms,
    so a chunk ranked well by BOTH arms rises to the top.
    Returns candidates sorted by fused score, each carrying its per-arm
    ranks for tracing: {"chunk_id", "score", "ranks": {arm: rank}}.
    """
    fused = {}
    for arm_name, ranked_ids in rankings.items():
        for position, chunk_id in enumerate(ranked_ids, start=1):
            if chunk_id not in fused:
                fused[chunk_id] = {"chunk_id": chunk_id,
                                   "score": 0.0,
                                   "ranks": {}}
            fused[chunk_id]["score"] += 1.0 / (RRF_K + position)
            fused[chunk_id]["ranks"][arm_name] = position

    candidates = list(fused.values())
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def retrieve(query, index_dir, chunk_paths=None,
             mode="hybrid", top_n=75, k=10):
    """Run the chosen retrieval arms, fuse, and attach full chunk records."""
    rankings = {}
    if mode in ("bm25", "hybrid"):
        rankings["bm25"] = bm25_search(index_dir, query, top_n)
    if mode in ("vector", "hybrid"):
        collection = collection_for(index_dir)
        rankings["vector"] = vector_search(query, top_n, collection)

    candidates = rrf_fuse(rankings)[:k]

    store = load_chunks(chunk_paths or CHUNKS_DEFAULT)
    for candidate in candidates:
        candidate["chunk"] = store[candidate["chunk_id"]]
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
        chunk = r["chunk"]
        if chunk["source"] == "book":
            where = f"pp.{chunk['page_start']}-{chunk['page_end']}"
        else:
            where = chunk["url"]
        ranks = ", ".join(f"{arm}#{rank}"
                          for arm, rank in r["ranks"].items())
        snippet = chunk["text"][:140].replace("\n", " ")
        print(f"{r['score']:.4f}  {r['chunk_id']:32s} [{ranks}]")
        print(f"        {chunk['section_path'][:70]}  ({where})")
        print(f"        {snippet}")
        print()


if __name__ == "__main__":
    main()
