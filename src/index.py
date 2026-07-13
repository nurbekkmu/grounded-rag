"""
Build the dual retrieval index: BM25 (local pickle) + Weaviate (vectors).

BM25 terms are words, not subwords: lowercase, split on punctuation,
but internal -._ in technical identifiers survive, so exact-term queries
like "text-embedding-3-small" or "gpt-4" keep working — that exact-match
power is BM25's whole job in this pipeline. No stemming (corpus-scale
doesn't need it and identifiers must stay intact).

Both indexes are built from the same composed text as embed.py
(--with-context prepends the generated context), so the two retrieval
arms always see the same corpus representation.

The Weaviate half pushes embed.py's vectors plus chunk metadata into a
"Chunk" collection (vectorizer none — vectors are ours). It needs the
docker-compose Weaviate running; --bm25-only skips it.

Usage:  python index.py <index_dir> <chunks.jsonl> [more.jsonl ...]
            [--with-context] [--bm25-only]
        (index_dir must already hold embed.py's output for the same
         chunks + flag; the manifest text_hash is cross-checked)
Deps:   pip install rank-bm25 weaviate-client numpy
"""

import argparse
import hashlib
import json
import os
import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi

TERM_RE = re.compile(r"[a-z0-9]+(?:[-._][a-z0-9]+)*")


def collection_for(index_dir: str) -> str:
    """One Weaviate collection per index dir (baseline / contextual /
    book-only variants coexist for ablations)."""
    base = os.path.basename(os.path.normpath(index_dir))
    return "Chunk_" + re.sub(r"[^A-Za-z0-9]", "_", base)


def bm25_terms(text: str) -> list:
    return TERM_RE.findall(text.lower())


def composed_texts(chunks: list, with_context: bool) -> list:
    out = []
    for c in chunks:
        body = c["text"]
        if with_context:
            body = c["context"].strip() + "\n\n" + body
        out.append(body)
    return out


def build_bm25(index_dir: str, chunks: list, texts: list):
    corpus = [bm25_terms(t) for t in texts]
    bm25 = BM25Okapi(corpus)
    payload = {
        "bm25": bm25,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "term_regex": TERM_RE.pattern,
    }
    path = os.path.join(index_dir, "bm25.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"bm25: {len(corpus)} chunks, "
          f"avg {sum(map(len, corpus)) // len(corpus)} terms -> {path}")


def push_weaviate(chunks: list, vectors: np.ndarray, collection: str):
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property

    client = weaviate.connect_to_local()
    try:
        if client.collections.exists(collection):
            client.collections.delete(collection)
        col = client.collections.create(
            name=collection,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="section_path", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="page_start", data_type=DataType.INT),
                Property(name="page_end", data_type=DataType.INT),
                Property(name="url", data_type=DataType.TEXT),
            ],
        )
        with col.batch.dynamic() as batch:
            for c, v in zip(chunks, vectors):
                batch.add_object(
                    properties={
                        "chunk_id": c["chunk_id"],
                        "doc_id": c["doc_id"],
                        "source": c["source"],
                        "section_path": c.get("section_path", ""),
                        "text": c["text"],
                        "page_start": c.get("page_start", 0),
                        "page_end": c.get("page_end", 0),
                        "url": c.get("url", ""),
                    },
                    vector=v.tolist(),
                )
        print(f"weaviate: {len(chunks)} objects -> collection {collection}")
    finally:
        client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index_dir")
    ap.add_argument("chunks", nargs="+")
    ap.add_argument("--with-context", action="store_true")
    ap.add_argument("--bm25-only", action="store_true")
    a = ap.parse_args()

    chunks = []
    for path in a.chunks:
        chunks += [json.loads(l) for l in open(path, encoding="utf-8")]
    texts = composed_texts(chunks, a.with_context)

    # the index must describe the same corpus state as the embeddings
    manifest = json.load(open(os.path.join(a.index_dir, "manifest.json"),
                              encoding="utf-8"))
    text_hash = hashlib.sha256(("\x00".join(
        manifest["doc_prefix"] + t for t in texts)).encode()).hexdigest()[:16]
    if text_hash != manifest["text_hash"]:
        raise SystemExit("text hash mismatch: chunks or --with-context flag "
                         "differ from what embed.py embedded — re-run embed.py")

    build_bm25(a.index_dir, chunks, texts)
    if not a.bm25_only:
        vectors = np.load(os.path.join(a.index_dir, "vectors.npy"))
        push_weaviate(chunks, vectors, collection_for(a.index_dir))


if __name__ == "__main__":
    main()
