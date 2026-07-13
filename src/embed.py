"""
Embed chunks locally with nomic-embed-text-v1.5 (sentence-transformers).

Model choice (see spec §3/§5.2): Nomic's 8,192-token window fits
chunk + contextual prefix; BGE-small would truncate at 512. The model
requires task prefixes — "search_document: " at index time here, and
"search_query: " at query time in retrieve.py. Forgetting either
degrades retrieval silently.

Runs fully offline after the first model download; CI needs no API key.

Writes to <out_dir>:
    vectors.npy    float32 (N, 768), L2-normalized (cosine = dot)
    ids.jsonl      chunk_id / doc_id / source per row, same order
    manifest.json  model, dim, with_context flag, input files, text hash

--with-context prepends each chunk's generated context (contextual
retrieval, Anthropic 2024) before embedding. Off = the ablation baseline.
The context field must exist in every record when the flag is on;
a missing one fails loudly rather than silently mixing configurations.

Usage:  python embed.py <out_dir> <chunks.jsonl> [more.jsonl ...] [--with-context]
Deps:   pip install sentence-transformers einops numpy
"""

import argparse
import hashlib
import json
import os
from datetime import date

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DOC_PREFIX = "search_document: "
BATCH = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("chunks", nargs="+")
    ap.add_argument("--with-context", action="store_true",
                    help="prepend generated contexts before embedding")
    a = ap.parse_args()

    chunks = []
    for path in a.chunks:
        chunks += [json.loads(l) for l in open(path, encoding="utf-8")]

    texts = []
    for c in chunks:
        body = c["text"]
        if a.with_context:
            body = c["context"].strip() + "\n\n" + body
        texts.append(DOC_PREFIX + body)

    text_hash = hashlib.sha256(
        "\x00".join(texts).encode("utf-8")).hexdigest()[:16]

    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    vectors = model.encode(texts, batch_size=BATCH, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)

    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "vectors.npy"),
            vectors.astype(np.float32))
    with open(os.path.join(a.out_dir, "ids.jsonl"), "w",
              encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c["chunk_id"],
                                "doc_id": c["doc_id"],
                                "source": c["source"]}) + "\n")
    with open(os.path.join(a.out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "dim": int(vectors.shape[1]),
            "count": int(vectors.shape[0]),
            "with_context": a.with_context,
            "doc_prefix": DOC_PREFIX,
            "inputs": a.chunks,
            "text_hash": text_hash,
            "built": date.today().isoformat(),
        }, f, indent=2)

    print(f"embedded {vectors.shape[0]} chunks -> {a.out_dir} "
          f"(dim {vectors.shape[1]}, with_context={a.with_context})")


if __name__ == "__main__":
    main()
