"""
Contextual retrieval preprocessing (Anthropic, 2024) for the chunked corpus.

For every chunk, ask Gemini for a short (50-100 token) context situating the
chunk within its document (= book chapter / blog post). The context is stored
next to the chunk and later prepended to it for BOTH indexes (embeddings and
BM25). Citations keep pointing at the original chunk text only.

Prompt lives in prompts/contextualize.yaml (versioned config). Document-first
ordering in the template keeps the big document prefix identical across all
per-chunk calls so prompt caching can reuse it.

Every generated context is cached in data/cache/contexts.jsonl keyed on
sha256(doc_text, chunk_text, prompt_version, model) - re-runs only pay for
new or changed chunks; chunking changes invalidate the key automatically.

Usage:  python contextualize.py <chunks.jsonl> <out.jsonl> [limit]
Deps:   pip install google-genai python-dotenv pyyaml
Env:    GEMINI_API_KEYS=key1,key2,...   (rotates on quota errors)
"""

import hashlib
import json
import os
import sys
import time

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

PROMPT_FILE = "prompts/contextualize.yaml"
CACHE_FILE = "data/cache/contexts.jsonl"
MAX_CYCLES = 20            # full key-cycles before giving up on a chunk
BACKOFF_S = 65             # sleep after a full key-cycle (Gemini RPM window)
CACHE_TTL = "1800s"        # explicit content-cache lifetime per document


def split_template(template: str):
    """Split the prompt at </document>: the document half is uploaded once
    per (key, doc) as an explicit Gemini content cache; the chunk half is the
    only part sent per call. Concatenated they equal the original prompt."""
    head, tail = template.split("</document>", 1)
    return head + "</document>", tail.lstrip("\n")


def load_prompt(path=PROMPT_FILE):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cache_key(doc_text: str, chunk_text: str, version: int, model: str) -> str:
    h = hashlib.sha256()
    for part in (doc_text, chunk_text, str(version), model):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_cache(path=CACHE_FILE) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        rows = (json.loads(line) for line in f if line.strip())
        return {r["key"]: r["context"] for r in rows}


def append_cache(key: str, context: str, path=CACHE_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "context": context},
                           ensure_ascii=False) + "\n")


class KeyRotator:
    """Round-robin over API keys; advance on quota/rate errors."""

    def __init__(self, keys):
        self.clients = [genai.Client(api_key=k) for k in keys]
        self.i = 0

    def client(self):
        return self.clients[self.i % len(self.clients)]

    def rotate(self):
        self.i += 1
        return self.i % len(self.clients) == 0   # True = full cycle done


def gen_context(rotator: KeyRotator, cfg: dict, doc_id: str, doc_text: str,
                chunk_text: str, cache_map: dict):
    """Returns (text, prompt_tokens, cached_tokens).

    cache_map[(key_index, doc_id)] -> Gemini cache name, or None if this
    key/doc pair can't cache (too small, unsupported) -> full-prompt fallback.
    Caches belong to the key's project, so each key needs its own.
    """
    head, tail = split_template(cfg["template"])
    cache_text = head.replace("{{WHOLE_DOCUMENT}}", doc_text)
    request_text = tail.replace("{{CHUNK_CONTENT}}", chunk_text)
    n_keys = len(rotator.clients)
    last_err = ""
    for attempt in range(MAX_CYCLES * n_keys):
        ki = rotator.i % n_keys
        client = rotator.client()
        ck = (ki, doc_id)
        if ck not in cache_map:
            try:
                cc = client.caches.create(
                    model=cfg["model"],
                    config={"contents": [cache_text], "ttl": CACHE_TTL},
                )
                cache_map[ck] = cc.name
            except genai_errors.APIError as e:
                cache_map[ck] = None         # can't cache -> full prompt
                if not any(v is None for k, v in cache_map.items()
                           if k != ck):      # first refusal: say why once
                    print(f"    [caches.create refused ({e.code}): "
                          f"{str(e)[:160]}]", flush=True)
        gen_cfg = {
            "temperature": cfg["temperature"],
            "max_output_tokens": cfg["max_output_tokens"],
            # 2.5 Flash thinks by default; the reasoning budget would eat
            # max_output_tokens before the answer emits. Contextualization
            # is summarization, not reasoning.
            "thinking_config": {"thinking_budget": 0},
        }
        cache_name = cache_map.get(ck)
        if cache_name:
            gen_cfg["cached_content"] = cache_name
            contents = request_text
        else:
            contents = cache_text + "\n" + request_text
        try:
            resp = client.models.generate_content(
                model=cfg["model"], contents=contents, config=gen_cfg)
            text = (resp.text or "").strip()
            if text:
                m = resp.usage_metadata
                return (text,
                        getattr(m, "prompt_token_count", 0) or 0,
                        getattr(m, "cached_content_token_count", 0) or 0)
        except genai_errors.APIError as e:
            if cache_name and e.code in (400, 403, 404):
                cache_map.pop(ck, None)      # cache expired/invalid: rebuild
                continue
            if e.code not in (429, 500, 503):
                raise
            last_err = str(e)[:800]
        if rotator.rotate():
            print(f"    [quota cycle {attempt // n_keys + 1}, "
                  f"sleeping {BACKOFF_S}s]", flush=True)
            time.sleep(BACKOFF_S)
    raise RuntimeError(f"all attempts exhausted; last error: {last_err}")


def main(chunks_path: str, out_path: str, limit: int = 0):
    load_dotenv()
    keys = [k.strip() for k in os.environ["GEMINI_API_KEYS"].split(",")
            if k.strip()]
    rotator = KeyRotator(keys)
    cfg = load_prompt()

    chunks = [json.loads(line) for line in open(chunks_path, encoding="utf-8")]

    # document text = ALL its chunks joined (before any limit, so the model
    # always sees the complete document; paragraph overlap is harmless here)
    docs = {}
    for c in chunks:
        docs.setdefault(c["doc_id"], []).append(c["text"])
    docs = {d: "\n\n".join(texts) for d, texts in docs.items()}

    if limit:
        chunks = chunks[:limit]

    cache = load_cache()
    cache_map = {}             # (key_index, doc_id) -> Gemini cache name
    hits = calls = 0
    prompt_toks = cached_toks = 0
    for i, c in enumerate(chunks, 1):
        key = cache_key(docs[c["doc_id"]], c["text"],
                        cfg["version"], cfg["model"])
        if key in cache:
            c["context"] = cache[key]
            hits += 1
        else:
            c["context"], p, k = gen_context(rotator, cfg, c["doc_id"],
                                             docs[c["doc_id"]], c["text"],
                                             cache_map)
            prompt_toks += p
            cached_toks += k
            append_cache(key, c["context"])
            cache[key] = c["context"]
            calls += 1
        if i % 25 == 0 or i == len(chunks):
            hit_rate = f"{100 * cached_toks / prompt_toks:.0f}%" \
                if prompt_toks else "n/a"
            print(f"  {i}/{len(chunks)}  disk cache hits {hits}, "
                  f"new calls {calls}  |  gemini prompt-cache: "
                  f"{cached_toks}/{prompt_toks} tokens ({hit_rate})")

    # best-effort cleanup of explicit caches (TTL would expire them anyway)
    for (ki, _), name in cache_map.items():
        if name:
            try:
                rotator.clients[ki].caches.delete(name=name)
            except Exception:
                pass
    made = sum(1 for v in cache_map.values() if v)
    if cache_map:
        print(f"explicit caches: {made} created, "
              f"{sum(1 for v in cache_map.values() if v is None)} fell back")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    ctx_toks = [len(c["context"].split()) for c in chunks]
    print(f"contexts: {len(chunks)}  |  words min/avg/max: "
          f"{min(ctx_toks)}/{sum(ctx_toks)//len(ctx_toks)}/{max(ctx_toks)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 0)
