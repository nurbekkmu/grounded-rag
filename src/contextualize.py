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

Usage:  python src/contextualize.py <chunks.jsonl> <out.jsonl>
            [--limit N] [--prompt prompts/contextualize.yaml]
Deps:   pip install google-genai python-dotenv pyyaml
Env:    GEMINI_API_KEYS=key1,key2,...   (rotates on quota errors)
"""

import argparse
import hashlib
import json
import os
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


def split_template(template):
    """Split the prompt at </document>.

    The document half is identical for every chunk of one document, so
    it can be uploaded once as an explicit Gemini content cache; the
    chunk half is the only part that changes per call. Concatenated,
    the two halves equal the original prompt.
    """
    head, tail = template.split("</document>", 1)
    return head + "</document>", tail.lstrip("\n")


def load_prompt(path=PROMPT_FILE):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cache_key(doc_text, chunk_text, version, model):
    """Fingerprint of everything that determines a context's content.

    If the document, the chunk, the prompt version, or the model
    changes, the key changes — so stale cache entries can never be
    mistaken for current ones.
    """
    hasher = hashlib.sha256()
    for part in (doc_text, chunk_text, str(version), model):
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def load_cache(path=CACHE_FILE):
    """Read the on-disk context cache into a dict: key -> context."""
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = row["context"]
    return cache


def append_cache(key, context, path=CACHE_FILE):
    """Persist one generated context immediately (crash-safe)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        row = {"key": key, "context": context}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class KeyRotator:
    """Round-robin over API keys; advance on quota/rate errors."""

    def __init__(self, keys):
        self.clients = [genai.Client(api_key=key) for key in keys]
        self.i = 0

    def client(self):
        return self.clients[self.i % len(self.clients)]

    def rotate(self):
        """Move to the next key. Returns True when a full cycle of all
        keys has been tried (the caller then sleeps out the window)."""
        self.i += 1
        return self.i % len(self.clients) == 0


def gen_context(rotator, prompt_cfg, doc_id, doc_text, chunk_text,
                cache_map):
    """Generate one chunk's context. Returns (text, prompt_tokens,
    cached_tokens).

    cache_map[(key_index, doc_id)] holds a Gemini explicit-cache name,
    or None when that key/doc pair can't cache (free tier refuses;
    document too small) — then the full prompt is sent instead.
    Explicit caches belong to the key's project, so each key needs its
    own.
    """
    head, tail = split_template(prompt_cfg["template"])
    cache_text = head.replace("{{WHOLE_DOCUMENT}}", doc_text)
    request_text = tail.replace("{{CHUNK_CONTENT}}", chunk_text)

    n_keys = len(rotator.clients)
    last_error = ""

    for attempt in range(MAX_CYCLES * n_keys):
        key_index = rotator.i % n_keys
        client = rotator.client()

        # Try to create an explicit content cache for this key + doc,
        # once. A quota error (429) is not stored, so it is retried
        # later; any other refusal means "fall back to full prompts".
        cache_slot = (key_index, doc_id)
        if cache_slot not in cache_map:
            try:
                created = client.caches.create(
                    model=prompt_cfg["model"],
                    config={"contents": [cache_text], "ttl": CACHE_TTL},
                )
                cache_map[cache_slot] = created.name
            except genai_errors.APIError as error:
                cache_map[cache_slot] = None
                is_first_refusal = not any(
                    value is None for slot, value in cache_map.items()
                    if slot != cache_slot)
                if is_first_refusal:
                    print(f"    [caches.create refused ({error.code}): "
                          f"{str(error)[:160]}]", flush=True)

        generation_config = {
            "temperature": prompt_cfg["temperature"],
            "max_output_tokens": prompt_cfg["max_output_tokens"],
            # 2.5 Flash thinks by default; the reasoning budget would
            # eat max_output_tokens before the answer emits.
            # Contextualization is summarization, not reasoning.
            "thinking_config": {"thinking_budget": 0},
        }
        cache_name = cache_map.get(cache_slot)
        if cache_name:
            generation_config["cached_content"] = cache_name
            contents = request_text
        else:
            contents = cache_text + "\n" + request_text

        try:
            response = client.models.generate_content(
                model=prompt_cfg["model"], contents=contents,
                config=generation_config)
            text = (response.text or "").strip()
            if text:
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                cached_tokens = getattr(usage, "cached_content_token_count",
                                        0) or 0
                return text, prompt_tokens, cached_tokens
        except genai_errors.APIError as error:
            cache_became_invalid = (cache_name
                                    and error.code in (400, 403, 404))
            if cache_became_invalid:
                cache_map.pop(cache_slot, None)   # rebuild next attempt
                continue
            retryable = error.code in (429, 500, 503)
            if not retryable:
                raise
            last_error = str(error)[:800]

        finished_full_cycle = rotator.rotate()
        if finished_full_cycle:
            cycle_number = attempt // n_keys + 1
            print(f"    [quota cycle {cycle_number}, "
                  f"sleeping {BACKOFF_S}s]", flush=True)
            time.sleep(BACKOFF_S)

    raise RuntimeError(f"all attempts exhausted; last error: {last_error}")


def main(chunks_path, out_path, limit=0, prompt_path=PROMPT_FILE):
    load_dotenv()
    keys = []
    for key in os.environ["GEMINI_API_KEYS"].split(","):
        if key.strip():
            keys.append(key.strip())
    rotator = KeyRotator(keys)
    prompt_cfg = load_prompt(prompt_path)

    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    # Document text = ALL its chunks joined, built BEFORE any limit is
    # applied so the model always sees the complete document.
    # (Paragraph overlap between chunks is harmless here.)
    doc_texts = {}
    for chunk in chunks:
        doc_texts.setdefault(chunk["doc_id"], []).append(chunk["text"])
    for doc_id, texts in doc_texts.items():
        doc_texts[doc_id] = "\n\n".join(texts)

    if limit:
        chunks = chunks[:limit]

    disk_cache = load_cache()
    cache_map = {}          # (key_index, doc_id) -> Gemini cache name
    hits = 0
    calls = 0
    prompt_tokens_total = 0
    cached_tokens_total = 0

    for i, chunk in enumerate(chunks, 1):
        key = cache_key(doc_texts[chunk["doc_id"]], chunk["text"],
                        prompt_cfg["version"], prompt_cfg["model"])
        if key in disk_cache:
            chunk["context"] = disk_cache[key]
            hits += 1
        else:
            context, prompt_tokens, cached_tokens = gen_context(
                rotator, prompt_cfg, chunk["doc_id"],
                doc_texts[chunk["doc_id"]], chunk["text"], cache_map)
            chunk["context"] = context
            prompt_tokens_total += prompt_tokens
            cached_tokens_total += cached_tokens
            append_cache(key, context)
            disk_cache[key] = context
            calls += 1

        if i % 25 == 0 or i == len(chunks):
            if prompt_tokens_total:
                percent = 100 * cached_tokens_total / prompt_tokens_total
                rate = f"{percent:.0f}%"
            else:
                rate = "n/a"
            print(f"  {i}/{len(chunks)}  disk cache hits {hits}, "
                  f"new calls {calls}  |  gemini prompt-cache: "
                  f"{cached_tokens_total}/{prompt_tokens_total} "
                  f"tokens ({rate})")

    # Best-effort cleanup of explicit caches (TTL expires them anyway).
    for (key_index, _), cache_name in cache_map.items():
        if cache_name:
            try:
                rotator.clients[key_index].caches.delete(name=cache_name)
            except Exception:
                pass
    if cache_map:
        created = sum(1 for name in cache_map.values() if name)
        fell_back = sum(1 for name in cache_map.values() if name is None)
        print(f"explicit caches: {created} created, {fell_back} fell back")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    word_counts = [len(chunk["context"].split()) for chunk in chunks]
    print(f"contexts: {len(chunks)}  |  words min/avg/max: "
          f"{min(word_counts)}/{sum(word_counts) // len(word_counts)}"
          f"/{max(word_counts)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", default=PROMPT_FILE)
    a = ap.parse_args()
    main(a.chunks, a.out, a.limit, a.prompt)
