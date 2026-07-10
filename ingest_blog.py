"""
Fetch Chip Huyen's blog posts and extract clean article text.

Corpus scope (user decision, 2026-07-06): ALL posts listed on
huyenchip.com/blog — technical, career, everything.

Raw HTML is cached in data/raw/blog/ and article text is extracted with
trafilatura as markdown (headings survive, which chunk_blog.py needs).
Nothing fetched is ever committed: anyone can rebuild this half of the
corpus by running the script ("public code, private data" repo pattern).

Output: data/processed/blog_docs.jsonl, one post per line:
    {"doc_id": "blog/agents", "source": "blog", "title": "Agents",
     "url": "https://huyenchip.com/2025/01/07/agents.html",
     "date": "2025-01-07", "text": "..."}

Usage:  python ingest_blog.py [out.jsonl]
Deps:   pip install httpx trafilatura
"""

import json
import os
import re
import sys
import time

import httpx
import trafilatura

BASE = "https://huyenchip.com"
INDEX = f"{BASE}/blog/"
RAW_DIR = "data/raw/blog"
OUT_DEFAULT = "data/processed/blog_docs.jsonl"
DELAY_S = 1.0              # politeness delay between live fetches
POST_RE = re.compile(r'href="(/\d{4}/\d{2}/\d{2}/[^"]+\.html)"')
HEADERS = {"User-Agent": "personal-rag-portfolio-project (research use)"}


def post_paths(client) -> list:
    """All dated post links on the blog index, order preserved."""
    html = client.get(INDEX).text
    seen, out = set(), []
    for path in POST_RE.findall(html):
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def fetch(client, path: str):
    """Return (html, slug, fetched_live); serves from raw cache when present."""
    slug = path.rstrip("/").split("/")[-1].removesuffix(".html")
    cache = os.path.join(RAW_DIR, f"{slug}.html")
    if os.path.exists(cache):
        return open(cache, encoding="utf-8").read(), slug, False
    html = client.get(BASE + path).text
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        f.write(html)
    return html, slug, True


def main(out_path=OUT_DEFAULT):
    docs, skipped = [], []
    with httpx.Client(headers=HEADERS, timeout=30,
                      follow_redirects=True) as client:
        paths = post_paths(client)
        print(f"found {len(paths)} posts on {INDEX}")
        for i, path in enumerate(paths, 1):
            html, slug, live = fetch(client, path)
            text = trafilatura.extract(
                html, output_format="markdown", include_formatting=True,
                include_tables=True, include_comments=False) or ""
            if not text.strip():
                skipped.append(path)
                continue
            meta = trafilatura.extract_metadata(html)
            title = (meta.title if meta and meta.title
                     else slug.replace("-", " "))
            docs.append({
                "doc_id": f"blog/{slug}",
                "source": "blog",
                "title": title,
                "url": BASE + path,
                "date": "-".join(path.strip("/").split("/")[:3]),
                "text": text,
            })
            if i % 10 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)}")
            if live:
                time.sleep(DELAY_S)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    words = [len(d["text"].split()) for d in docs]
    print(f"docs: {len(docs)}  |  words min/avg/max: "
          f"{min(words)}/{sum(words) // len(words)}/{max(words)}")
    if skipped:
        print(f"skipped (empty extraction): {skipped}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT)
