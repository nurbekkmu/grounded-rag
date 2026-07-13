"""
Structure-aware chunker for blog posts (markdown from ingest_blog.py).

Same strategy and tunables as the book chunker (imported from
chunk_book.py so the two can never drift): headings are hard section
walls; paragraphs pack to ~TARGET tokens; paragraph-level overlap;
oversize paragraphs get sentence-split; fenced code blocks are atomic.

Blog chunks carry url + date instead of page numbers; chunk_id / doc_id /
source / section_path fields match the book chunks so every downstream
stage (contextualize, index, retrieve) sees one uniform schema.

Usage:  python chunk_blog.py [docs.jsonl] [out.jsonl]
"""

import json
import os
import re
import sys

from chunk_book import (MAX_TOKENS, MIN_TOKENS, OVERLAP_TOKENS,
                        TARGET_TOKENS, n_tokens, split_oversize)

DOCS_DEFAULT = "data/processed/blog_docs.jsonl"
OUT_DEFAULT = "data/processed/blog_chunks.jsonl"

HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")


def blocks(md_text: str):
    """Yield (heading, block_text) in reading order.

    A block is a paragraph (blank-line separated) or a whole fenced code
    block. heading is the most recent markdown heading (None before the
    first one).
    """
    heading, buf, in_code, out = None, [], False, []

    def flush():
        nonlocal buf
        t = "\n".join(buf).strip()
        if t:
            out.append((heading, t))
        buf = []

    for ln in md_text.splitlines():
        if ln.lstrip().startswith("```"):
            if not in_code:
                flush()                # prose before the fence ends here
            buf.append(ln)
            if in_code:
                flush()                # closing fence ends the block
            in_code = not in_code
            continue
        if in_code:
            buf.append(ln)
            continue
        m = HEADING_RE.match(ln)
        if m:
            flush()
            heading = m.group(1).strip()
            continue
        if not ln.strip():
            flush()
            continue
        buf.append(ln)
    flush()
    return out


def chunk_doc(doc: dict) -> list:
    """Pack one post's blocks into chunks; heading change = hard wall."""
    chunks, buf, seq = [], [], 0
    cur_head = None

    def section_path(head):
        return f"{doc['title']} > {head}" if head else doc["title"]

    def emit(text, head):
        nonlocal seq
        chunks.append({
            "chunk_id": f"{doc['doc_id']}#{seq:03d}",
            "doc_id": doc["doc_id"],
            "source": "blog",
            "title": doc["title"],
            "section": head,
            "section_path": section_path(head),
            "url": doc["url"],
            "date": doc["date"],
            "n_tokens": n_tokens(text),
            "text": text,
        })
        seq += 1

    def flush(seed_overlap=True):
        nonlocal buf
        if not buf:
            return
        emit("\n\n".join(t for _, t in buf), buf[-1][0])
        if not seed_overlap:
            buf = []
            return
        keep, tok = [], 0
        for item in reversed(buf):
            t = n_tokens(item[1])
            if keep and tok + t > OVERLAP_TOKENS:
                break
            if not keep and t > OVERLAP_TOKENS * 2:
                break
            keep.append(item)
            tok += t
        buf = list(reversed(keep))

    for head, block in blocks(doc["text"]):
        if head != cur_head:
            flush(seed_overlap=False)   # never overlap across a heading
            cur_head = head
        pieces = (split_oversize(block)
                  if n_tokens(block) > MAX_TOKENS else [block])
        for piece in pieces:
            cur = sum(n_tokens(t) for _, t in buf)
            t = n_tokens(piece)
            if buf and (cur + t > MAX_TOKENS
                        or (cur + t > TARGET_TOKENS
                            and cur >= TARGET_TOKENS // 2)):
                flush()
            buf.append((head, piece))
    flush(seed_overlap=False)
    return merge_tiny(chunks, doc["doc_id"])


def merge_tiny(chunks: list, doc_id: str) -> list:
    """Merge chunks under MIN_TOKENS forward into the next chunk.

    Blog sections are often a heading plus one short intro paragraph; that
    intro belongs WITH the content that follows it, not alone. The final
    chunk (no next) merges backward instead. IDs are renumbered after.
    """
    merged, pending = [], None
    for c in chunks:
        if pending:
            c["text"] = pending["text"] + "\n\n" + c["text"]
            c["n_tokens"] = n_tokens(c["text"])
            pending = None
        if c["n_tokens"] < MIN_TOKENS:
            pending = c
            continue
        merged.append(c)
    if pending:
        if merged:
            merged[-1]["text"] += "\n\n" + pending["text"]
            merged[-1]["n_tokens"] = n_tokens(merged[-1]["text"])
        else:
            merged.append(pending)      # whole post is tiny: keep as-is
    for i, c in enumerate(merged):
        c["chunk_id"] = f"{doc_id}#{i:03d}"
    return merged


def main(docs_path=DOCS_DEFAULT, out_path=OUT_DEFAULT):
    docs = [json.loads(line) for line in open(docs_path, encoding="utf-8")]
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_doc(doc))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    toks = [c["n_tokens"] for c in all_chunks]
    big = sum(1 for t in toks if t > MAX_TOKENS)
    print(f"chunks: {len(all_chunks)} from {len(docs)} posts  |  "
          f"tokens min/avg/max: {min(toks)}/{sum(toks)//len(toks)}/{max(toks)}"
          f"  |  >{MAX_TOKENS}: {big}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DOCS_DEFAULT,
         sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT)
