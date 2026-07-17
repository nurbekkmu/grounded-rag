"""
Structure-aware chunker for blog posts (markdown from ingest_blog.py).

Same strategy and tunables as the book chunker (imported from
chunk_book.py so the two can never drift): headings are hard section
walls; paragraphs pack to ~TARGET tokens; paragraph-level overlap;
oversize paragraphs get sentence-split; fenced code blocks are atomic.

Blog chunks carry url + date instead of page numbers; chunk_id / doc_id /
source / section_path fields match the book chunks so every downstream
stage (contextualize, index, retrieve) sees one uniform schema.

Usage:  python src/chunk_blog.py [docs.jsonl] [out.jsonl]
"""

import json
import os
import re
import sys

from chunk_book import MIN_TOKENS, ParagraphPacker, n_tokens

DOCS_DEFAULT = "data/processed/blog_docs.jsonl"
OUT_DEFAULT = "data/processed/blog_chunks.jsonl"

HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")


def blocks(md_text):
    """Split markdown into (heading, block) pairs in reading order.

    A block is one paragraph (blank-line separated) or one whole fenced
    code block. heading is the most recent markdown heading, or None
    before the first one. Lines inside a code fence are never treated
    as headings, even if they start with '#'.
    """
    heading = None
    current_lines = []
    inside_code_fence = False
    result = []

    def finish_block():
        text = "\n".join(current_lines).strip()
        if text:
            result.append((heading, text))
        current_lines.clear()

    for line in md_text.splitlines():
        if line.lstrip().startswith("```"):
            if not inside_code_fence:
                finish_block()            # prose before the fence ends here
            current_lines.append(line)
            if inside_code_fence:
                finish_block()            # the closing fence ends the block
            inside_code_fence = not inside_code_fence
            continue

        if inside_code_fence:
            current_lines.append(line)
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            finish_block()
            heading = heading_match.group(1).strip()
            continue

        if not line.strip():
            finish_block()
            continue

        current_lines.append(line)

    finish_block()
    return result


def chunk_doc(doc):
    """Pack one post's blocks into chunks; a heading change is a hard wall."""
    packer = ParagraphPacker()
    current_heading = None
    for heading, block in blocks(doc["text"]):
        if heading != current_heading:
            # Never let overlap leak across a heading boundary — the
            # text on either side belongs to different topics.
            packer.close_group(keep_overlap=False)
            current_heading = heading
        packer.add(heading, block)
    packer.close_group(keep_overlap=False)

    chunks = []
    for seq, group in enumerate(packer.groups):
        text = "\n\n".join(block for _, block in group)
        heading = group[-1][0]
        if heading:
            section_path = f"{doc['title']} > {heading}"
        else:
            section_path = doc["title"]
        chunks.append({
            "chunk_id": f"{doc['doc_id']}#{seq:03d}",
            "doc_id": doc["doc_id"],
            "source": "blog",
            "title": doc["title"],
            "section": heading,
            "section_path": section_path,
            "url": doc["url"],
            "date": doc["date"],
            "n_tokens": n_tokens(text),
            "text": text,
        })
    return merge_tiny(chunks, doc["doc_id"])


def merge_tiny(chunks, doc_id):
    """Merge chunks under MIN_TOKENS forward into the next chunk.

    Blog sections are often a heading plus one short intro paragraph;
    that intro belongs WITH the content that follows it, not alone. The
    final chunk (no next) merges backward instead. IDs are renumbered
    after merging.
    """
    merged = []
    pending = None                 # a tiny chunk waiting to join the next one
    for chunk in chunks:
        if pending is not None:
            chunk["text"] = pending["text"] + "\n\n" + chunk["text"]
            chunk["n_tokens"] = n_tokens(chunk["text"])
            pending = None
        if chunk["n_tokens"] < MIN_TOKENS:
            pending = chunk
        else:
            merged.append(chunk)

    if pending is not None:
        if merged:
            merged[-1]["text"] += "\n\n" + pending["text"]
            merged[-1]["n_tokens"] = n_tokens(merged[-1]["text"])
        else:
            merged.append(pending)     # the whole post is tiny: keep as-is

    for i, chunk in enumerate(merged):
        chunk["chunk_id"] = f"{doc_id}#{i:03d}"
    return merged


def main(docs_path=DOCS_DEFAULT, out_path=OUT_DEFAULT):
    docs = []
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_doc(doc))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    token_counts = [chunk["n_tokens"] for chunk in all_chunks]
    over_cap = sum(1 for count in token_counts if count > 800)
    print(f"chunks: {len(all_chunks)} from {len(docs)} posts  |  "
          f"tokens min/avg/max: {min(token_counts)}"
          f"/{sum(token_counts) // len(token_counts)}/{max(token_counts)}"
          f"  |  >800: {over_cap}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DOCS_DEFAULT,
         sys.argv[2] if len(sys.argv) > 2 else OUT_DEFAULT)
