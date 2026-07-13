"""
Structure-aware chunker for "AI Engineering" (Chip Huyen) PDF.

Strategy:
  1. Use the PDF's embedded TOC (3 levels) to find exact section boundaries.
  2. Extract text block-by-block per page; drop running footers by position + regex.
  3. Clean: de-hyphenate (U+2010 soft hyphens), normalize whitespace.
  4. Pack paragraphs into chunks (~TARGET tokens, hard MAX), never splitting a
     paragraph/code block; small paragraph overlap between chunks.
  5. Emit JSONL with metadata (chapter / section / subsection / pages) ready for
     contextual retrieval (Gemini) + Weaviate ingestion.

Usage:  python chunk_book.py <book.pdf> <out.jsonl>
Deps:   pip install pymupdf
"""

import json
import os
import re
import sys
import unicodedata
from functools import lru_cache

import fitz  # PyMuPDF
import tiktoken

# ---- tunables -------------------------------------------------------------
TARGET_TOKENS = 650        # aim per chunk (spec: 500-800 band)
MAX_TOKENS = 800           # hard cap; oversize paragraphs get sentence-split
OVERLAP_TOKENS = 100       # carry-over between adjacent chunks
MIN_TOKENS = 100           # tail chunks smaller than this merge into predecessor
FOOTER_Y = 595             # blocks starting below this y are footer candidates
SKIP_TOP_TITLES = {"Cover", "Copyright", "Table of Contents", "Index",
                   "About the Author", "Colophon"}

FOOTER_RE = re.compile(r"^\s*\d+\s*\n?\s*\|\s*\n?.*$|^.*\|\s*\n?\s*\d+\s*$",
                       re.DOTALL)


_ENC = tiktoken.get_encoding("o200k_base")


@lru_cache(maxsize=None)
def n_tokens(text: str) -> int:
    """Token count with the project's one measuring ruler (tiktoken o200k)."""
    return len(_ENC.encode(text))


def doc_id_for(chapter: str) -> str:
    """'Chapter 6. RAG and Agents' -> 'book/ch06'; 'Preface' -> 'book/preface'."""
    m = re.match(r"^Chapter (\d+)\b", chapter)
    if m:
        return f"book/ch{int(m.group(1)):02d}"
    return "book/" + re.sub(r"[^a-z0-9]+", "-", chapter.lower()).strip("-")


def clean_block(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # de-hyphenate: soft hyphen (U+2010) or '-' at line end joins the word
    text = re.sub(r"[\u2010\u00ad-]\n(?=[A-Za-z])", "", text)
    text = text.replace("\u2010", "-")  # normalize remaining unicode hyphens
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_sections(doc):
    """TOC -> ordered leaf sections with hierarchy + start page."""
    toc = doc.get_toc()
    sections, ch, sec = [], None, None
    for level, title, page in toc:
        title = title.strip()
        if level == 1:
            if title in SKIP_TOP_TITLES:
                ch = None
                continue
            ch, sec = title, None
            sections.append({"chapter": ch, "section": None,
                             "subsection": None, "page": page, "title": title})
        elif ch and level == 2:
            sec = title
            sections.append({"chapter": ch, "section": sec,
                             "subsection": None, "page": page, "title": title})
        elif ch and level == 3:
            sections.append({"chapter": ch, "section": sec,
                             "subsection": title, "page": page, "title": title})
    return sections


def heading_y(page, title: str):
    """y-coordinate of a section heading on its start page (None if not found)."""
    needle = title[:40].strip()
    hits = page.search_for(needle)
    return min(h.y0 for h in hits) if hits else None


def page_blocks(page):
    """Content blocks of a page: (y0, text), footers dropped, reading order."""
    out = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if not text.strip():
            continue
        if y0 > FOOTER_Y and FOOTER_RE.match(text.strip()):
            continue
        out.append((y0, text))
    out.sort(key=lambda b: b[0])
    return out


def collect_section_paragraphs(doc, sections, last_page):
    """Assign cleaned paragraphs to each leaf section using page + y bounds."""
    bounds = []
    for i, s in enumerate(sections):
        start_p = s["page"]
        y0 = heading_y(doc[start_p - 1], s["title"])
        if i + 1 < len(sections):
            nxt = sections[i + 1]
            end_p = nxt["page"]
            y_end = heading_y(doc[end_p - 1], nxt["title"])
        else:
            end_p, y_end = last_page, None
        bounds.append((s, start_p, y0, end_p, y_end))

    for s, start_p, y0, end_p, y_end in bounds:
        paras = []
        for pno in range(start_p, end_p + 1):
            for y, text in page_blocks(doc[pno - 1]):
                if pno == start_p and y0 is not None and y < y0:
                    continue                      # before this heading
                if pno == end_p and y_end is not None and y >= y_end:
                    continue                      # next section started
                t = clean_block(text)
                if not t or t == s["title"] or re.match(r"^CHAPTER \d+\b", t):
                    continue
                # word split across blocks: previous para ends mid-word
                if paras and paras[-1][1].endswith(("-", "\u2010")) \
                        and t[:1].isalpha():
                    ppno, prev = paras[-1]
                    paras[-1] = (ppno, prev[:-1] + t)
                else:
                    paras.append((pno, t))
        s["paras"], s["page_end"] = paras, end_p
    return sections


def split_oversize(text: str):
    """Sentence-split a paragraph that alone exceeds MAX_TOKENS."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for sent in sents:
        if cur and n_tokens(cur + " " + sent) > MAX_TOKENS:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_section(s, seq_start: int):
    """Pack paragraphs into chunks; paragraph-level overlap; never split blocks."""
    doc_id = doc_id_for(s["chapter"])
    section_path = " > ".join(
        x for x in (s["chapter"], s["section"], s["subsection"]) if x)
    chunks, buf, seq = [], [], seq_start

    def emit(text, page_a, page_b):
        nonlocal seq
        chunks.append({
            "chunk_id": f"{doc_id}#{seq:03d}",
            "doc_id": doc_id,
            "source": "book",
            "chapter": s["chapter"],
            "section": s["section"],
            "subsection": s["subsection"],
            "section_path": section_path,
            "page_start": page_a,
            "page_end": page_b,
            "n_tokens": n_tokens(text),
            "text": text,
        })
        seq += 1

    def flush(seed_overlap=True):
        nonlocal buf
        if not buf:
            return
        emit("\n\n".join(p for _, p in buf), buf[0][0], buf[-1][0])
        if not seed_overlap:
            buf = []
            return
        # paragraph overlap: trailing paras up to OVERLAP_TOKENS; a lone
        # paragraph over 2x the budget is skipped (don't duplicate big blocks)
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

    for pno, para in s["paras"]:
        pieces = split_oversize(para) if n_tokens(para) > MAX_TOKENS else [para]
        for piece in pieces:
            cur = sum(n_tokens(p) for _, p in buf)
            t = n_tokens(piece)
            if buf and (cur + t > MAX_TOKENS
                        or (cur + t > TARGET_TOKENS
                            and cur >= TARGET_TOKENS // 2)):
                flush()
            buf.append((pno, piece))
    flush(seed_overlap=False)

    # a tiny tail chunk carries too little to stand alone: merge it back
    if len(chunks) >= 2 and chunks[-1]["n_tokens"] < MIN_TOKENS:
        tail = chunks.pop()
        chunks[-1]["text"] += "\n\n" + tail["text"]
        chunks[-1]["page_end"] = tail["page_end"]
        chunks[-1]["n_tokens"] = n_tokens(chunks[-1]["text"])
    return chunks, seq


def main(pdf_path: str, out_path: str):
    doc = fitz.open(pdf_path)
    sections = load_sections(doc)
    # last content page = page before Index (or doc end)
    idx = next((p for lv, t, p in doc.get_toc()
                if lv == 1 and t.strip() == "Index"), doc.page_count + 1)
    sections = collect_section_paragraphs(doc, sections, idx - 1)

    all_chunks, seqs = [], {}
    for s in sections:
        if not s["paras"]:
            continue
        did = doc_id_for(s["chapter"])
        chunks, seqs[did] = chunk_section(s, seqs.get(did, 0))
        all_chunks.extend(chunks)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    toks = [c["n_tokens"] for c in all_chunks]
    big = sum(1 for t in toks if t > MAX_TOKENS)
    print(f"chunks: {len(all_chunks)}  |  tokens min/avg/max: "
          f"{min(toks)}/{sum(toks)//len(toks)}/{max(toks)}  |  >{MAX_TOKENS}: {big}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
