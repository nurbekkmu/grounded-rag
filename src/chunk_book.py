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

Usage:  python src/chunk_book.py <book.pdf> <out.jsonl>
Deps:   pip install pymupdf tiktoken
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
def n_tokens(text):
    """Token count with the project's one measuring ruler (tiktoken o200k)."""
    return len(_ENC.encode(text))


def doc_id_for(chapter):
    """'Chapter 6. RAG and Agents' -> 'book/ch06'; 'Preface' -> 'book/preface'."""
    match = re.match(r"^Chapter (\d+)\b", chapter)
    if match:
        number = int(match.group(1))
        return f"book/ch{number:02d}"
    slug = re.sub(r"[^a-z0-9]+", "-", chapter.lower()).strip("-")
    return "book/" + slug


def clean_block(text):
    """Normalize one PDF text block: fix hyphenation and whitespace."""
    text = unicodedata.normalize("NFKC", text)
    # A word split across lines ends in a hyphen; joining the lines
    # rejoins the word ("retrie-\nval" -> "retrieval").
    text = re.sub(r"[‐­-]\n(?=[A-Za-z])", "", text)
    text = text.replace("‐", "-")
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_sections(doc):
    """Turn the PDF's table of contents into an ordered list of sections.

    Each entry keeps its chapter / section / subsection names and the
    page it starts on. Front/back matter (cover, index, ...) is skipped.
    """
    sections = []
    chapter = None
    section = None
    for level, title, page in doc.get_toc():
        title = title.strip()
        if level == 1:
            if title in SKIP_TOP_TITLES:
                chapter = None
                continue
            chapter = title
            section = None
            sections.append({"chapter": chapter, "section": None,
                             "subsection": None, "page": page,
                             "title": title})
        elif chapter and level == 2:
            section = title
            sections.append({"chapter": chapter, "section": section,
                             "subsection": None, "page": page,
                             "title": title})
        elif chapter and level == 3:
            sections.append({"chapter": chapter, "section": section,
                             "subsection": title, "page": page,
                             "title": title})
    return sections


def heading_y(page, title):
    """Vertical position of a section heading on its first page.

    Used to know where one section ends and the next begins when both
    share a page. Returns None when the heading text can't be found.
    """
    needle = title[:40].strip()
    hits = page.search_for(needle)
    if not hits:
        return None
    return min(hit.y0 for hit in hits)


def page_blocks(page):
    """Content blocks of one page in reading order, with footers dropped."""
    blocks = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if not text.strip():
            continue
        if y0 > FOOTER_Y and FOOTER_RE.match(text.strip()):
            continue
        blocks.append((y0, text))
    blocks.sort(key=lambda block: block[0])
    return blocks


def collect_section_paragraphs(doc, sections, last_page):
    """Assign each cleaned paragraph to the section it belongs to.

    A section spans from its heading (page + y position) up to the next
    section's heading. Everything between those two points is this
    section's text.
    """
    # First work out where each section starts and ends.
    bounds = []
    for i, s in enumerate(sections):
        start_page = s["page"]
        start_y = heading_y(doc[start_page - 1], s["title"])
        if i + 1 < len(sections):
            next_section = sections[i + 1]
            end_page = next_section["page"]
            end_y = heading_y(doc[end_page - 1], next_section["title"])
        else:
            end_page = last_page
            end_y = None
        bounds.append((s, start_page, start_y, end_page, end_y))

    # Then walk each section's pages and keep the paragraphs inside it.
    for s, start_page, start_y, end_page, end_y in bounds:
        paragraphs = []
        for page_number in range(start_page, end_page + 1):
            for y, text in page_blocks(doc[page_number - 1]):
                before_this_section = (page_number == start_page
                                       and start_y is not None
                                       and y < start_y)
                if before_this_section:
                    continue
                next_section_started = (page_number == end_page
                                        and end_y is not None
                                        and y >= end_y)
                if next_section_started:
                    continue
                cleaned = clean_block(text)
                if not cleaned:
                    continue
                if cleaned == s["title"]:
                    continue
                if re.match(r"^CHAPTER \d+\b", cleaned):
                    continue
                # If the previous paragraph ended mid-word (trailing
                # hyphen), glue this one onto it instead of starting new.
                ends_mid_word = (paragraphs
                                 and paragraphs[-1][1].endswith(("-", "‐"))
                                 and cleaned[:1].isalpha())
                if ends_mid_word:
                    prev_page, prev_text = paragraphs[-1]
                    paragraphs[-1] = (prev_page, prev_text[:-1] + cleaned)
                else:
                    paragraphs.append((page_number, cleaned))
        s["paras"] = paragraphs
        s["page_end"] = end_page
    return sections


def split_oversize(text):
    """Sentence-split a paragraph that alone exceeds MAX_TOKENS."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces = []
    current = ""
    for sentence in sentences:
        if current and n_tokens(current + " " + sentence) > MAX_TOKENS:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


class ParagraphPacker:
    """Packs (marker, paragraph) pairs into chunk-sized groups.

    The marker travels with each paragraph and can be anything the
    caller wants to remember — the book chunker uses page numbers, the
    blog chunker uses section headings.

    Paragraphs are added in reading order into a buffer. When adding
    the next paragraph would push the buffer past the limits, the
    buffer is closed into a finished group first. A small tail of the
    closed group (~OVERLAP_TOKENS) stays in the buffer so an idea that
    spans the boundary appears intact in at least one chunk.
    """

    def __init__(self):
        self.groups = []   # finished groups: each is a list of (marker, text)
        self.buffer = []   # paragraphs waiting to be packed

    def add(self, marker, paragraph):
        if n_tokens(paragraph) > MAX_TOKENS:
            pieces = split_oversize(paragraph)
        else:
            pieces = [paragraph]
        for piece in pieces:
            if self._would_overflow(piece):
                self.close_group(keep_overlap=True)
            self.buffer.append((marker, piece))

    def _would_overflow(self, piece):
        """Would adding this piece make the buffer too big to keep open?"""
        if not self.buffer:
            return False
        current = sum(n_tokens(text) for _, text in self.buffer)
        combined = current + n_tokens(piece)
        if combined > MAX_TOKENS:
            return True
        past_target = combined > TARGET_TOKENS
        big_enough_to_close = current >= TARGET_TOKENS // 2
        return past_target and big_enough_to_close

    def close_group(self, keep_overlap):
        """Move the buffer into a finished group.

        With keep_overlap=True the last ~OVERLAP_TOKENS worth of
        paragraphs stays in the buffer as the start of the next group.
        """
        if not self.buffer:
            return
        self.groups.append(list(self.buffer))
        if keep_overlap:
            self.buffer = self._overlap_tail()
        else:
            self.buffer = []

    def _overlap_tail(self):
        """The trailing paragraphs worth carrying into the next group.

        Walk backwards collecting paragraphs until the overlap budget
        is spent. A lone paragraph bigger than twice the budget is not
        carried at all — duplicating a huge block isn't overlap.
        """
        tail = []
        tokens_kept = 0
        for marker, text in reversed(self.buffer):
            tokens = n_tokens(text)
            if tail and tokens_kept + tokens > OVERLAP_TOKENS:
                break
            if not tail and tokens > OVERLAP_TOKENS * 2:
                break
            tail.append((marker, text))
            tokens_kept += tokens
        tail.reverse()
        return tail


def chunk_section(s, seq_start):
    """Pack one section's paragraphs into finished chunk records."""
    packer = ParagraphPacker()
    for page, paragraph in s["paras"]:
        packer.add(page, paragraph)
    packer.close_group(keep_overlap=False)

    doc_id = doc_id_for(s["chapter"])
    name_parts = [s["chapter"], s["section"], s["subsection"]]
    section_path = " > ".join(part for part in name_parts if part)

    chunks = []
    seq = seq_start
    for group in packer.groups:
        text = "\n\n".join(paragraph for _, paragraph in group)
        chunks.append({
            "chunk_id": f"{doc_id}#{seq:03d}",
            "doc_id": doc_id,
            "source": "book",
            "chapter": s["chapter"],
            "section": s["section"],
            "subsection": s["subsection"],
            "section_path": section_path,
            "page_start": group[0][0],
            "page_end": group[-1][0],
            "n_tokens": n_tokens(text),
            "text": text,
        })
        seq += 1

    # A tiny tail chunk carries too little to stand alone: merge it back.
    if len(chunks) >= 2 and chunks[-1]["n_tokens"] < MIN_TOKENS:
        tail = chunks.pop()
        chunks[-1]["text"] += "\n\n" + tail["text"]
        chunks[-1]["page_end"] = tail["page_end"]
        chunks[-1]["n_tokens"] = n_tokens(chunks[-1]["text"])
    return chunks, seq


def main(pdf_path, out_path):
    doc = fitz.open(pdf_path)
    sections = load_sections(doc)

    # The last content page is the one right before the Index.
    index_page = doc.page_count + 1
    for level, title, page in doc.get_toc():
        if level == 1 and title.strip() == "Index":
            index_page = page
            break
    sections = collect_section_paragraphs(doc, sections, index_page - 1)

    all_chunks = []
    seq_per_doc = {}
    for s in sections:
        if not s["paras"]:
            continue
        doc_id = doc_id_for(s["chapter"])
        start = seq_per_doc.get(doc_id, 0)
        chunks, seq_per_doc[doc_id] = chunk_section(s, start)
        all_chunks.extend(chunks)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    token_counts = [chunk["n_tokens"] for chunk in all_chunks]
    over_cap = sum(1 for count in token_counts if count > MAX_TOKENS)
    print(f"chunks: {len(all_chunks)}  |  tokens min/avg/max: "
          f"{min(token_counts)}/{sum(token_counts) // len(token_counts)}"
          f"/{max(token_counts)}  |  >{MAX_TOKENS}: {over_cap}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
