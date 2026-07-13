"""
Generate a side-by-side review document for golden-set verification.

For every question: the expected answer next to the full text of each
anchored evidence chunk, plus a checklist. Output goes to data/ (it
contains book excerpts, which stay out of the repo by design).

After reviewing an entry, flip its "verified" to true in
eval/golden_set.jsonl — run_eval drops its provisional warning once all
entries pass.

Usage:  python eval/make_review.py [--out data/golden_review.md]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

CHUNK_FILES = ["data/processed/book_chunks.jsonl",
               "data/processed/blog_chunks.jsonl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_set.jsonl")
    ap.add_argument("--out", default="data/golden_review.md")
    a = ap.parse_args()

    store = {}
    for p in CHUNK_FILES:
        for line in open(p, encoding="utf-8"):
            c = json.loads(line)
            store[c["chunk_id"]] = c

    golden = [json.loads(l) for l in open(a.golden, encoding="utf-8")]
    lines = [
        "# Golden set review",
        "",
        "For each entry: does the evidence actually support the expected",
        "answer, and is the question fair? Fix or bless, then set",
        '`"verified": true` in eval/golden_set.jsonl.',
        "",
    ]
    for q in golden:
        state = "VERIFIED" if q.get("verified") else "UNVERIFIED"
        lines += [f"## {q['qid']} ({q['category']}, {q['split']}) — {state}",
                  "", f"**Q:** {q['question']}", ""]
        if q["category"] == "refusal":
            lines += ["**Expected: refusal** — confirm the corpus really "
                      "does not contain this. Search terms worth trying "
                      "in data/processed/*.jsonl before blessing.", ""]
        else:
            lines += [f"**Expected answer:** {q['expected_answer']}", ""]
            for cid in q["evidence_chunk_ids"]:
                ch = store.get(cid)
                if not ch:
                    lines += [f"### {cid} — !! NOT FOUND IN CORPUS", ""]
                    continue
                where = (f"pp.{ch['page_start']}-{ch['page_end']}"
                         if ch["source"] == "book" else ch["url"])
                lines += [f"### evidence {cid} ({ch['section_path']} — "
                          f"{where})", "", "> " +
                          ch["text"].replace("\n", "\n> "), ""]
        lines += ["- [ ] evidence supports the answer",
                  "- [ ] question is answerable from evidence alone",
                  "- [ ] no better evidence chunk exists", "", "---", ""]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {a.out}: {len(golden)} entries, "
          f"{sum(1 for q in golden if not q.get('verified'))} unverified")


if __name__ == "__main__":
    main()
