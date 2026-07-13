"""
Citation-grounded answer generation (spec §5.7).

Assembles the versioned answer prompt (prompts/answer_with_citations.yaml)
from the reranked shortlist, calls Gemini at temperature 0, and returns
the answer plus a citation audit: which provided chunks were cited, and
— the red flag — any cited IDs that were never provided (a fabricated
citation is worse than no citation). The full enforcement pass lives in
guardrails.py; this module's audit is the first line.

The response object carries a trace (per-stage candidates and ranks)
so the observability layer can plug in later without refactoring.

Usage:  python generate.py "your question"
            [--mode hybrid] [--top-n 75] [--keep 8]
            [--index data/index/baseline]
Deps:   pip install google-genai python-dotenv pyyaml sentence-transformers
Env:    GEMINI_API_KEYS=key1,key2,...
"""

import argparse
import json
import os
import re
import sys

import yaml
from dotenv import load_dotenv

from contextualize import KeyRotator, genai_errors
from rerank import rerank
from retrieve import CHUNKS_DEFAULT, retrieve

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROMPT_FILE = "prompts/answer_with_citations.yaml"
CITE_RE = re.compile(r"\[([a-z0-9_/#.-]+#\d{3})\]")


def load_prompt(path=PROMPT_FILE):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def chunks_block(candidates: list) -> str:
    parts = []
    for c in candidates:
        ch = c["chunk"]
        where = (f"pp.{ch['page_start']}-{ch['page_end']}"
                 if ch["source"] == "book" else ch["url"])
        parts.append(f"[{ch['chunk_id']}] ({ch['section_path']} — {where})\n"
                     f"{ch['text']}")
    return "\n\n---\n\n".join(parts)


def call_gemini(rotator: KeyRotator, cfg: dict, system: str, user: str) -> str:
    n_keys = len(rotator.clients)
    last_err = ""
    for attempt in range(3 * n_keys):
        try:
            resp = rotator.client().models.generate_content(
                model=cfg["model"],
                contents=user,
                config={
                    "system_instruction": system,
                    "temperature": cfg["temperature"],
                    "max_output_tokens": cfg["max_output_tokens"],
                    "thinking_config": {
                        "thinking_budget": cfg.get("thinking_budget", 0)},
                },
            )
            text = (resp.text or "").strip()
            if text:
                return text
        except genai_errors.APIError as e:
            if e.code not in (429, 500, 503):
                raise
            last_err = str(e)[:400]
        rotator.rotate()
    raise RuntimeError(f"generation failed on all keys; last: {last_err}")


def answer(query: str, candidates: list, cfg: dict = None,
           rotator: KeyRotator = None) -> dict:
    cfg = cfg or load_prompt()
    if rotator is None:
        load_dotenv()
        rotator = KeyRotator([k.strip() for k in
                              os.environ["GEMINI_API_KEYS"].split(",")
                              if k.strip()])
    user = (cfg["user_template"]
            .replace("{{CHUNKS}}", chunks_block(candidates))
            .replace("{{QUESTION}}", query))
    text = call_gemini(rotator, cfg, cfg["system_prompt"], user)

    provided = {c["chunk_id"] for c in candidates}
    cited = set(CITE_RE.findall(text))
    return {
        "answer": text,
        "refused": cfg["refusal"] in text,
        "citations": sorted(cited & provided),
        "fabricated_citations": sorted(cited - provided),
        "trace": {
            "prompt_version": cfg["version"],
            "model": cfg["model"],
            "candidates": [{"chunk_id": c["chunk_id"],
                            "ranks": c["ranks"],
                            "rerank_score": c.get("rerank_score")}
                           for c in candidates],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"],
                    default="hybrid")
    ap.add_argument("--top-n", type=int, default=75)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--index", default="data/index/baseline")
    ap.add_argument("--chunks", nargs="+", default=CHUNKS_DEFAULT)
    ap.add_argument("--json", action="store_true",
                    help="print the full response object")
    a = ap.parse_args()

    net = retrieve(a.query, a.index, a.chunks, a.mode, a.top_n, k=a.top_n)
    shortlist = rerank(a.query, net, keep=a.keep)
    result = answer(a.query, shortlist)

    if a.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(result["answer"])
    print()
    if result["refused"]:
        print("(refused: insufficient evidence)")
    store = {c["chunk_id"]: c["chunk"] for c in shortlist}
    for cid in result["citations"]:
        ch = store[cid]
        where = (f"pp.{ch['page_start']}-{ch['page_end']}"
                 if ch["source"] == "book" else ch["url"])
        print(f"  [{cid}]  {ch['section_path'][:60]}  ({where})")
    if result["fabricated_citations"]:
        print(f"  !! fabricated citation ids: "
              f"{result['fabricated_citations']}")


if __name__ == "__main__":
    main()
