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

Usage:  python src/generate.py "your question"
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

from config import cfg as _cfg
from contextualize import KeyRotator, genai_errors
from rerank import rerank
from retrieve import CHUNKS_DEFAULT, retrieve

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROMPT_FILE = _cfg("generation.prompt")

# Matches citations like [book/ch06#014] but not [12] or [foo].
CITE_RE = re.compile(r"\[([a-z0-9_/#.-]+#\d{3})\]")


def load_prompt(path=PROMPT_FILE):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def chunks_block(candidates):
    """Format the shortlist as the prompt's evidence section.

    Each chunk is labeled with its ID (which the model must cite) and
    its human-readable source (section path plus pages or URL).
    """
    parts = []
    for candidate in candidates:
        chunk = candidate["chunk"]
        if chunk["source"] == "book":
            where = f"pp.{chunk['page_start']}-{chunk['page_end']}"
        else:
            where = chunk["url"]
        header = f"[{chunk['chunk_id']}] ({chunk['section_path']} — {where})"
        parts.append(header + "\n" + chunk["text"])
    return "\n\n---\n\n".join(parts)


def call_gemini(rotator, prompt_cfg, system_text, user_text):
    """One generation call, rotating across API keys on quota errors."""
    total_attempts = 3 * len(rotator.clients)
    last_error = ""
    for attempt in range(total_attempts):
        try:
            response = rotator.client().models.generate_content(
                model=prompt_cfg["model"],
                contents=user_text,
                config={
                    "system_instruction": system_text,
                    "temperature": prompt_cfg["temperature"],
                    "max_output_tokens": prompt_cfg["max_output_tokens"],
                    "thinking_config": {
                        "thinking_budget":
                            prompt_cfg.get("thinking_budget", 0)},
                },
            )
            text = (response.text or "").strip()
            if text:
                return text
        except genai_errors.APIError as error:
            retryable = error.code in (429, 500, 503)
            if not retryable:
                raise
            last_error = str(error)[:400]
        rotator.rotate()
    raise RuntimeError(f"generation failed on all keys; last: {last_error}")


def answer(query, candidates, prompt_cfg=None, rotator=None):
    """Generate a cited answer from the shortlist and audit its citations."""
    if prompt_cfg is None:
        prompt_cfg = load_prompt()
    if rotator is None:
        load_dotenv()
        keys = []
        for key in os.environ["GEMINI_API_KEYS"].split(","):
            if key.strip():
                keys.append(key.strip())
        rotator = KeyRotator(keys)

    user_text = (prompt_cfg["user_template"]
                 .replace("{{CHUNKS}}", chunks_block(candidates))
                 .replace("{{QUESTION}}", query))
    answer_text = call_gemini(rotator, prompt_cfg,
                              prompt_cfg["system_prompt"], user_text)

    # Audit the citations: which provided chunks were actually cited,
    # and did the model invent any IDs it was never given?
    provided_ids = set()
    for candidate in candidates:
        provided_ids.add(candidate["chunk_id"])
    cited_ids = set(CITE_RE.findall(answer_text))

    trace_candidates = []
    for candidate in candidates:
        trace_candidates.append({
            "chunk_id": candidate["chunk_id"],
            "ranks": candidate["ranks"],
            "rerank_score": candidate.get("rerank_score"),
        })

    return {
        "answer": answer_text,
        "refused": prompt_cfg["refusal"] in answer_text,
        "citations": sorted(cited_ids & provided_ids),
        "fabricated_citations": sorted(cited_ids - provided_ids),
        "trace": {
            "prompt_version": prompt_cfg["version"],
            "model": prompt_cfg["model"],
            "candidates": trace_candidates,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"],
                    default=_cfg("retrieval.mode"))
    ap.add_argument("--top-n", type=int, default=_cfg("retrieval.top_n"))
    ap.add_argument("--keep", type=int, default=_cfg("rerank.keep"))
    ap.add_argument("--index", default=_cfg("retrieval.index"))
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
    for chunk_id in result["citations"]:
        chunk = store[chunk_id]
        if chunk["source"] == "book":
            where = f"pp.{chunk['page_start']}-{chunk['page_end']}"
        else:
            where = chunk["url"]
        print(f"  [{chunk_id}]  {chunk['section_path'][:60]}  ({where})")
    if result["fabricated_citations"]:
        print(f"  !! fabricated citation ids: "
              f"{result['fabricated_citations']}")


if __name__ == "__main__":
    main()
