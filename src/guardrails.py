"""
Citation enforcement guardrails (spec §5.8) — the two layers that make
citations enforced rather than politely requested.

Layer 1, pre-generation: if even the best reranked candidate scores
below MIN_RERANK, the evidence simply isn't there — refuse before
spending a generation call. The default threshold is a placeholder
until the golden set calibrates it.

Layer 2, post-generation: every substantive sentence of the answer must
be entailed by the chunk it cites. A local NLI cross-encoder scores the
claim against EACH sentence (and adjacent sentence pair) of the cited
chunk, and the best entailment wins — SummaC-style max aggregation.
Whole-chunk premises don't work: NLI models are trained on sentence
pairs (MNLI), and a 300-word premise drifts to "neutral" even when it
contains the claim verbatim (measured: p=0.001 on an identity claim).
Any unsupported or uncited claim downgrades the whole answer to the
fixed refusal string: a wrong answer wearing confident citations is
the worst output this system could produce.

The runtime guardrail here and the offline faithfulness metric in the
eval harness are the same idea measured in two places.

Usage:
  python guardrails.py "question" [--mode hybrid] [--keep 8] ...   full pipeline
  python guardrails.py --selftest                                  offline check
Deps:  pip install sentence-transformers
"""

import argparse
import re
import sys

import numpy as np
from sentence_transformers import CrossEncoder

from generate import CITE_RE, answer, load_prompt
from rerank import rerank
from retrieve import CHUNKS_DEFAULT, retrieve

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
MIN_RERANK = 0.0        # pre-generation floor on the best candidate
ENTAIL_MIN = 0.5        # min P(entailment) for a claim to count as supported
SENT_RE = re.compile(r"(?<=[.!?])\s+")
MAX_PREMISE_UNITS = 80  # cap on sentence units scored per claim

# Attribution phrasing kills NLI: "named in the book are X and Y" scores
# p=0.003 against a premise that says exactly "are X and Y" (measured) —
# the premise never calls itself "the book", so the model rightly can't
# verify the meta-claim. The citation is the attribution; strip the
# phrasing before scoring the factual content. Prompt v2 also tells the
# generator not to write these phrases in the first place.
ATTRIB_SUBS = [
    (re.compile(r"^\s*according to the (book|blog|author|corpus|provided "
                r"(?:context|documents))[,:]?\s*", re.I), ""),
    (re.compile(r"\b(named|mentioned|described|listed|discussed|stated|"
                r"noted)\s+(?:in|by)\s+the\s+(?:book|blog|author|corpus|"
                r"documents?)\b", re.I), r"\1"),
]


def normalize_claim(text: str) -> str:
    for pat, rep in ATTRIB_SUBS:
        text = pat.sub(rep, text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r",\s*\.", ".", text)
    return text.strip()


def evidence_too_weak(shortlist: list, min_rerank: float = MIN_RERANK) -> bool:
    return not shortlist or max(c["rerank_score"] for c in shortlist) < min_rerank


class NliVerifier:
    def __init__(self, model_name: str = NLI_MODEL):
        self.model = CrossEncoder(model_name)
        id2label = self.model.model.config.id2label
        self.entail_idx = next(i for i, lab in id2label.items()
                               if lab.lower().startswith("entail"))

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        """Best P(entailment) of the hypothesis against each premise
        sentence and each adjacent-sentence pair (max aggregation)."""
        sents = [s.strip() for s in SENT_RE.split(premise) if s.strip()]
        units = sents + [" ".join(p) for p in zip(sents, sents[1:])]
        units = units[:MAX_PREMISE_UNITS] or [premise]
        logits = self.model.predict([(u, hypothesis) for u in units])
        logits = np.atleast_2d(logits)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return float(probs[:, self.entail_idx].max())


def verify(answer_text: str, store: dict, verifier: NliVerifier) -> dict:
    """Sentence-level audit. store: chunk_id -> chunk dict (the shortlist
    the generator saw). Returns {passed, sentences: [...]} where each
    sentence gets a status: supported / unsupported / uncited / skipped."""
    sentences = [s.strip() for s in SENT_RE.split(answer_text) if s.strip()]
    report = []
    for sent in sentences:
        cited = CITE_RE.findall(sent)
        bare = normalize_claim(CITE_RE.sub("", sent))
        if not cited:
            status = "skipped" if len(bare.split()) < 7 else "uncited"
            report.append({"sentence": sent, "citations": [],
                           "status": status})
            continue
        known = [cid for cid in cited if cid in store]
        if not known:
            report.append({"sentence": sent, "citations": cited,
                           "status": "unsupported", "entail_p": 0.0})
            continue
        premise = "\n\n".join(store[cid]["text"] for cid in known)
        p = verifier.entail_prob(premise, bare)
        report.append({"sentence": sent, "citations": cited,
                       "status": "supported" if p >= ENTAIL_MIN
                       else "unsupported", "entail_p": round(p, 3)})
    passed = all(r["status"] in ("supported", "skipped") for r in report)
    return {"passed": passed, "sentences": report}


def selftest():
    """Offline check with a real chunk: a verbatim claim must pass, a
    fabricated one must fail, an uncited one must be flagged."""
    import json
    chunk = None
    for line in open("data/processed/book_chunks.jsonl", encoding="utf-8"):
        c = json.loads(line)
        if c["chunk_id"] == "book/ch06#014":     # the chunking-strategy chunk
            chunk = c
            break
    store = {chunk["chunk_id"]: chunk}
    supported_claim = ("The chunking strategy you use can significantly "
                       "impact the performance of your retrieval system "
                       f"[{chunk['chunk_id']}].")
    fabricated_claim = ("The book requires all chunks to be exactly 123 "
                        f"tokens long with zero overlap [{chunk['chunk_id']}].")
    uncited_claim = ("Retrieval-augmented generation was first invented "
                     "by researchers back in 1985.")
    text = " ".join([supported_claim, fabricated_claim, uncited_claim])

    result = verify(text, store, NliVerifier())
    for r in result["sentences"]:
        p = r.get("entail_p", "-")
        print(f"  [{r['status']:11s}] p={p}  {r['sentence'][:70]}")
    expected = ["supported", "unsupported", "uncited"]
    got = [r["status"] for r in result["sentences"]]
    print("selftest:", "PASS" if got == expected else f"FAIL {got}")


def main():
    from config import cfg as rcfg
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"],
                    default=rcfg("retrieval.mode"))
    ap.add_argument("--top-n", type=int, default=rcfg("retrieval.top_n"))
    ap.add_argument("--keep", type=int, default=rcfg("rerank.keep"))
    ap.add_argument("--min-rerank", type=float,
                    default=rcfg("guardrails.min_rerank"))
    ap.add_argument("--index", default=rcfg("retrieval.index"))
    ap.add_argument("--chunks", nargs="+", default=CHUNKS_DEFAULT)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.query:
        ap.error("query required unless --selftest")

    cfg = load_prompt()
    net = retrieve(a.query, a.index, a.chunks, a.mode, a.top_n, k=a.top_n)
    shortlist = rerank(a.query, net, keep=a.keep)

    if evidence_too_weak(shortlist, a.min_rerank):
        print(cfg["refusal"])
        print("(refused before generation: best rerank score "
              f"{max((c['rerank_score'] for c in shortlist), default=None)})")
        return

    result = answer(a.query, shortlist)
    if result["refused"]:
        print(result["answer"])
        return

    store = {c["chunk_id"]: c["chunk"] for c in shortlist}
    audit = verify(result["answer"], store, NliVerifier())
    if audit["passed"]:
        print(result["answer"])
        print("\n(citation check: all claims supported)")
    else:
        print(cfg["refusal"])
        print("\n(downgraded: unsupported or uncited claims)")
    for r in audit["sentences"]:
        if r["status"] not in ("supported", "skipped"):
            print(f"  [{r['status']}] {r['sentence'][:90]}")


if __name__ == "__main__":
    main()
