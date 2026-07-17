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
  python src/guardrails.py "question" [--mode hybrid] [--keep 8]   full pipeline
  python src/guardrails.py --selftest                              offline check
Deps:  pip install sentence-transformers
"""

import argparse
import re
import sys

import numpy as np
from sentence_transformers import CrossEncoder

from generate import CITE_RE, answer, load_prompt
from rerank import rerank
from retrieve import add_retrieval_args, retrieve

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


def normalize_claim(text):
    """Strip attribution phrasing and tidy the punctuation left behind."""
    for pattern, replacement in ATTRIB_SUBS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r",\s*\.", ".", text)
    return text.strip()


def evidence_too_weak(shortlist, min_rerank=MIN_RERANK):
    """Layer 1: is the best candidate still below the evidence floor?"""
    if not shortlist:
        return True
    best_score = max(candidate["rerank_score"] for candidate in shortlist)
    return best_score < min_rerank


class NliVerifier:
    """Scores whether a premise text entails a claim, using a local
    NLI cross-encoder."""

    def __init__(self, model_name=NLI_MODEL):
        self.model = CrossEncoder(model_name)
        # Find which output column of the 3-class model means
        # "entailment" (the others are contradiction and neutral).
        id2label = self.model.model.config.id2label
        self.entail_idx = None
        for index, label in id2label.items():
            if label.lower().startswith("entail"):
                self.entail_idx = index
                break

    def entail_prob(self, premise, hypothesis):
        """Best P(entailment) of the hypothesis against the premise.

        The premise is split into sentences; the claim is scored
        against each sentence and each adjacent pair, and the best
        score wins (max aggregation).
        """
        sentences = []
        for part in SENT_RE.split(premise):
            if part.strip():
                sentences.append(part.strip())

        units = list(sentences)
        for first, second in zip(sentences, sentences[1:]):
            units.append(first + " " + second)
        units = units[:MAX_PREMISE_UNITS]
        if not units:
            units = [premise]

        pairs = [(unit, hypothesis) for unit in units]
        logits = np.atleast_2d(self.model.predict(pairs))

        # Softmax turns the 3-class logits into probabilities per row.
        exponentials = np.exp(logits)
        probabilities = exponentials / exponentials.sum(axis=1,
                                                        keepdims=True)
        return float(probabilities[:, self.entail_idx].max())


def verify(answer_text, store, verifier):
    """Layer 2: sentence-level audit of a generated answer.

    store maps chunk_id -> chunk dict (the shortlist the generator
    saw). Every sentence gets a status:
      supported    - cited, and the cited chunks entail it
      unsupported  - cited, but the evidence does not back it
      uncited      - a substantive claim with no citation at all
      skipped      - short boilerplate not worth checking
    Returns {"passed": bool, "sentences": [...]}.
    """
    sentences = []
    for part in SENT_RE.split(answer_text):
        if part.strip():
            sentences.append(part.strip())

    report = []
    for sentence in sentences:
        cited_ids = CITE_RE.findall(sentence)
        bare_claim = normalize_claim(CITE_RE.sub("", sentence))

        if not cited_ids:
            is_boilerplate = len(bare_claim.split()) < 7
            status = "skipped" if is_boilerplate else "uncited"
            report.append({"sentence": sentence, "citations": [],
                           "status": status})
            continue

        known_ids = [cid for cid in cited_ids if cid in store]
        if not known_ids:
            report.append({"sentence": sentence, "citations": cited_ids,
                           "status": "unsupported", "entail_p": 0.0})
            continue

        premise = "\n\n".join(store[cid]["text"] for cid in known_ids)
        probability = verifier.entail_prob(premise, bare_claim)
        if probability >= ENTAIL_MIN:
            status = "supported"
        else:
            status = "unsupported"
        report.append({"sentence": sentence, "citations": cited_ids,
                       "status": status,
                       "entail_p": round(probability, 3)})

    passed = all(entry["status"] in ("supported", "skipped")
                 for entry in report)
    return {"passed": passed, "sentences": report}


def selftest():
    """Offline check with a real chunk: a verbatim claim must pass, a
    fabricated one must fail, an uncited one must be flagged."""
    import json
    chunk = None
    with open("data/processed/book_chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["chunk_id"] == "book/ch06#014":
                chunk = record       # the chunking-strategy chunk
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
    for entry in result["sentences"]:
        probability = entry.get("entail_p", "-")
        print(f"  [{entry['status']:11s}] p={probability}  "
              f"{entry['sentence'][:70]}")
    expected = ["supported", "unsupported", "uncited"]
    got = [entry["status"] for entry in result["sentences"]]
    print("selftest:", "PASS" if got == expected else f"FAIL {got}")


def main():
    from config import cfg as rcfg
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--min-rerank", type=float,
                    default=rcfg("guardrails.min_rerank"))
    add_retrieval_args(ap, keep=True)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not a.query:
        ap.error("query required unless --selftest")

    prompt_cfg = load_prompt()

    net = retrieve(a.query, a.index, a.chunks, a.mode, a.top_n, k=a.top_n)
    shortlist = rerank(a.query, net, keep=a.keep)

    if evidence_too_weak(shortlist, a.min_rerank):
        best = max((c["rerank_score"] for c in shortlist), default=None)
        print(prompt_cfg["refusal"])
        print(f"(refused before generation: best rerank score {best})")
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
        print(prompt_cfg["refusal"])
        print("\n(downgraded: unsupported or uncited claims)")
    for entry in audit["sentences"]:
        if entry["status"] not in ("supported", "skipped"):
            print(f"  [{entry['status']}] {entry['sentence'][:90]}")


if __name__ == "__main__":
    main()
