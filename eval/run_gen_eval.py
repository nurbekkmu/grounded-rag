"""
Generation-quality evaluation: refusal correctness + faithfulness.

Spends Gemini quota (one generation call per golden question) — run it
on fresh quota windows, not in CI. Retrieval metrics live in run_eval.py.

Metrics:
  refusal_recall       refusal-category questions correctly refused.
  false_refusal_rate   answerable questions wrongly refused — over-
                       refusing destroys trust too; both directions count.
  faithfulness         mean fraction of answer sentences entailed by the
                       chunks they cite, scored by the SAME NLI verifier
                       the runtime guardrail uses (one idea, two places:
                       live enforcement there, offline measurement here).
  fabricated_citations total citations of chunk ids never provided.

RAGAS judge-based metrics are a planned complement (needs an LLM judge
and more quota); the NLI scorer is the quota-free workhorse.

Usage:  python eval/run_gen_eval.py [--mode hybrid] [--keep 8]
            [--split blog] [--json-out eval/gen_results.json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from generate import answer, load_prompt  # noqa: E402
from guardrails import NliVerifier, verify  # noqa: E402
from rerank import rerank  # noqa: E402
from retrieve import CHUNKS_DEFAULT, retrieve  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def faithfulness(audit: dict) -> float:
    """Fraction of substantive sentences that are supported."""
    counted = [s for s in audit["sentences"] if s["status"] != "skipped"]
    if not counted:
        return 1.0
    return sum(s["status"] == "supported" for s in counted) / len(counted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_set.jsonl")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"],
                    default="hybrid")
    ap.add_argument("--top-n", type=int, default=75)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--split", choices=["book", "blog", "mixed"])
    ap.add_argument("--index", default="data/index/baseline")
    ap.add_argument("--chunks", nargs="+", default=CHUNKS_DEFAULT)
    ap.add_argument("--json-out")
    a = ap.parse_args()

    golden = [json.loads(l) for l in open(a.golden, encoding="utf-8")]
    if a.split:
        golden = [q for q in golden if q["split"] == a.split]

    cfg = load_prompt()
    verifier = NliVerifier()
    rows = []
    for q in golden:
        net = retrieve(q["question"], a.index, a.chunks, a.mode,
                       a.top_n, k=a.top_n)
        shortlist = rerank(q["question"], net, keep=a.keep)
        res = answer(q["question"], shortlist)
        row = {"qid": q["qid"], "category": q["category"],
               "refused": res["refused"],
               "fabricated": len(res["fabricated_citations"])}
        if q["category"] == "refusal":
            row["correct_refusal"] = res["refused"]
        elif not res["refused"]:
            store = {c["chunk_id"]: c["chunk"] for c in shortlist}
            audit = verify(res["answer"], store, verifier)
            row["faithfulness"] = faithfulness(audit)
        rows.append(row)
        print(f"  {q['qid']} [{q['category']:10s}] "
              f"refused={res['refused']!s:5s} "
              f"faith={row.get('faithfulness', '-')} "
              f"fabricated={row['fabricated']}")

    refusal_rows = [r for r in rows if r["category"] == "refusal"]
    answerable = [r for r in rows if r["category"] != "refusal"]
    faith_rows = [r for r in answerable if "faithfulness" in r]
    summary = {
        "n": len(rows),
        "refusal_recall": (sum(r["correct_refusal"] for r in refusal_rows)
                           / len(refusal_rows)) if refusal_rows else None,
        "false_refusal_rate": (sum(r["refused"] for r in answerable)
                               / len(answerable)) if answerable else None,
        "faithfulness": (sum(r["faithfulness"] for r in faith_rows)
                         / len(faith_rows)) if faith_rows else None,
        "fabricated_citations": sum(r["fabricated"] for r in rows),
        "config": {"mode": a.mode, "keep": a.keep,
                   "prompt_version": cfg["version"], "model": cfg["model"]},
    }
    print()
    print(json.dumps(summary, indent=2))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2)
        print(f"wrote {a.json_out}")


if __name__ == "__main__":
    main()
