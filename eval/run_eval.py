"""
Offline retrieval evaluation against the golden set (spec §7.2).

Retrieval metrics only — they need no LLM, no API key, and no quota, so
they run on every invocation and (later) every CI pull request:

  failure-rate@20  % of questions where ANY evidence chunk is missing
                   from the top-20. Kept definitionally identical to
                   Anthropic's contextual-retrieval metric so our
                   ablation numbers compare directly to their published
                   35% / 49% / 67% reductions.
  recall@k         mean fraction of evidence chunks present in top-k.
  MRR              mean reciprocal rank of the first evidence hit.

Generation metrics (faithfulness, refusal correctness) are a separate
quota-spending script — different failure surface, measured separately.

Refusal-category questions are skipped here: no evidence chunks to
find. They get their turn in the generation eval.

Usage:  python eval/run_eval.py [--mode bm25|vector|hybrid] [--rerank]
            [--golden eval/golden_set.jsonl] [--top-n 75]
            [--index data/index/baseline] [--json-out results.json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieve import CHUNKS_DEFAULT, retrieve  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

K_EVAL = 20
RECALL_KS = (5, 10, 20)


def eval_question(q: dict, ranked_ids: list) -> dict:
    evidence = set(q["evidence_chunk_ids"])
    top = {k: ranked_ids[:k] for k in RECALL_KS}
    found_at_20 = evidence.issubset(set(top[K_EVAL]))
    first_hit = next((i + 1 for i, cid in enumerate(ranked_ids)
                      if cid in evidence), None)
    return {
        "qid": q["qid"],
        "category": q["category"],
        "failed_at_20": not found_at_20,
        "recall": {k: len(evidence & set(top[k])) / len(evidence)
                   for k in RECALL_KS},
        "reciprocal_rank": 1.0 / first_hit if first_hit else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_set.jsonl")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"],
                    default="hybrid")
    ap.add_argument("--rerank", action="store_true",
                    help="apply the cross-encoder before measuring")
    ap.add_argument("--top-n", type=int, default=75)
    ap.add_argument("--index", default="data/index/baseline")
    ap.add_argument("--chunks", nargs="+", default=CHUNKS_DEFAULT)
    ap.add_argument("--json-out")
    a = ap.parse_args()

    golden = [json.loads(l) for l in open(a.golden, encoding="utf-8")]
    questions = [q for q in golden if q["category"] != "refusal"]
    unverified = sum(1 for q in golden if not q.get("verified"))
    if unverified:
        print(f"WARNING: {unverified}/{len(golden)} golden entries are "
              f"not yet manually verified — numbers are provisional\n")

    rows = []
    for q in questions:
        candidates = retrieve(q["question"], a.index, a.chunks,
                              a.mode, a.top_n, k=a.top_n)
        if a.rerank:
            from rerank import rerank
            candidates = rerank(q["question"], candidates, keep=K_EVAL)
        ranked_ids = [c["chunk_id"] for c in candidates]
        rows.append(eval_question(q, ranked_ids))

    def agg(subset):
        n = len(subset)
        return {
            "n": n,
            "failure_rate@20": sum(r["failed_at_20"] for r in subset) / n,
            **{f"recall@{k}": sum(r["recall"][k] for r in subset) / n
               for k in RECALL_KS},
            "mrr": sum(r["reciprocal_rank"] for r in subset) / n,
        }

    config = {"mode": a.mode, "rerank": a.rerank, "top_n": a.top_n,
              "index": a.index}
    overall = agg(rows)
    by_cat = {cat: agg([r for r in rows if r["category"] == cat])
              for cat in sorted({r["category"] for r in rows})}

    print(f"config: {config}")
    print(f"overall (n={overall['n']}):  "
          f"failure@20={overall['failure_rate@20']:.0%}  "
          + "  ".join(f"recall@{k}={overall[f'recall@{k}']:.0%}"
                      for k in RECALL_KS)
          + f"  mrr={overall['mrr']:.3f}")
    for cat, m in by_cat.items():
        print(f"  {cat:10s} (n={m['n']}):  "
              f"failure@20={m['failure_rate@20']:.0%}  "
              f"recall@5={m['recall@5']:.0%}  mrr={m['mrr']:.3f}")
    failed = [r["qid"] for r in rows if r["failed_at_20"]]
    if failed:
        print(f"failed@20: {failed}")

    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump({"config": config, "overall": overall,
                       "by_category": by_cat, "rows": rows}, f, indent=2)
        print(f"wrote {a.json_out}")


if __name__ == "__main__":
    main()
