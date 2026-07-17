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

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from retrieve import CHUNKS_DEFAULT, retrieve  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

K_EVAL = 20
RECALL_KS = (5, 10, 20)


def eval_question(question, ranked_ids):
    """Score one golden question against a ranked chunk_id list."""
    evidence = set(question["evidence_chunk_ids"])

    # Did every evidence chunk make it into the top 20?
    top_20 = set(ranked_ids[:K_EVAL])
    all_evidence_found = evidence.issubset(top_20)

    # recall@k: what fraction of the evidence is in the first k results?
    recall = {}
    for k in RECALL_KS:
        top_k = set(ranked_ids[:k])
        found = evidence & top_k
        recall[k] = len(found) / len(evidence)

    # Reciprocal rank: 1/position of the first evidence hit (0 if none).
    reciprocal_rank = 0.0
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in evidence:
            reciprocal_rank = 1.0 / position
            break

    return {
        "qid": question["qid"],
        "category": question["category"],
        "failed_at_20": not all_evidence_found,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
    }


def _agg(rows):
    """Average the per-question results into one metrics row."""
    n = len(rows)
    metrics = {"n": n}
    metrics["failure_rate@20"] = sum(r["failed_at_20"] for r in rows) / n
    for k in RECALL_KS:
        metrics[f"recall@{k}"] = sum(r["recall"][k] for r in rows) / n
    metrics["mrr"] = sum(r["reciprocal_rank"] for r in rows) / n
    return metrics


def evaluate(golden_path, mode, use_rerank, top_n=75,
             index_dir="data/index/baseline", chunks=None, split=None,
             rerank_model=None):
    """Run retrieval metrics for one configuration.

    Importable — the ablation harness drives this same function across
    the config matrix. split="blog" restricts to blog-evidence
    questions (what CI can run: the book never leaves the author's
    machine).
    """
    golden = []
    with open(golden_path, encoding="utf-8") as f:
        for line in f:
            golden.append(json.loads(line))

    # Refusal questions have no evidence chunks to find, so retrieval
    # metrics skip them; they get their turn in the generation eval.
    questions = [q for q in golden if q["category"] != "refusal"]
    if split:
        questions = [q for q in questions if q["split"] == split]
    if not questions:
        raise SystemExit(f"no golden questions with split={split!r}")

    rows = []
    for question in questions:
        candidates = retrieve(question["question"], index_dir,
                              chunks or CHUNKS_DEFAULT, mode,
                              top_n, k=top_n)
        if use_rerank:
            from rerank import MODEL_DEFAULT, rerank
            candidates = rerank(question["question"], candidates,
                                rerank_model or MODEL_DEFAULT,
                                keep=K_EVAL)
        ranked_ids = [c["chunk_id"] for c in candidates]
        rows.append(eval_question(question, ranked_ids))

    by_category = {}
    for category in sorted({row["category"] for row in rows}):
        matching = [row for row in rows if row["category"] == category]
        by_category[category] = _agg(matching)

    unverified = sum(1 for q in golden if not q.get("verified"))
    return {
        "config": {"mode": mode, "rerank": use_rerank, "top_n": top_n,
                   "index": index_dir},
        "overall": _agg(rows),
        "by_category": by_category,
        "rows": rows,
        "unverified": unverified,
        "total_golden": len(golden),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_set.jsonl")
    ap.add_argument("--mode", choices=["vector", "bm25", "hybrid"],
                    default="hybrid")
    ap.add_argument("--rerank", action="store_true",
                    help="apply the cross-encoder before measuring")
    ap.add_argument("--rerank-model",
                    help="override the cross-encoder (default MiniLM)")
    ap.add_argument("--top-n", type=int, default=75)
    ap.add_argument("--index", default="data/index/baseline")
    ap.add_argument("--chunks", nargs="+", default=CHUNKS_DEFAULT)
    ap.add_argument("--split", choices=["book", "blog", "mixed"],
                    help="restrict to one evidence split (CI uses blog)")
    ap.add_argument("--max-failure", type=float,
                    help="CI gate: exit 1 if failure-rate@20 exceeds this")
    ap.add_argument("--json-out")
    a = ap.parse_args()

    result = evaluate(a.golden, a.mode, a.rerank, a.top_n, a.index,
                      a.chunks, a.split, a.rerank_model)
    if result["unverified"]:
        print(f"WARNING: {result['unverified']}/{result['total_golden']} "
              f"golden entries are not yet manually verified — numbers "
              f"are provisional\n")
    overall, by_cat, rows = (result["overall"], result["by_category"],
                             result["rows"])

    print(f"config: {result['config']}")
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
            json.dump(result, f, indent=2)
        print(f"wrote {a.json_out}")

    if a.max_failure is not None and \
            overall["failure_rate@20"] > a.max_failure:
        print(f"GATE FAILED: failure-rate@20 "
              f"{overall['failure_rate@20']:.0%} > {a.max_failure:.0%}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
