"""
Latency flight recorder: time every pipeline stage per query, append
traces to data/logs/queries.jsonl, print P50/P95 per stage.

Real-time applications start from a latency budget; this measures ours
so the README can state one. The first query loads models (embedder,
reranker, NLI) and is logged as warmup=true and excluded from the
percentiles — cold-start is reported separately, once.

Default run is quota-free: retrieval + rerank only (the local stages).
--generate adds cited generation + NLI verification per question,
spending one Gemini call each — run it on fresh quota windows.

Usage:  python eval/measure_latency.py [--mode hybrid] [--keep 8]
            [--generate] [--n 12] [--golden eval/golden_set.jsonl]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rerank import rerank  # noqa: E402
from retrieve import retrieve  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

LOG_PATH = "data/logs/queries.jsonl"


def pct(values, p):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))
    return ordered[idx]


def run_one(query, a, do_generate):
    stages = {}
    t0 = time.perf_counter()
    net = retrieve(query, a.index, a.chunks, a.mode, a.top_n, k=a.top_n)
    stages["retrieve_ms"] = round((time.perf_counter() - t0) * 1000)

    t1 = time.perf_counter()
    shortlist = rerank(query, net, keep=a.keep)
    stages["rerank_ms"] = round((time.perf_counter() - t1) * 1000)

    if do_generate:
        from generate import answer
        from guardrails import NliVerifier, verify
        t2 = time.perf_counter()
        res = answer(query, shortlist)
        stages["generate_ms"] = round((time.perf_counter() - t2) * 1000)
        if not res["refused"]:
            t3 = time.perf_counter()
            store = {c["chunk_id"]: c["chunk"] for c in shortlist}
            verify(res["answer"], store, _verifier())
            stages["verify_ms"] = round((time.perf_counter() - t3) * 1000)

    stages["total_ms"] = round((time.perf_counter() - t0) * 1000)
    return stages


_V = []


def _verifier():
    if not _V:
        from guardrails import NliVerifier
        _V.append(NliVerifier())
    return _V[0]


def main():
    from retrieve import add_retrieval_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_set.jsonl")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--generate", action="store_true",
                    help="include generation + NLI verification (quota)")
    add_retrieval_args(ap, keep=True)
    a = ap.parse_args()

    golden = [json.loads(l) for l in open(a.golden, encoding="utf-8")]
    queries = [q["question"] for q in golden
               if q["category"] != "refusal"][:a.n]

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    rows = []
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        for i, query in enumerate(queries):
            warmup = i == 0
            stages = run_one(query, a, a.generate)
            row = {"ts": datetime.now(timezone.utc).isoformat(),
                   "query": query, "mode": a.mode,
                   "generate": a.generate, "warmup": warmup, **stages}
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            tag = " (warmup, excluded)" if warmup else ""
            print(f"  {stages['total_ms']:6d} ms  {query[:55]}{tag}")

    warm = [r for r in rows if not r["warmup"]]
    print(f"\nwarm queries: {len(warm)}  |  mode={a.mode}  "
          f"generate={a.generate}")
    print(f"cold start (first query, model loads): "
          f"{rows[0]['total_ms']:,} ms")
    stage_keys = [k for k in ("retrieve_ms", "rerank_ms", "generate_ms",
                              "verify_ms", "total_ms")
                  if any(k in r for r in warm)]
    print(f"{'stage':12s} {'P50':>8s} {'P95':>8s} {'mean':>8s}")
    for k in stage_keys:
        vals = [r[k] for r in warm if k in r]
        mean = round(sum(vals) / len(vals))
        print(f"{k:12s} {pct(vals, 50):8,} {pct(vals, 95):8,} {mean:8,}")
    print(f"\nappended {len(rows)} traces -> {LOG_PATH}")


if __name__ == "__main__":
    main()
