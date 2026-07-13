# Contextual retrieval: book-only comparison

Early comparison run before the blog half of the contextual corpus was
generated: both indexes cover exactly the 449 book chunks, golden set
restricted to book-evidence questions (n=12, hardened set including
paraphrase and cross-chapter multi-hop; entries not yet fully
manually verified — treat as provisional).

| index | mode | rerank | failure@20 | recall@5 | MRR |
|---|---|---|---|---|---|
| baseline | bm25 | no | 8% | 75% | 0.704 |
| baseline | vector | no | **17%** | 79% | 0.564 |
| baseline | hybrid | no | 8% | 88% | 0.819 |
| baseline | hybrid | yes | 8% | 88% | 0.694 |
| contextual | bm25 | no | 8% | **83%** | 0.632 |
| contextual | vector | no | **8%** | 79% | 0.628 |
| contextual | hybrid | no | 8% | 88% | 0.736 |
| contextual | hybrid | yes | 8% | 88% | 0.694 |

## Findings

1. **Contextual embeddings halved the vector arm's failure rate**
   (17% -> 8%). The rescued question is q017, the cross-chapter
   KV-cache multi-hop — precisely the ambiguous-chunk case the
   technique targets. Anthropic reported a 35% relative reduction for
   contextual embeddings alone; we measure ~53% relative on a far
   smaller and harder set. Directionally consistent.
2. **Contextual BM25 lifted lexical recall@5 by 8 points** (75% -> 83%):
   context prefixes add matchable vocabulary to chunks.
3. **Hybrid fusion masks part of the gain here**: baseline-hybrid
   already catches q017 through its BM25 arm, so the fused failure
   rate is 8% either way. On this corpus the arms cover each other;
   contextualization mainly strengthens each arm individually.
4. **q014 fails all eight configurations** ("sampling several responses
   and keeping the most consistent one" -> self-consistency). The
   anchored evidence chunk mentions self-consistency only in a
   footnote. Either the evidence anchor should point elsewhere
   (verification task) or this is a genuine retrieval hole. Marked for
   manual review.
5. **Reranking lowered MRR on the harder set** (hybrid 0.819 -> 0.694
   baseline). The MiniLM reranker demotes some correct top-1 evidence
   on paraphrase questions — the first measured argument for testing a
   stronger reranker via the --model flag.

Regenerate: the eight runs are `eval/run_eval.py --split book` against
`data/index/book-baseline` and `data/index/book-contextual`.
