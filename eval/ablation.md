# Retrieval ablation

Golden set: 11 questions (11 still unverified — numbers provisional until manual verification is complete).
Metrics follow Anthropic's contextual-retrieval definitions; failure-rate@20 is directly comparable to their published baseline 5.7% → 1.9% progression.

| index | mode | rerank | failure@20 | recall@5 | recall@20 | MRR |
|---|---|---|---|---|---|---|
| baseline | bm25 | no | 0% | 78% | 100% | 0.571 |
| baseline | bm25 | yes | 0% | 94% | 100% | 0.889 |
| baseline | vector | no | 0% | 72% | 100% | 0.681 |
| baseline | vector | yes | 0% | 94% | 100% | 0.889 |
| baseline | hybrid | no | 0% | 94% | 100% | 0.800 |
| baseline | hybrid | yes | 0% | 94% | 100% | 0.889 |
| contextual | — | — | _pending: index not built_ | | | |
