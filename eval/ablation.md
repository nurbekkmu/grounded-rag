# Retrieval ablation

Golden set: 19 questions (0 still unverified — numbers provisional until manual verification is complete).
Metrics follow Anthropic's contextual-retrieval definitions; failure-rate@20 is directly comparable to their published baseline 5.7% → 1.9% progression.

| index | mode | rerank | failure@20 | recall@5 | recall@20 | MRR |
|---|---|---|---|---|---|---|
| baseline | bm25 | no | 13% | 67% | 87% | 0.567 |
| baseline | bm25 | yes | 13% | 80% | 90% | 0.697 |
| baseline | vector | no | 20% | 67% | 83% | 0.535 |
| baseline | vector | yes | 13% | 80% | 87% | 0.689 |
| baseline | hybrid | no | 13% | 80% | 87% | 0.702 |
| baseline | hybrid | yes | 20% | 80% | 83% | 0.689 |
| contextual | bm25 | no | 13% | 73% | 87% | 0.632 |
| contextual | bm25 | yes | 13% | 80% | 90% | 0.697 |
| contextual | vector | no | 13% | 67% | 87% | 0.582 |
| contextual | vector | yes | 13% | 80% | 87% | 0.689 |
| contextual | hybrid | no | 13% | 77% | 87% | 0.633 |
| contextual | hybrid | yes | 20% | 80% | 83% | 0.689 |
