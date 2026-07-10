# Production-Grade RAG System — Complete Project Specification

**Document purpose:** This is the single source of truth for this project. It is written to be self-contained: a reader (human or LLM) with no prior conversation context should be able to understand the goal, architecture, evaluation methodology, and build plan, and give useful help on any part of it.

**Status:** Design complete, implementation not started. Last updated: 2026-07-06.

---

## 1. Project goal

The author is a student / new graduate applying for **AI/ML engineer roles** and is building this as the centerpiece **portfolio project**. It follows Project 1 of Aishwarya Srinivasan's "5 AI Engineer Projects to Build in 2026" guide (ex-Google/Microsoft/IBM, former Head of AI DevRel at Fireworks AI), extended with **Anthropic's contextual retrieval** technique.

**The one-sentence pitch:** a production-grade, citation-enforced RAG system over Chip Huyen's *AI Engineering* book and blog, built on a $0 model stack, with an ablation study that reproduces Anthropic's published contextual-retrieval results on a new domain, and a CI pipeline that fails the build when answer quality regresses.

**The thesis behind it:** the gap between a RAG demo and a production RAG system is enormous, and that gap is where a candidate differentiates. Depth is demonstrated by **measured decisions, not feature count** — every component must justify its existence with a number in an ablation table.

### Non-goals (deliberate scope exclusions, to be stated in the README)
- No web UI until retrieval quality is proven (a chat interface in front of a mediocre retriever is still a mediocre retriever).
- No agents, no GraphRAG (scope discipline beats buzzword count).
- No fine-tuning (that is Project 4 in the source guide's sequence).
- No LangChain/LlamaIndex for pipeline logic (see §9, Design decisions).

---

## 2. Corpus

**Content:** Chip Huyen's *AI Engineering* book (O'Reilly) + her long-form technical blog posts from huyenchip.com/blog (e.g., "Agents" (Jan 2025, adapted from the book), "Common pitfalls when building generative AI applications", "Building A Generative AI Platform", "Building LLM applications for production", "RLHF", "Open challenges in LLM research").

**Why this corpus:**
- Interviewers know the book — it is *the* current book in the field. Answer quality is instantly judgeable, and the meta-signal is strong (a production RAG system built on the book about building production AI systems).
- Writing the golden evaluation set doubles as interview study.
- It genuinely exercises hybrid retrieval: full of exact terms (metric names, technique names) where BM25 beats embeddings, and concepts spanning chapters that require multi-hop retrieval.

**Copyright handling — "public code, private data" pattern:**
- The book text is copyrighted and is **never committed**. It exists only on the author's machine (`data/` is git-ignored).
- Blog posts are **fetched live** by `src/ingest.py` at build time, never committed. Anyone cloning the repo reproduces a working system with one command, no book required.
- CI runs its evaluations against the **blog subset** of the golden set.
- Corpus scale estimate: ~20 long blog posts + ~150–200k-token book ≈ **1,000–2,000 chunks** total.

**Open item:** whether the author owns a digital copy (EPUB/PDF) of the book is unconfirmed. Blog-only changes nothing structurally — the book can be added to the local corpus at any time.

---

## 3. Model stack ($0 total cost — hard constraint)

| Layer | Tool | Rationale |
|---|---|---|
| Embeddings | `nomic-embed-text-v1.5` via sentence-transformers, **local** | $0, no rate limits, deterministic, CI runs without API keys. Chosen over BGE-small because BGE truncates at 512 tokens — a ~650-token chunk + 50–100-token contextual prefix would be silently cut. Nomic handles 8,192 tokens. Requires task prefixes: `search_document:` at index time, `search_query:` at query time |
| Vector store | Weaviate, self-hosted via Docker Compose | Open source, production-grade. Note: Weaviate's embedded mode does not support Windows, so local dev runs it in Docker Desktop; CI runs it as a GitHub Actions service container. Weaviate's built-in hybrid search is deliberately NOT used for the main pipeline (custom BM25+RRF is the depth signal) but serves as a bonus ablation row |
| Lexical search | `rank_bm25` | Simple, pure-Python BM25 |
| Reranker | sentence-transformers **cross-encoder** (e.g., `BAAI/bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`), local | $0, the guide's own suggestion |
| Contextualization LLM | Gemini Flash (free tier) | One call per chunk at index time; free tier absorbs it |
| Generation LLM | Gemini Flash (free tier), temperature 0 | Groq / OpenRouter free tiers as fallback |
| Citation verification | Local NLI cross-encoder (entailment), or Gemini judge | Free runtime guardrail |
| Eval framework | RAGAS (faithfulness etc.) with Gemini as judge + custom retrieval metrics script | RAGAS is purpose-built for RAG evaluation |
| CI | GitHub Actions | Eval gate on every PR; API keys as repo secrets |

Local embeddings are a deliberate depth signal, not just a cost saving: the entire retrieval layer runs offline, and retrieval evals in CI need no API key or quota.

---

## 4. Architecture overview

Three planes. Every stage is a **separable pure function** with JSONL files as interfaces. Artifacts are cached by content hash.

```
OFFLINE INDEXING (runs on corpus change)
  sources (blog fetched live + book local)
    → ingest → chunk (500–800 tok, ~100 overlap, structure-aware)
    → contextualize (LLM prepends 50–100 token situating context per chunk)
    → dual index: [Weaviate vector index] + [BM25 index]   ← both index the CONTEXTUALIZED text

ONLINE QUERY (per request)
  user question
    → vector search (top 75) + BM25 search (top 75), in parallel
    → Reciprocal Rank Fusion (k=60)
    → cross-encoder rerank → keep top 5–10
    → grounded generation (temp 0, versioned citation prompt)
    → citation check (claim-level entailment verification)
    → answer with [chunk_id] citations  |  OR explicit refusal (fixed string)

EVALUATION & CI (every pull request)
  golden set (50–200 verified QA pairs)
    → run_eval.py (retrieval metrics + generation metrics, kept separate)
    → CI gate: build FAILS below thresholds
```

### Cache-invalidation chain (document this in the README)
`chunking change → invalidates contexts → invalidates embeddings → invalidates indexes`. Changing the context prompt re-runs contextualization but not chunking. Never mix old and new vectors in one index — re-index the whole corpus after any upstream change.

---

## 5. Component specifications

### 5.1 Ingest (`src/ingest.py`)
- `BlogLoader`: fetches the post list from huyenchip.com/blog, downloads each post, extracts clean article text (strip nav/boilerplate).
- `BookLoader`: reads a local EPUB/PDF, splits into chapters.
- Output `data/processed/docs.jsonl`, one document per line:
  ```json
  {"doc_id": "blog/agents", "source": "blog", "title": "Agents",
   "url": "https://huyenchip.com/2025/01/07/agents.html", "date": "2025-01-07", "text": "..."}
  ```

### 5.2 Chunk (`src/chunk.py`)
- Target ~650 tokens per chunk (band 500–800), ~100-token overlap between consecutive chunks. Intuition: one chunk ≈ one printed book page (~480 words, 2–4 paragraphs). Chunking is by tokens, never by pages — pages are a layout artifact (blog posts have none; EPUB has no fixed pages).
- **Structure-aware:** split on headings/paragraph boundaries first, then pack paragraphs to target size; never split a table or code block mid-way (the corpus is dense with both). Overlap applies within continuous prose only — overlapping across a section boundary joins two unrelated thoughts. (Overlap exists to prevent slicing an idea at a chunk boundary; structure-awareness prevents damage overlap can't fix.)
- If the book source is PDF, keep page numbers as chunk metadata so citations can render "book, p. 143". EPUB yields chapter/section paths instead.
- Output `chunks.jsonl`:
  ```json
  {"chunk_id": "blog/agents#012", "doc_id": "blog/agents", "text": "...",
   "section_path": "Planning > Tool selection", "char_start": 14203, "char_end": 16891, "token_count": 640}
  ```
- `char_start/char_end` offsets are what allow citations to point at the exact source paragraph.
- **Embedding-window constraint (why the embedder is Nomic, not BGE-small):** chunk (~650 tok) + contextual prefix (50–100 tok) ≈ 750 tokens. BGE-small truncates at 512 and would silently drop the tail of every chunk — contextual retrieval would *hurt* retrieval invisibly. nomic-embed-text-v1.5 (8,192-token window) removes the constraint. Document this decision in the README.
- **Tokenizer roles (all free; do not conflate):** (1) *Measuring ruler* for chunk sizing/stats/budgets: tiktoken `o200k_base` — one ruler used consistently everywhere; exactness doesn't matter, consistency does. (2) *Model tokenizers* (Nomic, reranker, NLI): bundled with each model, applied automatically — always pass raw text. (3) *BM25 terms*: word-level, not subword — lowercase, split on whitespace/punctuation but preserve internal `-._` in technical identifiers (regex `[a-z0-9]+(?:[-._][a-z0-9]+)*`) so exact-term matching on names like `text-embedding-3-small` survives; no stemming.

### 5.3 Contextualize (`src/contextualize.py`) — Anthropic contextual retrieval
- For each chunk: LLM receives the **full document + the chunk** and produces a 50–100 token context situating the chunk within the document ("This chunk is from Chip Huyen's post on agents, in the section about tool-selection failure modes...").
- The context is **prepended to the chunk text for BOTH indexes** (embeddings and BM25). This is the core of the technique — it fixes chunks that are ambiguous out of context ("this approach breaks at scale" — which approach?).
- The context is retrieval fuel only: citations always point at the **original** chunk boundaries (`char_start`/`char_end`), and the context prefix is never presented as evidence.
- Prompt lives in `prompts/contextualize.yaml` (versioned).
- Results cached to disk keyed on `hash(doc_text, chunk_text, prompt_version)` — the LLM cost is paid once per corpus/prompt version. At ~1,000–2,000 chunks this is an evening of rate-limited free-tier calls.
- Anthropic's reference numbers (their metric: **failure rate = % of golden questions whose needed chunk misses the top-20 retrieved**): baseline 5.7% → contextual embeddings 3.7% (−35%) → + contextual BM25 2.9% (−49%) → + reranking 1.9% (−67%). This project reproduces the same ablation on a new domain and compares directly.

### 5.4 Index (`src/index.py`)
- Vector index: Weaviate collection, nomic-embed-text-v1.5 embeddings of contextualized chunks (embeddings computed locally and pushed — Weaviate's own vectorizer modules are not used, keeping the embedding layer keyless and local).
- Weaviate runs from a `docker-compose.yml` in the repo root (local dev) and as a GitHub Actions service container (CI).
- BM25 index: `rank_bm25` over tokenized contextualized chunks, pickled to disk. Custom BM25+RRF is used for the main pipeline instead of Weaviate's built-in hybrid search — owning the fusion logic is the point; Weaviate's native hybrid is kept as a bonus ablation comparison.
- An index manifest records the config hash it was built from (guards against stale-index use).

### 5.5 Retrieve (`src/retrieve.py`)
- Runs vector top-N and BM25 top-N (default N=75 each).
- **Reciprocal Rank Fusion:** each document scores `Σ 1/(k + rank)` across the two ranked lists, k=60. Rank-based fusion sidesteps normalizing BM25's unbounded scores against cosine similarities.
- Output: fused candidate list (~100 candidates max) with per-retriever ranks preserved (needed for tracing and ablations).

### 5.6 Rerank (`src/rerank.py`)
- Cross-encoder scores (query, contextualized chunk) pairs — the model attends to query and chunk **together**, far more accurate than bi-encoder similarity, affordable only on a short list.
- Pattern: wide cheap net first (hybrid retrieval), expensive scoring on ~75–150 candidates, keep top 5–10 for generation (configurable).

### 5.7 Generate (`src/generate.py`)
- Assembles the prompt from `prompts/answer_with_citations.yaml`. Chunks are presented with their IDs; every claim in the answer must cite `[chunk_id]`; temperature 0.
- If context is insufficient, the model must return the **exact fixed refusal string**: `"I don't have enough information in the provided documents to answer this."`
- Example prompt config:
  ```yaml
  version: 1
  name: answer_with_citations
  temperature: 0.0
  system_prompt: |
    Answer ONLY using the provided context chunks below.
    Every claim must cite the chunk ID it came from, e.g. [blog/agents#012].
    If the context does not contain enough information to answer confidently,
    respond exactly with:
    "I don't have enough information in the provided documents to answer this."
    Never use outside knowledge.
  ```

### 5.8 Guardrails (`src/guardrails.py`) — citation enforcement
Two layers, because prompt instructions alone are not enforcement:
1. **Pre-generation refusal:** if no candidate clears a rerank-score threshold, refuse before calling the LLM.
2. **Post-generation verification pass:** extract each claim and its cited chunk; verify entailment (local NLI cross-encoder, or LLM judge); if any claim is unsupported, downgrade the response to the refusal string (or strip the unsupported claim).

The runtime guardrail and the offline faithfulness metric are the same idea measured in two places — one is the live check, the other measures how well the check works.

### 5.9 Response object
Every response carries: `answer_md`, `citations` (chunk_id, quote, source url/chapter, char offsets), and a `trace` dict (per-stage timings, candidate sets, scores). The trace is unused for now but is the instrumentation hook for the observability layer (Project 3 of the guide's sequence) — do not tangle the stages together.

---

## 6. Configuration (`config.yaml`)

Single config file; **the ablation harness is the production path with toggles**, not separate code:

```yaml
corpus:
  sources: [blog, book]          # book only present locally
chunking:
  target_tokens: 650             # within 500-800
  overlap_tokens: 100
contextualize:
  enabled: true                  # ablation toggle
  prompt_version: 1
retrieval:
  mode: hybrid                   # vector | bm25 | hybrid  (ablation toggle)
  top_n_per_retriever: 75
  rrf_k: 60
rerank:
  enabled: true                  # ablation toggle
  model: BAAI/bge-reranker-base
  keep_top: 8
generation:
  model: gemini-flash            # free tier
  temperature: 0.0
  prompt_version: 1
guardrails:
  min_rerank_score: 0.2
  verify_citations: true
```

---

## 7. Evaluation methodology

### 7.1 Golden set (`eval/golden_set.jsonl`)
50–200 manually verified QA pairs across **four deliberately hard categories** (an all-easy golden set catches no real regressions):

| Category | Tests | Example shape |
|---|---|---|
| Single-chunk factual | Baseline retrieval + generation | "What does Chip Huyen say X metric measures?" |
| Multi-hop | Fusion + reranking across documents | Answer requires combining two chapters/posts |
| Exact-term lookup | BM25's contribution (embeddings blur identifiers) | Question about a specific named technique/tool/number |
| Should-refuse | Refusal correctness | Question whose answer is NOT in the corpus |

Each entry:
```json
{"qid": "q042", "question": "...", "expected_answer": "...",
 "evidence_chunk_ids": ["book/ch4#031", "blog/agents#012"],
 "category": "multi-hop", "split": "blog"}
```
`evidence_chunk_ids` let retrieval metrics run **without any LLM**. `split` marks which entries CI can run (blog-only).

### 7.2 Metrics — two families, kept strictly separate (different failure surfaces)

**Retrieval metrics** (no LLM needed; run on every PR over the full golden set):
- **failure-rate@20** — % of questions where a needed evidence chunk is absent from the top-20 retrieved. Kept identical to Anthropic's metric for direct comparison with their published numbers.
- recall@5, recall@10 — is the evidence in what generation actually sees?
- MRR — how high does the first relevant chunk rank?

**Generation metrics** (LLM judge needed; run on blog-subset sample in CI, full set locally):
- **RAGAS faithfulness** — are the claims in the answer supported by the retrieved chunks?
- RAGAS context precision/recall.
- **Refusal correctness** — % of should-refuse questions correctly refused AND % of answerable questions incorrectly refused (both directions matter).

### 7.3 Ablation study (the centerpiece README artifact)
`eval/ablate.py` runs the eval over a config matrix and **generates the README table automatically**:

| Configuration | failure-rate@20 | recall@5 | faithfulness | latency |
|---|---|---|---|---|
| Vector only (baseline) | measure | measure | measure | measure |
| + contextual embeddings | … | … | … | … |
| + contextual BM25 (hybrid, RRF) | … | … | … | … |
| + cross-encoder rerank | … | … | … | … |

Compare each delta against Anthropic's published −35% / −49% / −67% failure-rate reductions. "Anthropic reported 49%; I measured X% on my corpus" is the project's strongest single line.

### 7.4 CI gate (`.github/workflows/rag-eval.yml`)
- Triggers on every pull request.
- Steps: restore cached models/embeddings → build index from fetched blog corpus → run retrieval metrics (full) + generation metrics (sample, `GEMINI_API_KEY` from repo secrets) → **exit non-zero if failure-rate@20 exceeds threshold or faithfulness < 0.85** (thresholds tightened once baselines exist).
- Target: full CI run under ~10 minutes via aggressive caching.
- Prompts and config are versioned alongside code because a prompt change can alter behavior as much as a code change — same review, diff, and rollback path.

---

## 8. Repository structure

```
production-rag/
├── src/
│   ├── ingest.py          # BlogLoader (live fetch) + BookLoader (local)
│   ├── chunk.py           # structure-aware chunking
│   ├── contextualize.py   # Anthropic contextual retrieval, disk-cached
│   ├── index.py           # Weaviate + BM25 index builds, manifest
│   ├── retrieve.py        # hybrid retrieval + RRF
│   ├── rerank.py          # cross-encoder
│   ├── generate.py        # prompt assembly + Gemini call
│   └── guardrails.py      # refusal threshold + citation verification
├── prompts/
│   ├── answer_with_citations.yaml
│   └── contextualize.yaml
├── config.yaml
├── eval/
│   ├── golden_set.jsonl
│   ├── run_eval.py        # retrieval + generation metrics, threshold asserts
│   └── ablate.py          # config matrix → README ablation table
├── tests/                 # pytest on tiny committed fixture corpus; no network, no keys
├── data/                  # GIT-IGNORED: raw/, processed/, indexes/
├── docker-compose.yml     # Weaviate for local dev
├── .github/workflows/rag-eval.yml   # runs Weaviate as a service container
└── README.md
```

---

## 9. Key design decisions (and the reasoning to defend in interviews)

1. **No LangChain/LlamaIndex.** The retrieval logic is ~400 lines of plain Python the author can defend line-by-line. Frameworks hide exactly the parts that demonstrate knowledge. Libraries are used for *models and storage* (sentence-transformers, weaviate-client, rank_bm25, google-genai), never for *logic*.
2. **Config-driven ablations.** The ablation harness and the production pipeline are the same code — a config matrix, not forked scripts. The README table is generated, not typed.
3. **Public code, private data.** Solves the copyrighted-corpus problem with the same pattern real companies use; documented as a feature ("how I handled a corpus I couldn't redistribute"), not hidden as a workaround.
4. **Local embeddings + local reranker.** Retrieval is fully offline: deterministic, free, keyless CI.
5. **Pure-function stages with file interfaces.** Enables unit tests on fixtures, per-stage caching, honest ablations, and drop-in observability later (Project 3).
6. **Retrieval and generation evaluated separately.** If retrieval pulls the wrong chunks, no prompt tweak fixes the answer — they are different failure surfaces and are measured as such.
7. **Refusal is a feature.** Declining to answer without evidence is enforced (pre-generation threshold + post-generation verification), not requested politely in a prompt.

## 10. Known pitfalls being consciously avoided (from the source guide)
- Evaluating only the final answer and never retrieval in isolation.
- Chunking blind to document structure (cutting tables/clauses in half).
- Forgetting to re-index the whole corpus after chunking/embedding changes.
- An all-easy golden set that catches no real regressions.
- Polishing a UI before retrieval quality is solid.

---

## 11. Build plan (3–4 weeks)

| Week | Phase | Deliverables | Done when |
|---|---|---|---|
| 1 | Phase 1 — fundamentals | ingest (blog) → chunk → embed → vector-only retrieval → cited generation; CLI demo; **baseline eval numbers recorded** | Any question in → answer + exact source paragraph out, end to end |
| 2 | Phase 2 — production retrieval | contextualize; BM25 + RRF; reranker; citation enforcement; prompts to versioned YAML; re-measure after each addition | Hybrid + rerank live; system refuses deliberately out-of-scope questions; prompts versioned |
| 3 | Phase 3 — shippable | golden set (50–200 QAs, all four categories); run_eval.py; CI gate; ablate.py + final table | CI fails a deliberately weakened PR |
| 4 | Polish | README (architecture diagram, ablation table, 3–5 worked examples: question / retrieved chunks / cited answer side-by-side, headline eval numbers in plain English, honest limitations section); add book to local corpus | A reviewer understands the system in 10 seconds from the README |

**Demo script for interviews:** ask a real question → show the exact source paragraph the answer cites → show CI failing a deliberately weakened PR because faithfulness dropped. Covers "can you build it" and "do you understand production AI" in under two minutes.

---

## 12. References
- Source guide: "Production-Grade RAG System — Portfolio Project Guide" (based on Aishwarya Srinivasan's video "5 AI Engineer Projects to Build in 2026").
- Anthropic, "Introducing Contextual Retrieval": https://www.anthropic.com/engineering/contextual-retrieval
- Corpus: https://huyenchip.com/blog/ + Chip Huyen, *AI Engineering* (O'Reilly) — local copy only, never redistributed.
- RAGAS: https://docs.ragas.io — RAG evaluation framework.

## 13. Current status and open questions
- Design finalized as specified above; **no code written yet, repo not created**.
- Open: confirm the author owns a digital copy of the *AI Engineering* book (EPUB/PDF). If not, start blog-only — architecture unchanged.
- Author constraint: everything must run on free tiers / local models ($0 total).
- Author preference: repo docs and commits in a plain engineer voice — no AI-style emoji headers, should read human-written.
