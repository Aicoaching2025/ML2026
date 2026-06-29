# Hybrid RAG + evals

A retrieval/memory service over a corpus, with a **measured** eval harness. The
point is to go past naive cosine-similarity RAG: dense + sparse **hybrid**
retrieval, an LLM reranker, citation-grounded answers that **abstain** on weak
evidence, and metrics that show each piece earns its place.

## Architecture

```
docs ─► ingest (chunk + embed) ─► pgvector store
                                    ├─ embedding vector(D)   (dense / semantic)
                                    └─ ts tsvector           (sparse / lexical)

query ─► dense_search ┐
        sparse_search ┴─► RRF fusion ─► LLM rerank ─► top-k ─► grounded answer
                                                                 (cite or abstain)
```

| File | Role |
|---|---|
| `embeddings.py` | Pluggable embedder: Voyage `voyage-3` (default) or local sentence-transformers |
| `store.py` | pgvector hybrid store — dense `vector` + sparse `tsvector` in one table |
| `ingest.py` | Chunk → embed → index |
| `retrieve.py` | Dense + sparse → **Reciprocal Rank Fusion** → **LLM reranker** |
| `generate.py` | Citation-grounded answer with **abstention** on weak retrieval |
| `api.py` | FastAPI: `/search`, `/answer`, `/health` |
| `evals/` | Synthetic QA generation + retrieval/faithfulness metrics |

## Setup

```bash
# 1. Postgres with pgvector. Easiest:
docker run -d --name rag-pg -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag pgvector/pgvector:pg16

# 2. Env (in repo root .env)
#    ANTHROPIC_API_KEY=...        (required)
#    DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag
#    VOYAGE_API_KEY=...  EMBED_DIM=1024     (Voyage)  — OR —
#    leave VOYAGE_API_KEY empty + EMBED_DIM=384       (local MiniLM fallback)
```

> **Embedding dim must match the embedder.** `voyage-3` → `EMBED_DIM=1024`;
> `all-MiniLM-L6-v2` → `EMBED_DIM=384`. `get_embedder()` errors loudly on a
> mismatch.

## Use

```bash
# Ingest a corpus (.txt/.md). The repo's own docs make a fine demo corpus:
python -m rag.ingest --reset .

# Ask (citation-grounded, abstains when evidence is weak):
python -c "from rag.generate import Generator; import json; \
  print(json.dumps(Generator().answer('What does the agent harness do?'), indent=2))"

# Or serve it:
uvicorn rag.api:app --reload
# POST /answer  {"query": "..."}
```

## Evaluate

The eval harness is the differentiator — most RAG demos show retrieval; few show
*measured* retrieval.

```bash
# 1. Generate a synthetic QA set from the ingested corpus (gold passage = source chunk)
python -m rag.evals.synth --n 40 -o rag/evals/out/qa.jsonl

# 2. Score retrieval (recall@1/5, MRR — with vs without reranker) + answer
#    faithfulness/correctness (LLM-as-judge)
python -m rag.evals.run_eval rag/evals/out/qa.jsonl
```

Sample output shape:

```json
{
  "n": 40,
  "retrieval_no_rerank":   {"recall@1": 0.62, "recall@5": 0.88, "mrr": 0.71},
  "retrieval_with_rerank": {"recall@1": 0.78, "recall@5": 0.90, "mrr": 0.83},
  "generation": {"faithful": 0.95, "correct": 0.85, "abstained": 0.10}
}
```

The `no_rerank` vs `with_rerank` columns are the A/B that justifies the
reranker's latency; `faithful` and `abstained` are the hallucination-mitigation
signal.

## Design notes

- **Why RRF for fusion.** Reciprocal Rank Fusion combines the dense and sparse
  rankings by rank, not score, so it needs no calibration between two retrievers
  whose scores aren't comparable. Robust and parameter-light.
- **Why an LLM reranker.** Embeddings miss relevance that reading the passage
  catches. The rerank pass (`retrieve._llm_rerank`, structured output) reorders
  the fused top-k and is the single biggest recall@1 lever — measured above.
- **Why abstention.** `generate.py` refuses to answer when the best reranked
  score is below a floor, and the prompt forbids using non-passage knowledge.
  Grounded-or-silent beats confidently-wrong.
- **Online evals.** The same `Judge` runs on production traffic (sample a % of
  live answers, judge faithfulness, alert on drops) — the offline harness here
  is the same machinery pointed at a synthetic set.
