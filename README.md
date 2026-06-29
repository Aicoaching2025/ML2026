# LLM Agent + Hybrid RAG Portfolio

Two production-shaped reference projects that together demonstrate agentic
tool-using LLM systems, memory/retrieval, evals, and guardrails — built on the
**Claude API** (`claude-opus-4-8`) with structured tool-calling.

| Dir | Project | What it shows |
|---|---|---|
| [`agent/`](agent/) | Tool-using agent harness | Custom agent loop, context management, permission gating, structured-output guardrails |
| [`rag/`](rag/) | Hybrid-retrieval RAG + evals | Dense + sparse hybrid retrieval (pgvector), reranking, citation-grounded answers with abstention, a measured eval harness |

Both are deliberately small but real: the harness is a stripped-down Claude
Code, and the RAG service is a measured pipeline rather than naive
cosine-similarity-only retrieval.

## Why these two

They cover the full set of skills a modern LLM-engineering role screens for in
one place:

- **Agent frameworks / tool use** → `agent/` is a hand-written tool-call loop.
- **Harness architecture** → context compaction, tool-result truncation,
  permission gating, sub-agent-ready design.
- **Memory / retrieval (RAG, vector DBs, hybrid retrieval)** → `rag/` does dense
  (pgvector) + sparse (Postgres full-text / BM25-style) fused with Reciprocal
  Rank Fusion, then an LLM reranker.
- **LLM evals (offline + online, synthetic data, LLM-as-judge)** → `rag/evals/`
  generates a synthetic QA set and measures retrieval recall@k / MRR and answer
  faithfulness.
- **Guardrails / structured output / hallucination mitigation** →
  `agent/guardrails.py` enforces Pydantic-typed, validated tool/LLM output;
  `rag/generate.py` grounds answers in citations and abstains on weak retrieval.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY (+ VOYAGE_API_KEY / DATABASE_URL for rag)
```

- Agent: `python -m agent.cli "Summarize what this repo does and list the Python files"`
- RAG: see [`rag/README.md`](rag/README.md) for ingest → query → eval.

## Design defaults (and why)

- **LLM: Claude API, `claude-opus-4-8`, structured tool-calling.** Tools are
  forced through JSON schemas; structured extraction uses
  `client.messages.parse()` with Pydantic so the model's output is validated at
  the SDK layer.
- **Vector store: pgvector.** A single Postgres instance holds both the dense
  embedding column (`vector`) and a `tsvector` full-text column, so hybrid
  retrieval is one store and one transaction — no second service to operate.
  (Qdrant is the obvious alternative if you'd rather run a dedicated vector DB;
  the `rag.store` interface is small enough to swap.)
- **Embeddings: pluggable.** Defaults to Voyage AI (`voyage-3`, Anthropic's
  recommended embeddings partner); falls back to a local
  `sentence-transformers` model when `VOYAGE_API_KEY` is unset, so the pipeline
  runs with no extra paid key.

See each subproject's README for details.
