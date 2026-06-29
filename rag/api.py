"""FastAPI service exposing retrieval and grounded generation.

    uvicorn rag.api:app --reload

Endpoints:
  GET  /health
  POST /search   {query}            -> fused + reranked passages (no generation)
  POST /answer   {query}            -> citation-grounded answer with abstention
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .generate import Generator
from .retrieve import Retriever
from .store import Store

app = FastAPI(title="Hybrid RAG", version="0.1.0")

_retriever = Retriever()
_generator = Generator(retriever=_retriever)


class Query(BaseModel):
    query: str
    rerank: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": Store().count()}


@app.post("/search")
def search(q: Query) -> dict:
    hits = _retriever.hybrid(q.query, rerank=q.rerank)
    return {
        "query": q.query,
        "results": [
            {
                "doc_id": h.chunk.doc_id,
                "text": h.chunk.text,
                "rrf_score": round(h.rrf_score, 4),
                "rerank_score": h.rerank_score,
                "metadata": h.chunk.metadata,
            }
            for h in hits
        ],
    }


@app.post("/answer")
def answer(q: Query) -> dict:
    return _generator.answer(q.query)
