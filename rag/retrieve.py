"""Hybrid retrieval: dense + sparse, fused with RRF, then an LLM reranker.

Pipeline:
  1. dense_search  (semantic, pgvector cosine)   -> candidate_k results
  2. sparse_search (lexical, Postgres full-text)  -> candidate_k results
  3. Reciprocal Rank Fusion merges the two ranked lists into one
  4. an LLM reranker (Claude, structured output) reorders the top rerank_k
  5. keep final_k

The reranker is the differentiator: most demos stop at cosine similarity. RRF
gives a robust fusion that needs no score calibration between the two
retrievers, and the LLM rerank pass catches relevance the embeddings miss.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
from pydantic import BaseModel, Field

from .config import CONFIG
from .embeddings import get_embedder
from .store import Chunk, Store


@dataclass
class Retrieved:
    chunk: Chunk
    rrf_score: float
    rerank_score: float | None = None


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]], k: int = CONFIG.rrf_k
) -> list[tuple[Chunk, float]]:
    """Fuse multiple ranked lists. RRF score = sum(1 / (k + rank)).

    Rank-based, so it needs no score normalization between retrievers.
    """
    scores: dict[int, float] = {}
    by_id: dict[int, Chunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            by_id[chunk.id] = chunk
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(by_id[cid], score) for cid, score in fused]


class _RerankItem(BaseModel):
    index: int = Field(description="0-based index of the candidate.")
    relevance: float = Field(ge=0.0, le=1.0, description="Relevance to the query, 0-1.")


class _RerankResult(BaseModel):
    rankings: list[_RerankItem]


class Retriever:
    def __init__(self, store: Store | None = None, client: anthropic.Anthropic | None = None):
        self.store = store or Store()
        self.embedder = get_embedder()
        self.client = client or anthropic.Anthropic()

    def hybrid(self, query: str, rerank: bool = True) -> list[Retrieved]:
        q_emb = self.embedder.embed([query], input_type="query")[0]
        dense = self.store.dense_search(q_emb, CONFIG.candidate_k)
        sparse = self.store.sparse_search(query, CONFIG.candidate_k)

        fused = reciprocal_rank_fusion([dense, sparse])
        results = [Retrieved(chunk=c, rrf_score=s) for c, s in fused]

        if not rerank or not results:
            return results[: CONFIG.final_k]

        top = results[: CONFIG.rerank_k]
        reranked = self._llm_rerank(query, top)
        return reranked[: CONFIG.final_k]

    def _llm_rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        """Score each candidate's relevance with Claude (structured output)."""
        listing = "\n\n".join(
            f"[{i}] {r.chunk.text[:600]}" for i, r in enumerate(candidates)
        )
        try:
            resp = self.client.messages.parse(
                model=CONFIG.anthropic_model,
                max_tokens=1_500,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Rate how well each candidate passage answers the "
                            f"query. Query: {query!r}\n\nCandidates:\n{listing}"
                        ),
                    }
                ],
                output_format=_RerankResult,
            )
            scores = {item.index: item.relevance for item in resp.parsed_output.rankings}
        except Exception:
            # Reranker is best-effort; fall back to RRF order on any failure.
            return candidates

        for i, r in enumerate(candidates):
            r.rerank_score = scores.get(i, 0.0)
        return sorted(candidates, key=lambda r: r.rerank_score or 0.0, reverse=True)
