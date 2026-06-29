"""Central configuration for the RAG service, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    anthropic_model: str = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")
    database_url: str = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag"
    )
    # Embeddings
    voyage_api_key: str | None = os.environ.get("VOYAGE_API_KEY") or None
    voyage_model: str = os.environ.get("VOYAGE_MODEL", "voyage-3")
    local_embed_model: str = os.environ.get(
        "LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embed_dim: int = int(os.environ.get("EMBED_DIM", "1024"))

    # Retrieval knobs
    candidate_k: int = 30  # how many to pull from each retriever before fusion
    rerank_k: int = 8  # how many fused candidates to send to the reranker
    final_k: int = 5  # how many to keep after rerank
    rrf_k: int = 60  # RRF damping constant


CONFIG = Config()
