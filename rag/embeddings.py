"""Pluggable embedding backend.

Defaults to Voyage AI (`voyage-3`, Anthropic's recommended embeddings partner)
when VOYAGE_API_KEY is set; otherwise falls back to a local
sentence-transformers model so the pipeline runs with no extra paid key.

The two backends produce different dimensions (voyage-3=1024, MiniLM=384), so
keep EMBED_DIM in .env in sync with whichever is active.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Protocol

from .config import CONFIG

InputType = Literal["query", "document"]


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]: ...


class VoyageEmbedder:
    def __init__(self) -> None:
        import voyageai

        self._client = voyageai.Client(api_key=CONFIG.voyage_api_key)
        self.model = CONFIG.voyage_model
        self.dim = CONFIG.embed_dim

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        # Voyage distinguishes query vs document embeddings for retrieval quality.
        result = self._client.embed(texts, model=self.model, input_type=input_type)
        return result.embeddings


class LocalEmbedder:
    """sentence-transformers fallback. No query/document distinction."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(CONFIG.local_embed_model)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if CONFIG.voyage_api_key:
        emb = VoyageEmbedder()
        backend = f"voyage:{CONFIG.voyage_model}"
    else:
        emb = LocalEmbedder()
        backend = f"local:{CONFIG.local_embed_model}"
    if emb.dim != CONFIG.embed_dim:
        raise RuntimeError(
            f"EMBED_DIM={CONFIG.embed_dim} but {backend} produces {emb.dim}-dim "
            f"vectors. Set EMBED_DIM={emb.dim} in .env."
        )
    return emb
