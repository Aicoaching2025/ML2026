"""pgvector-backed hybrid store: dense vectors + sparse full-text in one table.

A single Postgres table holds:
- `embedding vector(EMBED_DIM)` for dense / semantic search (cosine distance)
- `ts tsvector` (generated from the chunk text) for sparse / lexical search

Keeping both columns in one row means hybrid retrieval is one store and one
connection — no second service to operate. (Swap this module for a Qdrant
client if you'd rather run a dedicated vector DB; the public methods below are
the whole interface the rest of the package depends on.)
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from .config import CONFIG


@dataclass
class Chunk:
    id: int
    doc_id: str
    text: str
    metadata: dict
    score: float = 0.0  # populated by retrievers


class Store:
    def __init__(self, dsn: str = CONFIG.database_url, dim: int = CONFIG.embed_dim):
        self.dsn = dsn
        self.dim = dim

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id        bigserial PRIMARY KEY,
                    doc_id    text NOT NULL,
                    text      text NOT NULL,
                    metadata  jsonb NOT NULL DEFAULT '{{}}',
                    embedding vector({self.dim}),
                    ts        tsvector GENERATED ALWAYS AS
                                  (to_tsvector('english', text)) STORED
                )
                """
            )
            # IVFFlat for dense ANN (cosine); GIN for the sparse full-text index.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                "ON chunks USING ivfflat (embedding vector_cosine_ops) "
                "WITH (lists = 100)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_ts_idx ON chunks USING gin(ts)")

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("TRUNCATE chunks RESTART IDENTITY")

    def insert(
        self, doc_id: str, texts: list[str], embeddings: list[list[float]], metadatas: list[dict]
    ) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (doc_id, text, metadata, embedding) "
                "VALUES (%s, %s, %s, %s)",
                [
                    (doc_id, t, psycopg.types.json.Jsonb(m), e)
                    for t, e, m in zip(texts, embeddings, metadatas)
                ],
            )
        return len(texts)

    # --- retrievers ---------------------------------------------------------

    def dense_search(self, query_embedding: list[float], k: int) -> list[Chunk]:
        """Semantic search by cosine distance (lower distance = better)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, doc_id, text, metadata, "
                "1 - (embedding <=> %s) AS score "
                "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (query_embedding, query_embedding, k),
            ).fetchall()
        return [Chunk(r[0], r[1], r[2], r[3], float(r[4])) for r in rows]

    def sparse_search(self, query: str, k: int) -> list[Chunk]:
        """Lexical search via Postgres full-text ranking (BM25-style ts_rank)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, doc_id, text, metadata, "
                "ts_rank(ts, plainto_tsquery('english', %s)) AS score "
                "FROM chunks WHERE ts @@ plainto_tsquery('english', %s) "
                "ORDER BY score DESC LIMIT %s",
                (query, query, k),
            ).fetchall()
        return [Chunk(r[0], r[1], r[2], r[3], float(r[4])) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
