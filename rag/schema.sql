-- Reference schema for the hybrid chunk store.
-- `rag.store.Store.init_schema()` creates this programmatically with the
-- configured EMBED_DIM; this file documents the shape. Replace 1024 with your
-- embedder's dimension if you apply it by hand.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id        bigserial PRIMARY KEY,
    doc_id    text NOT NULL,
    text      text NOT NULL,
    metadata  jsonb NOT NULL DEFAULT '{}',
    embedding vector(1024),                       -- dense / semantic
    ts        tsvector GENERATED ALWAYS AS        -- sparse / lexical
                  (to_tsvector('english', text)) STORED
);

-- Dense ANN index (cosine). Tune `lists` to ~sqrt(rows).
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Sparse full-text index.
CREATE INDEX IF NOT EXISTS chunks_ts_idx ON chunks USING gin(ts);
