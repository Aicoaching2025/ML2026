"""Ingestion: chunk documents, embed, and index into the hybrid store.

    python -m rag.ingest path/to/docs/            # a directory of .txt/.md
    python -m rag.ingest --reset path/to/file.md  # wipe and re-ingest

Chunking is simple paragraph-packing with overlap — adequate for prose/docs.
Swap in a structure-aware splitter for code or HTML.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .embeddings import get_embedder
from .store import Store

CHUNK_CHARS = 1_200
OVERLAP_CHARS = 200


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Pack paragraphs into ~`size`-char chunks with `overlap` carryover."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if buf and len(buf) + len(para) + 2 > size:
            chunks.append(buf)
            buf = buf[-overlap:] + "\n\n" + para if overlap else para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def iter_files(path: Path):
    if path.is_file():
        yield path
    else:
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in {".txt", ".md", ".markdown"} and p.is_file():
                yield p


def ingest(path: str, reset: bool = False, batch: int = 64) -> int:
    store = Store()
    store.init_schema()
    if reset:
        store.clear()
    embedder = get_embedder()

    total = 0
    for file in iter_files(Path(path)):
        text = file.read_text(errors="replace")
        chunks = chunk_text(text)
        if not chunks:
            continue
        for i in range(0, len(chunks), batch):
            window = chunks[i : i + batch]
            embeddings = embedder.embed(window, input_type="document")
            metas = [
                {"source": str(file), "chunk_index": i + j} for j in range(len(window))
            ]
            store.insert(doc_id=str(file), texts=window, embeddings=embeddings, metadatas=metas)
        total += len(chunks)
        print(f"  ingested {len(chunks):4d} chunks from {file}")

    print(f"done. {total} chunks; store now holds {store.count()}.")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest docs into the hybrid store")
    parser.add_argument("path", help="File or directory of .txt/.md docs")
    parser.add_argument("--reset", action="store_true", help="Truncate before ingesting")
    args = parser.parse_args(argv)
    if not Path(args.path).exists():
        print(f"error: no such path: {args.path}", file=sys.stderr)
        return 2
    ingest(args.path, reset=args.reset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
