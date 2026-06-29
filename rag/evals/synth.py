"""Generate a synthetic QA eval set from the ingested corpus.

For a sample of chunks, ask Claude (structured output) to write a question that
the chunk answers. The chunk's id becomes the gold passage for that question, so
we can measure retrieval recall@k / MRR without hand-labeling.

    python -m rag.evals.synth --n 50 -o rag/evals/out/qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from ..config import CONFIG
from ..store import Store


class _QA(BaseModel):
    question: str = Field(description="A specific question answerable from the passage.")
    answer: str = Field(description="The short ground-truth answer.")
    answerable: bool = Field(
        description="False if the passage is too sparse to form a good question."
    )


def _sample_chunks(store: Store, n: int) -> list[tuple[int, str, str]]:
    with store._connect() as conn:
        rows = conn.execute(
            "SELECT id, doc_id, text FROM chunks "
            "WHERE length(text) > 200 ORDER BY random() LIMIT %s",
            (n,),
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def generate(n: int, out_path: str) -> int:
    store = Store()
    client = anthropic.Anthropic()
    samples = _sample_chunks(store, n)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w") as f:
        for chunk_id, doc_id, text in samples:
            try:
                resp = client.messages.parse(
                    model=CONFIG.anthropic_model,
                    max_tokens=600,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Write one specific question that this passage "
                                "answers, plus the short ground-truth answer. "
                                "Avoid pronouns; the question must stand alone.\n\n"
                                f"Passage:\n{text[:1500]}"
                            ),
                        }
                    ],
                    output_format=_QA,
                )
                qa: _QA = resp.parsed_output
            except Exception as e:
                print(f"  skip chunk {chunk_id}: {e}")
                continue
            if not qa.answerable:
                continue
            f.write(
                json.dumps(
                    {
                        "question": qa.question,
                        "answer": qa.answer,
                        "gold_chunk_id": chunk_id,
                        "gold_doc_id": doc_id,
                    }
                )
                + "\n"
            )
            written += 1
    print(f"wrote {written} QA pairs to {out_path}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic QA eval set")
    parser.add_argument("--n", type=int, default=50, help="Number of chunks to sample")
    parser.add_argument("-o", "--out", default="rag/evals/out/qa.jsonl")
    args = parser.parse_args(argv)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY not set")
        return 2
    generate(args.n, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
