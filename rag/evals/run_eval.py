"""Run the eval harness over a synthetic QA set.

Measures, per query and aggregated:
- retrieval: recall@1/5, MRR  (with vs without the LLM reranker — the A/B that
  shows the reranker earns its latency)
- generation: faithfulness + correctness (LLM-as-judge)

    python -m rag.evals.run_eval rag/evals/out/qa.jsonl
    python -m rag.evals.run_eval rag/evals/out/qa.jsonl --no-judge   # retrieval only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..generate import Generator
from ..retrieve import Retriever
from . import metrics
from .judge import Judge


def _ids(hits) -> list[int]:
    return [h.chunk.id for h in hits]


def run(qa_path: str, judge_answers: bool = True) -> dict:
    rows = [json.loads(line) for line in Path(qa_path).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no QA rows in {qa_path}")

    retriever = Retriever()
    generator = Generator(retriever=retriever)
    judge = Judge() if judge_answers else None

    retr_no_rerank, retr_rerank, gen_metrics = [], [], []

    for i, row in enumerate(rows, 1):
        q, gold = row["question"], row["gold_chunk_id"]

        base = retriever.hybrid(q, rerank=False)
        reranked = retriever.hybrid(q, rerank=True)

        retr_no_rerank.append(
            {
                "recall@1": metrics.recall_at_k(_ids(base), gold, 1),
                "recall@5": metrics.recall_at_k(_ids(base), gold, 5),
                "mrr": metrics.reciprocal_rank(_ids(base), gold),
            }
        )
        retr_rerank.append(
            {
                "recall@1": metrics.recall_at_k(_ids(reranked), gold, 1),
                "recall@5": metrics.recall_at_k(_ids(reranked), gold, 5),
                "mrr": metrics.reciprocal_rank(_ids(reranked), gold),
            }
        )

        if judge is not None:
            result = generator.answer(q)
            verdict = judge.judge(
                q, result["answer"], row["answer"], [s["preview"] for s in result["sources"]]
            )
            gen_metrics.append(
                {
                    "faithful": 1.0 if verdict.faithful else 0.0,
                    "correct": 1.0 if verdict.correct else 0.0,
                    "abstained": 1.0 if result["abstained"] else 0.0,
                }
            )
        print(f"  [{i}/{len(rows)}] {q[:70]}")

    summary = {
        "n": len(rows),
        "retrieval_no_rerank": metrics.aggregate(retr_no_rerank),
        "retrieval_with_rerank": metrics.aggregate(retr_rerank),
    }
    if gen_metrics:
        summary["generation"] = metrics.aggregate(gen_metrics)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG eval harness")
    parser.add_argument("qa", help="Path to the synthetic QA jsonl")
    parser.add_argument("--no-judge", action="store_true", help="Retrieval metrics only")
    args = parser.parse_args(argv)

    summary = run(args.qa, judge_answers=not args.no_judge)
    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
