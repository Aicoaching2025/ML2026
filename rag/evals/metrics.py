"""Retrieval metrics: recall@k and mean reciprocal rank (MRR)."""

from __future__ import annotations


def recall_at_k(retrieved_ids: list[int], gold_id: int, k: int) -> float:
    """1.0 if the gold chunk is in the top-k retrieved, else 0.0."""
    return 1.0 if gold_id in retrieved_ids[:k] else 0.0


def reciprocal_rank(retrieved_ids: list[int], gold_id: int) -> float:
    """1/rank of the gold chunk (1-based), or 0 if not retrieved."""
    for i, cid in enumerate(retrieved_ids, 1):
        if cid == gold_id:
            return 1.0 / i
    return 0.0


def aggregate(per_query: list[dict]) -> dict:
    """Average a list of per-query metric dicts into a summary."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: round(sum(q[k] for q in per_query) / len(per_query), 4) for k in keys}
