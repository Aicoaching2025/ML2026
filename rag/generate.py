"""Citation-grounded answer generation with abstention.

Hallucination mitigation, concretely:
- The answer must cite the retrieved passages it used, by index.
- If retrieval is weak (top rerank/RRF score below a floor) or the model finds
  no supporting passage, it abstains ("I don't have enough information") rather
  than guessing.
- Output is structured (Pydantic via client.messages.parse), so `answer`,
  `citations`, and `abstained` are validated fields, not parsed from prose.
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel, Field

from .config import CONFIG
from .retrieve import Retrieved, Retriever

# If the best candidate scores below this, treat retrieval as too weak to answer.
MIN_RERANK_SCORE = 0.35
MIN_RRF_SCORE = 0.0  # RRF is always >0 when anything matched; rely on rerank floor


class Answer(BaseModel):
    answer: str = Field(description="The grounded answer, or an abstention message.")
    citations: list[int] = Field(
        default_factory=list,
        description="0-based indices of the passages that support the answer.",
    )
    abstained: bool = Field(description="True if there was not enough evidence to answer.")


SYSTEM = """You answer strictly from the provided passages. Rules:
- Use ONLY information present in the passages. Do not use prior knowledge.
- Cite the passages you used by their [index].
- If the passages do not contain the answer, set abstained=true and say you \
don't have enough information. Do not guess."""


class Generator:
    def __init__(self, retriever: Retriever | None = None, client: anthropic.Anthropic | None = None):
        self.retriever = retriever or Retriever()
        self.client = client or anthropic.Anthropic()

    def answer(self, query: str) -> dict:
        hits = self.retriever.hybrid(query, rerank=True)
        if not hits or not self._strong_enough(hits):
            return {
                "answer": "I don't have enough information in the retrieved context to answer that.",
                "citations": [],
                "abstained": True,
                "sources": [self._source(h) for h in hits],
            }

        passages = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits))
        resp = self.client.messages.parse(
            model=CONFIG.anthropic_model,
            max_tokens=1_500,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Question: {query}\n\nPassages:\n{passages}"}],
            output_format=Answer,
        )
        parsed: Answer = resp.parsed_output
        return {
            "answer": parsed.answer,
            "citations": parsed.citations,
            "abstained": parsed.abstained,
            "sources": [self._source(h) for h in hits],
        }

    @staticmethod
    def _strong_enough(hits: list[Retrieved]) -> bool:
        best = hits[0]
        if best.rerank_score is not None:
            return best.rerank_score >= MIN_RERANK_SCORE
        return best.rrf_score > MIN_RRF_SCORE

    @staticmethod
    def _source(h: Retrieved) -> dict:
        return {
            "doc_id": h.chunk.doc_id,
            "metadata": h.chunk.metadata,
            "rrf_score": round(h.rrf_score, 4),
            "rerank_score": h.rerank_score,
            "preview": h.chunk.text[:160],
        }
