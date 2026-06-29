"""LLM-as-judge for answer quality: faithfulness + correctness.

- faithfulness: is every claim in the answer supported by the retrieved
  passages? (catches hallucination)
- correctness: does the answer match the gold answer?

Both are scored by Claude with structured output. In a real pipeline you'd add
a human spot-check loop over a sample of these judgments to calibrate the judge.
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel, Field

from ..config import CONFIG


class Judgment(BaseModel):
    faithful: bool = Field(description="True if every claim is supported by the passages.")
    correct: bool = Field(description="True if the answer matches the expected answer.")
    reason: str = Field(description="One-sentence justification.")


_SYSTEM = """You are a strict evaluator of RAG answers. Judge two things:
1. faithfulness — is every claim in the answer supported by the passages?
2. correctness — does the answer match the expected answer?
Be conservative: if a claim is unsupported, faithful=false."""


class Judge:
    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()

    def judge(self, question: str, answer: str, expected: str, passages: list[str]) -> Judgment:
        ctx = "\n\n".join(f"[{i}] {p[:600]}" for i, p in enumerate(passages))
        resp = self.client.messages.parse(
            model=CONFIG.anthropic_model,
            max_tokens=500,
            system=_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\nExpected answer: {expected}\n"
                        f"Model answer: {answer}\n\nPassages:\n{ctx}"
                    ),
                }
            ],
            output_format=Judgment,
        )
        return resp.parsed_output
