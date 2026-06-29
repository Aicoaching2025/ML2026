"""The agent harness: a custom tool-call loop over the Claude API.

Responsibilities (the things a harness owns, that the model does not):
- run the plan -> tool_use -> tool_result loop until the task is done
- gate mutating tools behind a permission callback
- validate tool inputs before executing (guardrails)
- manage context: truncate oversized tool results and compact old turns so a
  long run doesn't blow the context window
- produce a schema-validated final answer

Uses Claude `claude-opus-4-8` with adaptive thinking and forced tool-calling.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from .guardrails import FinalAnswer, validate_tool_input
from .tools import MUTATING_TOOLS, TOOL_DEFS, Toolbox, ToolError

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")

SYSTEM_PROMPT = """You are a focused coding/research agent operating inside a \
sandboxed workdir via tools.

Work in small steps: inspect with read-only tools (list_dir, glob, grep, \
read_file) before writing or running anything. Prefer the most direct path to \
the goal. When the task is complete, stop calling tools and give your final \
answer as prose — the harness will capture and structure it.

Be honest about uncertainty. If you could not complete part of the task, say so \
plainly rather than claiming success."""

# Per-tool-result cap. Oversized results are truncated so one big file read
# doesn't dominate the context window.
MAX_TOOL_RESULT_CHARS = 8_000

# When the running message list exceeds this many turns, compact the oldest
# tool exchanges into a short note. Keeps long agentic runs bounded.
COMPACT_AFTER_MESSAGES = 40
KEEP_RECENT_MESSAGES = 16

# A permission callback returns True to allow a mutating tool call.
PermissionFn = Callable[[str, dict[str, Any]], bool]


def allow_all(_name: str, _args: dict[str, Any]) -> bool:
    return True


@dataclass
class RunResult:
    final: FinalAnswer | None
    raw_text: str
    turns: int
    input_tokens: int
    output_tokens: int
    transcript: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        workdir: str | os.PathLike[str] | None = None,
        *,
        model: str = DEFAULT_MODEL,
        permission: PermissionFn = allow_all,
        max_turns: int = 30,
        client: anthropic.Anthropic | None = None,
        verbose: bool = True,
    ):
        self.toolbox = Toolbox(workdir or os.environ.get("AGENT_WORKDIR", "."))
        self.model = model
        self.permission = permission
        self.max_turns = max_turns
        self.client = client or anthropic.Anthropic()
        self.verbose = verbose
        self._schema_by_tool = {t["name"]: t["input_schema"] for t in TOOL_DEFS}

    # --- logging ------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # --- context management -------------------------------------------------

    def _compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Summarize old tool exchanges so the context stays bounded.

        We keep the first user message (the task) and the most recent turns
        verbatim, and replace the middle with a one-line synthetic note. This is
        deliberately simple; a production harness would summarize with a cheap
        model call instead of dropping content.
        """
        if len(messages) <= COMPACT_AFTER_MESSAGES:
            return messages
        head = messages[:1]
        tail = messages[-KEEP_RECENT_MESSAGES:]
        dropped = len(messages) - len(head) - len(tail)
        note = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"[harness: compacted {dropped} earlier tool "
                    f"exchanges to save context. Earlier steps explored the "
                    f"workdir and gathered context for the current task.]",
                }
            ],
        }
        self._log(f"  · compacted {dropped} old messages")
        return head + [note] + tail

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_TOOL_RESULT_CHARS:
            return text
        head = text[: MAX_TOOL_RESULT_CHARS // 2]
        tail = text[-MAX_TOOL_RESULT_CHARS // 2 :]
        return f"{head}\n... [harness: truncated {len(text)} chars] ...\n{tail}"

    # --- tool execution -----------------------------------------------------

    def _run_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Execute one tool call. Returns (result_text, is_error)."""
        schema = self._schema_by_tool.get(name)
        if schema is None:
            return f"unknown tool: {name}", True

        err = validate_tool_input(schema, args)
        if err:
            return f"invalid input: {err}", True

        if name in MUTATING_TOOLS and not self.permission(name, args):
            return "denied by user; do not retry this exact call", True

        try:
            return self._truncate(self.toolbox.dispatch(name, args)), False
        except ToolError as e:
            return e.message, True
        except Exception as e:  # surface unexpected failures to the model
            return f"tool crashed: {type(e).__name__}: {e}", True

    # --- main loop ----------------------------------------------------------

    def run(self, task: str) -> RunResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        in_tokens = out_tokens = 0
        last_text = ""

        for turn in range(1, self.max_turns + 1):
            messages = self._compact(messages)
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=8_000,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                tools=TOOL_DEFS,
                messages=messages,
            )
            in_tokens += resp.usage.input_tokens
            out_tokens += resp.usage.output_tokens

            # Echo assistant turn back into history (including thinking blocks).
            messages.append({"role": "assistant", "content": resp.content})

            text_blocks = [b.text for b in resp.content if b.type == "text"]
            if text_blocks:
                last_text = "\n".join(text_blocks)
                self._log(f"[turn {turn}] {last_text[:400]}")

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                # Model is done. Structure the final answer.
                self._log(f"[done] stop_reason={resp.stop_reason}")
                final = self._finalize(task, last_text)
                return RunResult(
                    final=final,
                    raw_text=last_text,
                    turns=turn,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    transcript=messages,
                )

            results = []
            for tu in tool_uses:
                self._log(f"  → {tu.name}({json.dumps(tu.input)[:160]})")
                text, is_error = self._run_tool(tu.name, dict(tu.input))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": text,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})

        self._log("[stopped] hit max_turns")
        return RunResult(
            final=self._finalize(task, last_text),
            raw_text=last_text,
            turns=self.max_turns,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            transcript=messages,
        )

    def _finalize(self, task: str, agent_text: str) -> FinalAnswer | None:
        """Coerce the free-text result into a schema-validated FinalAnswer.

        Uses structured outputs (client.messages.parse) so the terminal payload
        is guaranteed to match the FinalAnswer schema.
        """
        try:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=2_000,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Convert this agent run into the structured schema. "
                            f"Task:\n{task}\n\nAgent's final notes:\n{agent_text}"
                        ),
                    }
                ],
                output_format=FinalAnswer,
            )
            return resp.parsed_output
        except Exception as e:  # never let finalization crash the run
            self._log(f"[warn] could not structure final answer: {e}")
            return None
