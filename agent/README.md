# Agent harness

A stripped-down, hand-written tool-using agent loop on the Claude API
(`claude-opus-4-8`). It is intentionally *not* built on a framework — the point
is to show the harness internals a framework would otherwise hide.

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m agent.cli "List the Python files and summarize what each does"
python -m agent.cli --workdir ./rag --auto-approve "Run the test suite and report failures"
```

## What it demonstrates

| Concern | Where |
|---|---|
| Custom tool-call loop (plan → tool_use → tool_result) | `harness.py: Agent.run` |
| Tool surface (read/write/list/glob/grep/bash) | `tools.py` |
| Permission gating for mutating tools | `harness.py` + `cli.py._interactive_permission` |
| Path-sandboxing (no reads/writes outside workdir) | `tools.py: Toolbox._resolve` |
| Context management (tool-result truncation + compaction) | `harness.py: _truncate`, `_compact` |
| Structured-output guardrails | `guardrails.py` (input validation + `FinalAnswer`) |
| Adaptive thinking + forced tool-calling | `harness.py` (`thinking={"type": "adaptive"}`) |

## Harness design notes

- **The model proposes, the harness disposes.** Every tool call is validated
  against its JSON schema (`guardrails.validate_tool_input`) and — for
  `write_file`/`bash` — checked against a permission callback before it runs. A
  bad call comes back as a recoverable `is_error` tool_result, not a crash.
- **Context stays bounded.** Individual tool results are truncated at
  `MAX_TOOL_RESULT_CHARS`; once the message list grows past
  `COMPACT_AFTER_MESSAGES`, the oldest exchanges are compacted to a short note.
  This is the lever that lets a long agentic run not blow the context window.
- **The final answer is typed.** The run ends by coercing the model's free text
  into a Pydantic `FinalAnswer` via `client.messages.parse()`, so downstream
  callers get `{summary, steps_taken, files_touched, confidence}` — validated,
  not hoped-for.

### Extending toward a real harness

- **Sub-agents:** spawn a second `Agent` with a cheaper model for read-heavy
  exploration and feed its `FinalAnswer.summary` back as a tool result — the
  loop is already structured for it.
- **Smarter compaction:** replace the synthetic note in `_compact` with a cheap
  summarization call instead of dropping content.
- **Parallel-safe tools:** mark read-only tools so the harness can run multiple
  `tool_use` blocks concurrently (they already arrive batched per turn).
