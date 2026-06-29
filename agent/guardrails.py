"""Structured-output guardrails for the agent.

This is the "safety + structure" layer (portfolio project #3) bolted onto the
harness. It does two things:

1. Validates tool *inputs* against their declared JSON schema before execution,
   so a malformed tool call becomes a recoverable `is_error` result instead of
   a Python exception.
2. Enforces a Pydantic-typed *final answer* via `client.messages.parse()`, so
   the agent's terminal output is guaranteed to be schema-valid — not free text
   we hope is well-formed.

Both keep the model's output inside a contract the rest of the system can rely
on, which is the core of hallucination/structure mitigation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FinalAnswer(BaseModel):
    """The validated terminal output of an agent run."""

    summary: str = Field(description="Concise answer to the user's task.")
    steps_taken: list[str] = Field(
        default_factory=list,
        description="Short bullet list of the key actions the agent took.",
    )
    files_touched: list[str] = Field(
        default_factory=list,
        description="Paths created or modified, if any.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Self-rated confidence in the answer, 0-1."
    )


def validate_tool_input(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Lightweight required-field / type check against a tool's input_schema.

    Returns an error string if invalid, else None. We keep this dependency-free
    (no jsonschema) because the tool schemas here are flat; for nested schemas
    swap in `jsonschema.validate`.
    """
    props = schema.get("properties", {})
    required = schema.get("required", [])

    for field in required:
        if field not in args:
            return f"missing required field: {field!r}"

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for key, value in args.items():
        spec = props.get(key)
        if not spec or "type" not in spec:
            continue
        expected = type_map.get(spec["type"])
        if expected and not isinstance(value, expected):
            return f"field {key!r} must be {spec['type']}, got {type(value).__name__}"
    return None
