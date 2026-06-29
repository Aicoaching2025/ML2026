"""CLI entry point for the agent harness.

    python -m agent.cli "your task here"
    python -m agent.cli --workdir ./somedir --auto-approve "fix the failing test"

Permission gating: mutating tools (write_file, bash) prompt for confirmation
unless --auto-approve / AGENT_AUTO_APPROVE=1 is set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

from .harness import Agent, allow_all


def _interactive_permission(name: str, args: dict[str, Any]) -> bool:
    print(f"\n  ⚠ allow {name}? {json.dumps(args)[:200]}")
    answer = input("    [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Tool-using Claude agent harness")
    parser.add_argument("task", help="Natural-language task for the agent")
    parser.add_argument("--workdir", default=os.environ.get("AGENT_WORKDIR", "."))
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        default=os.environ.get("AGENT_AUTO_APPROVE") == "1",
        help="Skip confirmation for write_file/bash.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set (see .env.example)", file=sys.stderr)
        return 2

    agent = Agent(
        workdir=args.workdir,
        permission=allow_all if args.auto_approve else _interactive_permission,
        max_turns=args.max_turns,
        verbose=not args.quiet,
    )
    result = agent.run(args.task)

    print("\n" + "=" * 60)
    if result.final is not None:
        print("SUMMARY     :", result.final.summary)
        if result.final.steps_taken:
            print("STEPS       :")
            for s in result.final.steps_taken:
                print("   -", s)
        if result.final.files_touched:
            print("FILES       :", ", ".join(result.final.files_touched))
        print("CONFIDENCE  :", result.final.confidence)
    else:
        print(result.raw_text or "(no output)")
    print(
        f"\nturns={result.turns}  "
        f"tokens in/out={result.input_tokens}/{result.output_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
