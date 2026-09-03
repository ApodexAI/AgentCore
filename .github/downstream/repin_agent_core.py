"""Repoint this product's AgentCore pin at a released tag.

Copied into the product repository alongside the bump workflow.

Deliberately an in-place rev substitution rather than `uv add`: `uv add` rewrites
the standard PEP 508 direct-URL dependency into uv's proprietary
`[tool.uv.sources]` table, which pip and other installers ignore. That would
silently break any non-uv install path (a Dockerfile running `pip install .`, for
one) and put a structural diff in every bump pull request.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Anchor on the repository path so the '@' inside 'git@github.com' is never
# mistaken for the rev separator.
PIN = re.compile(r'(AgentCore\.git@)[^"\'\s]+')


def repin(text: str, version: str) -> tuple[str, int]:
    return PIN.subn(rf"\g<1>{version}", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Release tag to pin, e.g. v0.2.0")
    parser.add_argument("--file", default="pyproject.toml")
    args = parser.parse_args(argv)

    path = Path(args.file)
    original = path.read_text(encoding="utf-8")
    updated, count = repin(original, args.version)

    if count != 1:
        print(
            f"Expected exactly one AgentCore pin in {path}, found {count}.\n"
            "The dependency declaration changed shape; update this script rather "
            "than letting the bump land a half-edited pin.",
            file=sys.stderr,
        )
        return 1

    if updated == original:
        print(f"Already pinned to {args.version}.")
        return 0

    path.write_text(updated, encoding="utf-8")
    print(f"Repinned to {args.version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
