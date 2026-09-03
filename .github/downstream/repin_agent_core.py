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
# AgentCore follows a three-part release scheme and may publish standard PEP 440
# prereleases, post releases, dev releases, or local versions. Keeping this an
# allow-list also makes the value safe for GITHUB_ENV and git branch names.
VERSION = re.compile(
    r"v[0-9]+\.[0-9]+\.[0-9]+"
    r"(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?"
    r"(?:\.dev[0-9]+)?"
    r"(?:\+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?"
)


def is_valid_version(version: str) -> bool:
    return VERSION.fullmatch(version) is not None


def repin(text: str, version: str) -> tuple[str, int]:
    return PIN.subn(rf"\g<1>{version}", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Release tag to pin, e.g. v0.2.0")
    parser.add_argument("--file", default="pyproject.toml")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the release tag without editing the dependency file.",
    )
    args = parser.parse_args(argv)

    if not is_valid_version(args.version):
        print(
            f"Refusing to pin {args.version!r}: expected an AgentCore release tag "
            "such as v0.2.0 or v0.3.0rc1.",
            file=sys.stderr,
        )
        return 1

    if args.validate_only:
        print(f"Validated release tag {args.version}.")
        return 0

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
