"""Single source of truth for reading the distribution version.

CI and the release workflow both need the version declared in
``pyproject.toml``. Parsing it with ``tomllib`` rather than ``grep`` keeps the
two from disagreeing when the file is reformatted, and makes the failure mode a
clear traceback instead of a silently empty string.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_version(pyproject: Path | None = None) -> str:
    path = pyproject or ROOT / "pyproject.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-tag",
        metavar="TAG",
        help="Verify a git tag ('v1.2.3' or '1.2.3') matches the declared version.",
    )
    args = parser.parse_args(argv)

    version = read_version()

    if args.check_tag is None:
        print(version)
        return 0

    tagged = args.check_tag.removeprefix("refs/tags/").removeprefix("v")
    if tagged != version:
        print(
            f"tag {args.check_tag!r} does not match pyproject version {version!r}.\n"
            "A release tag must name the version it publishes: either move the tag "
            "or bump [project].version (and re-run `uv lock`).",
            file=sys.stderr,
        )
        return 1

    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
