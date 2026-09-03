"""Print one version's section from CHANGELOG.md.

The release workflow uses this as the GitHub Release body, which makes a missing
CHANGELOG entry a hard release failure rather than a silently empty release
note. Consumers reading a bump PR need to know what changed without diffing the
tag range by hand.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]")


def extract(version: str, text: str) -> str | None:
    lines = text.splitlines()
    start: int | None = None

    for index, line in enumerate(lines):
        match = HEADING.match(line)
        if match is None:
            continue
        if start is None and match.group("version") == version:
            start = index + 1
            continue
        if start is not None:
            # The next version heading terminates the section.
            return "\n".join(lines[start:index]).strip()

    if start is None:
        return None
    return "\n".join(lines[start:]).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Version to extract, without a leading 'v'.")
    args = parser.parse_args(argv)

    changelog = ROOT / "CHANGELOG.md"
    section = extract(args.version, changelog.read_text(encoding="utf-8"))

    if not section:
        print(
            f"CHANGELOG.md has no entry for {args.version!r}.\n"
            f"Add a '## [{args.version}] - <YYYY-MM-DD>' section describing what "
            "consumers must know before upgrading.",
            file=sys.stderr,
        )
        return 1

    print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
