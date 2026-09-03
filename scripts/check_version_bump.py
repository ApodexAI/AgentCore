"""Fail a pull request that changes shared runtime code without increasing the version.

Two products consume AgentCore by pinning a revision. When ``agent_core/``
changes but ``[project].version`` does not increase, both products can end up reporting the
same version for different code: the installed ``dist-info`` stops identifying
what is actually running, and no version constraint downstream can mean
anything. This check is the enforcement point for that rule.

Docs-only, test-only, and tooling-only pull requests are exempt, because they
change nothing a consumer can import.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

# Support being run as a plain script from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from version import read_version

# Paths whose contents are importable by a consumer. A change under any of these
# alters the published artifact and therefore requires a new version.
PUBLISHED_PATHS = ("agent_core/", "pyproject.toml")
VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def changed_files(base: str) -> list[str]:
    # Two-dot diff: what this branch's tip looks like against the merge base,
    # which is what the merge would actually land.
    merge_base = _git("merge-base", base, "HEAD").strip()
    out = _git("diff", "--name-only", f"{merge_base}..HEAD")
    return [line for line in out.splitlines() if line]


def base_version(base: str) -> str | None:
    try:
        blob = _git("show", f"{base}:pyproject.toml")
    except subprocess.CalledProcessError:
        # No pyproject on the base ref: nothing to compare against, so nothing
        # this check can meaningfully assert.
        return None
    return tomllib.loads(blob)["project"]["version"]


def version_key(value: str) -> tuple[int, int, int]:
    """Return a comparable key for the repository's three-part version scheme."""
    match = VERSION.fullmatch(value)
    if match is None:
        raise ValueError(
            f"invalid version {value!r}; expected three numeric components such as '0.2.1'"
        )
    return tuple(int(part) for part in match.groups())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base ref or SHA of the pull request.")
    args = parser.parse_args(argv)

    touched = [f for f in changed_files(args.base) if f.startswith(PUBLISHED_PATHS)]
    if not touched:
        print("No published code changed; version bump not required.")
        return 0

    current = read_version()
    previous = base_version(args.base)

    if previous is None:
        print(f"Published code changed and version moved {previous} -> {current}.")
        return 0

    try:
        increased = version_key(current) > version_key(previous)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    if increased:
        print(f"Published code changed and version increased {previous} -> {current}.")
        return 0

    listed = "\n  ".join(touched[:20])
    overflow = f"\n  ... and {len(touched) - 20} more" if len(touched) > 20 else ""
    print(
        "This pull request changes published code but does not increase "
        f"[project].version ({previous!r} -> {current!r}).\n\n"
        f"Changed:\n  {listed}{overflow}\n\n"
        "Bump [project].version in pyproject.toml, then run `uv lock` so the "
        "lockfile's self-entry matches (otherwise `uv sync --frozen` fails), and "
        "add a CHANGELOG.md entry. See docs/versioning.md for how to choose the "
        "new number. If this change genuinely cannot affect consumers, apply the "
        "'skip-version-bump' label.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
