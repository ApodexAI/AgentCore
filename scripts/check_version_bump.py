"""Require independent change fragments for code PRs and validate release PRs."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

# Support being run as a plain script from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.release_notes import fragments, read_fragment, validate_lock, validate_release
from scripts.version import read_version

ROOT = Path(__file__).resolve().parent.parent

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

    changed = changed_files(args.base)
    touched = [f for f in changed if f.startswith(PUBLISHED_PATHS)]
    current = read_version()
    previous = base_version(args.base)
    try:
        current_key = version_key(current)
        previous_key = version_key(previous) if previous is not None else current_key
        validate_lock(ROOT, current)
        for path in fragments(ROOT):
            read_fragment(path)
        if current_key < previous_key:
            raise ValueError(f"Version must not decrease: {previous} -> {current}")
        if current_key > previous_key:
            base_fragments = _git("ls-tree", "-r", "--name-only", args.base, "--", "changes/").splitlines()
            if any(p.endswith((".feature.md", ".breaking.md")) for p in base_fragments):
                minimum = (previous_key[0], previous_key[1] + 1, 0)
                if current_key < minimum:
                    raise ValueError("Feature or breaking fragments require a MINOR release")
            validate_release(ROOT, current)
            print(f"Release validated: {previous} -> {current}")
            return 0
        merge_base = _git("merge-base", args.base, "HEAD").strip()
        changed_existing = _git("diff", "--no-renames", "--diff-filter=MD", "--name-only", f"{merge_base}..HEAD").splitlines()
        if any(p.startswith("changes/") and p.endswith(".md") for p in changed_existing):
            raise ValueError("Only a release PR may modify or remove existing change fragments")
        if "CHANGELOG.md" in changed:
            raise ValueError("Keep CHANGELOG.md for release PRs; add a changes/<id>.<kind>.md fragment instead")
        if not touched:
            print("No published code changed; release fragment not required.")
            return 0
        added = _git("diff", "--diff-filter=A", "--name-only", f"{merge_base}..HEAD").splitlines()
        new_fragments = [p for p in fragments(ROOT) if p.relative_to(ROOT).as_posix() in added]
        if not new_fragments:
            raise ValueError("Published code changed: add changes/<id>.fix.md, .feature.md or .breaking.md; do not bump the version in a feature PR")
        print("Published change has a new release fragment; version stays unchanged until release.")
        return 0
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
