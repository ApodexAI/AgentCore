"""Collect pending fragments into one release; run uv lock afterwards."""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_version_bump import version_key
from scripts.release_notes import fragments, read_fragment
from scripts.version import read_version

ROOT = Path(__file__).resolve().parent.parent


def prepare(root: Path, *, version: str | None = None, dry_run: bool = False) -> str:
    paths = fragments(root)
    if not paths:
        raise ValueError("No pending change fragments")
    notes = [read_fragment(path) for path in paths]
    current = read_version(root / "pyproject.toml")
    major, minor, patch = version_key(current)
    minimum = (major, minor + 1, 0) if any(k != "fix" for k, _ in notes) else (major, minor, patch + 1)
    target = version or ".".join(map(str, minimum))
    if version_key(target) < minimum:
        raise ValueError(f"Version {target} is below the required {'.'.join(map(str, minimum))}")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## [{target}]" in changelog:
        raise ValueError(f"Changelog already contains {target}")
    first_heading = re.search(r"^## \[", changelog, re.MULTILINE)
    if first_heading is None:
        raise ValueError("Cannot find the first version heading in CHANGELOG.md")
    section = f"## [{target}] - {datetime.date.today().isoformat()}\n\n"
    for kind, heading in (("breaking", "Changed"), ("feature", "Added"), ("fix", "Fixed")):
        entries = [text for k, text in notes if k == kind]
        if entries:
            section += f"### {heading}\n\n"
            section += "\n".join("- " + text.replace("\n", "\n  ") for text in entries) + "\n\n"
    project_path = root / "pyproject.toml"
    project = project_path.read_text(encoding="utf-8")
    # Limit replacement to [project], so another table's version stays untouched.
    match = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", project)
    if match is None:
        raise ValueError("Missing [project] table")
    body, count = re.subn(r'^version\s*=\s*[\'\"][^\'\"]+[\'\"]\s*$', f'version = "{target}"', match.group(1), count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("Cannot find [project].version")
    if dry_run:
        return section
    project_path.write_text(project[:match.start(1)] + body + project[match.end(1):], encoding="utf-8")
    (root / "CHANGELOG.md").write_text(changelog[:first_heading.start()] + section + changelog[first_heading.start():], encoding="utf-8")
    for path in paths:
        path.unlink()
    return section


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="Override the automatically selected version (may only increase it)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(prepare(ROOT, version=args.version, dry_run=args.dry_run))
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    if not args.dry_run:
        print("Run uv lock, then commit pyproject.toml, uv.lock, CHANGELOG.md and removed fragments in a release PR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
