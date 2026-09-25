"""Independent change fragments and validation shared by CI and release tooling."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from scripts.changelog_section import extract

FRAGMENT_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*\.(fix|feature|breaking)\.md$")


def fragments(root: Path) -> list[Path]:
    return sorted((root / "changes").glob("*.md"))


def read_fragment(path: Path) -> tuple[str, str]:
    match = FRAGMENT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid change fragment: {path.name}; use <id>.fix|feature|breaking.md")
    text = path.read_text(encoding="utf-8").strip()
    if not text or any(line.startswith("#") for line in text.splitlines()):
        raise ValueError(f"{path.name}: write a non-empty release-note paragraph without headings")
    return match.group(1), text


def validate_lock(root: Path, version: str) -> None:
    data = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    own = [p for p in data["package"] if p["name"] == "apodex-agent-core"]
    if len(own) != 1 or own[0]["version"] != version:
        raise ValueError("uv.lock project version is stale; run uv lock")


def validate_release(root: Path, version: str) -> None:
    pending = fragments(root)
    if pending:
        raise ValueError("Unreleased change fragments remain; run scripts/prepare_release.py")
    validate_lock(root, version)
    if not extract(version, (root / "CHANGELOG.md").read_text(encoding="utf-8")):
        raise ValueError(f"CHANGELOG.md has no non-empty entry for {version}")
