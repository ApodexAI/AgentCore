"""Small product-neutral LLM helpers."""

from __future__ import annotations


def extract_boxed_content(text: str) -> str:
    r"""Return the content of the last balanced ``\boxed{...}``."""
    if not text:
        return ""
    matches: list[str] = []
    offset = 0
    while (start := text.find(r"\boxed{", offset)) >= 0:
        content_start = start + 7
        depth = 1
        pos = content_start
        while pos < len(text) and depth:
            depth += (text[pos] == "{") - (text[pos] == "}")
            pos += 1
        if depth == 0:
            matches.append(text[content_start : pos - 1])
            offset = pos
        else:
            offset = content_start
    return matches[-1] if matches else ""


__all__ = ["extract_boxed_content"]
