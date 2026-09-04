"""Fail a pull request that adds a host-supplied field nothing consumes or documents.

AgentCore was extracted by canary-merging a product branch against it and closing
whatever refused to build. That finds hard conflicts — a missing attribute, a
required monkeypatch — and is blind to the opposite shape: a field carried into
``ToolResult`` from a product hook, correctly typed, fully tested, and read by
nothing. Every consumer of it stayed behind in the product's own loop copy, so a
product adopting ``run_agent_loop`` loses whatever it worded from that field and
loses it *silently*. Nothing raises. No test fails. The sentence the model used
to read is simply gone.

Four fields reached 0.4.0 in exactly that state. The rule that keeps it from
recurring: a field AgentCore does not read must be named in a boundary document,
which forces whoever adds it to write down who is expected to consume it.

Exit 0 when every unconsumed field is documented, 1 otherwise.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "agent_core"
DOCS = ROOT / "docs"

# Models whose fields are populated by a product hook rather than by core logic.
# Add a model here when it grows a host-supplied field; the check then holds it
# to the same rule.
WATCHED = {"ToolResult": PACKAGE / "loop_types.py"}


def declared_fields(class_name: str, module: Path) -> list[str]:
    tree = ast.parse(module.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
    raise SystemExit(f"{class_name} not found in {module.relative_to(ROOT)}")


def consumer_count(field: str, declared_in: Path) -> int:
    """Attribute reads of *field* anywhere in the package but its own module.

    Attribute access is the whole signal: a keyword argument at the construction
    site (``repeat_count=repeat_count``) is the field being FILLED, which is
    exactly the state this check exists to catch, so it must not count.
    """
    pattern = re.compile(rf"\.{re.escape(field)}\b")
    total = 0
    for path in PACKAGE.rglob("*.py"):
        if path == declared_in:
            continue
        total += len(pattern.findall(path.read_text()))
    return total


def documented(field: str) -> bool:
    return any(field in doc.read_text() for doc in DOCS.glob("*-boundary.md"))


def main() -> int:
    undocumented: list[str] = []
    for class_name, module in WATCHED.items():
        for field in declared_fields(class_name, module):
            if consumer_count(field, module) or documented(field):
                continue
            undocumented.append(f"{class_name}.{field}")

    if not undocumented:
        return 0

    print("Host-supplied fields with no consumer in agent_core/ and no mention")
    print("in any docs/*-boundary.md:")
    for name in undocumented:
        print(f"  - {name}")
    print()
    print("A field AgentCore does not read reaches a product only if that product")
    print("wires it up. Name it in the boundary document that owns it and say who")
    print("consumes it — or read it here. Silence is how the note a product used")
    print("to word from it disappears without a single test failing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
