"""Every emitted property must carry a ``type``.

Strict function-schema validators — Gemini-compatible gateways in particular —
reject a property with no ``type`` rather than treating it as "anything goes".
So an under-annotated parameter has to degrade to a concrete type, not to an
empty schema, or a fallback leg that used to work starts returning 400 on tool
validation alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_core.tool import tool


def _properties(t: Any) -> dict[str, Any]:
    return t.to_openai_schema()["function"]["parameters"]["properties"]


def _walk(schema: dict[str, Any], path: str) -> list[str]:
    """Return the paths of every sub-schema that declares no ``type``."""
    missing = [] if "type" in schema or "$ref" in schema or "anyOf" in schema else [path]
    if isinstance(items := schema.get("items"), dict):
        missing += _walk(items, f"{path}.items")
    for name, sub in (schema.get("properties") or {}).items():
        if isinstance(sub, dict):
            missing += _walk(sub, f"{path}.{name}")
    return missing


@pytest.mark.parametrize("annotation", [Any, list[Any], dict[str, Any], list[dict[str, Any]]])
def test_no_property_is_emitted_without_a_type(annotation: Any) -> None:
    async def fn(value):  # annotation injected below
        return ""
    fn.__annotations__ = {"value": annotation, "return": str}
    t = tool(fn)

    props = _properties(t)
    assert _walk(props["value"], "value") == []


def test_bare_any_degrades_to_string_not_an_empty_schema() -> None:
    @tool
    async def probe(value: Any) -> str:
        """Probe.

        Args:
            value: anything.
        """
        return ""

    assert _properties(probe)["value"]["type"] == "string"


def test_dict_annotation_is_how_a_tool_asks_for_an_object() -> None:
    @tool
    async def probe(items: list[dict[str, Any]]) -> str:
        """Probe.

        Args:
            items: a list of objects.
        """
        return ""

    assert _properties(probe)["items"]["items"] == {"type": "object"}
