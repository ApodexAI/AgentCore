"""Regression tests for the host-field boundary guard."""

from __future__ import annotations

from scripts import check_unconsumed_fields as check


def test_consumer_count_counts_only_real_attribute_reads(tmp_path, monkeypatch):
    package = tmp_path / "agent_core"
    package.mkdir()
    declared_in = package / "types.py"
    declared_in.write_text("class Model:\n    field: str\n")
    consumer = package / "consumer.py"
    consumer.write_text(
        "# A comment mentioning .field is not a consumer.\n"
        "note = 'A string mentioning .field is not a consumer either.'\n"
        "obj.field = 'write-only'\n"
    )
    monkeypatch.setattr(check, "PACKAGE", package)

    assert check.consumer_count("field") == 0

    consumer.write_text(consumer.read_text() + "value = obj.field\n")
    assert check.consumer_count("field") == 1

    declared_in.write_text(
        "class Model:\n"
        "    field: str\n"
        "    def read(self) -> str:\n"
        "        return self.field\n"
    )
    assert check.consumer_count("field") == 2


def test_documented_requires_the_qualified_model_field(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    boundary = docs / "tool-boundary.md"
    monkeypatch.setattr(check, "DOCS", docs)

    boundary.write_text("AnotherTool.result_id and ToolResult.result_identifier\n")
    assert check.documented("ToolResult", "result_id") is False

    boundary.write_text("The host consumes `ToolResult.result_id`.\n")
    assert check.documented("ToolResult", "result_id") is True
