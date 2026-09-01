"""Loader robustness for the host-configured model registry.

The registry is the PR's headline host-configuration seam and had no coverage:
a malformed row or a missing YAML parser decided every model's thinking format
silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent_core.runtime.loop.model_profile import (
    configure_model_registry,
    infer_thinking_format,
    reset_thinking_format_cache,
)


@pytest.fixture(autouse=True)
def _restore_registry():
    yield
    configure_model_registry(None)
    reset_thinking_format_cache()


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "model_registry.yaml"
    path.write_text(body, encoding="utf-8")
    configure_model_registry(path)
    return path


def test_valid_registry_drives_inference(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "thinking_formats:\n"
        "  - pattern: '^claude-'\n"
        "    format: content_block\n",
    )
    assert infer_thinking_format("claude-opus-5", default="tag") == "content_block"
    assert infer_thinking_format("qwen3-32b", default="tag") == "tag"


def test_unhashable_format_value_skips_the_row_instead_of_crashing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``format: [content_block]`` is an easy indentation typo.

    ``fmt in _VALID_FORMATS`` raises ``TypeError: unhashable type`` for a list,
    which aborted the whole load and took every other row down with it.
    """
    _write(
        tmp_path,
        "thinking_formats:\n"
        "  - pattern: '^qwen'\n"
        "    format: [content_block]\n"
        "  - pattern: '^claude-'\n"
        "    format: content_block\n",
    )
    with caplog.at_level("WARNING"):
        assert infer_thinking_format("qwen3-32b", default="tag") == "tag"
        # The healthy row still loaded.
        assert (
            infer_thinking_format("claude-opus-5", default="tag") == "content_block"
        )
    assert any("skipped invalid entry" in r.message for r in caplog.records)


def test_non_string_pattern_skips_the_row(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "thinking_formats:\n"
        "  - pattern: {oops: 1}\n"
        "    format: content_block\n"
        "  - pattern: '^claude-'\n"
        "    format: content_block\n",
    )
    assert infer_thinking_format("claude-opus-5", default="tag") == "content_block"


def test_missing_yaml_parser_is_reported_not_silently_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently returning the default hides a fully disabled registry.

    For a ``content_block`` provider that fallback inlines reasoning as
    ``<think>`` and drops the signed blocks, while the host believes its
    configured registry is in effect.
    """
    _write(
        tmp_path,
        "thinking_formats:\n  - pattern: '^claude-'\n    format: content_block\n",
    )
    reset_thinking_format_cache()
    monkeypatch.setitem(sys.modules, "yaml", None)

    with caplog.at_level("ERROR"):
        assert infer_thinking_format("claude-opus-5", default="tag") == "tag"

    assert any("PyYAML is not installed" in r.message for r in caplog.records)
