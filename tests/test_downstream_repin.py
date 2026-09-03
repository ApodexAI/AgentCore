from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_repin_module() -> ModuleType:
    path = Path(__file__).parents[1] / ".github" / "downstream" / "repin_agent_core.py"
    spec = importlib.util.spec_from_file_location("repin_agent_core", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repin_agent_core = _load_repin_module()


@pytest.mark.parametrize(
    "version",
    [
        "v0.2.0",
        "v1.0.0rc1",
        "v1.0.0.post1",
        "v1.0.0.dev1",
        "v1.0.0+linux.x86",
    ],
)
def test_accepts_release_tags(version: str) -> None:
    assert repin_agent_core.is_valid_version(version)


@pytest.mark.parametrize(
    "version",
    [
        "0.2.0",
        "v1",
        "v1.2",
        "v1.2.3 ",
        "v1.2.3/other",
        "v1.2.3\nINJECTED=value",
        'v1.2.3"$(echo injected)"',
    ],
)
def test_rejects_unsafe_or_malformed_tags(version: str) -> None:
    assert not repin_agent_core.is_valid_version(version)


def test_validation_fails_before_opening_dependency_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"

    result = repin_agent_core.main(
        ["v1.2.3\nINJECTED=value", "--file", str(missing)]
    )

    assert result == 1
    assert not missing.exists()


def test_repin_replaces_exactly_one_agent_core_revision() -> None:
    original = (
        'dependencies = [\n'
        '  "apodex-agent-core @ '
        'git+ssh://git@github.com/ApodexAI/AgentCore.git@abc123",\n'
        ']\n'
    )

    updated, count = repin_agent_core.repin(original, "v0.2.0")

    assert count == 1
    assert "AgentCore.git@v0.2.0" in updated
