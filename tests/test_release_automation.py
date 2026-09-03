from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_version_bump

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.2.1", (0, 2, 1)),
        ("1.0.0", (1, 0, 0)),
        ("10.20.300", (10, 20, 300)),
    ],
)
def test_version_key_parses_three_numeric_components(
    value: str, expected: tuple[int, int, int]
) -> None:
    assert check_version_bump.version_key(value) == expected


@pytest.mark.parametrize("value", ["0.2", "v0.2.1", "0.02.1", "0.2.1rc1", "latest"])
def test_version_key_rejects_values_outside_the_version_scheme(value: str) -> None:
    with pytest.raises(ValueError, match="expected three numeric components"):
        check_version_bump.version_key(value)


@pytest.mark.parametrize(("previous", "current"), [("0.2.0", "0.2.0"), ("0.2.0", "0.1.9")])
def test_version_gate_rejects_equal_or_decreasing_versions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    previous: str,
    current: str,
) -> None:
    monkeypatch.setattr(check_version_bump, "changed_files", lambda _base: ["agent_core/x.py"])
    monkeypatch.setattr(check_version_bump, "base_version", lambda _base: previous)
    monkeypatch.setattr(check_version_bump, "read_version", lambda: current)

    assert check_version_bump.main(["--base", "base-sha"]) == 1
    assert "does not increase" in capsys.readouterr().err


def test_version_gate_accepts_an_increase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_version_bump, "changed_files", lambda _base: ["agent_core/x.py"])
    monkeypatch.setattr(check_version_bump, "base_version", lambda _base: "0.2.0")
    monkeypatch.setattr(check_version_bump, "read_version", lambda: "0.2.1")

    assert check_version_bump.main(["--base", "base-sha"]) == 0


def test_version_label_changes_retrigger_ci() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "types: [opened, synchronize, reopened, labeled, unlabeled]" in workflow


def test_downstream_version_input_is_not_interpolated_into_shell_source() -> None:
    workflow = (ROOT / ".github/downstream/bump-agent-core.yml").read_text(encoding="utf-8")

    assert "INPUT_VERSION: ${{ github.event.client_payload.version || inputs.version }}" in workflow
    assert 'version="$INPUT_VERSION"' in workflow
    assert 'version="${{ github.event.client_payload.version || inputs.version }}"' not in workflow
    assert "persist-credentials: false" in workflow


def test_workflows_mint_short_lived_github_app_tokens() -> None:
    downstream = (ROOT / ".github/downstream/bump-agent-core.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

    action = "actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1"
    assert downstream.count(action) == 2
    assert release.count(action) == 1
    assert "DOWNSTREAM_BUMP_TOKEN" not in release
    assert "BUMP_PR_TOKEN" not in downstream
    assert "AGENT_CORE_REPO_TOKEN: ${{ secrets." not in downstream
