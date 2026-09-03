from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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


def _release_workflow() -> str:
    return (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")


def test_pypi_publishing_uses_trusted_publishing_without_any_token() -> None:
    """A stored PyPI token is the thing Trusted Publishing exists to remove.

    Reintroducing one would be a silent downgrade: publishing keeps working, so
    nothing fails to reveal that a long-lived credential is back in the repo.
    """
    release = _release_workflow()

    assert "pypa/gh-action-pypi-publish@" in release
    for forbidden in ("PYPI_API_TOKEN", "PYPI_TOKEN", "TWINE_PASSWORD", "password:"):
        assert forbidden not in release, forbidden


def test_id_token_permission_is_scoped_to_the_publish_job_alone() -> None:
    """`id-token: write` mints the identity PyPI trusts.

    Granted workflow-wide, every third-party action in the build could request
    that identity, so the permission must sit on the publishing job only.
    """
    workflow = yaml.safe_load(_release_workflow())
    jobs = workflow["jobs"]

    assert jobs["publish-pypi"]["permissions"] == {"id-token": "write"}
    assert jobs["publish-pypi"]["needs"] == "build"
    assert "id-token" not in jobs["build"]["permissions"]
    assert jobs["build"]["permissions"] == {"contents": "read"}
    # Workflow-level permissions would apply to both jobs.
    assert "permissions" not in workflow


def test_github_release_is_published_last_and_is_retry_safe() -> None:
    """A failed upload must not advertise a release that is absent from PyPI."""
    workflow = yaml.safe_load(_release_workflow())
    jobs = workflow["jobs"]

    assert jobs["publish-github"]["needs"] == "publish-pypi"
    assert jobs["publish-github"]["permissions"] == {"contents": "write"}
    release = _release_workflow()
    assert 'gh release view "$TAG"' in release
    assert 'gh release upload "$TAG" --clobber' in release
    assert "packages-dir: release-artifacts/dist/" in release


def test_metadata_is_validated_before_a_version_number_is_consumed() -> None:
    """A PyPI version can never be reused, not even after deletion.

    Invalid metadata must fail the build rather than burn the number.
    """
    assert "twine check dist/*" in _release_workflow()


def test_the_removed_dispatch_machinery_has_not_returned() -> None:
    """Downstream bumps are Dependabot's job now.

    The dispatch/repin path needed a cross-repository write credential, which is
    exactly what publishing to PyPI removed the need for.
    """
    assert not (ROOT / ".github/downstream").exists()

    release = _release_workflow()
    for forbidden in ("repository_dispatch", "DOWNSTREAM_BUMP_TOKEN", "create-github-app-token"):
        assert forbidden not in release, forbidden


def test_release_verifies_the_wheel_installs_and_imports() -> None:
    """The artifact, not just the tree, is what consumers get.

    A wheel that builds but cannot be imported would consume the version number
    before anyone noticed, and PyPI never releases a number back.
    """
    release = _release_workflow()

    assert "uv pip install --python /tmp/wheel-smoke/bin/python dist/*.whl" in release
    assert "import agent_core" in release


def test_typed_marker_backs_the_typing_classifier() -> None:
    """`Typing :: Typed` is a promise consumers' type checkers rely on."""
    import tomllib

    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    assert "Typing :: Typed" in project["classifiers"]
    assert (ROOT / "agent_core/py.typed").is_file()


def test_license_is_declared_as_an_spdx_expression() -> None:
    """PEP 639: an SPDX expression, and no redundant License:: classifier.

    Declaring both makes PyPI reject the upload — after the tag already exists.
    """
    import tomllib

    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]


def test_github_release_job_names_the_repository_explicitly() -> None:
    """The GitHub Release job never checks out the repository.

    Without GH_REPO, `gh release` tries to infer the target from a git remote
    and dies with "not a git repository". That failure is not recoverable by
    rerunning: a rerun replays the workflow definition at the tag, and by then
    PyPI has already published the version — which can never be reused. This
    happened on v0.3.0; the release had to be created by hand.
    """
    workflow = yaml.safe_load(_release_workflow())
    job = workflow["jobs"]["publish-github"]

    assert not any("checkout" in str(step.get("uses", "")) for step in job["steps"])

    publish = next(s for s in job["steps"] if s.get("name") == "Publish GitHub Release")
    assert publish["env"]["GH_REPO"] == "${{ github.repository }}"
