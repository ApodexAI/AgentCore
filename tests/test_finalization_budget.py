"""Wall-clock split between research and the finalization phase.

The invariant this file protects: research stop + a full late-tool overrun + the
finalization phase must fit inside the externally enforced ceiling. It does NOT
hold on the research deadline alone, because ``soft_wall_deadline_s`` floors
research at half the wall — on a short wall the planned reserve is silently
larger than what survives. ``remaining_phase_budget_s`` is what closes the gap,
so these tests pin both halves together.
"""

from __future__ import annotations

import time

import pytest

from agent_core.components.finalization import (
    check_wall_feasibility,
    remaining_phase_budget_s,
    resolve_report_wall_time,
    resolve_research_wall,
    soft_wall_deadline_s,
)

# ── hard_total_s: which config keys are real external ceilings ────────────

def test_env_wall_is_a_hard_ceiling(monkeypatch):
    monkeypatch.setenv("MIROHARNESS_TASK_WALL_TIME_S", "7200")
    wall = resolve_research_wall({}, reserve_s=1800, label_prefix="t")
    assert wall.hard_total_s == 7200
    assert wall.research_deadline_s == 5400


def test_research_only_budget_is_not_a_hard_ceiling(monkeypatch):
    """The reporter deliberately runs *outside* ``research_wall_time_s``."""
    monkeypatch.delenv("MIROHARNESS_TASK_WALL_TIME_S", raising=False)
    wall = resolve_research_wall(
        {"research_wall_time_s": 9000}, reserve_s=1800, label_prefix="t",
    )
    assert wall.hard_total_s == 0
    assert wall.research_deadline_s == 9000


def test_legacy_wall_deadline_s_is_a_hard_ceiling(monkeypatch):
    monkeypatch.delenv("MIROHARNESS_TASK_WALL_TIME_S", raising=False)
    wall = resolve_research_wall(
        {"wall_deadline_s": 3600}, reserve_s=600, label_prefix="t",
    )
    assert wall.hard_total_s == 3600
    assert wall.research_deadline_s == 3000


def test_shortest_positive_ceiling_wins(monkeypatch):
    monkeypatch.setenv("MIROHARNESS_TASK_WALL_TIME_S", "3600")
    wall = resolve_research_wall(
        {"wall_deadline_s": 9000}, reserve_s=600, label_prefix="t",
    )
    assert wall.hard_total_s == 3600


def test_no_ceiling_anywhere(monkeypatch):
    monkeypatch.delenv("MIROHARNESS_TASK_WALL_TIME_S", raising=False)
    wall = resolve_research_wall({}, reserve_s=600, label_prefix="t")
    assert wall.hard_total_s == 0
    assert wall.research_deadline_s == 0


# ── operator-friendly phase overrides ────────────────────────────────────

def test_research_wall_env_replaces_profile_budget(monkeypatch):
    monkeypatch.delenv("MIROHARNESS_TASK_WALL_TIME_S", raising=False)
    monkeypatch.setenv("RESEARCH_WALL_TIME", "300")

    wall = resolve_research_wall(
        {"research_wall_time_s": 9000}, reserve_s=1800, label_prefix="t",
    )

    assert wall.research_deadline_s == 300
    assert wall.hard_total_s == 0


def test_research_wall_env_can_intentionally_extend_profile(monkeypatch):
    monkeypatch.delenv("MIROHARNESS_TASK_WALL_TIME_S", raising=False)
    monkeypatch.setenv("RESEARCH_WALL_TIME", "12000")

    wall = resolve_research_wall(
        {"research_wall_time_s": 9000}, reserve_s=1800, label_prefix="t",
    )

    assert wall.research_deadline_s == 12000


@pytest.mark.parametrize("raw", ["", "0", "-1", "not-a-number"])
def test_invalid_research_wall_env_preserves_profile(monkeypatch, raw):
    monkeypatch.delenv("MIROHARNESS_TASK_WALL_TIME_S", raising=False)
    monkeypatch.setenv("RESEARCH_WALL_TIME", raw)

    wall = resolve_research_wall(
        {"research_wall_time_s": 9000}, reserve_s=1800, label_prefix="t",
    )

    assert wall.research_deadline_s == 9000


def test_task_wall_still_caps_research_env(monkeypatch):
    monkeypatch.setenv("RESEARCH_WALL_TIME", "9000")
    monkeypatch.setenv("MIROHARNESS_TASK_WALL_TIME_S", "7200")

    wall = resolve_research_wall({}, reserve_s=1800, label_prefix="t")

    assert wall.research_deadline_s == 5400
    assert wall.hard_total_s == 7200


def test_report_wall_env_replaces_profile_reserve(monkeypatch):
    monkeypatch.setenv("REPORT_WALL_TIME", "600")

    reserve = resolve_report_wall_time(
        {"wall_deadline_reserve_s": 1800}, default=180, label_prefix="t",
    )

    assert reserve == 600


@pytest.mark.parametrize("raw", ["", "-1", "not-a-number"])
def test_invalid_report_wall_env_preserves_profile(monkeypatch, raw):
    monkeypatch.setenv("REPORT_WALL_TIME", raw)

    reserve = resolve_report_wall_time(
        {"wall_deadline_reserve_s": 1800}, default=180, label_prefix="t",
    )

    assert reserve == 1800


# ── remaining_phase_budget_s ──────────────────────────────────────────────

def test_no_deadline_leaves_the_requested_budget_alone():
    assert remaining_phase_budget_s(1800, None) == 1800


def test_budget_is_clamped_to_the_time_left():
    deadline = time.monotonic() + 100
    assert remaining_phase_budget_s(1800, deadline) == pytest.approx(100, abs=2)


def test_budget_is_not_extended_past_the_request():
    deadline = time.monotonic() + 9000
    assert remaining_phase_budget_s(1800, deadline) == 1800


def test_expired_deadline_still_gets_one_short_attempt():
    """Zero would abort before the fail-open baseline is ever emitted."""
    assert remaining_phase_budget_s(1800, time.monotonic() - 500) == 1.0
    assert remaining_phase_budget_s(1800, time.monotonic() - 500, minimum_s=5) == 5


# ── the invariant the 0.5 floor used to break ─────────────────────────────

@pytest.mark.parametrize("hard_wall_s", [3600, 5400, 7200, 10800])
def test_research_plus_tool_overrun_plus_finalization_fits_the_wall(
    monkeypatch, hard_wall_s,
):
    """Worst case must not exceed the ceiling.

    Reproduces the online shape: ``tool_timeout_s: 1800`` and a 1800s reporter
    phase. Below 7200s the ``total * 0.5`` floor keeps research alive at the
    reserve's expense, so the finalization phase — not research — is what has
    to give. Holds while ``tool_timeout_s <= hard_wall / 2``; the case below
    that bound is covered by
    :func:`test_tool_timeout_over_half_the_wall_is_reported_infeasible`.
    """
    tool_timeout_s = 1800.0
    requested_phase_s = 1800.0
    reserve_s = max(3600.0, tool_timeout_s + requested_phase_s)

    monkeypatch.setenv("MIROHARNESS_TASK_WALL_TIME_S", str(hard_wall_s))
    wall = resolve_research_wall(
        {}, reserve_s=reserve_s, label_prefix="t",
    )

    # Research always keeps at least half the wall — a generous reserve must
    # never starve it.
    assert wall.research_deadline_s >= hard_wall_s * 0.5

    # Worst case: the deadline observer only fires at turn end, so a tool
    # started just before it can burn its whole timeout.
    started = time.monotonic()
    phase_start = wall.research_deadline_s + tool_timeout_s
    effective_phase_s = remaining_phase_budget_s(
        requested_phase_s,
        started + wall.hard_total_s - phase_start,
    )
    worst_case_finish = phase_start + effective_phase_s
    assert worst_case_finish <= hard_wall_s + 1.0


def test_soft_deadline_floor_is_why_the_clamp_is_required():
    """Pin the floor's behaviour so the pairing above cannot be dropped."""
    # Reserve fits: research gets the full remainder.
    assert soft_wall_deadline_s(10800, 3600) == 7200
    # Reserve does not fit: research is floored and the reserve is truncated
    # from 3600s to 1800s without any error.
    assert soft_wall_deadline_s(3600, 3600) == 1800


# ── the residual case no arithmetic can fix ───────────────────────────────

def test_tool_timeout_over_half_the_wall_is_reported_infeasible(caplog):
    """A tool already running cannot be shrunk — the config must change."""
    with caplog.at_level("WARNING"):
        feasible = check_wall_feasibility(
            hard_total_s=1800,
            research_deadline_s=900,
            tool_timeout_s=1800,
            landing_budget_s=1800,
            label_prefix="stateful",
        )
    assert feasible is False
    assert "cannot guarantee a final answer" in caplog.text
    assert "tool_timeout_s" in caplog.text


def test_feasible_config_is_silent(caplog):
    with caplog.at_level("WARNING"):
        feasible = check_wall_feasibility(
            hard_total_s=10800,
            research_deadline_s=7200,
            tool_timeout_s=1800,
            landing_budget_s=1800,
            label_prefix="stateful",
        )
    assert feasible is True
    assert caplog.text == ""


def test_no_external_ceiling_is_always_feasible(caplog):
    with caplog.at_level("WARNING"):
        assert check_wall_feasibility(
            hard_total_s=0,
            research_deadline_s=9000,
            tool_timeout_s=1800,
            landing_budget_s=1800,
            label_prefix="stateful",
        ) is True
    assert caplog.text == ""
