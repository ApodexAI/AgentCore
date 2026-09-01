"""Failure-aware stuck-target guard regression tests."""

from __future__ import annotations

import pytest

from agent_core.components.observers.stuck_target_guard import StuckTargetGuard
from agent_core.loop_types import ToolResult, TurnContext


def _ctx(turn: int, *tool_calls: dict) -> TurnContext:
    return TurnContext(
        turn=turn,
        max_turns=600,
        task_id="t",
        role_id="react_solver",
        ai_text="",
        thinking="",
        tool_calls=list(tool_calls),
        messages=[],
        usage=None,
        metadata={},
    )


def _bash(command: str) -> dict:
    return {"name": "bash", "args": {"command": command}}


def _fetch(url: str) -> dict:
    return {"name": "web_fetch", "args": {"url": url, "info_to_extract": "x"}}


def _search(query: str) -> dict:
    return {"name": "web_search", "args": {"q": query}}


async def _finish_turn(
    guard: StuckTargetGuard,
    turn: int,
    call: dict,
    *,
    output: str = "[ERROR]: target unavailable",
    is_error: bool = False,
):
    ctx = _ctx(turn, call)
    await guard.on_tool_result(
        ctx,
        ToolResult(
            name=call["name"],
            args=call.get("args") or {},
            result=output,
            duration_ms=1,
            tool_call_id=f"call-{turn}",
            is_error=is_error,
        ),
    )
    return await guard.on_turn_end(ctx)


@pytest.mark.asyncio
async def test_failed_target_hints_then_escalates_and_quarantines():
    guard = StuckTargetGuard(hint_after=3, escalate_after=6, window=20)
    fired: list[int] = []

    attempts = [
        _fetch("https://icmconjectures.com/1983-prob-8"),
        _bash('curl -sfL "https://icmconjectures.com/1983-prob-8"'),
        _bash('curl -sfL "https://icmconjectures.com/mnt/app.js"'),
        _bash('curl -sfL "https://icmconjectures.com/api/conjectures"'),
        _bash('curl -sfL "http://icmconjectures.com/data.json"'),
        _bash('curl -sfL "https://icmconjectures.com/"'),
    ]
    for turn, call in enumerate(attempts, 1):
        result = await _finish_turn(guard, turn, call)
        if result is not None:
            fired.append(turn)

    assert fired == [3, 6]
    blocked = await guard.on_tool_call(_ctx(7), _fetch(
        "https://icmconjectures.com/another-path",
    ))
    assert blocked is not None
    assert blocked.skip_with_result
    assert "disabled" in blocked.skip_with_result


@pytest.mark.asyncio
async def test_successful_pages_on_one_host_never_trip_the_guard():
    """Host frequency alone is not no-progress."""
    guard = StuckTargetGuard(hint_after=3, escalate_after=6, window=20)
    for turn in range(1, 10):
        result = await _finish_turn(
            guard,
            turn,
            _fetch(f"https://docs.python.org/3/library/page-{turn}.html"),
            output=f"Useful documentation page {turn}",
        )
        assert result is None


@pytest.mark.asyncio
async def test_success_resets_prior_failure_history():
    guard = StuckTargetGuard(hint_after=3, escalate_after=6, window=20)
    target = "https://docs.example.com"

    assert await _finish_turn(guard, 1, _fetch(f"{target}/a")) is None
    assert await _finish_turn(guard, 2, _fetch(f"{target}/b")) is None
    assert await _finish_turn(
        guard,
        3,
        _fetch(f"{target}/working"),
        output="Recovered content",
    ) is None

    # The next failure is the first failure in a new sequence, not the third.
    assert await _finish_turn(guard, 4, _fetch(f"{target}/c")) is None


@pytest.mark.asyncio
async def test_scheme_less_search_domain_counts_when_search_fails():
    guard = StuckTargetGuard(hint_after=2, escalate_after=4, window=20)
    first = await _finish_turn(
        guard,
        1,
        _search("site:icmconjectures.com 1983 problem 8"),
        output="No results found.",
    )
    second = await _finish_turn(
        guard,
        2,
        _search("icmconjectures.com exact problem statement"),
        output="No results found for query",
    )
    assert first is None
    assert second is not None
    assert "icmconjectures.com" in second.inject_messages[0]


@pytest.mark.asyncio
async def test_a_search_hit_does_not_vouch_for_a_host():
    """Only a fetch-class tool can clear a host's failure history.

    A search engine returning a snippet about ``example.com`` proves the index
    has it, not that the page can be read. Measured on the 2026-07-28 trace:
    6 searches and 7 shell commands "succeeded" against a host that never once
    returned its content, and each one reset the guard — which is why the guard
    never fired on the very run it was written for.
    """
    guard = StuckTargetGuard(hint_after=2, escalate_after=4, window=20)
    await _finish_turn(
        guard, 1, _search("site:example.com first"), output="No results found.",
    )
    assert await _finish_turn(
        guard,
        2,
        _search("site:example.com second"),
        output="Useful result https://example.com/answer",
    ) is None, "a search hit is neutral — neither a failure nor a vouch"
    third = await _finish_turn(
        guard, 3, _search("site:example.com third"), output="No results found.",
    )
    assert third is not None, "the earlier failure must still be on the books"
    assert "example.com" in third.inject_messages[0]


@pytest.mark.asyncio
async def test_only_a_fetch_success_clears_the_history():
    guard = StuckTargetGuard(hint_after=2, escalate_after=4, window=20)
    await _finish_turn(
        guard, 1, _search("site:example.com first"), output="No results found.",
    )
    assert await _finish_turn(
        guard,
        2,
        _fetch("https://example.com/answer"),
        output="R" * 900,
    ) is None
    assert await _finish_turn(
        guard, 3, _search("site:example.com third"), output="No results found.",
    ) is None, "the fetch proved the host readable, so the count restarts"


@pytest.mark.asyncio
async def test_a_shell_http_refusal_is_a_failure_not_a_success():
    """``bash`` exits 0 while printing the refusal — 34 of the trace's 39
    attempts on the dead host looked like tool-level successes this way."""
    guard = StuckTargetGuard(hint_after=3, escalate_after=99, window=20)
    outputs = [
        "403 --  /api/conjectures\n403 --  /styles.css\n",
        "/ HTTP 403 b'error code: 1010'\n",
        "ERR <HTTPError 403: 'Forbidden'>\n",
    ]
    fired = []
    for turn, out in enumerate(outputs, 1):
        got = await _finish_turn(
            guard, turn, _bash(f'curl -sS "https://dead.test/x{turn}"'), output=out,
        )
        if got is not None:
            fired.append(turn)
    assert fired == [3]


@pytest.mark.asyncio
async def test_a_markup_dump_neither_accuses_nor_vouches():
    """A hand-rolled scrape that printed HTML may or may not hold the content."""
    guard = StuckTargetGuard(hint_after=2, escalate_after=99, window=20)
    await _finish_turn(
        guard, 1, _bash('curl -sS "https://spa.test/x"'),
        output="[ERROR]: Scraping failed: boom",
    )
    assert await _finish_turn(
        guard,
        2,
        _bash('curl -sS "https://spa.test/y" -o p.html; head -c 200 p.html'),
        output='<!doctype html>\n<html><head><script src="/m.js"></script></head>',
    ) is None, "a markup dump must not vouch for the host"
    third = await _finish_turn(
        guard, 3, _bash('curl -sS "https://spa.test/z"'),
        output="[ERROR]: Scraping failed: boom",
    )
    assert third is not None, "…and must not clear the earlier failure either"


@pytest.mark.asyncio
async def test_search_only_failures_never_quarantine_a_host():
    """Not being able to FIND a page must not disable FETCHING it.

    Otherwise a poorly-indexed domain gets cut off before it is ever tried.
    """
    guard = StuckTargetGuard(hint_after=2, escalate_after=4, window=20)
    for turn in range(1, 13):
        await _finish_turn(
            guard, turn, _search(f"site:goodsite.com q{turn}"),
            output="No results found.",
        )
    fetch = {"name": "web_fetch", "args": {"url": "https://goodsite.com/real-page"}}
    assert await guard.on_tool_call(_ctx(13, fetch), fetch) is None
    assert guard._blocked_hosts == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "is_error"),
    [
        ("search backend timeout", True),
        ("", False),
    ],
)
async def test_search_backend_failures_are_soft_not_quarantine(
    output: str,
    is_error: bool,
):
    """Search infrastructure cannot prove that the target host is dead."""
    guard = StuckTargetGuard(hint_after=2, escalate_after=3, window=10)
    for turn in range(1, 7):
        await _finish_turn(
            guard,
            turn,
            _search(f"site:healthy.example query {turn}"),
            output=output,
            is_error=is_error,
        )

    fetch = _fetch("https://healthy.example/real-page")
    assert await guard.on_tool_call(_ctx(7, fetch), fetch) is None
    assert guard._blocked_hosts == set()


@pytest.mark.asyncio
async def test_batched_fetch_attributes_each_result_to_its_own_host():
    """One failed URL must not quarantine a successful sibling in the batch."""
    guard = StuckTargetGuard(hint_after=2, escalate_after=3, window=10)
    call = {
        "name": "web_fetch",
        "args": {
            "url": [
                "https://good.example/article",
                "https://dead.example/article",
            ],
            "info_to_extract": ["article", "article"],
        },
    }
    output = (
        "[1] URL: https://good.example/article\n"
        f"    Info: {'useful content ' * 60}\n\n"
        "[2] URL: https://dead.example/article\n"
        "    Info: [ERROR]: Scraping failed: origin refused the request"
    )
    for turn in range(1, 4):
        await _finish_turn(guard, turn, call, output=output)

    assert guard._blocked_hosts == {"dead.example"}
    good = _fetch("https://good.example/another")
    dead = _fetch("https://dead.example/another")
    assert await guard.on_tool_call(_ctx(4, good), good) is None
    assert await guard.on_tool_call(_ctx(4, dead), dead) is not None


@pytest.mark.asyncio
async def test_fetch_focus_url_is_not_treated_as_a_network_target():
    guard = StuckTargetGuard(hint_after=2, escalate_after=3, window=10)
    guard._blocked_hosts.add("dead.example")
    call = {
        "name": "web_fetch",
        "args": {
            "url": "https://good.example/article",
            "info_to_extract": "Compare the claim with https://dead.example/source",
        },
    }

    assert await guard.on_tool_call(_ctx(1, call), call) is None


@pytest.mark.asyncio
async def test_a_short_but_delivered_page_never_quarantines_its_host():
    """``[POSSIBLY NOT RENDERED]`` ships content and says it may be useful.

    A small JSON API answering 12 times must not be cut off for being terse.
    """
    guard = StuckTargetGuard(hint_after=2, escalate_after=4, window=20)
    url = "https://api.test/v1/status"
    for turn in range(1, 13):
        await _finish_turn(
            guard, turn, _fetch(url),
            output=(
                f"[POSSIBLY NOT RENDERED] The page at {url} remained "
                'suspiciously short…\n\n{"ok":true,"n":3}'
            ),
        )
    call = {"name": "web_fetch", "args": {"url": url}}
    assert await guard.on_tool_call(_ctx(13, call), call) is None
    assert guard._blocked_hosts == set()


@pytest.mark.asyncio
async def test_filenames_in_a_search_query_are_not_hosts():
    guard = StuckTargetGuard(hint_after=2, escalate_after=99, window=20)
    for turn in (1, 2, 3):
        got = await _finish_turn(
            guard, turn, _search(f"read report.pdf and data.json part {turn}"),
            output="No results found.",
        )
        assert got is None, "report.pdf / data.json are filenames, not targets"


@pytest.mark.asyncio
async def test_interleaving_other_hosts_does_not_hide_failed_target():
    guard = StuckTargetGuard(hint_after=4, escalate_after=99, window=20)
    fired = []
    for turn in range(1, 9):
        if turn % 2:
            call = _bash(f'curl -sfL "https://target.test/x?try={turn}"')
            output = "[Exit code 22]\ncurl: HTTP 404"
        else:
            call = _fetch(f"https://elsewhere-{turn}.test/y")
            output = "Useful unrelated content"
        if await _finish_turn(guard, turn, call, output=output) is not None:
            fired.append(turn)
    assert fired == [7]


@pytest.mark.asyncio
async def test_local_only_turns_are_neutral_and_do_not_age_the_window():
    guard = StuckTargetGuard(hint_after=3, escalate_after=99, window=3)
    target = _fetch("https://spa.test/x")
    assert await _finish_turn(guard, 1, target) is None
    assert await _finish_turn(guard, 2, target) is None

    for turn, command in enumerate(
        ("ls /workspace", "python3 solve.py", "pytest -q", "cat out.txt"),
        3,
    ):
        assert await _finish_turn(
            guard,
            turn,
            _bash(command),
            output="local success",
        ) is None

    # Local work did not age either failed attempt out of the network window.
    assert await _finish_turn(guard, 7, target) is not None


@pytest.mark.asyncio
async def test_unrelated_network_turns_age_old_failures_out():
    guard = StuckTargetGuard(hint_after=3, escalate_after=4, window=4)
    target = _fetch("https://spa.test/x")
    assert await _finish_turn(guard, 1, target) is None
    assert await _finish_turn(guard, 2, target) is None

    for turn in (3, 4, 5, 6):
        assert await _finish_turn(
            guard,
            turn,
            _search(f"unrelated query {turn}"),
            output="Useful search results",
        ) is None

    # Both target failures aged out of the last four network turns.
    assert await _finish_turn(guard, 7, target) is None


@pytest.mark.asyncio
async def test_hint_rearms_after_an_old_failure_burst_ages_out():
    guard = StuckTargetGuard(hint_after=2, escalate_after=4, window=4)
    target = _fetch("https://spa.test/x")
    assert await _finish_turn(guard, 1, target) is None
    assert await _finish_turn(guard, 2, target) is not None

    for turn in (3, 4, 5, 6):
        await _finish_turn(
            guard,
            turn,
            _search(f"unrelated query {turn}"),
            output="Useful search results",
        )

    assert await _finish_turn(guard, 7, target) is None
    assert await _finish_turn(guard, 8, target) is not None


@pytest.mark.asyncio
async def test_reader_and_archive_hosts_do_not_mask_the_real_target():
    guard = StuckTargetGuard(hint_after=3, escalate_after=99, window=20)
    seq = [
        _fetch("https://icmconjectures.com/1983-prob-8"),
        _bash('curl "https://r.jina.ai/https://icmconjectures.com/1983-prob-8"'),
        _bash(
            'curl "https://web.archive.org/web/2024id_/'
            'https://icmconjectures.com/x"',
        ),
    ]
    results = [
        await _finish_turn(guard, turn, call)
        for turn, call in enumerate(seq, 1)
    ]
    assert [result is not None for result in results] == [False, False, True]


@pytest.mark.asyncio
async def test_transparent_hosts_are_not_quarantined():
    guard = StuckTargetGuard(hint_after=2, escalate_after=3, window=20)
    for turn in range(1, 6):
        result = await _finish_turn(
            guard,
            turn,
            _fetch("https://r.jina.ai/http://archive.org/page"),
        )
        assert result is None


@pytest.mark.asyncio
async def test_possibly_not_rendered_marker_is_a_failure():
    guard = StuckTargetGuard(hint_after=2, escalate_after=4, window=20)
    output = (
        "[POSSIBLY NOT RENDERED] The result remained suspiciously short.\n\n"
        "navigation only"
    )
    assert await _finish_turn(
        guard, 1, _fetch("https://spa.test/x"), output=output,
    ) is None
    assert await _finish_turn(
        guard, 2, _fetch("https://spa.test/x"), output=output,
    ) is not None


@pytest.mark.asyncio
async def test_shipped_defaults_quarantine_the_real_trace_shape():
    """End-to-end on the shape that defeated two earlier versions of this guard.

    Condensed from the 2026-07-28 trace: attempts on one host, no two alike,
    interleaved with searches that DO return snippets about it, unrelated hosts,
    local steps, and shell commands whose refusals are printed on stdout with a
    zero exit code. The first version (consecutive streak) never fired; the
    second (any outcome vouches) never fired either. With the shipped defaults
    the host must be nudged and then quarantined, and the quarantine must
    actually stop the next attempt.
    """
    guard = StuckTargetGuard()  # 6 / 10 / 20
    dead = "https://icmconjectures.com"
    steps: list[tuple[dict, str]] = [
        (_fetch(f"{dead}/1983-prob-8"), "[ERROR]: Scraping failed: shell"),
        (_bash(f'curl -sSL "{dead}/1983-prob-8" -o p.html; head -c 99 p.html'),
         "<!doctype html><html><script src=/m.js></script>"),          # neutral
        (_bash("cat p.html | head -5"), "<!doctype html>"),             # local
        (_search("icmconjectures.com 1983 problem 8"),
         "[1] Title: ICM Conjectures\n    Snippet: This dataset comes from…"),
        (_bash(f'curl -sS "{dead}/api/conjectures"'), "403 --  /api/conjectures"),
        (_bash(f'curl -sS "{dead}/data.json"'), "/ HTTP 403 b'error code: 1010'"),
        (_fetch("https://en.wikipedia.org/wiki/Percolation"), "R" * 900),
        (_bash(f'curl -sS "{dead}/styles.css"'), "ERR <HTTPError 403: 'Forbidden'>"),
        (_bash("python3 -c 'print(1)'"), "1"),                          # local
        (_bash(f'curl -sS -H "UA: bot" "{dead}/"'), "403 --  /"),
        (_search("site:icmconjectures.com 1983"), "No results found."),  # soft
        (_bash(f'curl -sS "{dead}/robots.txt"'), "403 --  /robots.txt"),
        (_bash(f'curl -sS "{dead}/sitemap.xml"'), "curl: (22) HTTP 403"),
        (_bash(f'curl -sS "{dead}/1983-prob-8.json"'), "403 --  /1983-prob-8.json"),
        (_bash(f'curl -sS --http1.1 "{dead}/"'), "Forbidden"),
        (_bash(f'curl -sS "{dead}/index.html"'), "HTTP 403"),
    ]
    hint_turn = block_turn = None
    for turn, (call, output) in enumerate(steps, 1):
        got = await _finish_turn(guard, turn, call, output=output)
        if got is None:
            continue
        message = got.inject_messages[0]
        if "blocked" in message and block_turn is None:
            block_turn = turn
        elif hint_turn is None:
            hint_turn = turn

    assert hint_turn is not None, "the guard must speak up"
    assert block_turn is not None, "the quarantine must be reachable, not decorative"
    assert hint_turn < block_turn
    assert "icmconjectures.com" in guard._blocked_hosts
    # Wikipedia was read successfully and must stay available.
    assert "en.wikipedia.org" not in guard._blocked_hosts
    nxt = _bash(f'curl -sS "{dead}/last-try"')
    intervention = await guard.on_tool_call(_ctx(len(steps) + 1, nxt), nxt)
    assert intervention is not None
    assert "[STUCK TARGET BLOCKED]" in intervention.skip_with_result


@pytest.mark.asyncio
async def test_loop_start_resets_failures_and_quarantine():
    guard = StuckTargetGuard(hint_after=2, escalate_after=3, window=20)
    target = _fetch("https://spa.test/x")
    for turn in (1, 2, 3):
        await _finish_turn(guard, turn, target)
    assert await guard.on_tool_call(_ctx(4), target) is not None

    await guard.on_loop_start(None)
    assert await guard.on_tool_call(_ctx(1), target) is None
    assert await _finish_turn(guard, 1, target) is None
