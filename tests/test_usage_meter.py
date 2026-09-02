from __future__ import annotations

from unittest.mock import patch

from agent_core.runtime.usage_meter import (
    ExternalAPIMeter,
    bind_usage_meter,
    close_meter_span,
    get_usage_meter,
    open_meter_span,
    record_api_request,
    record_llm_usage,
    record_tool_call,
    reset_usage_meter,
    set_meter_gauge,
)


def test_meter_records_requests_tools_and_isolates_snapshots() -> None:
    meter = ExternalAPIMeter()
    meter.record_api_request(
        "search",
        requests=2,
        cache_hits=1,
        retries=1,
        bytes_read=2.5,
    )
    meter.record_tool_call("web_search")
    first = meter.snapshot()
    first["external_apis"]["search"]["requests"] = 99
    assert meter.snapshot() == {
        "external_apis": {
            "search": {
                "requests": 2.0,
                "cache_hits": 1.0,
                "retries": 1.0,
                "errors": 0.0,
                "bytes_read": 2.5,
            },
        },
        "tools": {"web_search": 1},
    }


def test_open_spans_and_gauges_are_monotonic_and_idempotent() -> None:
    meter = ExternalAPIMeter()
    with patch("agent_core.runtime.usage_meter.time.monotonic") as clock:
        clock.side_effect = [10.0, 12.5, 14.0, 16.0]
        meter.open_span("sandbox", "one")
        meter.open_span("sandbox", "one")
        meter.set_gauge("sandbox", "ttl", 30)
        meter.set_gauge("sandbox", "ttl", 20)
        current = meter.snapshot()["external_apis"]["sandbox"]
        assert current["sandbox_seconds"] == 4.0
        assert current["spans_open"] == 1
        assert current["ttl"] == 30.0
        meter.close_span("sandbox", "one")
    assert meter.snapshot()["external_apis"]["sandbox"][
        "sandbox_seconds"
    ] == 6.0


def test_context_helpers_are_noop_safe_and_resettable() -> None:
    calls: list[dict[str, object]] = []
    meter = ExternalAPIMeter(llm_recorder=lambda **kw: calls.append(kw))
    token = bind_usage_meter(meter)
    try:
        assert get_usage_meter() is meter
        record_api_request("api", errors=1)
        record_tool_call("tool")
        record_llm_usage(model="m", prompt_tokens=2)
        set_meter_gauge("api", "limit", 5)
        with patch(
            "agent_core.runtime.usage_meter.time.monotonic",
            side_effect=[1.0, 2.0],
        ):
            open_meter_span("api", "span", field="seconds")
            close_meter_span("api", "span")
    finally:
        reset_usage_meter(token)
    assert get_usage_meter() is None
    assert calls == [{"model": "m", "prompt_tokens": 2}]
    assert meter.snapshot()["tools"] == {"tool": 1}


def test_llm_recorder_failure_is_suppressed() -> None:
    def fail(**_kwargs: object) -> None:
        raise RuntimeError("accounting unavailable")

    ExternalAPIMeter(llm_recorder=fail).record_llm_usage(model="m")
