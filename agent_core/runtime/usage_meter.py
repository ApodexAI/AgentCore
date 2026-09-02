"""Task-local, thread-safe metering for external APIs, tools, and raw LLMs.

The meter records transport-level facts without knowing product prices,
billing policy, or telemetry backends. Hosts bind one meter per execution
context and may supply an LLM recorder callback for their own aggregation.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections.abc import Callable
from typing import Any

_BASE_FIELDS: tuple[str, ...] = (
    "requests",
    "cache_hits",
    "retries",
    "errors",
)


def _empty_slot() -> dict[str, Any]:
    """A provider view for gauge/span-only providers: counts stay integral."""
    return dict.fromkeys(_BASE_FIELDS, 0)


def _render(field: str, value: float) -> Any:
    """Report the four wire counters as ints; durations keep 2 decimals."""
    if field in _BASE_FIELDS:
        return int(value)
    return round(value, 2)


class ExternalAPIMeter:
    """Accumulate external request, tool-call, gauge, and span measurements."""

    def __init__(self, *, llm_recorder: Callable[..., None] | None = None) -> None:
        self._lock = threading.Lock()
        self._providers: dict[str, dict[str, float]] = {}
        self._tool_counts: dict[str, int] = {}
        self._open_spans: dict[tuple[str, str], tuple[float, str]] = {}
        self._gauges: dict[tuple[str, str], float] = {}
        self._llm_recorder = llm_recorder

    def record_api_request(
        self,
        provider: str,
        *,
        requests: int = 1,
        cache_hits: int = 0,
        retries: int = 0,
        errors: int = 0,
        **extra: float,
    ) -> None:
        """Fold one or more wire-level events into a provider slot."""
        with self._lock:
            slot = self._slot_locked(provider)
            slot["requests"] += requests
            slot["cache_hits"] += cache_hits
            slot["retries"] += retries
            slot["errors"] += errors
            for field, value in extra.items():
                slot[field] = slot.get(field, 0.0) + float(value)

    def record_tool_call(self, name: str) -> None:
        with self._lock:
            self._tool_counts[name] = self._tool_counts.get(name, 0) + 1

    def set_gauge(self, provider: str, field: str, value: float) -> None:
        """Record a non-additive value using conservative max-wins folding."""
        with self._lock:
            key = (provider, field)
            self._gauges[key] = max(self._gauges.get(key, 0.0), float(value))

    def record_llm_usage(self, **kwargs: Any) -> None:
        """Forward usage to the host callback without breaking the measured call."""
        if self._llm_recorder is not None:
            with contextlib.suppress(Exception):
                self._llm_recorder(**kwargs)

    def open_span(
        self,
        provider: str,
        key: str,
        *,
        field: str = "sandbox_seconds",
    ) -> None:
        """Start an idempotent monotonic-clock span."""
        with self._lock:
            self._open_spans.setdefault(
                (provider, key),
                (time.monotonic(), field),
            )

    def close_span(self, provider: str, key: str) -> None:
        """Close a span and fold its elapsed time into the provider slot."""
        with self._lock:
            span = self._open_spans.pop((provider, key), None)
            if span is None:
                return
            started, field = span
            slot = self._slot_locked(provider)
            slot[field] = slot.get(field, 0.0) + (time.monotonic() - started)

    def snapshot(self) -> dict[str, Any]:
        """Return an isolated snapshot including elapsed still-open spans."""
        with self._lock:
            now = time.monotonic()
            apis: dict[str, dict[str, Any]] = {
                provider: {key: _render(key, value) for key, value in slot.items()}
                for provider, slot in self._providers.items()
            }
            for (provider, _key), (started, field) in self._open_spans.items():
                view = apis.setdefault(provider, _empty_slot())
                view[field] = round(
                    float(view.get(field, 0.0)) + now - started,
                    2,
                )
                view["spans_open"] = int(view.get("spans_open", 0)) + 1
            for (provider, field), value in self._gauges.items():
                view = apis.setdefault(provider, _empty_slot())
                view[field] = round(value, 2)
            return {
                "external_apis": apis,
                "tools": dict(self._tool_counts),
            }

    def _slot_locked(self, provider: str) -> dict[str, float]:
        slot = self._providers.get(provider)
        if slot is None:
            slot = dict.fromkeys(_BASE_FIELDS, 0.0)
            self._providers[provider] = slot
        return slot


_CURRENT_METER: contextvars.ContextVar[ExternalAPIMeter | None] = (
    contextvars.ContextVar("agent_core_usage_meter", default=None)
)


def bind_usage_meter(
    meter: ExternalAPIMeter,
) -> contextvars.Token[ExternalAPIMeter | None]:
    """Bind ``meter`` to the current context and return its reset token."""
    return _CURRENT_METER.set(meter)


def reset_usage_meter(
    token: contextvars.Token[ExternalAPIMeter | None],
) -> None:
    _CURRENT_METER.reset(token)


def get_usage_meter() -> ExternalAPIMeter | None:
    return _CURRENT_METER.get()


def record_api_request(provider: str, **kwargs: Any) -> None:
    meter = get_usage_meter()
    if meter is not None:
        meter.record_api_request(provider, **kwargs)


def record_tool_call(name: str) -> None:
    meter = get_usage_meter()
    if meter is not None:
        meter.record_tool_call(name)


def record_llm_usage(**kwargs: Any) -> None:
    meter = get_usage_meter()
    if meter is not None:
        meter.record_llm_usage(**kwargs)


def open_meter_span(
    provider: str,
    key: str,
    *,
    field: str = "sandbox_seconds",
) -> None:
    meter = get_usage_meter()
    if meter is not None:
        meter.open_span(provider, key, field=field)


def close_meter_span(provider: str, key: str) -> None:
    meter = get_usage_meter()
    if meter is not None:
        meter.close_span(provider, key)


def set_meter_gauge(provider: str, field: str, value: float) -> None:
    meter = get_usage_meter()
    if meter is not None:
        meter.set_gauge(provider, field, value)


__all__ = [
    "ExternalAPIMeter",
    "bind_usage_meter",
    "close_meter_span",
    "get_usage_meter",
    "open_meter_span",
    "record_api_request",
    "record_llm_usage",
    "record_tool_call",
    "reset_usage_meter",
    "set_meter_gauge",
]
