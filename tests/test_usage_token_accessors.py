"""One place to read token counts out of a usage mapping.

``TurnContext.usage`` is deliberately an open ``Mapping`` (see the
``UsageMetadata`` docstring): ``extract_usage`` normalises to
``prompt_tokens`` / ``completion_tokens``, but hosts hand in raw provider
payloads and test doubles use the ``input_tokens`` / ``output_tokens``
aliases. Every consumer that read only one spelling has been silently inert at
some point — the context-size guard and the budget observer both were — so the
accessors are the single place that knows about both.
"""

from __future__ import annotations

import pytest

from agent_core.runtime.loop import usage_input_tokens, usage_output_tokens


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"prompt_tokens": 120}, 120),
        ({"input_tokens": 120}, 120),
        ({"prompt_tokens": 120, "input_tokens": 999}, 120),
        ({"prompt_tokens": 0, "input_tokens": 120}, 120),
        ({}, 0),
        (None, 0),
        ({"prompt_tokens": None}, 0),
    ],
)
def test_input_tokens(usage, expected):
    assert usage_input_tokens(usage) == expected


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"completion_tokens": 40}, 40),
        ({"output_tokens": 40}, 40),
        ({"completion_tokens": 40, "output_tokens": 999}, 40),
        ({"completion_tokens": 0, "output_tokens": 40}, 40),
        ({}, 0),
        (None, 0),
    ],
)
def test_output_tokens(usage, expected):
    assert usage_output_tokens(usage) == expected


def test_reads_what_extract_usage_produces():
    """Pins the accessors against the real producer, not a hand-written dict."""
    from agent_core.llm import LLMResponse
    from agent_core.runtime.loop import extract_usage

    usage = extract_usage(LLMResponse(
        content="hi",
        usage={"prompt_tokens": 31, "completion_tokens": 7},
    ))

    assert usage is not None
    assert usage_input_tokens(usage) == 31
    assert usage_output_tokens(usage) == 7
