"""Consistency of the three token budgets a loop is configured with.

``config/sglang/README.md`` already describes the context/output relationship,
and ``docker/sglang-doctor.sh`` checks it — but only for the sglang compose path,
expressed in ``SGLANG_*`` variables, by a script an operator has to remember to
run. The values that actually reach the loop are the profile's
``max_len`` / ``max_input_tokens`` and the LLM's ``max_tokens``, and those can be
set straight through ``OPENAI_CONTEXT_WINDOW`` / ``OPENAI_MAX_INPUT_TOKENS`` /
``OPENAI_MAX_TOKENS`` against any endpoint, with nothing checking them at all.

A violation is not visibly a misconfiguration. It surfaces as a provider
rejection mid-run, or as a reasoning watchdog that cannot reliably pre-empt the
provider's output cap. Both read as a runtime fault.

Warnings only. A deploy that is running today with an unusual combination must
keep running; the point is to say so once, at startup, in terms of the knob to
turn.
"""

from __future__ import annotations

import logging

from agent_core.runtime.loop.tiered_compact import (
    DEFAULT_TRIGGER_RATIO as COMPACTION_TRIGGER_RATIO,
)
from agent_core.runtime.loop.tiered_compact import compaction_trigger_tokens

logger = logging.getLogger(__name__)

__all__ = ["COMPACTION_TRIGGER_RATIO", "check_context_budget"]

# The ratio of ``max_len`` at which tiered compaction triggers is re-exported,
# not restated: a second literal here could silently drift from the trigger, and
# a check that disagreed with the mechanism it describes would be worse than no
# check. A host that passes its own ``ratio`` to ``compaction_trigger_tokens``
# must pass the same one here.


def check_context_budget(
    *,
    max_len: int,
    max_input_tokens: int | None,
    max_tokens: int | None,
    reasoning_only_max_tokens: int | None = None,
    ratio: float = COMPACTION_TRIGGER_RATIO,
    label: str,
) -> list[str]:
    """Warn about token budgets that cannot all hold at once.

    Returns the warning strings, so a caller (or a test) can assert on them
    rather than scraping the log. Anything unset or non-positive is skipped: a
    missing bound is a deliberate configuration, not an inconsistency.

    ``ratio`` must be the one the loop's compaction trigger was built with; the
    margin reported below is meaningless against any other.
    """
    problems: list[str] = []

    # This relationship does not depend on knowing the context window. Keep it
    # outside the max_len-gated checks so profiles without tiered compaction
    # still get the warning.
    if (
        max_tokens is not None
        and max_tokens > 0
        and reasoning_only_max_tokens is not None
        and reasoning_only_max_tokens > 0
        and reasoning_only_max_tokens >= max_tokens
    ):
        # The reasoning-only watchdog cancels a stream that is still ONLY
        # reasoning. At or above max_tokens it cannot reliably pre-empt the
        # provider's completion cap, so the runaway path handles the reply
        # instead — a different mechanism with a different recovery.
        problems.append(
            f"{label}: reasoning_only_max_tokens "
            f"({reasoning_only_max_tokens:,}) is not below max_tokens "
            f"({max_tokens:,}); the reasoning watchdog cannot reliably fire "
            f"before the provider truncates the reply"
        )

    if max_len <= 0:
        for problem in problems:
            logger.warning("%s", problem)
        return problems

    trigger = compaction_trigger_tokens(max_len, ratio)

    if (
        max_input_tokens is not None
        and max_input_tokens > 0
        and max_tokens is not None
        and max_tokens > 0
        and max_input_tokens + max_tokens > max_len
    ):
        # A prompt sitting at the guard limit plus a full-length completion does
        # not fit what the endpoint serves. Nothing catches this before the
        # provider does, and by then the turn is lost.
        problems.append(
            f"{label}: max_input_tokens ({max_input_tokens:,}) + max_tokens "
            f"({max_tokens:,}) = {max_input_tokens + max_tokens:,} exceeds "
            f"max_len ({max_len:,}); a full-length reply to a prompt at the "
            f"input guard cannot fit the served context"
        )

    if max_tokens is not None and max_tokens > 0 and max_len - trigger > 0:
        # The margin the compaction trigger has before the hard wall, and what
        # one full-length reply costs against it. INFORMATIONAL, deliberately not
        # a threshold: whether a turn actually crosses this is dynamic (reply +
        # its tool result + whatever the reasoning guard really allowed), and no
        # static rule separates a safe configuration from an unsafe one.
        #
        # It is logged because the diagnosis that needed it (PR #66) had to be
        # done by hand: 51 of 51 dead sub-agents had a last-successful prompt
        # BELOW the trigger and a next request past the wall — a whole turn's
        # growth fitting inside the margin. `should_compact` reads the request
        # already sent, so it cannot see that coming; the standing fix is to
        # project the next request, not to bound this ratio.
        margin = max_len - trigger
        logger.info(
            "%s: compaction trigger %s leaves %s tokens before max_len %s; "
            "one full-length reply (max_tokens %s) is %.0f%% of that margin",
            label, f"{trigger:,}", f"{margin:,}", f"{max_len:,}",
            f"{max_tokens:,}", 100.0 * max_tokens / margin,
        )

    for problem in problems:
        logger.warning("%s", problem)
    return problems
