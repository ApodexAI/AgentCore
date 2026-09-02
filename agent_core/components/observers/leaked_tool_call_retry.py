# pyright: reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportUnnecessaryIsInstance=false, reportUnusedFunction=false, reportAttributeAccessIssue=false, reportUnnecessaryComparison=false
"""Observer that retries leaked-text tool calls with escalating temperature.

When an LLM response is parsed and yields **no** tool calls, but the visible
content clearly refers to an available tool ("I will call create_subagent…",
"<seed:tool_call>…</seed:tool_call>", "<function name=...>"), the most likely
explanation is a format-leak the parser could not recover.

``MultiFormatToolCallParser`` already catches structured leaks (Qwen + Seed
XML). This observer covers the harder free-text / exotic-wrapper cases by
re-prompting the LLM **on the same turn** with:

1. An explicit user-message nudge ("your last response leaked a tool call —
   call it properly").
2. A per-turn ``_llm_temp_override`` stashed in loop metadata. The agent
   loop picks it up and binds the LLM with that temperature for the retry.
3. ``continue_to_next_turn=True`` so the loop replays the turn (consuming
   the ``attempts`` budget, NOT the turn budget) regardless of the policy's
   ``no_tool_behavior``. Without this flag, a loop configured with
   ``no_tool_behavior="stop"`` (e.g. ``heavy_reporter``) would still break
   out of the loop right after the nudge was injected — the message would
   be appended but never sent.

Temperatures escalate through ``[0.3, 0.6, 1.0]`` — borrowed verbatim from
``workflows/AgentTeam/llm_client.py``. After the sequence is exhausted the
observer stops nudging and lets the normal ``no_tool`` handler take over.

Gated at construction time: pass ``enabled=False`` (or construct with an
empty temperature sequence) to disable without removing the observer from
the stack.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from agent_core.loop_types import BaseObserver, Intervention, TurnContext

logger = logging.getLogger(__name__)

_METADATA_COUNT_KEY = "_leak_retry_count"
_METADATA_TEMP_KEY = "_llm_temp_override"

_DEFAULT_TEMPERATURES: tuple[float, ...] = (0.3, 0.6, 1.0)

# Prose allowed around a tool-named fence before it stops looking like a bare
# leaked call and starts looking like a real answer that documents a command.
# Room for a lead-in ("Run this to reproduce:") but not for findings.
_MAX_INCIDENTAL_PROSE_CHARS = 120

# XML wrappers the parser already tries — detect-only for signalling.
_LEAK_XML_RE = re.compile(
    r"<\s*(?:tool_call\b"
    r"|function\s+name\s*="
    r"|seed:tool_call\b"
    r"|seedtool_call\b"
    r"|seed:tool[-_]name\b"
    r")",
    re.IGNORECASE,
)

# A long-context model can preserve the *body* of a requested tool call but
# lose the native function-call envelope, emitting it as Markdown instead:
#
#     ```bash
#     cp /workspace/final.png /outputs/final.png
#     ```
#
# The generic tool-name heuristic below deliberately needs an additional
# prose/call-shape hint to avoid flagging ordinary summaries.  A fenced block
# whose info string is itself an available tool name is already an explicit
# execution shape, so detect that separately.  The exact name match keeps
# normal fenced examples such as ```python inert when ``python`` is not a
# bound tool.
_FENCED_BLOCK_RE = re.compile(
    r"(?ms)^\s*(?P<fence>`{3,}|~{3,})"
    r"[ \t]*(?P<info>[A-Za-z_][A-Za-z0-9_.-]*)[^\n]*\n"
    r"(?P<body>.*?)^\s*(?P=fence)[ \t]*$",
)

LEAKED_TOOL_CALL_NUDGE = (
    "Your previous response mentioned calling a tool in free text but no "
    "structured tool_call was emitted. Invoke the tool through the proper "
    "tool_calls interface now — do not describe the call in prose."
)


class LeakedToolCallRetryObserver(BaseObserver):
    """Detect text-only tool-call leaks and trigger a next-turn retry.

    Args:
        tool_names: iterable of allowed tool identifiers. Used to guard the
            free-text mention check ("I will run_python_code(...)" only
            counts as a leak if ``run_python_code`` is in this set).
        temperatures: escalation sequence. Defaults to ``(0.3, 0.6, 1.0)``.
        enabled: master switch. Disables without removing from the stack.
        max_nudges_per_task: total cap on retries triggered per loop run
            (regardless of how many distinct turns leak). Prevents
            pathological tight loops when the model ignores the nudge.
        treat_any_tool_fence_as_leak: when True, ANY non-empty fence named
            after a bound tool is a leak, regardless of surrounding prose.
            Default False keeps the conservative rule (see
            :meth:`_fenced_tool_call_only`) that protects loops whose final
            answer legitimately documents a command. Turn it on only for a
            task whose deliverable is a *file* rather than text — a publisher
            has no reason to end on a ```bash block, so there the strict read
            costs nothing and catches the leak shape that motivated this
            observer.
    """

    critical = True

    def __init__(
        self,
        tool_names: Iterable[str],
        *,
        temperatures: tuple[float, ...] = _DEFAULT_TEMPERATURES,
        enabled: bool = True,
        max_nudges_per_task: int | None = None,
        treat_any_tool_fence_as_leak: bool = False,
    ) -> None:
        self._any_tool_fence_is_leak = bool(treat_any_tool_fence_as_leak)
        self._tool_names: frozenset[str] = frozenset(tool_names)
        self._tool_names_lower = frozenset(name.lower() for name in self._tool_names)
        self._temperatures = temperatures
        self._enabled = enabled and bool(temperatures)
        # Default cap = len(temperatures) so each temperature is tried once.
        self._max_nudges = (
            max_nudges_per_task if max_nudges_per_task is not None else len(temperatures)
        )
        # Pre-compile the free-text tool-name probe once.
        if self._tool_names:
            pattern = r"\b(" + "|".join(re.escape(n) for n in self._tool_names) + r")\b"
            self._tool_name_re: re.Pattern | None = re.compile(pattern, re.IGNORECASE)
        else:
            self._tool_name_re = None

    # ------------------------------------------------------------------ API

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        if not self._enabled:
            return None

        # The parser found a real call, but the landing-turn policy denied it.
        # Retrying would re-bind tools on the next iteration and turn the
        # denied action into a real execution, defeating LastTurnForcer.
        if ctx.blocked_tool_calls:
            ctx.metadata.pop(_METADATA_COUNT_KEY, None)
            ctx.metadata.pop(_METADATA_TEMP_KEY, None)
            return None

        # This turn ran with no tools bound at all — ``LastTurnForcer`` strips
        # them for the landing turn. Nothing could be parsed, so
        # ``blocked_tool_calls`` is empty and the guard above does not cover
        # it, yet a prose answer that merely mentions ``finalize_answer(...)``
        # trips ``_looks_like_leak``. Replaying would hand tools back on the
        # one turn that was supposed to have none, because
        # ``_llm_strip_tools`` is one-shot and already consumed.
        if ctx.metadata.get("_llm_tools_stripped"):
            ctx.metadata.pop(_METADATA_COUNT_KEY, None)
            ctx.metadata.pop(_METADATA_TEMP_KEY, None)
            return None

        # A structured tool call was parsed — nothing to retry. Also reset
        # the escalation counter so a future leak starts at 0.3 again.
        if ctx.tool_calls:
            ctx.metadata.pop(_METADATA_COUNT_KEY, None)
            return None

        if not self._looks_like_leak(ctx.ai_text):
            return None

        count = int(ctx.metadata.get(_METADATA_COUNT_KEY, 0))
        if count >= self._max_nudges:
            logger.info(
                "[LeakedToolCallRetry] Exhausted retry budget (%d) — "
                "letting default no_tool handler take over.",
                self._max_nudges,
            )
            return None

        temp_index = min(count, len(self._temperatures) - 1)
        next_temp = float(self._temperatures[temp_index])

        ctx.metadata[_METADATA_COUNT_KEY] = count + 1
        ctx.metadata[_METADATA_TEMP_KEY] = next_temp

        logger.warning(
            "[LeakedToolCallRetry] turn=%d | leaked content, scheduling retry "
            "with temperature=%.1f (retry %d/%d)",
            ctx.turn,
            next_temp,
            count + 1,
            self._max_nudges,
        )

        return Intervention(
            inject_messages=[LEAKED_TOOL_CALL_NUDGE],
            continue_to_next_turn=True,
        )

    # --------------------------------------------------------------- helpers

    def _fenced_tool_call_only(self, sample: str) -> bool:
        """Whether the text is *nothing but* a fenced block named after a tool.

        A response that consists solely of ```` ```bash …``` ```` is an
        execution request that lost its envelope. The SAME fence inside a real
        answer is ordinary documentation — "here is the command to reproduce
        this" — and ``bash`` is a real tool name, so matching the fence alone
        misfires on every workflow that ends on bare text with
        ``no_tool_behavior="stop"``: the retry would override the stop, discard
        a finished answer, and re-prompt the model to actually run it.

        So the fence only counts when it is the whole response. Only matching
        tool fences are stripped; a JSON/Python/result fence is substantive
        answer content and must remain in the prose budget.

        ``treat_any_tool_fence_as_leak`` opts a task out of all of that. The
        two shapes are not separable by prose volume — a status fence plus a
        leaked ``cp`` into ``/outputs`` can be SHORTER than a real answer that
        carries its findings in a result fence — so the caller distinguishes
        them by who is speaking instead of by a threshold.
        """
        matched_tool_fence = False
        substantive_non_tool_fence = False
        remainder: list[str] = []
        cursor = 0
        for match in _FENCED_BLOCK_RE.finditer(sample):
            info = match.group("info").lower()
            body = match.group("body").strip()
            if info not in self._tool_names_lower:
                substantive_non_tool_fence = substantive_non_tool_fence or bool(body)
                continue
            if not body:
                continue
            matched_tool_fence = True
            remainder.append(sample[cursor : match.start()])
            cursor = match.end()
        if not matched_tool_fence:
            return False
        if self._any_tool_fence_is_leak:
            return True
        if substantive_non_tool_fence:
            return False
        remainder.append(sample[cursor:])
        prose = "".join(remainder).strip()
        return len(prose) <= _MAX_INCIDENTAL_PROSE_CHARS

    def _looks_like_leak(self, text: str) -> bool:
        """Heuristic detector: free-text tool mention OR XML-wrapper residue."""
        if not text:
            return False
        sample = text[:4000]
        if _LEAK_XML_RE.search(sample):
            return True
        if self._fenced_tool_call_only(sample):
            return True
        if self._tool_name_re is None:
            return False
        # Fences have their own stricter classifier above. Remove all of them
        # before the generic prose heuristic so JSON braces and a ``bash`` info
        # string inside a legitimate documented example cannot manufacture the
        # tool-name + call-shape pair this fallback looks for.
        heuristic_sample = _FENCED_BLOCK_RE.sub("", sample)
        # Require both a tool-name mention AND either a call-shape hint
        # ("(" or "{") or a directive verb — avoids flagging plain prose
        # that happens to include the tool name in a summary.
        if not self._tool_name_re.search(heuristic_sample):
            return False
        lowered = heuristic_sample.lower()
        call_hints = ("(", "{", "args:", "arguments:", "parameter")
        verb_hints = (
            "i will",
            "let me",
            "i'll call",
            "now call",
            "i'm going to",
            "next, i",
            "calling ",
        )
        if any(hint in lowered for hint in call_hints):
            return True
        return bool(any(verb in lowered for verb in verb_hints))


__all__ = ["LEAKED_TOOL_CALL_NUDGE", "LeakedToolCallRetryObserver"]
