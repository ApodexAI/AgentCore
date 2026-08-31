"""``_strip_thinking_blocks`` removes ``<think>`` artifacts from final answers.

The kernel's ``extract_final_content`` calls this helper before handing
content to downstream judges / scorers. Three artifact shapes appear in
the wild:

1. **Proper pair** ``<think>...</think>answer`` — vanilla case.
2. **Dangling opener** ``<think>...EOF`` — model truncated mid-thought.
3. **Orphan closer** ``thinking text</think>answer`` — SGLang ``enable_thinking
   + preserve_thinking`` chat-template sometimes drops the opening tag.
   Until 2026-05-09 this case was mishandled (only the 8-char tag was
   removed; the thinking trace before it leaked into the answer and
   confused the judge). Confirmed against trial 65 of an apodex BC-200
   run where the model's "Wait—wait no!" indecision rambling reached the
   judge as the final answer.
"""
from __future__ import annotations

from agent_core.runtime.loop._response import _strip_thinking_blocks


def test_strips_proper_pair() -> None:
    assert _strip_thinking_blocks(
        "<think>reasoning</think>\nFinal answer: 42"
    ) == "Final answer: 42"


def test_strips_dangling_opener() -> None:
    assert _strip_thinking_blocks(
        "answer\n<think>cut off mid-thought"
    ) == "answer"


def test_strips_orphan_closer_sglang_quirk() -> None:
    """SGLang preserve_thinking sometimes emits </think> with no opener."""
    raw = "lots of thinking text including doubts</think>\n**Final: 42**"
    assert _strip_thinking_blocks(raw) == "**Final: 42**"


def test_strips_orphan_closer_keeps_only_tail_after_last() -> None:
    """When multiple stray </think> exist, keep only what's after the last."""
    raw = "thought one</think>second pass thinking</think>\nanswer"
    assert _strip_thinking_blocks(raw) == "answer"


def test_no_think_passthrough() -> None:
    assert _strip_thinking_blocks("plain answer") == "plain answer"


def test_strips_pair_then_orphan_closer() -> None:
    """Mixed: a proper pair earlier and a stray closer later."""
    raw = "<think>pair</think>\ntext<br/>more thinking</think>\nfinal"
    # _THINK_BLOCK_RE removes the pair → "text<br/>more thinking</think>\nfinal"
    # Then orphan-closer logic keeps after last </think> → "final"
    assert _strip_thinking_blocks(raw) == "final"


def test_empty_string() -> None:
    assert _strip_thinking_blocks("") == ""


def test_all_thinking_no_answer() -> None:
    """Worst case: model emitted only thinking, never an answer."""
    assert _strip_thinking_blocks("<think>just reasoning</think>") == ""
    assert _strip_thinking_blocks("just reasoning</think>") == ""
