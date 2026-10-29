"""The JSON trajectory follows the loop's history, including engine replies."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from agent_core.components.observers.last_turn_forcer import LastTurnForcer
from agent_core.components.observers.trajectory import TrajectoryFileObserver
from agent_core.llm import LLMResponse
from agent_core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    Intervention,
    LoopConfig,
    LoopPolicy,
    TurnContext,
)
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.model_profile import HistoryPolicy, ModelProfile


class EchoTool:
    name = "echo"

    def to_openai_schema(self):
        return {"type": "function", "function": {
            "name": self.name, "parameters": {"type": "object"},
        }}

    async def ainvoke(self, args):
        return args.get("value", "ok")


def _call(cid, arguments='{"value":"ok"}', name="echo"):
    return {"id": cid, "type": "function", "function": {
        "name": name, "arguments": arguments,
    }}


def _messages(path):
    return json.loads(path.read_text())["messages"]


def _without_diagnostics(messages):
    return [{k: v for k, v in msg.items()
             if k not in {"usage", "thinking_blocks", "finish_reason", "stop_details"}}
            for msg in messages]


def _assert_balanced(messages):
    pending = set()
    for msg in messages:
        if msg["role"] == "tool":
            assert msg["tool_call_id"] in pending
            pending.remove(msg["tool_call_id"])
        else:
            assert not pending
            pending.update(tc["id"] for tc in msg.get("tool_calls", []))
    assert not pending


async def _run(tmp_path, responses, *, observers=(), config=None,
               formats=("json", "jsonl"), verify_snapshots=True, **kwargs):
    class Client:
        def __init__(self):
            self.requests = []
            self.snapshots = []

        async def chat(self, messages, **_kwargs):
            self.requests.append(copy.deepcopy(messages))
            self.snapshots.append(_messages(tmp_path / "trace.json"))
            return responses.pop(0)

    client = Client()
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=formats)
    result = await run_agent_loop(
        system_prompt="system", user_message="start", llm=client, tools=[EchoTool()],
        observers=[trajectory, *observers],
        config=config or LoopConfig(max_turns=5, max_llm_retries=1,
                                    loop_policy=LoopPolicy(no_tool_behavior="stop")),
        **kwargs,
    )
    messages = _messages(tmp_path / "trace.json")
    assert _without_diagnostics(messages) == result.messages
    _assert_balanced(messages)
    if verify_snapshots:
        for snapshot, request in zip(client.snapshots, client.requests, strict=True):
            assert _without_diagnostics(snapshot) == request
    assert not (tmp_path / "trace.messages.spool").exists()
    assert trajectory._history is None
    assert not trajectory._response_metadata
    return messages, client, result


@pytest.fixture(autouse=True)
def _uncoalesced(monkeypatch):
    monkeypatch.setattr("agent_core.components.observers.trajectory._COALESCE_N", 0)
    monkeypatch.setattr("agent_core.components.observers.trajectory._COALESCE_MS", 0)


@pytest.mark.parametrize("turn,call_count", [(1, 6), (1, 7), (2, 6)])
@pytest.mark.parametrize("formats", [("json",), ("json", "jsonl")])
async def test_over_cap_keeps_every_native_call_and_reply(tmp_path, turn, call_count, formats):
    responses = ([LLMResponse(tool_calls=[_call("warmup")])] if turn == 2 else [])
    responses += [
        LLMResponse(tool_calls=[_call(f"c{i}") for i in range(call_count)]),
        LLMResponse(content="done"),
    ]
    messages, _, _ = await _run(tmp_path, responses, formats=formats, config=LoopConfig(
        max_turns=4, max_llm_retries=1, max_tool_calls_per_turn=5,
        loop_policy=LoopPolicy(no_tool_behavior="stop"),
    ))
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert len(assistants) == turn + 1
    assert len(assistants[turn - 1]["tool_calls"]) == call_count
    skipped = [m for m in messages if m["role"] == "tool" and "[tool call skipped]" in m["content"]]
    assert len(skipped) == call_count - 5
    assert (tmp_path / "trace.jsonl").exists() == ("jsonl" in formats)


@pytest.mark.parametrize("scenario", ["unknown", "final", "skip", "restart"])
async def test_engine_generated_tool_replies_are_recorded(tmp_path, scenario):
    class Control(BaseObserver):
        critical = True
        fired = False

        async def on_llm_response(self, ctx):
            if ctx.turn == 2 and not self.fired:
                self.fired = True
                if scenario == "skip":
                    return Intervention(skip_tool_execution=True)
                if scenario == "restart":
                    return Intervention(continue_to_next_turn=True)
            return None

    responses = [
        LLMResponse(tool_calls=[_call("warmup")]),
        LLMResponse(tool_calls=[_call("target", name="missing" if scenario == "unknown" else "echo"),
                                *([_call("known")] if scenario == "unknown" else [])]),
    ]
    if scenario != "final":
        responses.append(LLMResponse(content="done"))
    messages, _, _ = await _run(tmp_path, responses, observers=[Control(), LastTurnForcer()], config=LoopConfig(
        max_turns=2 if scenario == "final" else 5,
        max_llm_retries=1,
        loop_policy=LoopPolicy(no_tool_behavior="stop"),
    ))
    target = next(m for m in messages if m.get("tool_call_id") == "target")
    assert "[tool call" in target["content"]


async def test_invalid_arguments_follow_their_assistant_even_without_an_async_yield(tmp_path):
    messages, _, _ = await _run(tmp_path, [
        LLMResponse(content="partial", tool_calls=[_call("broken", "{")], finish_reason="length"),
        LLMResponse(content="done"),
    ])
    assistant_index = next(i for i, m in enumerate(messages) if m.get("tool_calls"))
    assert messages[assistant_index + 1]["tool_call_id"] == "broken"
    assert "[invalid tool call]" in messages[assistant_index + 1]["content"]
    assert messages[assistant_index]["finish_reason"] == "length"
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [e["t"] for e in events] == ["start", "llm", "result", "llm", "end"]


@pytest.mark.parametrize("rollback", [False, True])
async def test_injections_rollbacks_and_compaction_follow_rewritten_history(tmp_path, rollback):
    class Rewrite(BaseObserver):
        critical = True
        fired = False

        async def on_turn_end(self, ctx):
            if not self.fired:
                self.fired = True
                return Intervention(inject_messages=["continue"], pop_last_message=rollback,
                                    continue_to_next_turn=rollback)
            return None

    class Compact:
        def compact(self, messages, keep_recent):
            return [messages[0], {"role": "user", "content": "compacted"}]

    messages, _, _ = await _run(tmp_path, [
        LLMResponse(tool_calls=[_call("discarded")]), LLMResponse(content="done"),
    ], observers=[Rewrite()], config=LoopConfig(
        max_turns=4, max_llm_retries=1, compact_after_turns=0,
        compactor=Compact(), loop_policy=LoopPolicy(no_tool_behavior="stop"),
    ))
    assert "discarded" not in json.dumps(messages)
    assert any(m["role"] == "user" for m in messages)


async def test_tool_snapshot_uses_postprocessed_body_and_jsonl_keeps_original(tmp_path):
    class Processor:
        def process(self, result):
            return "model preview"

    messages, _, _ = await _run(tmp_path, [
        LLMResponse(tool_calls=[_call("c", '{"value":"original result"}')]),
        LLMResponse(content="done"),
    ], config=LoopConfig(max_turns=3, tool_result_post_processor=Processor(),
                         loop_policy=LoopPolicy(no_tool_behavior="stop")))
    assert next(m for m in messages if m["role"] == "tool")["content"] == "model preview"
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert next(e for e in events if e["t"] == "result")["result"] == "original result"


@pytest.mark.parametrize("thinking_format", ["tag", "reasoning_content", "content_block"])
@pytest.mark.parametrize("thinking_in_history", [False, True])
async def test_reasoning_uses_the_normalizers_history_policy(tmp_path, thinking_format, thinking_in_history):
    content = ([{"type": "thinking", "thinking": "reasoning", "signature": "signed"},
                {"type": "text", "text": "visible"}]
               if thinking_format == "content_block" else "visible")
    messages, _, _ = await _run(tmp_path, [
        LLMResponse(content=content, reasoning_content="reasoning", tool_calls=[_call("c")]),
        LLMResponse(content="done"),
    ], model_profile=ModelProfile(model_id="test", provider="test", thinking_format=thinking_format),
       history_policy=HistoryPolicy(thinking_in_history=thinking_in_history))
    first = next(m for m in messages if m["role"] == "assistant")
    if thinking_format == "content_block":
        assert first["content"] == content
    elif thinking_format == "reasoning_content" and thinking_in_history:
        assert first["reasoning_content"] == "reasoning"
    else:
        assert "reasoning_content" not in first


@pytest.mark.parametrize("early_stop", [False, True])
async def test_empty_run_and_before_llm_stop_keep_initial_history(tmp_path, early_stop):
    class Stop(BaseObserver):
        critical = True

        async def on_before_llm(self, ctx):
            return Intervention(stop_reason="stopped", inject_messages=["notice"])

    messages, client, _ = await _run(tmp_path, [], observers=[Stop()], config=LoopConfig(
        max_turns=3 if early_stop else 0,
    ))
    assert not client.requests
    assert [m["content"] for m in messages] == ["system", "start", *(["notice"] if early_stop else [])]


async def test_coalesced_terminal_snapshot_is_complete(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_core.components.observers.trajectory._COALESCE_N", 1000)
    messages, _, _ = await _run(tmp_path, [
        LLMResponse(tool_calls=[_call(f"c{i}") for i in range(7)]),
        LLMResponse(content="done"),
    ], verify_snapshots=False)
    assert len([m for m in messages if m["role"] == "tool"]) == 7


async def test_cancellation_flushes_current_history_and_preserves_spool(tmp_path):
    entered = asyncio.Event()

    class Client:
        async def chat(self, messages, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    trajectory = TrajectoryFileObserver(tmp_path, filename="trace")
    task = asyncio.create_task(run_agent_loop(
        system_prompt="system", user_message="start", llm=Client(), tools=[],
        observers=[trajectory], config=LoopConfig(max_turns=2),
    ))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [m["content"] for m in _messages(tmp_path / "trace.json")] == ["system", "start"]
    assert (tmp_path / "trace.messages.spool").exists()


async def test_failed_terminal_replace_keeps_readable_snapshot_and_forensic_spool(tmp_path, monkeypatch):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    await trajectory.on_loop_start(LoopConfig())
    previous = (tmp_path / "trace.json").read_bytes()

    def fail_replace(source, destination):
        raise OSError("disk failure")

    monkeypatch.setattr("agent_core.components.observers.trajectory.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        await trajectory.on_loop_end(AgentLoopResult(messages=[{"role": "user", "content": "latest"}]))
    assert (tmp_path / "trace.json").read_bytes() == previous
    assert json.loads((tmp_path / "trace.messages.spool").read_text())["content"] == "latest"
    assert trajectory._spool_entries == []
    assert trajectory._history is None


@pytest.mark.parametrize("formats", [("jsonl",), ()])
async def test_disabled_json_creates_no_snapshot_or_spool(tmp_path, formats):
    class Client:
        async def chat(self, messages, **kwargs):
            return LLMResponse(content="done")

    await run_agent_loop(
        system_prompt="system", user_message="start", llm=Client(), tools=[],
        observers=[TrajectoryFileObserver(tmp_path, filename="trace", formats=formats)],
        config=LoopConfig(max_turns=1, loop_policy=LoopPolicy(no_tool_behavior="stop")),
    )
    assert not (tmp_path / "trace.json").exists()
    assert not (tmp_path / "trace.messages.spool").exists()
    assert (tmp_path / "trace.jsonl").exists() == ("jsonl" in formats)


@pytest.mark.parametrize("limit", [0, 4])
def test_history_body_limits_preserve_blocks_and_do_not_mutate_history(tmp_path, monkeypatch, limit):
    monkeypatch.setattr("agent_core.components.observers.trajectory._BODY_MAX_CHARS", limit)
    message = {"role": "tool", "tool_call_id": "c", "content": [
        {"type": "text", "text": "long result"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,payload"}},
    ]}
    original = copy.deepcopy(message)
    trajectory = TrajectoryFileObserver(tmp_path)
    rendered = trajectory._message_to_dict(message)
    text = rendered["content"][0]["text"]
    if limit == 0:
        assert text == "long result"
    else:
        assert text.startswith("long\n... [truncated")
    assert "elided from trace" in rendered["content"][1]["image_url"]["url"]
    assert message == original


def _turn(messages, **kwargs):
    fields = {"turn": 1, "max_turns": 5, "task_id": "t", "role_id": "r", "ai_text": "",
              "thinking": "", "tool_calls": [], "messages": messages, "usage": None,
              "metadata": {}}
    fields.update(kwargs)
    return TurnContext(**fields)


async def test_cancellation_after_loop_end_keeps_terminal_snapshot(tmp_path):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    await trajectory.on_loop_start(LoopConfig())
    history = [{"role": "user", "content": "start"}, {"role": "assistant", "content": "done"}]
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    terminal = (tmp_path / "trace.json").read_bytes()
    # A later critical observer's on_loop_end is cancelled; the loop then
    # notifies on_loop_cancelled on every observer, this one included.
    await trajectory.on_loop_cancelled()
    assert (tmp_path / "trace.json").read_bytes() == terminal
    assert [m["content"] for m in _messages(tmp_path / "trace.json")] == ["start", "done"]


async def test_diagnostics_survive_compaction_that_copies_assistants(tmp_path):
    class CopyCompact:
        def compact(self, messages, keep_recent):
            # Like strip_stale_reasoning: same content, new dict objects.
            return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]

    messages, _, _ = await _run(tmp_path, [
        LLMResponse(tool_calls=[_call("c")], finish_reason="tool_use"),
        LLMResponse(content="done"),
    ], config=LoopConfig(max_turns=4, max_llm_retries=1, compact_after_turns=0,
                         compactor=CopyCompact(),
                         loop_policy=LoopPolicy(no_tool_behavior="stop")))
    first = next(m for m in messages if m["role"] == "assistant")
    assert first["finish_reason"] == "tool_use"


async def test_diagnostics_survive_a_signed_thinking_reset(tmp_path):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    blocks = [{"type": "thinking", "thinking": "r", "signature": "s"}]
    assistant = {"role": "assistant", "content": [*blocks, {"type": "text", "text": "a"}]}
    history = [{"role": "user", "content": "start"}, assistant]
    await trajectory.on_llm_response(_turn(history, ai_text="a", thinking_blocks=blocks,
                                           usage={"input_tokens": 3}))
    # The real reset substitutes a copy without the blocks.
    from agent_core.runtime.loop.agent_loop import _reset_signed_thinking_history

    _reset_signed_thinking_history(history)
    assert history[1] is not assistant
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    last = _messages(tmp_path / "trace.json")[-1]
    assert last["usage"] == {"input_tokens": 3}
    # The blocks are gone from content, so the annotation must carry them.
    assert last["thinking_blocks"] == blocks


async def test_native_thinking_blocks_are_not_written_twice(tmp_path):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    blocks = [{"type": "thinking", "thinking": "r", "signature": "s"}]
    history = [{"role": "assistant", "content": [*blocks, {"type": "text", "text": "a"}]}]
    await trajectory.on_llm_response(_turn(history, thinking_blocks=blocks,
                                           usage={"input_tokens": 3}))
    message = _messages(tmp_path / "trace.json")[0]
    assert "thinking_blocks" not in message
    assert message["usage"] == {"input_tokens": 3}


async def test_non_json_history_values_do_not_freeze_snapshots(tmp_path):
    import datetime

    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json", "jsonl"])
    await trajectory.on_loop_start(LoopConfig())
    stamp = datetime.datetime(2026, 1, 2, tzinfo=datetime.UTC)
    history = [
        {"role": "assistant", "content": "", "tool_calls": [_call("c")]},
        {"role": "tool", "tool_call_id": "c", "content": "ok", "seen_at": stamp,
         "raw": b"abc"},
    ]
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    tool = _messages(tmp_path / "trace.json")[-1]
    assert tool["seen_at"] == stamp.isoformat()
    assert tool["raw"] == "<3 bytes>"


async def test_hooks_without_loop_history_still_record_the_turn(tmp_path):
    from agent_core.loop_types import ToolResult

    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    await trajectory.on_loop_start(LoopConfig())
    await trajectory.on_llm_response(_turn(
        None, ai_text="calling", tool_calls=[{"name": "echo", "args": {"v": 1}}],
        finish_reason="tool_use",
    ))
    await trajectory.on_tool_result(_turn(None), ToolResult(
        name="echo", args={"v": 1}, result="ok", duration_ms=1, tool_call_id="", is_error=False,
    ))
    await trajectory.on_llm_response(_turn(None, turn=2, ai_text="done", tool_calls=None))
    await trajectory.on_loop_end(AgentLoopResult(messages=[], turns_used=2))
    messages = _messages(tmp_path / "trace.json")
    assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]
    assert messages[0]["finish_reason"] == "tool_use"
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"v":1}'
    assert messages[1]["tool_call_id"] == messages[0]["tool_calls"][0]["id"]
    assert messages[2]["content"] == "done"


async def test_tool_results_count_toward_coalesced_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr("agent_core.components.observers.trajectory._COALESCE_N", 5)
    _, client, _ = await _run(tmp_path, [
        LLMResponse(tool_calls=[_call(f"c{i}") for i in range(3)]),
        LLMResponse(content="done"),
    ], verify_snapshots=False)
    # Turn 1 is input + response + 3 results + turn end; without the results
    # it would only become due at the second input, after the snapshot.
    assert [m.get("tool_call_id") for m in client.snapshots[1] if m["role"] == "tool"] == [
        "c0", "c1", "c2",
    ]


async def test_snapshots_render_only_new_or_changed_messages(tmp_path, monkeypatch):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    rendered = []
    original = trajectory._message_to_dict

    def counting(message):
        rendered.append(message.get("content"))
        return original(message)

    monkeypatch.setattr(trajectory, "_message_to_dict", counting)
    history = [{"role": "user", "content": f"m{i}"} for i in range(3)]
    trajectory._history = history
    trajectory._flush_json()
    history.append({"role": "user", "content": "m3"})
    trajectory._flush_json()
    assert rendered == ["m0", "m1", "m2", "m3"]

    # Value replacement in place (as image eviction does) is detected.
    history[1]["content"] = "edited"
    trajectory._flush_json()
    assert rendered[4:] == ["edited", "m2", "m3"]
    # So is a rewrite that removes messages.
    history[:] = [history[0], history[3]]
    trajectory._flush_json()
    assert [m["content"] for m in _messages(tmp_path / "trace.json")] == ["m0", "m3"]
    assert rendered[7:] == ["m3"]


@pytest.mark.parametrize("field", ["text", "arguments", "image_meta", "usage"])
async def test_snapshots_detect_nested_history_edits(tmp_path, field):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    assistant = {
        "role": "assistant", "content": [{"type": "text", "text": "old"}],
        "tool_calls": [_call("c")], "image_meta": [{"label": "old", "tokens": 1}],
    }
    usage = {"prompt_tokens": 10, "details": {"cached": 1}}
    history = [assistant]
    await trajectory.on_llm_response(_turn(history, usage=usage))
    trajectory._flush_json(force=True)
    if field == "text":
        assistant["content"][0]["text"] = "new"
    elif field == "arguments":
        assistant["tool_calls"][0]["function"]["arguments"] = '{"value":"new"}'
    elif field == "image_meta":
        assistant["image_meta"][0]["label"] = "new"
    else:
        usage["details"]["cached"] = 2
    # Completion must detect edits even when every container keeps its identity.
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    saved = _messages(tmp_path / "trace.json")[0]
    assert _without_diagnostics([saved]) == history
    assert saved["usage"] == usage


@pytest.mark.parametrize("copies", [1, 2])
@pytest.mark.parametrize("first_has_details", [False, True])
async def test_duplicate_assistant_copies_do_not_guess_diagnostics(tmp_path, copies, first_has_details):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace")
    history = [{"role": "assistant", "content": "same"}]
    await trajectory.on_llm_response(_turn(
        history, usage={"prompt_tokens": 10} if first_has_details else None,
        finish_reason="length" if first_has_details else "stop",
    ))
    history.append({"role": "assistant", "content": "same"})
    await trajectory.on_llm_response(_turn(history, turn=2, usage={"prompt_tokens": 20},
                                           finish_reason="refusal"))
    # A compactor copies the retained tail, optionally dropping the first turn.
    history[:] = [dict(message) for message in history[-copies:]]
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    assert _messages(tmp_path / "trace.json") == history
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [event.get("usage") for event in events if event["t"] == "llm"] == [
        {"prompt_tokens": 10} if first_has_details else None, {"prompt_tokens": 20},
    ]


async def test_identity_keeps_diagnostics_when_other_assistant_has_same_text(tmp_path):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    history = [{"role": "assistant", "content": "same"}]
    await trajectory.on_llm_response(_turn(history, usage={"prompt_tokens": 10}))
    history.append({"role": "assistant", "content": "same"})
    await trajectory.on_llm_response(_turn(history, turn=2, usage={"prompt_tokens": 20}))
    history[:] = [history[-1]]
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    assert _messages(tmp_path / "trace.json")[0]["usage"] == {"prompt_tokens": 20}


async def test_copied_diagnostic_does_not_attach_to_multiple_candidates(tmp_path):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    history = [{"role": "assistant", "content": "same"}]
    await trajectory.on_llm_response(_turn(history, usage={"prompt_tokens": 10}))
    history[:] = [dict(history[0]), dict(history[0])]
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    assert _messages(tmp_path / "trace.json") == history


async def test_resumed_assistant_without_details_counts_as_an_ambiguous_source(tmp_path):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    history = [{"role": "assistant", "content": "same"}]
    await trajectory.on_before_llm(_turn(history))
    history.append({"role": "assistant", "content": "same"})
    await trajectory.on_llm_response(_turn(history, usage={"prompt_tokens": 20}))
    history[:] = [dict(history[0])]
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    assert _messages(tmp_path / "trace.json") == history


async def test_identity_disambiguates_a_copy_with_identical_text(tmp_path):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    history = [{"role": "assistant", "content": "same"}]
    await trajectory.on_llm_response(_turn(history, usage={"prompt_tokens": 10}))
    history.append({"role": "assistant", "content": "same"})
    await trajectory.on_llm_response(_turn(history, turn=2, usage={"prompt_tokens": 20}))
    history[-1] = dict(history[-1])
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    assert [m["usage"]["prompt_tokens"] for m in _messages(tmp_path / "trace.json")] == [10, 20]


async def test_loop_compactor_can_edit_a_native_block_in_place(tmp_path):
    class EditBlock:
        def compact(self, messages, keep_recent):
            for message in messages:
                if message.get("role") == "assistant" and isinstance(message.get("content"), list):
                    for block in message["content"]:
                        if block.get("type") == "text":
                            block["text"] = "compacted text"
            return messages

    messages, _, _ = await _run(tmp_path, [
        LLMResponse(content=[
            {"type": "thinking", "thinking": "r", "signature": "s"},
            {"type": "text", "text": "original"},
        ], tool_calls=[_call("c")]), LLMResponse(content="done"),
    ], model_profile=ModelProfile(model_id="test", provider="test", thinking_format="content_block"),
       config=LoopConfig(max_turns=3, compact_after_turns=0, compactor=EditBlock(),
                         loop_policy=LoopPolicy(no_tool_behavior="stop")))
    first = next(m for m in messages if m["role"] == "assistant")
    assert first["content"][-1]["text"] == "compacted text"


async def test_routine_snapshots_do_not_reserialize_the_prefix(tmp_path, monkeypatch):
    from agent_core.runtime.loop.compact import COMPACTION_SEQ_KEY

    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    rendered = []
    original = trajectory._message_to_dict

    def counting(message):
        rendered.append(message.get("content"))
        return original(message)

    monkeypatch.setattr(trajectory, "_message_to_dict", counting)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 4096}}
    block = {"type": "text", "text": "old"}
    history = [{"role": "user", "content": [block, image]}, {"role": "assistant", "content": "a"}]
    metadata = {}
    await trajectory.on_before_llm(_turn(history, metadata=metadata))
    await trajectory.on_llm_input(_turn(list(history), metadata=dict(metadata)))
    await trajectory.on_turn_end(_turn(history, metadata=metadata))
    assert len(rendered) == 2

    # A nested edit with no compaction is not looked for on routine snapshots.
    block["text"] = "edited"
    await trajectory.on_llm_input(_turn(list(history), metadata=dict(metadata)))
    assert len(rendered) == 2
    # A compactor run (its counter advances even without a notice) schedules
    # one deep comparison of the redacted lines, which finds the edit.
    metadata[COMPACTION_SEQ_KEY] = 1
    await trajectory.on_llm_input(_turn(list(history), metadata=dict(metadata)))
    assert _messages(tmp_path / "trace.json")[0]["content"][0]["text"] == "edited"
    assert "AAAA" not in (tmp_path / "trace.json").read_text()
    count = len(rendered)
    await trajectory.on_turn_end(_turn(history, metadata=metadata))
    assert len(rendered) == count


async def test_rollback_notice_schedules_a_deep_check(tmp_path):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    block = {"type": "text", "text": "old"}
    history = [{"role": "user", "content": [block]}]
    await trajectory.on_before_llm(_turn(history))
    await trajectory.on_turn_end(_turn(history))
    block["text"] = "edited"
    await trajectory.on_context_compacted(None)
    await trajectory.on_turn_end(_turn(history))
    assert _messages(tmp_path / "trace.json")[0]["content"][0]["text"] == "edited"


async def test_diagnostics_follow_an_in_place_edit_and_a_later_copy(tmp_path):
    from agent_core.runtime.loop.agent_loop import _reset_signed_thinking_history

    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    blocks = [{"type": "thinking", "thinking": "r", "signature": "s"}]
    text = {"type": "text", "text": "original"}
    history = [{"role": "assistant", "content": [*blocks, text], "tool_calls": [_call("c")]},
               {"role": "tool", "tool_call_id": "c", "content": "ok"}]
    await trajectory.on_llm_response(_turn(history, thinking_blocks=blocks,
                                           usage={"prompt_tokens": 10}))
    trajectory._flush_json(force=True)
    text["text"] = "compacted text"
    _reset_signed_thinking_history(history)
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    first = _messages(tmp_path / "trace.json")[0]
    assert first["content"] == [{"type": "text", "text": "compacted text"}]
    assert first["usage"] == {"prompt_tokens": 10}


async def test_a_new_assistant_with_equal_text_never_inherits_a_vanished_copy(tmp_path):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    # Built at runtime so the two strings are distinct objects, as model
    # output always is; literals would be interned and shared.
    first = {"role": "assistant", "content": "".join(["o", "k"])}
    history = [first]
    await trajectory.on_llm_response(_turn(history, usage={"prompt_tokens": 10},
                                           finish_reason="length"))
    history[0] = dict(first)
    trajectory._flush_json(force=True)
    assert _messages(tmp_path / "trace.json")[0]["usage"] == {"prompt_tokens": 10}
    # Within one coalesced window the copy is dropped and an unrelated
    # assistant with the same text arrives without its own diagnostics.
    history[:] = [{"role": "assistant", "content": "".join(["o", "k"])}]
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    assert _messages(tmp_path / "trace.json") == history


async def test_unencodable_history_values_keep_the_snapshot_and_pairing(tmp_path):
    trajectory = TrajectoryFileObserver(tmp_path, filename="trace", formats=["json"])
    await trajectory.on_loop_start(LoopConfig())
    cycle = {}
    cycle["self"] = cycle
    history = [
        {"role": "assistant", "content": "", "tool_calls": [_call("c")]},
        {"role": "tool", "tool_call_id": "c", "content": "ok", "meta": cycle},
        {"role": "tool", "tool_call_id": "d", "content": "ok", "meta": {(1, 2): "tuple key"}},
        {"role": "user", "content": "after"},
    ]
    await trajectory.on_loop_end(AgentLoopResult(messages=history))
    messages = _messages(tmp_path / "trace.json")
    assert [m.get("tool_call_id") for m in messages] == [None, "c", "d", None]
    assert "unserializable" in messages[1]["trace_error"]
    assert "unserializable" in messages[2]["trace_error"]
    assert messages[-1]["content"] == "after"
