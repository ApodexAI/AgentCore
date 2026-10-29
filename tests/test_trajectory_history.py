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
    assert trajectory._message_spool_handle is None
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
