"""Regression tests for the parallel-tool-call batch race that produced HTTP 409.

Observed in production (Hermes Agent → gate :8137 → this server → claude opus):
Claude dispatches a parallel tool batch over MCP with a stagger, but the first
``assistant`` stream event announced only ONE client tool_use. The turn loop
collected that one call and parked immediately, so the second call — arriving
45-57ms later in eight separate incidents — was orphaned in ``bridge._pending``.

The orphan then poisoned the turn: the client's genuinely-complete continuation
looked partial to ``resume``, which called ``fail_all``, and the *next*
continuation no longer intersected ``pending_ids`` and was rejected as a
``DuplicateContinuation`` → ``HTTP 409: these tool results were already
claimed``. Every one of those 409s knocked the client off Claude and onto its
paid fallback model.

These tests pin both halves of the fix: the straggler grace window, and raising
the quota when a follow-up ``AssistantToolUse`` announces more client calls.
"""

import asyncio
import time

import pytest

from app.conversation import (
    RUNNING,
    SUSPENDED,
    Conversation,
    ConversationManager,
    DoneChunk,
    ErrorChunk,
    ToolCallsChunk,
)
from app.claude_session import STREAM_CLOSED
from app.events import AssistantToolUse, ToolUseBlock, TurnDone
from app.mcp_bridge import ConversationBridge, McpBridge
from app.openai_models import FunctionDef, ToolDef

# The measured production stagger between the announced call and its straggler.
OBSERVED_STAGGER_S = 0.050


def _tools():
    return [
        ToolDef(function=FunctionDef(name="get_weather", description="w", parameters={})),
        ToolDef(function=FunctionDef(name="get_time", description="t", parameters={})),
    ]


def make_settings(**over):
    from app.config import Settings

    base = dict(
        suspended_ttl_s=300,
        idle_session_ttl_s=900,
        gc_interval_s=30,
        request_timeout_s=30,
        permission_mode="bypassPermissions",
        port=8787,
    )
    base.update(over)
    return Settings(**base)


class LiveFakeSession:
    """FakeSession whose stream stays OPEN once the preloaded events run out.

    The shared FakeSession in test_conversation.py returns STREAM_CLOSED as soon
    as its list empties, which would end the grace window instantly and mask the
    very behaviour under test. A real subprocess mid-turn simply has nothing to
    say yet, so block instead.
    """

    def __init__(self, events, *, close_after=None):
        self._events = list(events)
        self._close_after = close_after
        self.sent_turns = []
        self.closed = False

    async def start(self):
        pass

    async def send_user_turn(self, content):
        self.sent_turns.append(content)

    async def next_event(self, timeout=None):
        if self._events:
            return self._events.pop(0)
        if self._close_after is not None:
            await asyncio.sleep(self._close_after)
            return STREAM_CLOSED
        await asyncio.sleep(timeout if timeout is not None else 30)
        return None

    async def aclose(self):
        self.closed = True

    @property
    def running(self):
        return not self.closed


def _setup(mgr_settings, events, *, close_after=None):
    mcp = McpBridge()
    mgr = ConversationManager(mcp, mgr_settings)
    bridge = ConversationBridge("c1", _tools())
    mcp.register(bridge)
    sess = LiveFakeSession(events, close_after=close_after)
    conv = Conversation(conv_id="c1", session=sess, bridge=bridge, model="opus")
    mgr._conversations["c1"] = conv
    return mgr, bridge, conv


async def _dispatch_later(bridge, name, args, delay):
    await asyncio.sleep(delay)
    return await bridge.dispatch(name, args)


async def test_staggered_parallel_call_is_swept_into_the_batch():
    """The straggler arriving 50ms after the announced call must not be orphaned."""
    announced_one = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_1", name="mcp__client__get_weather", input={"city": "Paris"})])
    mgr, bridge, conv = _setup(make_settings(), [announced_one])

    first = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    await asyncio.sleep(0)  # let it reach the queue before the turn loop runs
    straggler = asyncio.create_task(
        _dispatch_later(bridge, "get_time", {"tz": "UTC"}, OBSERVED_STAGGER_S))

    chunks = [c async for c in mgr.run_turn(conv)]

    tc = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    assert len(tc) == 1, "expected exactly one ToolCallsChunk"
    names = sorted(c.name for c in tc[0].calls)
    assert names == ["get_time", "get_weather"], (
        f"straggler was orphaned; client only saw {names}")
    assert conv.state == SUSPENDED
    # Both ids must be routable, or the continuation cannot resolve them.
    for call in tc[0].calls:
        assert mgr._pending_index[call.id] == "c1"
    # Nothing left dangling in the bridge beyond what the client was told about.
    assert set(bridge.pending_ids) == {c.id for c in tc[0].calls}

    for t in (first, straggler):
        t.cancel()


async def test_orphaned_straggler_reproduces_the_409_without_the_grace_window():
    """With batch_grace_s=0 the old bug is still reproducible — pins the cause.

    This is the control: it proves the grace window is what fixes it, not some
    incidental change in ordering.
    """
    announced_one = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_1", name="mcp__client__get_weather", input={"city": "Paris"})])
    mgr, bridge, conv = _setup(make_settings(batch_grace_s=0.0), [announced_one])

    first = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    await asyncio.sleep(0)
    straggler = asyncio.create_task(
        _dispatch_later(bridge, "get_time", {"tz": "UTC"}, OBSERVED_STAGGER_S))

    chunks = [c async for c in mgr.run_turn(conv)]
    tc = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    assert len(tc[0].calls) == 1, "control case should show the old one-call batch"

    # The straggler lands after the park and is orphaned: pending_ids now holds
    # a call the client was never told about. That mismatch is exactly what made
    # a complete continuation look "partial" and poisoned the turn.
    await asyncio.sleep(OBSERVED_STAGGER_S * 2)
    assert len(bridge.pending_ids) == 2
    assert len(tc[0].calls) == 1

    for t in (first, straggler):
        t.cancel()


async def test_followup_assistant_tool_use_raises_the_quota():
    """A second assistant message announcing a NEW client call must be honoured."""
    first_ev = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_1", name="mcp__client__get_weather", input={"city": "Paris"})])
    second_ev = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_2", name="mcp__client__get_time", input={"tz": "UTC"})])
    mgr, bridge, conv = _setup(make_settings(batch_grace_s=0.30), [first_ev, second_ev])

    first = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    await asyncio.sleep(0)
    late = asyncio.create_task(
        _dispatch_later(bridge, "get_time", {"tz": "UTC"}, 0.10))

    chunks = [c async for c in mgr.run_turn(conv)]
    tc = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    names = sorted(c.name for c in tc[0].calls)
    assert names == ["get_time", "get_weather"], (
        f"follow-up announcement was discarded; client only saw {names}")

    for t in (first, late):
        t.cancel()


async def test_reannounced_block_does_not_raise_the_quota():
    """Re-announcing a block we already hold must not make the loop wait again.

    Claude Code emits one ``assistant`` line per ``tool_use`` block, so the same
    block can be described more than once. Counting the repeat as a new call
    made the loop wait for a dispatch that never comes — observed live as a full
    ``item_timeout`` (10s) hang before parking on the single real call.
    """
    ev = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_1", name="mcp__client__get_weather", input={"city": "Paris"})])
    # The identical block announced a second time.
    repeat = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_1", name="mcp__client__get_weather", input={"city": "Paris"})])
    mgr, bridge, conv = _setup(make_settings(batch_grace_s=0.15), [ev, repeat])

    first = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    await asyncio.sleep(0)

    started = time.monotonic()
    chunks = [c async for c in mgr.run_turn(conv)]
    elapsed = time.monotonic() - started

    tc = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    assert len(tc[0].calls) == 1
    assert elapsed < 1.0, (
        f"re-announced block stalled the batch for {elapsed:.2f}s "
        f"(the 10s item_timeout hang)")

    first.cancel()


async def test_deferred_announcement_is_replayed_on_the_next_turn():
    """The observed hang: a consumed announcement whose dispatch lands later.

    Claude Code announces block B while we are collecting block A, but only
    dispatches B once A's result comes back. The announcing ``assistant`` event
    is already consumed and never repeats, so if it is dropped the next
    ``run_turn`` waits on an event that cannot arrive — measured live as the
    turn hanging until the request timeout. B must be replayed instead.
    """
    ev_a = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_A", name="mcp__client__get_weather", input={"city": "Paris"})])
    ev_b = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_B", name="mcp__client__get_time", input={"tz": "UTC"})])
    mgr, bridge, conv = _setup(make_settings(batch_grace_s=0.15), [ev_a, ev_b])

    call_a = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    await asyncio.sleep(0)
    # B is announced during collection but NOT dispatched yet.

    chunks = [c async for c in mgr.run_turn(conv)]
    tc = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    assert [c.name for c in tc[0].calls] == ["get_weather"]
    assert conv.state == SUSPENDED
    assert conv.deferred_events, (
        "announcement for toolu_B was dropped; the next turn will hang")
    assert [b.id for b in conv.deferred_events[0].tool_uses] == ["toolu_B"]

    # Client answers A; the CLI now dispatches B, and a fresh run_turn starts.
    bridge.resolve(tc[0].calls[0].id, '{"temp_c":21}')
    call_b = asyncio.create_task(bridge.dispatch("get_time", {"tz": "UTC"}))
    await asyncio.sleep(0)

    started = time.monotonic()
    chunks2 = [c async for c in mgr.run_turn(conv)]
    elapsed = time.monotonic() - started

    tc2 = [c for c in chunks2 if isinstance(c, ToolCallsChunk)]
    assert tc2, f"turn produced no tool calls in {elapsed:.2f}s (the hang)"
    assert [c.name for c in tc2[0].calls] == ["get_time"]
    assert not conv.deferred_events
    assert elapsed < 1.0, f"replay took {elapsed:.2f}s"

    for t in (call_a, call_b):
        t.cancel()


async def test_replayed_announcement_that_never_dispatches_is_ignored():
    """A phantom announcement must not tear the turn down.

    Before this change the event was discarded outright, so the turn simply
    continued. A replay that yields nothing has to behave the same way rather
    than surfacing \"expected client tool calls did not arrive\" (502).
    """
    ev_a = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_A", name="mcp__client__get_weather", input={"city": "Paris"})])
    ghost = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_ghost", name="mcp__client__get_time", input={"tz": "UTC"})])
    mgr, bridge, conv = _setup(make_settings(batch_grace_s=0.15), [ev_a, ghost])

    call_a = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    await asyncio.sleep(0)

    chunks = [c async for c in mgr.run_turn(conv)]
    tc = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    assert len(tc[0].calls) == 1
    assert conv.deferred_events

    # Answer A. The ghost is never dispatched; the stream then ends normally.
    bridge.resolve(tc[0].calls[0].id, '{"temp_c":21}')
    conv.session._events = [TurnDone(stop_reason="end_turn")]

    started = time.monotonic()
    chunks2 = [c async for c in mgr.run_turn(conv)]
    elapsed = time.monotonic() - started

    assert not any(isinstance(c, ErrorChunk) for c in chunks2), (
        f"phantom replay produced {[c for c in chunks2 if isinstance(c, ErrorChunk)]}")
    assert any(isinstance(c, DoneChunk) for c in chunks2), "turn should finish cleanly"
    assert elapsed < 3.0, f"phantom replay stalled for {elapsed:.2f}s"

    call_a.cancel()


async def test_deferred_dispatch_landing_before_resume_is_not_poisoned():
    """A deferred call that dispatches between park and resume must survive.

    This is the residual failure mode the whole 409 class reduces to: Claude
    announces B, we defer it and park on A, and B's MCP dispatch lands while we
    are SUSPENDED — so ``pending_ids`` holds A *and* B when the client's
    continuation (which only knows about A) arrives. ``resume`` must leave B
    pending for the next turn instead of ``fail_all``-ing it, or the turn is
    poisoned and the client drops onto its fallback model.
    """
    ev_a = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_A", name="mcp__client__get_weather", input={"city": "Paris"})])
    ev_b = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_B", name="mcp__client__get_time", input={"tz": "UTC"})])
    mgr, bridge, conv = _setup(make_settings(batch_grace_s=0.15), [ev_a, ev_b])

    call_a = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    await asyncio.sleep(0)

    chunks = [c async for c in mgr.run_turn(conv)]
    tc = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    assert len(tc[0].calls) == 1, "should have parked on A alone"
    a_call = tc[0].calls[0]
    assert conv.state == SUSPENDED
    assert conv.deferred_events, "B must be deferred"

    # B dispatches now, while SUSPENDED — its future is pending and unresolved.
    call_b = asyncio.create_task(bridge.dispatch("get_time", {"tz": "UTC"}))
    await asyncio.sleep(0)
    assert len(bridge.pending_ids) == 2, "A and B are both pending at resume time"

    # Client answers only A (it was only told about A).
    from app.openai_models import ChatCompletionRequest, ChatMessage, FunctionCall, ToolCall
    cont = ChatCompletionRequest(tools=_tools(), messages=[
        ChatMessage(role="user", content="x"),
        ChatMessage(role="assistant", content=None, tool_calls=[
            ToolCall(id=a_call.id, function=FunctionCall(name="get_weather", arguments='{}'))]),
        ChatMessage(role="tool", tool_call_id=a_call.id, content='{"temp_c":21}'),
    ])
    resumed = await mgr.resume(cont)
    assert resumed.conv_id == "c1"
    assert conv.state == RUNNING
    # B's future must still be pending — NOT failed.
    assert not call_b.done(), "resume failed the deferred call B"

    # Next turn collects B and returns it to the client.
    chunks2 = [c async for c in mgr.run_turn(conv)]
    tc2 = [c for c in chunks2 if isinstance(c, ToolCallsChunk)]
    assert [c.name for c in tc2[0].calls] == ["get_time"]

    for t in (call_a, call_b):
        t.cancel()


async def test_announced_call_that_never_dispatches_is_bounded():
    """An announced call whose MCP dispatch never arrives must not hang the turn.

    The announcement is a hint, not a guarantee. Waiting for it unbounded costs
    a full ``item_timeout`` of dead air on every turn where Claude announces a
    call it then does not invoke through the bridge.
    """
    first_ev = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_1", name="mcp__client__get_weather", input={"city": "Paris"})])
    ghost_ev = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_ghost", name="mcp__client__get_time", input={"tz": "UTC"})])
    mgr, bridge, conv = _setup(make_settings(batch_grace_s=0.15), [first_ev, ghost_ev])

    first = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    await asyncio.sleep(0)
    # No dispatch is ever made for toolu_ghost.

    started = time.monotonic()
    chunks = [c async for c in mgr.run_turn(conv)]
    elapsed = time.monotonic() - started

    tc = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    assert len(tc[0].calls) == 1, "should park on the one call that really arrived"
    assert conv.state == SUSPENDED
    assert elapsed < 2.0, (
        f"ghost announcement stalled the turn for {elapsed:.2f}s; the wait for an "
        f"announced-but-undispatched call is not bounded")

    first.cancel()


async def test_single_call_still_parks_promptly():
    """No straggler ⇒ the batch must still park, bounded by the grace window."""
    announced_one = AssistantToolUse(tool_uses=[
        ToolUseBlock(id="toolu_1", name="mcp__client__get_weather", input={"city": "Paris"})])
    mgr, bridge, conv = _setup(make_settings(batch_grace_s=0.15), [announced_one])

    first = asyncio.create_task(bridge.dispatch("get_weather", {"city": "Paris"}))
    await asyncio.sleep(0)

    started = time.monotonic()
    chunks = [c async for c in mgr.run_turn(conv)]
    elapsed = time.monotonic() - started

    tc = [c for c in chunks if isinstance(c, ToolCallsChunk)]
    assert len(tc[0].calls) == 1
    assert conv.state == SUSPENDED
    assert elapsed < 1.0, f"parking took {elapsed:.2f}s, grace window is not bounding it"

    first.cancel()
