"""ConversationManager — reconciles stateless OpenAI requests with stateful,
suspendable Claude Code subprocesses for the tool-passthrough path.

A *conversation* owns one live ``claude`` subprocess plus its
:class:`~app.mcp_bridge.ConversationBridge`. It moves through:

    RUNNING ── tool calls ──▶ SUSPENDED ──(next request resolves results)──▶ RUNNING
            └────────────── clean result ──────────────▶ CLOSED

Request classification:

* **Continuation** — the message list ends with one or more ``tool`` results. We
  match them to a suspended conversation by ``tool_call_id`` (the ids we minted),
  resolve those Futures, and let Claude resume. No new user turn is sent.
* **Fresh turn** — otherwise. A new conversation is created: the bridge is
  registered, the subprocess spawned with an ``--mcp-config`` pointing at this
  conversation's MCP URL, and the (system-stripped) history is folded into one
  user turn.

The whole multi-step tool loop for one client turn is a single conversation kept
alive across the continuations; matching by ``tool_call_id`` needs no hashing.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from app.claude_session import STREAM_CLOSED, ClaudeSession, prompt_session_kwargs as _prompt_kwargs
from app.config import Settings
from app.events import (
    AssistantToolUse,
    ControlDialog,
    Error,
    PermissionRequest,
    QuestionRequest,
    TextDelta,
    ThinkingDelta,
    TurnDone,
)
from app.mcp_bridge import ConversationBridge, McpBridge, PendingCall
from app.openai_models import ChatCompletionRequest, ChatMessage
from app.translate import fold_conversation, message_text, split_system, usage_from_turn
from app.warmpool import Signature, WarmPool, tools_signature

logger = logging.getLogger("cci.conv")

RUNNING = "running"
SUSPENDED = "suspended"
CLOSED = "closed"


# ── turn-loop chunks (rendered to SSE / JSON by the chat route) ─────────────—


@dataclass
class TextChunk:
    text: str


@dataclass
class ThinkingChunk:
    """A chunk of Claude's extended-thinking (reasoning) text."""

    text: str


@dataclass
class ToolCallsChunk:
    calls: list[PendingCall]


@dataclass
class DoneChunk:
    finish_reason: str
    usage: dict


@dataclass
class ErrorChunk:
    message: str
    status_code: int = 502


@dataclass
class ToolBoundaryChunk:
    """Marks an internal (built-in) tool use between two assistant text
    segments. Carries no text; the route uses it to insert a blank-line seam so
    the next text block does not glue onto the previous one."""


TurnChunk = Union[TextChunk, ThinkingChunk, ToolCallsChunk, DoneChunk, ErrorChunk, ToolBoundaryChunk]


# ── conversation ────────────────────────────────────────────────────────────


@dataclass
class Conversation:
    conv_id: str
    session: ClaudeSession
    bridge: ConversationBridge
    model: str
    state: str = RUNNING
    last_activity: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_activity = time.monotonic()


class ExpiredContinuation(Exception):
    """A continuation whose tool_call_ids match no suspended conversation."""


class ConversationManager:
    def __init__(self, mcp: McpBridge, settings: Settings) -> None:
        self.mcp = mcp
        self.settings = settings
        self._conversations: dict[str, Conversation] = {}
        self._pending_index: dict[str, str] = {}  # tool_call_id -> conv_id
        self._lock = asyncio.Lock()
        self._counter = 0
        # Warm subprocess pool (Phase 2). None unless CCI_WARM_POOL_SIZE > 0, so
        # it ships dark. Lifespan owns its start/stop (see app/main.lifespan).
        self.pool: Optional[WarmPool] = (
            WarmPool(mcp, settings, settings.warm_pool_size)
            if settings.warm_pool_size > 0
            else None
        )

    # ── classification ───────────────────────────────────────────────────—

    @staticmethod
    def is_continuation(req: ChatCompletionRequest) -> bool:
        for m in reversed(req.messages):
            if m.role == "tool":
                return True
            return False
        return False

    @staticmethod
    def _trailing_tool_messages(req: ChatCompletionRequest) -> list[ChatMessage]:
        out: list[ChatMessage] = []
        for m in reversed(req.messages):
            if m.role == "tool":
                out.append(m)
            else:
                break
        out.reverse()
        return out

    # ── creation (fresh turn) ─────────────────────────────────────────────—

    def _next_conv_id(self) -> str:
        self._counter += 1
        return f"conv{self._counter}-{int(time.time())}"

    def _mcp_url(self, conv_id: str) -> str:
        prefix = self.settings.mcp_path_prefix.rstrip("/")
        return f"http://127.0.0.1:{self.settings.port}{prefix}/{conv_id}"

    async def create(
        self,
        req: ChatCompletionRequest,
        *,
        model: str,
        workdir: Path,
        effort: Optional[str],
    ) -> Conversation:
        convo, system = split_system(req.messages)
        content = fold_conversation(convo)

        # ── warm-pool fast path ───────────────────────────────────────────—
        # On a signature match, the proc is already spawned (and warm): late-bind
        # the request's tools onto its pre-registered bridge BEFORE the first user
        # turn (list_tools is only consulted after the turn starts, so the schema
        # is correct), then send the turn. No spawn, no register.
        if self.pool is not None:
            sig = Signature(
                model=model, effort=effort, workdir=str(workdir), system=system,
                tools_key=tools_signature(req.tools),
            )
            entry = await self.pool.acquire(sig, req.tools or [])
            if entry is not None:
                # Bridge already carries matching schemas (sig includes tools_key);
                # rebind to the request's exact tool objects for good measure.
                entry.bridge.tools = req.tools or []
                conv = Conversation(
                    conv_id=entry.conv_id, session=entry.session,
                    bridge=entry.bridge, model=model,
                )
                async with self._lock:
                    self._conversations[entry.conv_id] = conv
                logger.info("conv=%s adopted from warm pool (model=%s, %d tools)",
                            entry.conv_id, model, len(req.tools or []))
                await entry.session.send_user_turn(content)
                return conv

        async with self._lock:
            conv_id = self._next_conv_id()
        bridge = ConversationBridge(conv_id, req.tools or [])
        self.mcp.register(bridge)

        mcp_config = {
            "mcpServers": {
                "client": {"type": "http", "url": self._mcp_url(conv_id)}
            }
        }
        session = ClaudeSession(
            claude_bin=self.settings.claude_bin,
            model=model,
            permission_mode=self.settings.permission_mode,
            workdir=workdir,
            effort=effort,
            mcp_config=mcp_config,
            enable_tool_search=self.settings.enable_tool_search,
            timing_log=self.settings.timing_log,
            timing_label="tool",
            **_prompt_kwargs(self.settings, system),
        )
        await session.start()
        conv = Conversation(conv_id=conv_id, session=session, bridge=bridge, model=model)
        async with self._lock:
            self._conversations[conv_id] = conv
        logger.info("conv=%s created (model=%s, %d tools)", conv_id, model, len(req.tools or []))
        await session.send_user_turn(content)
        return conv

    # ── resume (continuation) ─────────────────────────────────────────────—

    async def resume(self, req: ChatCompletionRequest) -> Conversation:
        tool_msgs = self._trailing_tool_messages(req)
        if not tool_msgs:
            raise ExpiredContinuation("no trailing tool results in continuation request")
        ids = [m.tool_call_id for m in tool_msgs if m.tool_call_id]

        # Locate AND claim the conversation atomically: verify it is suspended
        # and flip it to RUNNING under one lock acquisition. A second concurrent
        # continuation for the same conv then sees RUNNING and is rejected,
        # instead of both passing the check and double-driving the one Claude
        # subprocess (the TOCTOU race). The claimed ids leave the global index
        # while we still hold the lock.
        async with self._lock:
            conv_id = next((self._pending_index[i] for i in ids if i in self._pending_index), None)
            conv = self._conversations.get(conv_id) if conv_id else None
            if conv is None or conv.state != SUSPENDED:
                raise ExpiredContinuation(
                    "tool results reference an expired or unknown conversation; retry the turn"
                )
            conv.state = RUNNING
            conv.touch()
            for tid in ids:
                self._pending_index.pop(tid, None)

        # Deliver results to the per-call Futures (outside the lock).
        outstanding = set(conv.bridge.pending_ids)
        answered: set[str] = set()
        resolved = 0
        for m in tool_msgs:
            if not m.tool_call_id:
                continue
            answered.add(m.tool_call_id)
            if conv.bridge.resolve(m.tool_call_id, message_text(m)):
                resolved += 1

        # Any call Claude is still blocked on that this continuation did not
        # answer would hang the subprocess until the request timeout. Fail those
        # Futures so the turn errors out promptly instead of stalling.
        missing = outstanding - answered
        if missing:
            logger.warning("conv=%s partial continuation: %d of %d tool calls unanswered",
                           conv.conv_id, len(missing), len(outstanding))
            conv.bridge.fail_all("continuation did not supply all tool results")

        logger.info("conv=%s resumed (%d/%d tool results)", conv.conv_id, resolved, len(tool_msgs))
        return conv

    # ── turn loop ─────────────────────────────────────────────────────────—

    async def run_turn(self, conv: Conversation) -> AsyncIterator[TurnChunk]:
        timeout = self.settings.request_timeout_s
        try:
            while True:
                ev = await conv.session.next_event(timeout=timeout)
                if ev is None:
                    yield ErrorChunk("upstream timeout", status_code=504)
                    await self._close(conv)
                    return
                if ev is STREAM_CLOSED:
                    yield DoneChunk("stop", {})
                    await self._close(conv)
                    return
                if isinstance(ev, TextDelta):
                    yield TextChunk(ev.text)
                elif isinstance(ev, ThinkingDelta):
                    yield ThinkingChunk(ev.text)
                elif isinstance(ev, AssistantToolUse):
                    client = ev.client_calls
                    if not client:
                        logger.debug("conv=%s internal tools: %s", conv.conv_id,
                                     [b.name for b in ev.builtin_calls])
                        # Built-in tool ran internally; surface a seam marker so
                        # the route's OutputFilter starts the next text segment
                        # on a fresh blank line instead of gluing it on.
                        yield ToolBoundaryChunk()
                        continue
                    batch = await self._collect_client_batch(conv, len(client))
                    if batch is None:
                        yield ErrorChunk("upstream timeout", status_code=504)
                        await self._close(conv)
                        return
                    if not batch:
                        yield ErrorChunk("expected client tool calls did not arrive", status_code=502)
                        await self._close(conv)
                        return
                    async with self._lock:
                        for pc in batch:
                            self._pending_index[pc.id] = conv.conv_id
                        conv.state = SUSPENDED
                    conv.touch()
                    logger.info("conv=%s suspended on %d tool call(s)", conv.conv_id, len(batch))
                    yield ToolCallsChunk(batch)
                    return  # park SUSPENDED; subprocess blocked in call_tool
                elif isinstance(ev, TurnDone):
                    yield DoneChunk(_finish(ev.stop_reason), usage_from_turn(ev))
                    await self._close(conv)
                    return
                elif isinstance(ev, Error):
                    yield ErrorChunk(ev.message, status_code=502)
                    await self._close(conv)
                    return
                elif isinstance(ev, (PermissionRequest, QuestionRequest, ControlDialog)):
                    await self._handle_control_event(conv, ev)
                    continue
                # Init / others: ignore.
        except asyncio.CancelledError:
            # Client disconnected mid-turn (not at a clean suspend point): tear
            # the conversation down so no subprocess is orphaned.
            if conv.state != SUSPENDED:
                await self._close(conv)
            raise

    async def _handle_control_event(
        self, conv: Conversation, ev: Union[PermissionRequest, QuestionRequest, ControlDialog]
    ) -> None:
        """Answer a control_request so the subprocess unblocks.

        Under bypassPermissions the CLI still asks for MCP tools (observed on
        2.1.206, unlike the built-in-tool path) — auto-allow so the subprocess
        calls the tool instead of hanging on an answer that never arrives. There
        is no human on the other end of bypassPermissions, so a question/dialog
        is cancelled rather than answered.
        """
        if isinstance(ev, PermissionRequest):
            logger.debug("conv=%s auto-allowing permission request for %s", conv.conv_id, ev.tool)
            await conv.session.send_control_response(
                ev.request_id, {"behavior": "allow", "updatedInput": ev.input}
            )
        elif isinstance(ev, QuestionRequest):
            await conv.session.send_control_response(
                ev.request_id, {"behavior": "deny", "message": "no interactive user available"}
            )
        else:
            await conv.session.send_control_response(
                ev.request_id, {"behavior": ev.cancel_behavior()}
            )

    async def _collect_client_batch(
        self, conv: Conversation, n: int, *, item_timeout: float = 10.0
    ) -> Optional[list[PendingCall]]:
        """Collect the ``n`` client calls Claude just announced via ``tool_use``.

        Claude does not actually invoke the MCP tool until it gets a
        control_response for the ``can_use_tool`` permission check the CLI
        raises for MCP tools even under bypassPermissions (see
        ``_handle_control_event``) — so this cannot simply block on the
        bridge's incoming queue. It races that queue against the session's own
        event stream so control_requests keep getting answered *while* waiting
        for the batch, or the two sides deadlock: the turn loop waiting on the
        bridge, the bridge waiting on a control_response only the turn loop
        can send. Returns ``None`` on a hard stop (timeout/EOF/error); a
        shorter-than-``n`` list only if ``item_timeout`` elapses with no new
        arrival, matching the old ``collect_batch`` contract.
        """
        batch: list[PendingCall] = []
        incoming_task: Optional[asyncio.Task] = None
        event_task: Optional[asyncio.Task] = None
        try:
            while len(batch) < n:
                if incoming_task is None:
                    incoming_task = asyncio.ensure_future(conv.bridge.next_incoming())
                if event_task is None:
                    event_task = asyncio.ensure_future(conv.session.next_event(timeout=item_timeout))
                done, _ = await asyncio.wait(
                    {incoming_task, event_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if incoming_task in done:
                    batch.append(incoming_task.result())
                    incoming_task = None
                if event_task in done:
                    ev = event_task.result()
                    event_task = None
                    if ev is None:
                        logger.warning(
                            "conv=%s collect_batch timed out at %d/%d", conv.conv_id, len(batch), n
                        )
                        break
                    if ev is STREAM_CLOSED or isinstance(ev, Error):
                        # Both tasks can land in the same wait round: if the
                        # batch is already complete, the calls are real and the
                        # client can still run them — don't throw them away.
                        return batch if len(batch) >= n else None
                    if isinstance(ev, (PermissionRequest, QuestionRequest, ControlDialog)):
                        await self._handle_control_event(conv, ev)
                    # TextDelta / further AssistantToolUse / others: ignore —
                    # the batch is what we're here for.
            return batch
        finally:
            for t in (incoming_task, event_task):
                if t is not None and not t.done():
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t

    # ── teardown ──────────────────────────────────────────────────────────—

    async def _close(self, conv: Conversation) -> None:
        if conv.state == CLOSED:
            return
        conv.state = CLOSED
        conv.bridge.fail_all("conversation closed")
        async with self._lock:
            self._conversations.pop(conv.conv_id, None)
            for tid in conv.bridge.pending_ids:
                self._pending_index.pop(tid, None)
            stale = [tid for tid, cid in self._pending_index.items() if cid == conv.conv_id]
            for tid in stale:
                self._pending_index.pop(tid, None)
        self.mcp.unregister(conv.conv_id)
        await conv.session.aclose()
        logger.info("conv=%s closed", conv.conv_id)

    async def close_all(self) -> None:
        for conv in list(self._conversations.values()):
            await self._close(conv)

    # ── garbage collection ────────────────────────────────────────────────—

    async def gc_once(self) -> int:
        """Reap conversations whose TTL elapsed. Returns the count closed.

        A SUSPENDED conversation (a client turn that never returned its tool
        results) past ``suspended_ttl_s`` is killed; failing its pending Futures
        unblocks the subprocess before teardown. Any conversation idle past
        ``idle_session_ttl_s`` is also evicted.
        """
        now = time.monotonic()
        susp_ttl = self.settings.suspended_ttl_s
        idle_ttl = self.settings.idle_session_ttl_s
        victims: list[Conversation] = []
        async with self._lock:
            for conv in list(self._conversations.values()):
                age = now - conv.last_activity
                if conv.state == SUSPENDED and age > susp_ttl:
                    victims.append(conv)
                elif age > idle_ttl:
                    victims.append(conv)
        for conv in victims:
            logger.info("conv=%s GC (state=%s)", conv.conv_id, conv.state)
            await self._close(conv)
        return len(victims)

    async def gc_loop(self) -> None:
        """Background task: periodically run :meth:`gc_once`."""
        interval = max(1, self.settings.gc_interval_s)
        while True:
            try:
                await asyncio.sleep(interval)
                await self.gc_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                logger.exception("GC loop iteration failed")


def _finish(stop_reason: Optional[str]) -> str:
    from app.translate import map_finish_reason

    return map_finish_reason(stop_reason)
