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
import hashlib
import json
import logging
import time
import uuid
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
IDLE = "idle"  # turn complete, subprocess parked alive for cross-turn reuse


# ── turn-loop chunks (rendered to SSE / JSON by the chat route) ─────────────—


@dataclass
class TextChunk:
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


TurnChunk = Union[TextChunk, ToolCallsChunk, DoneChunk, ErrorChunk, ToolBoundaryChunk]


# ── conversation ────────────────────────────────────────────────────────────


@dataclass
class Conversation:
    conv_id: str
    session: ClaudeSession
    bridge: ConversationBridge
    model: str
    state: str = RUNNING
    last_activity: float = field(default_factory=time.monotonic)
    # Cross-turn reuse bookkeeping (unused unless settings.cross_turn_reuse):
    # `current_msgs` is the message list of the request currently being served
    # (set at create/resume), `reuse_salt` pins the spawn signature, and
    # `reuse_key` is this conv's key in _idle_by_prefix while parked IDLE.
    current_msgs: list = field(default_factory=list)
    reuse_salt: str = ""
    reuse_key: Optional[str] = None
    # Events consumed off the session stream by ``_collect_client_batch`` that
    # ``run_turn`` still has to act on, replayed FIFO before anything new is
    # read. Two kinds land here:
    #   * an ``AssistantToolUse`` Claude announced but had not yet dispatched
    #     over MCP when we had to park — the CLI only issues it once the current
    #     batch's results come back, and the announcing event never repeats;
    #   * a terminal ``TurnDone`` that arrived while we were waiting.
    # Dropping either is what produced the historical failure modes: a late
    # dispatch became an orphan (→ HTTP 409), or the turn loop was left waiting
    # on an event that could not arrive (→ hang until the request timeout).
    deferred_events: list = field(default_factory=list)

    def touch(self) -> None:
        self.last_activity = time.monotonic()


class ExpiredContinuation(Exception):
    """A continuation whose tool_call_ids match no suspended conversation."""


class DuplicateContinuation(ExpiredContinuation):
    """A continuation whose conversation is alive but is not waiting on these
    tool results — a second copy of a turn already claimed. Distinct from a
    plain expiry because it must never be rebuilt: the first copy is (or was)
    answering it, and a rebuild would re-run the whole turn, side effects and
    all, on a second subprocess. Still renders as 409, as it always did."""


class ConversationManager:
    def __init__(self, mcp: McpBridge, settings: Settings) -> None:
        self.mcp = mcp
        self.settings = settings
        self._conversations: dict[str, Conversation] = {}
        self._pending_index: dict[str, str] = {}  # tool_call_id -> conv_id
        self._idle_by_prefix: dict[str, str] = {}  # reuse_key -> conv_id (IDLE convs)
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
        # The counter keeps ids readable in logs; the uuid4 suffix makes them
        # unguessable. A predictable id (e.g. counter + unix timestamp) is
        # enumerable, and the /mcp mount routes purely on it — so an id that can
        # be guessed is an access-control gap. See also the loopback-peer gate in
        # McpBridge.asgi_app; this is defense in depth behind it.
        self._counter += 1
        return f"conv{self._counter}-{uuid.uuid4().hex}"

    def _mcp_url(self, conv_id: str) -> str:
        prefix = self.settings.mcp_path_prefix.rstrip("/")
        return f"http://127.0.0.1:{self.settings.port}{prefix}/{conv_id}"

    # ── cross-turn reuse (opt-in) ─────────────────────────────────────────—

    @staticmethod
    def _norm_msgs(msgs: list[ChatMessage]) -> str:
        """Serialize the INPUT messages (user + tool) of a history for key
        matching. Assistant messages are deliberately excluded: they are the
        subprocess's own outputs, and the client re-sends a *filtered* copy of
        them (table-flattening, newline seams) that would not byte-match what we
        generated. The user/tool inputs, by contrast, are echoed verbatim, so
        keying on them makes a faithful continuation match reliably while two
        genuinely different conversations still differ. Because we only ever
        reuse the SAME subprocess, its assistant outputs already equal the
        client's history by construction — so dropping them from the key loses no
        safety."""
        parts: list[str] = []
        for m in msgs:
            if m.role not in ("user", "tool"):
                continue
            # Serialize list content (content parts) whole, not via message_text:
            # message_text drops non-text parts, so two histories differing only
            # in an attached image would otherwise collide on the same key.
            c = m.content
            txt = json.dumps(c, sort_keys=True) if isinstance(c, list) else (c or "")
            tid = getattr(m, "tool_call_id", "") or ""
            parts.append(f"{m.role}\x1f{txt}\x1f{tid}")
        return "\x1e".join(parts)

    def _reuse_salt(self, model: str, effort: Optional[str], workdir: Path,
                    system: str, tools) -> str:
        """Spawn signature: a subprocess may only be reused for a request that
        would have spawned an identical one (same model/effort/workdir/system/
        tools), so these are folded into the key."""
        return "\x1e".join([
            model, effort or "", str(workdir), tools_signature(tools), system or "",
        ])

    def _prefix_key(self, salt: str, msgs: list[ChatMessage]) -> str:
        return hashlib.sha256((salt + "\x1e" + self._norm_msgs(msgs)).encode("utf-8")).hexdigest()

    async def _try_reuse(
        self, req: ChatCompletionRequest, *, model: str, workdir: Path, effort: Optional[str],
    ) -> Optional[Conversation]:
        """If a parked IDLE conversation has processed exactly this request's
        prior turns, adopt it and send only the new user message. Returns None
        (→ caller spawns fresh) on any doubt — a miss is always safe."""
        if not self.settings.cross_turn_reuse:
            return None
        if not req.messages or req.messages[-1].role != "user":
            return None
        _, system = split_system(req.messages)
        salt = self._reuse_salt(model, effort, workdir, system, req.tools)
        key = self._prefix_key(salt, req.messages[:-1])
        async with self._lock:
            cid = self._idle_by_prefix.get(key)
            conv = self._conversations.get(cid) if cid else None
            if conv is None or conv.state != IDLE or not conv.session.running:
                return None
            # claim it
            self._idle_by_prefix.pop(key, None)
            conv.reuse_key = None
            conv.state = RUNNING
            conv.current_msgs = list(req.messages)
            conv.reuse_salt = salt
            conv.touch()
        conv.bridge.tools = req.tools or []
        tail = fold_conversation([req.messages[-1]])  # just the new user message, no history
        await conv.session.send_user_turn(tail)
        logger.info("conv=%s REUSED across turns (sent tail only, %d prior msgs skipped)",
                    conv.conv_id, len(req.messages) - 1)
        return conv

    async def create(
        self,
        req: ChatCompletionRequest,
        *,
        model: str,
        workdir: Path,
        effort: Optional[str],
    ) -> Conversation:
        reused = await self._try_reuse(req, model=model, workdir=workdir, effort=effort)
        if reused is not None:
            return reused

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
                    current_msgs=list(req.messages),
                    reuse_salt=self._reuse_salt(model, effort, workdir, system, req.tools),
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
        conv = Conversation(
            conv_id=conv_id, session=session, bridge=bridge, model=model,
            current_msgs=list(req.messages),
            reuse_salt=self._reuse_salt(model, effort, workdir, system, req.tools),
        )
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
            if conv is None:
                raise ExpiredContinuation(
                    "tool results reference an expired or unknown conversation; retry the turn"
                )
            # The ids are known but this conversation is not blocked on them:
            # another copy of this same continuation already claimed the turn
            # (or the conversation has since suspended on a later batch). That
            # is what the claim is for — reject, and never rebuild it.
            if conv.state != SUSPENDED or not set(ids) & set(conv.bridge.pending_ids):
                raise DuplicateContinuation(
                    "these tool results were already claimed; the turn is already running"
                )
            conv.state = RUNNING
            conv.touch()
            # The continuation carries the full history the client now holds
            # (incl. the assistant tool_calls and tool results); record it so a
            # later cross-turn reuse keys off what the subprocess has actually
            # processed by the time this turn ends.
            conv.current_msgs = list(req.messages)
            # The claimed ids stay in the index (``_close`` sweeps them) so a
            # duplicate copy of this continuation still resolves to this
            # conversation and is recognised as a duplicate, rather than looking
            # like an expired session and being rebuilt.

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
        # Futures so the turn errors out promptly instead of stalling — unless
        # they are deferred dispatches (announced before we parked, dispatched
        # after), which the next turn collects via the replayed announcement.
        missing = outstanding - answered
        if missing:
            if conv.deferred_events:
                # Deferred calls: leave them pending, the next run_turn collects
                # them from the bridge queue and hands them to the client.
                logger.info(
                    "conv=%s leaving %d deferred call(s) pending for the next turn "
                    "(%d answered)", conv.conv_id, len(missing), len(answered),
                )
            else:
                logger.warning("conv=%s partial continuation: %d of %d tool calls unanswered",
                               conv.conv_id, len(missing), len(outstanding))
                conv.bridge.fail_all("continuation did not supply all tool results")

        logger.info("conv=%s resumed (%d/%d tool results)", conv.conv_id, resolved, len(tool_msgs))
        return conv

    # ── resume-or-rebuild (expired-continuation recovery) ─────────────────—

    async def resume_or_rebuild(
        self,
        req: ChatCompletionRequest,
        *,
        model: str,
        workdir: Path,
        effort: Optional[str],
    ) -> Conversation:
        """Resume a suspended conversation, or rebuild it from full history.

        The happy path is :meth:`resume`: match the trailing tool results to the
        live suspended subprocess and hand them over. If that subprocess is gone
        (GC'd past ``suspended_ttl_s``, or wiped by a restart), ``resume`` raises
        :class:`ExpiredContinuation`, which the route renders as HTTP 409. That
        409 is unretryable by construction — the session it refers to no longer
        exists — so the client either loses the turn or fails it over to another
        model, even though nothing is actually wrong with the request.

        Since OpenAI clients are stateless, the continuation carries the entire
        transcript, so we can just start over: :meth:`create` folds the whole
        thing (the user turn, the assistant turn with tool_calls, and the tool
        results) into one fresh turn and Claude produces the next turn — final
        text or new tool calls — exactly as it would have. Gated by
        ``settings.rebuild_on_expiry`` so the legacy 409 can be restored.
        """
        try:
            return await self.resume(req)
        except DuplicateContinuation:
            # A live conversation already owns this turn: rebuilding would run
            # it a second time. Always the legacy 409.
            raise
        except ExpiredContinuation:
            if not self.settings.rebuild_on_expiry:
                raise
            logger.info(
                "continuation expired; rebuilding a fresh %s session from full "
                "history (%d messages) instead of 409",
                model, len(req.messages),
            )
            return await self.create(req, model=model, workdir=workdir, effort=effort)

    # ── turn loop ─────────────────────────────────────────────────────────—

    async def run_turn(self, conv: Conversation) -> AsyncIterator[TurnChunk]:
        timeout = self.settings.request_timeout_s
        # Terminal cleanup (close, or park for reuse) is done BEFORE yielding the
        # terminal chunk, never after: the route breaks out of this generator as
        # soon as it receives DoneChunk/ToolCallsChunk/ErrorChunk, so any code
        # after that `yield` would not run until the generator is GC'd — too late
        # to park a subprocess for the very next turn to reuse.
        try:
            while True:
                # Drain anything the collector consumed on our behalf before
                # reading new input: a deferred announcement's dispatch is
                # normally already queued on the bridge by now.
                if conv.deferred_events:
                    ev: object = conv.deferred_events.pop(0)
                    replayed = isinstance(ev, AssistantToolUse)
                else:
                    ev = await conv.session.next_event(timeout=timeout)
                    replayed = False
                if ev is None:
                    await self._close(conv)
                    yield ErrorChunk("upstream timeout", status_code=504)
                    return
                if ev is STREAM_CLOSED:
                    await self._close(conv)  # stdout closed => proc gone, not reusable
                    yield DoneChunk("stop", {})
                    return
                if isinstance(ev, TextDelta):
                    yield TextChunk(ev.text)
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
                    batch = await self._collect_client_batch(
                        conv,
                        announced_blocks=list(client),
                        # A replay is a best-effort second look, so bound it:
                        # an announcement is a hint, and Claude may simply never
                        # dispatch it.
                        initial_deadline_s=(
                            self._replay_deadline_s() if replayed else None),
                    )
                    if batch is None:
                        await self._close(conv)
                        yield ErrorChunk("upstream timeout", status_code=504)
                        return
                    if not batch:
                        if replayed or conv.deferred_events:
                            # Either the replayed announcement never became a
                            # real dispatch, or a terminal event overtook it and
                            # is queued to end the turn properly. Historically
                            # such an announcement was discarded outright and
                            # the loop carried on, so tearing the turn down here
                            # would be the regression.
                            logger.info(
                                "conv=%s announcement %s never dispatched; ignoring",
                                conv.conv_id, [b.id for b in client],
                            )
                            continue
                        await self._close(conv)
                        yield ErrorChunk("expected client tool calls did not arrive", status_code=502)
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
                    await self._park_or_close(conv)  # park for reuse, or close
                    yield DoneChunk(_finish(ev.stop_reason), usage_from_turn(ev))
                    return
                elif isinstance(ev, Error):
                    await self._close(conv)
                    yield ErrorChunk(ev.message, status_code=502)
                    return
                elif isinstance(ev, (PermissionRequest, QuestionRequest, ControlDialog)):
                    await self._handle_control_event(conv, ev)
                    continue
                # Init / others: ignore.
        except asyncio.CancelledError:
            # Client disconnected mid-turn (not at a clean suspend point): tear
            # the conversation down so no subprocess is orphaned.
            if conv.state not in (SUSPENDED, IDLE, CLOSED):
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

    def _replay_deadline_s(self) -> float:
        """How long a replayed announcement may wait for its dispatch."""
        return max(1.0, self.settings.batch_grace_s * 4)

    async def _collect_client_batch(
        self,
        conv: Conversation,
        *,
        announced_blocks: Optional[list] = None,
        initial_deadline_s: Optional[float] = None,
        item_timeout: float = 10.0,
    ) -> Optional[list[PendingCall]]:
        """Collect the client calls Claude just announced via ``tool_use``.

        Claude does not actually invoke the MCP tool until it gets a
        control_response for the ``can_use_tool`` permission check the CLI
        raises for MCP tools even under bypassPermissions (see
        ``_handle_control_event``) — so this cannot simply block on the
        bridge's incoming queue. It races that queue against the session's own
        event stream so control_requests keep getting answered *while* waiting
        for the batch, or the two sides deadlock: the turn loop waiting on the
        bridge, the bridge waiting on a control_response only the turn loop
        can send. Returns ``None`` on a hard stop (timeout/EOF/error); a
        shorter-than-announced list only if ``item_timeout`` elapses with no
        new arrival, matching the old ``collect_batch`` contract.

        The announced count is only Claude's *first* announcement, not a
        reliable total, and two realities break the naive "collect N, then
        park" reading:

        * A parallel batch is dispatched over MCP with a stagger (measured
          45-57ms on opus), so a call can arrive just after the quota is met.
        * Claude Code may announce a batch across several ``assistant`` events
          (one per tool_use block), and may dispatch them one per turn — the
          next ``tools/call`` only fires once the current batch's results come
          back.

        An orphaned dispatch is not a benign miss: it stays in
        ``bridge._pending``, makes a genuinely-complete continuation look
        partial, ``resume`` calls ``fail_all`` and poisons the turn, and the
        *next* continuation is rejected as a :class:`DuplicateContinuation` —
        HTTP 409 — knocking the client onto its fallback model. A dropped
        announcement is just as bad: its dispatch has no listener, and the turn
        loop is left waiting on an event that can never arrive.

        The fix tracks *every* announced block in an ordered ``deferred`` list
        and matches arrivals to it by order (the announcement and the MCP
        dispatch carry unrelated ids, so order is the only correlation, exactly
        as the original ``collect_batch(n)`` implicitly assumed). Whatever is
        still unmatched at the end is handed to ``conv.deferred_events`` so the
        next ``run_turn`` collects it, and a terminal ``TurnDone`` is likewise
        returned rather than consumed. Waiting is bounded by ``batch_grace_s``
        once the first arrival has landed, so an announcement that outpaces its
        dispatch (the serial case) can never stall the turn.
        """
        batch: list[PendingCall] = []
        incoming_task: Optional[asyncio.Task] = None
        event_task: Optional[asyncio.Task] = None
        grace = max(0.0, self.settings.batch_grace_s)
        # Ordered announcement blocks not yet matched to a dispatch. Every
        # arrival matches the front of this list; anything left at the end is
        # re-deferred. Starts with the caller's announced blocks so a replay is
        # tracked exactly like a fresh announcement.
        deferred: list = list(announced_blocks or [])
        seen: set[str] = {b.id for b in deferred if b.id}
        # Bounds the *discretionary* waiting: None until the first arrival lands
        # (we always wait for at least one dispatch, as the original code did,
        # bounded by ``item_timeout``), then every further wait is capped by the
        # grace window. A replay starts with a longer, still-bounded deadline.
        deadline: Optional[float] = (
            time.monotonic() + initial_deadline_s if initial_deadline_s else None
        )
        try:
            while True:
                now = time.monotonic()
                if deadline is not None and now >= deadline:
                    if deferred:
                        logger.warning(
                            "conv=%s %d client call(s) announced but only %d dispatched "
                            "within %.2fs; parking on what arrived",
                            conv.conv_id, len(deferred) + len(batch), len(batch), grace,
                        )
                    break
                if incoming_task is None:
                    incoming_task = asyncio.ensure_future(conv.bridge.next_incoming())
                if event_task is None:
                    event_task = asyncio.ensure_future(conv.session.next_event(timeout=item_timeout))
                wait_timeout = (deadline - now) if deadline is not None else None
                done, _ = await asyncio.wait(
                    {incoming_task, event_task},
                    timeout=wait_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    continue  # deadline expired; the top of the loop reports it
                if incoming_task in done:
                    arrived = incoming_task.result()
                    incoming_task = None
                    batch.append(arrived)
                    had_announcement = bool(deferred)
                    if deferred:
                        deferred.pop(0)
                    # Once we have our first dispatch, any further waiting is
                    # discretionary: bound it so a serialized dispatch cannot
                    # stall the turn.
                    deadline = time.monotonic() + grace
                    if not had_announcement:
                        logger.info(
                            "conv=%s swept a late parallel tool call into the batch (now %d)",
                            conv.conv_id, len(batch),
                        )
                if event_task in done:
                    ev = event_task.result()
                    event_task = None
                    if ev is None:
                        if not deferred:
                            break
                        logger.warning(
                            "conv=%s collect_batch timed out at %d/%d",
                            conv.conv_id, len(batch), len(deferred) + len(batch),
                        )
                        break
                    if ev is STREAM_CLOSED or isinstance(ev, Error):
                        # Both tasks can land in the same wait round: if the
                        # batch is already complete, the calls are real and the
                        # client can still run them — don't throw them away.
                        return batch if not deferred else None
                    if isinstance(ev, TurnDone):
                        # The turn ended while we were waiting. Nothing further
                        # will be dispatched, so stop expecting the announced
                        # calls — but hand the terminal event back, or run_turn
                        # blocks on a stream that has nothing left to say.
                        conv.deferred_events.append(ev)
                        deferred.clear()
                        break
                    if isinstance(ev, (PermissionRequest, QuestionRequest, ControlDialog)):
                        await self._handle_control_event(conv, ev)
                    elif isinstance(ev, AssistantToolUse):
                        # Claude Code emits one ``assistant`` line per tool_use
                        # block, so a single step announces itself across several
                        # events and may re-announce blocks we already hold.
                        # Count only blocks not seen before: raising the quota
                        # for a duplicate makes the loop wait for a dispatch that
                        # will never arrive (observed: a 10s hang).
                        new_blocks = [
                            b for b in ev.client_calls
                            if not b.id or b.id not in seen
                        ]
                        seen.update(b.id for b in new_blocks if b.id)
                        if new_blocks:
                            deferred.extend(new_blocks)
                            # An announcement is a hint, not a guarantee of a
                            # dispatch, so stop treating the wait as obligatory.
                            deadline = time.monotonic() + grace if grace > 0 else None
                            logger.info(
                                "conv=%s follow-up tool_use announced %d new client call(s) "
                                "%s; expected now %d",
                                conv.conv_id, len(new_blocks),
                                [b.id for b in new_blocks], len(deferred) + len(batch),
                            )
                        else:
                            logger.debug(
                                "conv=%s follow-up tool_use re-announced known block(s) %s; "
                                "quota unchanged", conv.conv_id,
                                [b.id for b in ev.client_calls],
                            )
                    # TextDelta / others: ignore — the batch is what we're here for.
            if deferred:
                # Hand the unfulfilled announcement to the next run_turn. The
                # CLI typically dispatches these only once the current batch's
                # results come back, and the announcing event never repeats.
                conv.deferred_events.append(AssistantToolUse(tool_uses=list(deferred)))
                logger.info(
                    "conv=%s deferring %d announced-but-undispatched call(s) %s to the next turn",
                    conv.conv_id, len(deferred), [b.id for b in deferred],
                )
            return batch
        finally:
            for t in (incoming_task, event_task):
                if t is not None and not t.done():
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t

    # ── park for reuse ────────────────────────────────────────────────────—

    async def _park_or_close(self, conv: Conversation) -> None:
        """On a clean turn end: if cross-turn reuse is on and the subprocess is
        still alive, park it IDLE keyed by the user/tool inputs it has now
        processed, so a matching next turn can adopt it. Otherwise tear it down
        as before."""
        if not self.settings.cross_turn_reuse or not conv.session.running:
            await self._close(conv)
            return
        key = self._prefix_key(conv.reuse_salt, conv.current_msgs)
        async with self._lock:
            if conv.state == CLOSED:
                return
            conv.state = IDLE
            conv.reuse_key = key
            self._idle_by_prefix[key] = conv.conv_id
        conv.touch()
        logger.info("conv=%s parked IDLE for cross-turn reuse", conv.conv_id)

    # ── teardown ──────────────────────────────────────────────────────────—

    async def _close(self, conv: Conversation) -> None:
        if conv.state == CLOSED:
            return
        conv.state = CLOSED
        conv.bridge.fail_all("conversation closed")
        async with self._lock:
            self._conversations.pop(conv.conv_id, None)
            if conv.reuse_key:
                # Pop only our own index entry: a later conv parked under the
                # same key overwrites it, and closing us must not strand that one.
                if self._idle_by_prefix.get(conv.reuse_key) == conv.conv_id:
                    self._idle_by_prefix.pop(conv.reuse_key)
                conv.reuse_key = None
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
        ``idle_session_ttl_s`` is also evicted — this includes IDLE conversations
        parked for cross-turn reuse, so ``idle_session_ttl_s`` bounds how long a
        reusable subprocess is kept alive between turns.
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
