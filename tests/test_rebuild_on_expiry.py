"""Tests for rebuild-on-expiry: answer an expired tool continuation by starting
a fresh session from the request's own history, instead of returning HTTP 409.

Background
----------
The server is stateful: a turn that emits tool calls parks a live ``claude``
subprocess in the SUSPENDED state until the tool results arrive in the next
request. If that suspended conversation is gone by then — GC'd past
``suspended_ttl_s``, or wiped by a server restart — ``resume()`` raises
``ExpiredContinuation`` and the route answers 409. The client cannot retry that
409 against the same session (it is physically gone), so the turn is either lost
or failed over to a different model, even though the request itself is fine.

OpenAI clients are stateless, so the continuation ALWAYS carries the full
message history (system, user, the assistant turn with tool_calls, and the tool
results). ``resume_or_rebuild`` uses it to rebuild a fresh conversation and
answer the turn normally, gated by ``settings.rebuild_on_expiry`` (default on).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.conversation import Conversation, ConversationManager, ExpiredContinuation
from app.main import create_app
from app.mcp_bridge import McpBridge
from app.openai_models import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionCall,
    FunctionDef,
    ToolCall,
    ToolDef,
)


class _FakeSession:
    """Stand-in for ClaudeSession: records the folded turn, streams nothing."""

    last_content = None

    def __init__(self, **kwargs):
        pass

    async def start(self):
        pass

    async def send_user_turn(self, content):
        _FakeSession.last_content = content

    async def next_event(self, timeout=None):
        from app.claude_session import STREAM_CLOSED

        return STREAM_CLOSED

    async def aclose(self):
        pass

    @property
    def running(self):
        return True


def _make_settings(**over):
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


def _expired_continuation_req() -> ChatCompletionRequest:
    """A continuation (ends with a tool result) whose tool_call_id matches no
    suspended conversation — i.e. the session expired / was wiped."""
    return ChatCompletionRequest(
        model="sonnet",
        stream=False,
        tools=[ToolDef(function=FunctionDef(name="calc", parameters={"type": "object", "properties": {}}))],
        messages=[
            ChatMessage(role="user", content="What is 7 times 6? Use calc then tell me the number."),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id="call_ghost_1", function=FunctionCall(name="calc", arguments='{"a": 7, "b": 6}'))],
            ),
            ChatMessage(role="tool", tool_call_id="call_ghost_1", content="42"),
        ],
    )


def _expired_continuation_json() -> dict:
    return {
        "model": "sonnet",
        "stream": False,
        "tools": [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object", "properties": {}}}}],
        "messages": [
            {"role": "user", "content": "What is 7 times 6? Use calc then tell me the number."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_ghost_1", "type": "function", "function": {"name": "calc", "arguments": '{"a": 7, "b": 6}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_ghost_1", "content": "42"},
        ],
    }


# ── unit: manager.resume_or_rebuild ────────────────────────────────────────—


async def test_resume_or_rebuild_rebuilds_when_expired(monkeypatch):
    """Flag ON: an expired continuation is rebuilt from full history (create),
    folding the whole transcript — user question + tool result — into the fresh
    turn. No ExpiredContinuation escapes."""
    import app.conversation as conv_mod

    _FakeSession.last_content = None
    monkeypatch.setattr(conv_mod, "ClaudeSession", _FakeSession)
    mgr = ConversationManager(McpBridge(), _make_settings(rebuild_on_expiry=True))
    req = _expired_continuation_req()
    assert mgr.is_continuation(req) is True

    conv = await mgr.resume_or_rebuild(req, model="sonnet", workdir=Path("/tmp"), effort=None)

    assert isinstance(conv, Conversation)
    assert isinstance(_FakeSession.last_content, str), "rebuild must fold history into a fresh turn"
    assert "42" in _FakeSession.last_content, "tool result must reach the rebuilt turn"
    assert "7 times 6" in _FakeSession.last_content, "user question must reach the rebuilt turn"


async def test_resume_or_rebuild_raises_when_flag_off(monkeypatch):
    """Flag OFF: legacy behaviour preserved — expired continuation still raises
    ExpiredContinuation (which the route renders as 409)."""
    import app.conversation as conv_mod

    monkeypatch.setattr(conv_mod, "ClaudeSession", _FakeSession)
    mgr = ConversationManager(McpBridge(), _make_settings(rebuild_on_expiry=False))
    req = _expired_continuation_req()

    with pytest.raises(ExpiredContinuation):
        await mgr.resume_or_rebuild(req, model="sonnet", workdir=Path("/tmp"), effort=None)


async def test_duplicate_continuation_is_never_rebuilt(monkeypatch):
    """A live conversation that already owns this turn must NOT be rebuilt: the
    resume claim exists to stop a second copy of a continuation double-running
    the turn, and Claude's built-in tools have real side effects. Flag on."""
    import app.conversation as conv_mod
    from app.conversation import Conversation, DuplicateContinuation, RUNNING
    from app.mcp_bridge import ConversationBridge

    _FakeSession.last_content = None
    monkeypatch.setattr(conv_mod, "ClaudeSession", _FakeSession)
    mgr = ConversationManager(McpBridge(), _make_settings(rebuild_on_expiry=True))
    # A conversation already driving the turn these tool results belong to.
    conv = Conversation(conv_id="c1", session=_FakeSession(), bridge=ConversationBridge("c1", []),
                        model="sonnet", state=RUNNING)
    mgr._conversations["c1"] = conv
    mgr._pending_index["call_ghost_1"] = "c1"

    with pytest.raises(DuplicateContinuation):
        await mgr.resume_or_rebuild(
            _expired_continuation_req(), model="sonnet", workdir=Path("/tmp"), effort=None
        )
    assert _FakeSession.last_content is None, "duplicate continuation must not spawn a rebuild"


# ── integration: full route (200 rebuild vs 409 legacy) ────────────────────—


def test_route_rebuilds_expired_continuation(monkeypatch):
    """Flag ON (default): the route answers 200 by rebuilding the session
    instead of 409. TestClient is entered as a context manager so the lifespan
    builds ``app.state.conv_manager``."""
    import app.conversation as conv_mod
    from app.config import get_settings

    _FakeSession.last_content = None
    monkeypatch.setattr(conv_mod, "ClaudeSession", _FakeSession)
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            r = client.post("/v1/chat/completions", json=_expired_continuation_json())
    finally:
        get_settings.cache_clear()
    assert r.status_code == 200, r.text
    assert isinstance(_FakeSession.last_content, str)
    assert "42" in _FakeSession.last_content


def test_route_returns_409_when_flag_off(monkeypatch):
    """Flag OFF: the route falls back to the legacy 409 so a client's own
    failover still triggers for operators who want it.

    The flag is driven through the env (``CCI_REBUILD_ON_EXPIRY=false``) with the
    settings cache cleared before and after, so the mutation can't leak into
    other tests via the process-wide ``get_settings`` lru_cache.
    """
    import app.conversation as conv_mod
    from app.config import get_settings

    monkeypatch.setattr(conv_mod, "ClaudeSession", _FakeSession)
    monkeypatch.setenv("CCI_REBUILD_ON_EXPIRY", "false")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            r = client.post("/v1/chat/completions", json=_expired_continuation_json())
    finally:
        get_settings.cache_clear()
    assert r.status_code == 409, r.text
    assert "expired" in r.text.lower()
