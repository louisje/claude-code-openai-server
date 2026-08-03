"""/mcp is loopback-only + conversation ids are unguessable.

The /mcp mount is unauthenticated by design (the auth middleware gates only /v1)
and is dialed only by the local claude subprocess over 127.0.0.1. A non-loopback
BIND is permitted when CCI_API_KEY is set, so /mcp must independently refuse any
non-loopback PEER — otherwise an off-box party can enumerate the client's tool
schemas and inject tool calls. The enumerable conv{counter}-{unixtime} id made
guessing the mount path trivial; ids now carry uuid4 entropy.
"""

import asyncio
import re

from app.config import Settings
from app.conversation import ConversationManager
from app.mcp_bridge import McpBridge, _peer_is_loopback


# ── conv-id entropy ──────────────────────────────────────────────────────────
def test_conv_id_is_unguessable():
    mgr = ConversationManager(McpBridge(), Settings(host="127.0.0.1"))
    a, b = mgr._next_conv_id(), mgr._next_conv_id()
    assert re.fullmatch(r"conv\d+-[0-9a-f]{32}", a), a
    assert a != b
    # no bare 10-digit unix timestamp (the old, enumerable form)
    assert not re.search(r"-1\d{9}$", a)


# ── peer classification ──────────────────────────────────────────────────────
def test_peer_is_loopback_classification():
    assert _peer_is_loopback(("127.0.0.1", 5000))
    assert _peer_is_loopback(("127.5.5.5", 1))
    assert _peer_is_loopback(("::1", 1))
    assert _peer_is_loopback(("localhost", 1))
    assert not _peer_is_loopback(("10.0.0.5", 1))
    assert not _peer_is_loopback(("0.0.0.0", 1))
    assert not _peer_is_loopback(None)      # fail closed on unknown peer


# ── the ASGI gate ────────────────────────────────────────────────────────────
async def _drive(app, scope):
    sent = []
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    async def send(msg):
        sent.append(msg)
    await app(scope, receive, send)
    return sent


def test_mcp_rejects_nonloopback_peer():
    app = McpBridge().asgi_app()
    scope = {"type": "http", "path": "/conv1-deadbeef", "headers": [],
             "client": ("203.0.113.9", 4444)}   # off-box peer
    sent = asyncio.run(_drive(app, scope))
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert starts and starts[0]["status"] == 403


def test_mcp_rejects_missing_peer():
    app = McpBridge().asgi_app()
    scope = {"type": "http", "path": "/conv1-deadbeef", "headers": [], "client": None}
    sent = asyncio.run(_drive(app, scope))
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert starts and starts[0]["status"] == 403


def test_mcp_allows_loopback_peer(monkeypatch):
    bridge = McpBridge()
    seen = {}
    async def fake_handle(scope, receive, send):
        seen["path"] = scope["path"]
    monkeypatch.setattr(bridge.session_manager, "handle_request", fake_handle)
    app = bridge.asgi_app()
    scope = {"type": "http", "path": "/conv1-deadbeef", "headers": [],
             "client": ("127.0.0.1", 5000)}
    sent = asyncio.run(_drive(app, scope))
    # no 403 emitted, and the request reached the transport (path rewritten to "/")
    assert not [m for m in sent if m.get("status") == 403]
    assert seen.get("path") == "/"
