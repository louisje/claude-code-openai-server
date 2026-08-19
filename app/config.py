"""Runtime configuration (env-driven, prefix ``CCI_``) and model-alias resolution.

All knobs are read from the environment with a ``CCI_`` prefix (e.g.
``CCI_PORT=9000``). Defaults are chosen so the server runs out of the box against
a local ``claude login`` session with no API key.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Model aliases the `claude` CLI accepts directly on `--model`. Anything in this
# set, or any id beginning with "claude", is passed through verbatim; everything
# else falls back to `default_model`. The CLI itself resolves an alias like
# "opus" to the current concrete id, so we never hard-code dated ids here.
KNOWN_MODEL_ALIASES = {
    "opus",
    "sonnet",
    "haiku",
    "fable",
    "opusplan",
    "default",
}


class Settings(BaseSettings):
    """Server settings. Override any field via a ``CCI_<FIELD>`` environment var."""

    model_config = SettingsConfigDict(env_prefix="CCI_", extra="ignore")

    # ── HTTP server ──────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8787

    # ── Auth ─────────────────────────────────────────────────────────────────
    # Optional bearer token. When set, every /v1 request must carry
    # `Authorization: Bearer <api_key>`. Left unset, the server is open — which
    # is only safe on a loopback bind (see the startup guard in main.create_app).
    api_key: str | None = None

    # ── Claude CLI ───────────────────────────────────────────────────────────
    claude_bin: str = "claude"
    # An ALIAS, never a dated concrete id: the CLI resolves "opus" to the current
    # model, so this default cannot silently go stale (a hardcoded id like
    # "claude-opus-4-8" does). Override with CCI_DEFAULT_MODEL.
    default_model: str = "opus"
    default_effort: str | None = None
    permission_mode: str = "bypassPermissions"
    # Force the CLI to inject tool schemas directly rather than behind a
    # tool-search indirection, so the client's functions are always visible.
    enable_tool_search: bool = False

    # ── "Bare model" mode ──────────────────────────────────────────────────—
    # When true, the spawned claude is stripped of its Claude Code identity and
    # native tooling so it behaves as a plain model fronted by the OpenAI client:
    #   * the request's system message REPLACES claude's default system prompt
    #     (--system-prompt) instead of being appended,
    #   * --exclude-dynamic-system-prompt-sections drops the env/git/identity
    #     context blocks the CLI normally injects,
    #   * --tools "" removes every built-in tool, leaving only the client MCP
    #     tools delivered via --mcp-config.
    # Set CCI_BARE_MODEL_MODE=false to restore the legacy append+full-tools behavior.
    bare_model_mode: bool = True
    # Fallback system prompt used only in bare mode when a request carries no
    # system message of its own (clients normally always send one).
    bare_model_system_prompt: str = "You are a helpful AI assistant."

    # ── Output formatting ──────────────────────────────────────────────────—
    # Rewrite Markdown pipe tables to fenced monospace ASCII so they render in
    # clients (e.g. Discord) that don't support Markdown tables.
    flatten_markdown_tables: bool = True

    # ── Workspace ──────────────────────────────────────────────────────────—
    default_workdir: Path = Path("~/cci-workspace").expanduser()
    # Per-request `workdir` overrides must resolve under one of these roots.
    # Empty => only `default_workdir` is allowed.
    allowed_workdir_roots: list[str] = []

    # ── MCP bridge ─────────────────────────────────────────────────────────—
    mcp_path_prefix: str = "/mcp"

    # ── Cross-turn session reuse ───────────────────────────────────────────—
    # When true, a completed conversation is parked (subprocess kept alive)
    # keyed by a hash of its full history; a later request whose prior turns match
    # that history reuses the live subprocess and sends only the new user message,
    # instead of spawning a fresh proc and refolding the whole conversation into
    # one prompt every turn. Cuts the per-turn history re-billing and preserves
    # structured tool history. Ships OFF: a positive key match always implies an
    # identical prefix, and any miss falls back to the refold path, so enabling it
    # cannot corrupt a conversation — it only trades idle RAM for token savings.
    cross_turn_reuse: bool = False

    # ── Lifecycle / timeouts (seconds) ─────────────────────────────────────—
    request_timeout_s: int = 600
    suspended_ttl_s: int = 300
    idle_session_ttl_s: int = 900
    gc_interval_s: int = 30

    # ── Expired-continuation recovery ──────────────────────────────────────—
    # When a tool continuation references a suspended conversation that is gone
    # (GC'd past suspended_ttl_s, or wiped by a server restart), rebuild a fresh
    # session from the request's FULL history and answer the turn, instead of
    # returning HTTP 409. OpenAI clients are stateless: the continuation already
    # carries the whole transcript (system + user + assistant tool_calls + tool
    # results), so the rebuild loses nothing. A 409, by contrast, is unretryable
    # by construction — the session it refers to is physically gone — so clients
    # that have a fallback model drop the turn onto it, and clients that don't
    # surface a hard error. Set CCI_REBUILD_ON_EXPIRY=false for the legacy 409.
    rebuild_on_expiry: bool = True

    # ── Logging ────────────────────────────────────────────────────────────—
    log_level: str = "INFO"
    # When true, emit per-turn latency metrics (spawn_ms / ttft_ms / total_ms /
    # tok_per_s) to the "cci.timing" logger at INFO. Default off so normal
    # operation pays nothing; scripts/bench.py sets it on the throwaway port.
    timing_log: bool = False

    # ── Warm subprocess pool ───────────────────────────────────────────────—
    # Number of pre-spawned, idle `claude` subprocesses kept ready to adopt on a
    # fresh tool/autonomous turn, eliminating cold-start latency. 0 disables the
    # pool entirely (ships dark). Each pooled proc holds a pre-minted conv_id and
    # an empty bridge whose tools are late-bound on acquire. See app/warmpool.py.
    warm_pool_size: int = 0

    @field_validator("default_workdir", mode="before")
    @classmethod
    def _expand_workdir(cls, v: object) -> object:
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    def is_loopback_host(self) -> bool:
        """True when the bind host only accepts connections from this machine."""
        h = self.host.strip().lower()
        return h in {"localhost", "::1", "0:0:0:0:0:0:0:1"} or h.startswith("127.")

    def resolved_workdir_roots(self) -> list[Path]:
        """The set of allowed workspace roots as resolved absolute paths."""
        roots = [self.default_workdir]
        roots.extend(Path(r).expanduser() for r in self.allowed_workdir_roots)
        return [p.resolve() for p in roots]

    def resolve_workdir(self, requested: str | None) -> Path:
        """Resolve and validate a per-request workdir override.

        Returns ``default_workdir`` when no override is given. Raises
        ``ValueError`` if the override escapes every allowed root (traversal
        guard).
        """
        if not requested:
            return self.default_workdir
        candidate = Path(requested).expanduser().resolve()
        for root in self.resolved_workdir_roots():
            if candidate == root or root in candidate.parents:
                return candidate
        raise ValueError(
            f"workdir {requested!r} is not under an allowed root "
            f"({[str(r) for r in self.resolved_workdir_roots()]})"
        )


def resolve_model(requested: str | None, settings: Settings) -> str:
    """Map an OpenAI ``model`` field to a value the `claude` CLI accepts.

    Known aliases (opus/sonnet/haiku/…) and any ``claude*`` id pass through. An
    empty/``None`` request uses ``default_model``. An UNRECOGNIZED id raises
    ``ValueError`` — the caller surfaces it as a 400 — instead of silently
    downgrading to the default, which would run a different model than the client
    asked for with no signal (e.g. ``gpt-4o`` quietly becoming the default).
    """
    if not requested:
        return settings.default_model
    r = requested.strip()
    if not r:
        return settings.default_model
    if r in KNOWN_MODEL_ALIASES or r.lower().startswith("claude"):
        return r
    raise ValueError(
        f"unknown model {requested!r}: this server speaks the `claude` CLI, so "
        f"model must be a known alias ({', '.join(sorted(KNOWN_MODEL_ALIASES))}) "
        f"or a 'claude*' id. Set the client's model accordingly, or leave it "
        f"empty to use the server default ({settings.default_model!r})."
    )


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()


def ensure_workdir(path: Path) -> Path:
    """Create the workspace dir if missing; return it. Best-effort."""
    os.makedirs(path, exist_ok=True)
    return path
