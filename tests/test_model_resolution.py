"""Model resolution: no silent downgrade, and a never-stale default alias.

resolve_model() must pass through known aliases and `claude*` ids, use the server
default for an empty/None request, and RAISE on an unrecognized non-`claude*` id
(e.g. "gpt-4o") rather than silently substituting the default — which would run a
different model than the client asked for with no signal. The route surfaces that
as a 400.
"""

import pytest
from fastapi.testclient import TestClient

from app import main as main_mod
from app.config import KNOWN_MODEL_ALIASES, Settings, resolve_model


def _settings(**over):
    base = {"host": "127.0.0.1"}
    base.update(over)
    return Settings(**base)


def test_known_aliases_pass_through():
    s = _settings()
    for alias in KNOWN_MODEL_ALIASES:
        assert resolve_model(alias, s) == alias


def test_claude_ids_pass_through():
    s = _settings()
    assert resolve_model("claude-sonnet-5", s) == "claude-sonnet-5"
    assert resolve_model("claude-opus-4-8", s) == "claude-opus-4-8"


def test_empty_or_none_uses_default():
    s = _settings(default_model="opus")
    assert resolve_model(None, s) == "opus"
    assert resolve_model("", s) == "opus"
    assert resolve_model("   ", s) == "opus"


@pytest.mark.parametrize("bad", ["gpt-4o", "gpt-4o-mini", "llama-3.1-70b", "mistral-large"])
def test_unknown_model_raises_not_downgrades(bad):
    s = _settings(default_model="opus")
    with pytest.raises(ValueError) as ei:
        resolve_model(bad, s)
    # the error names the offending model so the client can fix its config
    assert bad in str(ei.value)


def test_default_model_is_a_nonstale_alias():
    """The shipped default must be an alias (CLI-resolved to the current model),
    not a dated concrete id that silently rots."""
    assert Settings().default_model in KNOWN_MODEL_ALIASES


def test_route_returns_400_on_unknown_model(monkeypatch):
    monkeypatch.setattr(main_mod, "get_settings",
                        lambda: _settings(api_key=None, default_model="opus"))
    client = TestClient(main_mod.create_app())
    resp = client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["param"] == "model"
    assert "gpt-4o" in body["error"]["message"]
