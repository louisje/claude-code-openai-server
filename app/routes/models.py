"""``GET /v1/models`` — advertise the Claude models this server exposes.

We list both the short aliases the CLI accepts (``opus``/``sonnet``/…) and the
current concrete ids, so an OpenAI client can pick either. The list is static;
the actual model used per request is whatever the client sends, resolved through
``config.resolve_model``.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import KNOWN_MODEL_ALIASES
from app.openai_models import ModelCard, ModelList

router = APIRouter()

# Short aliases (from config.KNOWN_MODEL_ALIASES, the single source of truth for
# what the CLI accepts) first, then concrete ids limited to current
# (non-deprecated, non-legacy) models.
_MODEL_IDS = [
    *sorted(KNOWN_MODEL_ALIASES),
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
    "claude-fable-5-1",
]


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id=mid) for mid in _MODEL_IDS])


@router.get("/v1/models/{model_id}", response_model=ModelCard)
async def get_model(model_id: str) -> ModelCard:
    return ModelCard(id=model_id)
