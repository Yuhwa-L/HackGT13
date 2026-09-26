"""GenAI shopping assistant + deterministic offline fallback. Owner: A.

LLM: `openai` client with base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL (env vars; the
team plans to use Grok — fill from xAI docs). Never hardcode provider details.

Offline fallback (llm_used=False): if DEMO_OFFLINE=1, any env var is missing, or the call fails /
times out (5 s) -> templated message per decision. The expo demo must work with no network.

The system prompt says the LLM may only perform allowed actions, but shop/gate.enforce() is applied
to every response regardless.
"""
from __future__ import annotations

import os

LLM_TIMEOUT_S = 5.0

OFFLINE_TEMPLATES: dict[str, str] = {
    "trust": "TODO(A): offline copy for trust",
    "caution": "TODO(A): offline copy for caution",
    "reject": "TODO(A): offline copy for reject",
}


def llm_configured() -> bool:
    if os.getenv("DEMO_OFFLINE", "1") == "1":
        return False
    return all(os.getenv(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))


def build_prompt(decision: str, allowed: set[str], candidates: list[dict], products: list[dict], user_message: str) -> list[dict]:
    """-> chat messages [{role, content}, ...]. TODO(A)."""
    raise NotImplementedError("TODO(A): build_prompt")


def assist(photo: dict, user_message: str, confirmed_class: str | None, catalog: dict) -> dict:
    """Return a schemas.AssistResponse-shaped dict (after gate.enforce). TODO(A)."""
    raise NotImplementedError("TODO(A): assist")
