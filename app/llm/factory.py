"""LLM provider factory.

The rest of the application calls `get_llm_provider()` and receives a
`LLMProvider` — it never imports a concrete provider class directly. This
keeps the provider selection in one place and makes it trivially swappable
via the `LLM_PROVIDER` environment variable.

FastAPI endpoints use this via `Depends(get_llm_provider)` (Phase 11).
"""

from functools import lru_cache

from app.config.settings import get_settings
from app.core.exceptions import LLMProviderError
from app.llm.base import LLMProvider


# Test override — set to a mock instance to bypass the real provider
_llm_override: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider.

    Returns `_llm_override` if set (for tests), otherwise returns the
    cached real provider. The real provider is cached via `_get_real_provider`.
    """
    if _llm_override is not None:
        return _llm_override
    return _get_real_provider()


@lru_cache
def _get_real_provider() -> LLMProvider:
    """Cached real provider — never called when _llm_override is set."""

    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "groq":
        from app.llm.providers.groq_provider import GroqProvider
        return GroqProvider()

    if provider == "openai":
        from app.llm.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()

    raise LLMProviderError(
        f"Unknown LLM provider '{provider}'. "
        "Set LLM_PROVIDER to 'groq' or 'openai' in your .env file.",
        details={"configured_provider": provider},
    )