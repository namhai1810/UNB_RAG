"""LLM provider factory."""
from __future__ import annotations

from functools import lru_cache

from src.config import settings
from src.llm.base import LLMError, LLMProvider


@lru_cache(maxsize=None)
def get_llm(provider: str | None = None) -> LLMProvider:
    """Return the configured backend. Cached so clients are reused across agents."""
    provider = provider or settings.llm_provider
    if provider == "anthropic":
        from src.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if provider == "openai":
        from src.llm.openai_provider import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(f"Unknown LLM provider: {provider!r}")


__all__ = ["get_llm", "LLMProvider", "LLMError"]
