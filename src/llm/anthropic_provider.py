"""Claude API backend.

Uses native structured outputs (`client.messages.parse`) so agent responses are
schema-valid without prompt-level JSON coaxing. Adaptive thinking is on by
default for Claude Opus 5 - the parameter is deliberately omitted rather than
configured.
"""
from __future__ import annotations

import logging

import anthropic

from src.config import settings
from src.llm.base import LLMError, LLMProvider, T, strip_reasoning

log = logging.getLogger(__name__)

# Claude's safety classifiers can decline offensive-security phrasing that is
# perfectly legitimate in an incident-response corpus. When that happens the API
# returns HTTP 200 with stop_reason="refusal"; we retry once on a sibling model
# before giving up so a single triage call cannot stall the whole graph.
REFUSAL_FALLBACK_MODEL = "claude-opus-4-8"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,  # None -> resolved from env/profile
            timeout=settings.llm_timeout,
        )
        self.model = settings.anthropic_model

    def health(self) -> tuple[bool, str]:
        try:
            self.client.models.retrieve(self.model)
        except Exception as exc:
            return False, f"Claude API unavailable for {self.model}: {exc}"
        return True, self.model

    # ------------------------------------------------------------------ text
    def complete(self, system: str, user: str, *, effort: str = "medium") -> str:
        response = self._create(
            model=self.model,
            system=system,
            user=user,
            effort=effort,
        )
        if response.stop_reason == "refusal" and settings.anthropic_fallbacks:
            log.warning("Refusal on %s, retrying with %s", self.model, REFUSAL_FALLBACK_MODEL)
            response = self._create(
                model=REFUSAL_FALLBACK_MODEL, system=system, user=user, effort=effort
            )
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise LLMError(f"Claude declined the request (category={category})")

        text = "".join(b.text for b in response.content if b.type == "text")
        if not text.strip():
            raise LLMError("Claude returned no text content")
        return strip_reasoning(text)

    def _create(self, *, model: str, system: str, user: str, effort: str):
        return self.client.messages.create(
            model=model,
            max_tokens=settings.llm_max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
        )

    # ------------------------------------------------------------ structured
    def structured(
        self, system: str, user: str, schema: type[T], *, effort: str = "medium"
    ) -> T:
        response = self._parse(model=self.model, system=system, user=user, schema=schema)
        if response.stop_reason == "refusal" and settings.anthropic_fallbacks:
            log.warning("Refusal on %s, retrying with %s", self.model, REFUSAL_FALLBACK_MODEL)
            response = self._parse(
                model=REFUSAL_FALLBACK_MODEL, system=system, user=user, schema=schema
            )
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise LLMError(f"Claude declined the request (category={category})")

        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(f"Claude returned no parsed output for {schema.__name__}")
        return parsed

    def _parse(self, *, model: str, system: str, user: str, schema: type[T]):
        return self.client.messages.parse(
            model=model,
            max_tokens=settings.llm_max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
