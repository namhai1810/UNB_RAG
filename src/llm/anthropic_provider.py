"""Claude API backend.

Uses native structured outputs (`client.messages.parse`) so agent responses are
schema-valid without prompt-level JSON coaxing. Adaptive thinking is on by
default for Claude Opus 5 - the parameter is deliberately omitted rather than
configured.
"""
from __future__ import annotations

import logging
from time import perf_counter

import anthropic

from src.config import settings
from src.logging_utils import log_event, payload
from src.llm.base import LLMError, LLMProvider, T, strip_reasoning

log = logging.getLogger(__name__)


def _usage(response) -> dict[str, int]:
    """Extract Anthropic usage fields when returned by the SDK."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    keys = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    return {
        key: value
        for key in keys
        if (value := getattr(usage, key, None)) is not None
    }


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
        started = perf_counter()
        active_model = self.model
        log_event(
            log,
            "llm.request",
            provider=self.name,
            model=active_model,
            call_type="text",
            effort=effort,
            system=payload(system),
            user=payload(user),
        )
        try:
            response = self._create(
                model=active_model,
                system=system,
                user=user,
                effort=effort,
            )
            if response.stop_reason == "refusal" and settings.anthropic_fallbacks:
                active_model = REFUSAL_FALLBACK_MODEL
                log_event(
                    log,
                    "llm.fallback",
                    provider=self.name,
                    from_model=self.model,
                    to_model=active_model,
                    reason="refusal",
                )
                response = self._create(
                    model=active_model, system=system, user=user, effort=effort
                )
        except Exception:
            log.exception(
                "llm.error | provider=%s model=%s call_type=text", self.name, active_model
            )
            raise
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise LLMError(f"Claude declined the request (category={category})")

        raw_text = "".join(b.text for b in response.content if b.type == "text")
        text = strip_reasoning(raw_text)
        log_event(
            log,
            "llm.response",
            provider=self.name,
            model=active_model,
            call_type="text",
            duration_ms=round((perf_counter() - started) * 1000, 2),
            stop_reason=response.stop_reason,
            usage=_usage(response),
            raw_chars=len(raw_text),
            output=payload(text),
        )
        if not text.strip():
            raise LLMError("Claude returned no text content")
        return text

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
        started = perf_counter()
        active_model = self.model
        log_event(
            log,
            "llm.request",
            provider=self.name,
            model=active_model,
            call_type="structured",
            schema=schema.__name__,
            effort=effort,
            system=payload(system),
            user=payload(user),
        )
        try:
            response = self._parse(
                model=active_model, system=system, user=user, schema=schema
            )
            if response.stop_reason == "refusal" and settings.anthropic_fallbacks:
                active_model = REFUSAL_FALLBACK_MODEL
                log_event(
                    log,
                    "llm.fallback",
                    provider=self.name,
                    from_model=self.model,
                    to_model=active_model,
                    reason="refusal",
                    schema=schema.__name__,
                )
                response = self._parse(
                    model=active_model, system=system, user=user, schema=schema
                )
        except Exception:
            log.exception(
                "llm.error | provider=%s model=%s call_type=structured schema=%s",
                self.name,
                active_model,
                schema.__name__,
            )
            raise
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise LLMError(f"Claude declined the request (category={category})")

        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(f"Claude returned no parsed output for {schema.__name__}")
        log_event(
            log,
            "llm.response",
            provider=self.name,
            model=active_model,
            call_type="structured",
            schema=schema.__name__,
            duration_ms=round((perf_counter() - started) * 1000, 2),
            stop_reason=response.stop_reason,
            usage=_usage(response),
            output=payload(parsed),
        )
        return parsed

    def _parse(self, *, model: str, system: str, user: str, schema: type[T]):
        return self.client.messages.parse(
            model=model,
            max_tokens=settings.llm_max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
