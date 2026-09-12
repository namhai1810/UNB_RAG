"""OpenAI-compatible backend - vLLM, TGI, llama.cpp server, Ollama, or OpenAI.

Structured output is attempted in descending order of reliability:
  1. `response_format={"type": "json_schema", ...}`  (vLLM >= 0.6 / OpenAI)
  2. `extra_body={"guided_json": schema}`            (older vLLM)
  3. schema in the prompt + JSON extraction          (anything else)
Each tier is cached per-process, so a server that rejects tier 1 pays the
probe cost exactly once.
"""
from __future__ import annotations

import logging
from time import perf_counter

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from src.config import settings
from src.logging_utils import log_event, payload
from src.llm.base import (
    LLMError,
    LLMProvider,
    T,
    extract_json,
    json_instructions,
    strip_reasoning,
    validate,
)

log = logging.getLogger(__name__)


def _usage(response) -> dict[str, int]:
    """Extract token counts across OpenAI SDK/server versions."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {
        key: value
        for key in keys
        if (value := getattr(usage, key, None)) is not None
    }


# Failures that say nothing about whether a constraint style is supported.
# Probing the next mode after one of these just multiplies the wait.
_FATAL = (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
    InternalServerError,
)

_CORRECTIVE_RETRY = """

Your previous response was malformed or incomplete. Return the complete JSON
object again and make it concise. Include every required field, keep explanatory
strings to at most two short sentences, and do not repeat or summarize the
evidence beyond what the schema requires.
"""


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout,
        )
        self.model = settings.openai_model
        self.base_url_display = settings.openai_base_url
        self._mode: str | None = None  # "json_schema" | "guided_json" | "prompt"
        self._last_usage: dict[str, int] = {}
        self._last_finish_reason: str | None = None

    def health(self) -> tuple[bool, str]:
        try:
            models = [m.id for m in self.client.models.list().data]
        except Exception as exc:
            return False, f"cannot reach {self.base_url_display}: {exc}"
        if models and self.model not in models:
            return False, f"{self.model!r} not served; available: {', '.join(models[:5])}"
        return True, f"{self.model} @ {self.base_url_display}"

    # ------------------------------------------------------------------ text
    def complete(self, system: str, user: str, *, effort: str = "medium") -> str:
        started = perf_counter()
        log_event(
            log,
            "llm.request",
            provider=self.name,
            model=self.model,
            call_type="text",
            effort=effort,
            system=payload(system),
            user=payload(user),
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception:
            log.exception(
                "llm.error | provider=%s model=%s call_type=text",
                self.name,
                self.model,
            )
            raise

        raw_text = response.choices[0].message.content or ""
        text = strip_reasoning(raw_text)
        log_event(
            log,
            "llm.response",
            provider=self.name,
            model=self.model,
            call_type="text",
            duration_ms=round((perf_counter() - started) * 1000, 2),
            finish_reason=getattr(response.choices[0], "finish_reason", None),
            usage=_usage(response),
            raw_chars=len(raw_text),
            output=payload(text),
        )
        if not text.strip():
            raise LLMError("Model returned an empty response")
        return text

    # ------------------------------------------------------------ structured
    def structured(
        self, system: str, user: str, schema: type[T], *, effort: str = "medium"
    ) -> T:
        json_schema = schema.model_json_schema()
        last_error: Exception | None = None
        started = perf_counter()
        log_event(
            log,
            "llm.request",
            provider=self.name,
            model=self.model,
            call_type="structured",
            schema=schema.__name__,
            effort=effort,
            system=payload(system),
            user=payload(user),
        )

        for mode in self._modes():
            text = ""
            # Parsing is inside the probe: a server that silently ignores an
            # unknown parameter answers 200 with prose, which only shows up as a
            # parse failure. Treating that as "mode unsupported" is what lets the
            # chain fall through to schema-in-prompt instead of failing hard.
            try:
                text = self._call(mode, system, user, schema, json_schema)
                result = validate(schema, extract_json(text))
            except _FATAL as exc:
                log_event(
                    log,
                    "llm.error",
                    provider=self.name,
                    model=self.model,
                    call_type="structured",
                    schema=schema.__name__,
                    mode=mode,
                    error=str(exc),
                )
                raise LLMError(
                    f"{self.base_url_display} is not reachable or refused the request: {exc}"
                ) from exc
            except Exception as exc:
                last_error = exc
                log_event(
                    log,
                    "llm.structured_mode_failed",
                    provider=self.name,
                    model=self.model,
                    schema=schema.__name__,
                    mode=mode,
                    error=str(exc),
                    fallback=self._mode is None,
                    finish_reason=getattr(self, "_last_finish_reason", None),
                    usage=getattr(self, "_last_usage", {}),
                    raw_chars=len(text),
                    output=payload(text),
                )
                if self._mode is not None:
                    break            # a proven mode failed: retry below, once
                continue
            if self._mode != mode:
                log.info("Using %r structured-output mode for %s", mode, self.model)
            self._mode = mode
            log_event(
                log,
                "llm.response",
                provider=self.name,
                model=self.model,
                call_type="structured",
                schema=schema.__name__,
                mode=mode,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                finish_reason=getattr(self, "_last_finish_reason", None),
                usage=getattr(self, "_last_usage", {}),
                output=payload(result),
            )
            return result

        if self._mode is not None:
            # A deterministic local model often repeats malformed output when
            # given the identical prompt. Make the single retry corrective and
            # explicitly concise so an incomplete object is not reproduced.
            text = ""
            try:
                text = self._call(
                    self._mode, system + _CORRECTIVE_RETRY, user, schema, json_schema
                )
                result = validate(schema, extract_json(text))
                log_event(
                    log,
                    "llm.response",
                    provider=self.name,
                    model=self.model,
                    call_type="structured",
                    schema=schema.__name__,
                    mode=self._mode,
                    retry=True,
                    duration_ms=round((perf_counter() - started) * 1000, 2),
                    finish_reason=getattr(self, "_last_finish_reason", None),
                    usage=getattr(self, "_last_usage", {}),
                    output=payload(result),
                )
                return result
            except Exception as exc:
                log_event(
                    log,
                    "llm.structured_retry_failed",
                    provider=self.name,
                    model=self.model,
                    schema=schema.__name__,
                    mode=self._mode,
                    error=str(exc),
                    finish_reason=getattr(self, "_last_finish_reason", None),
                    usage=getattr(self, "_last_usage", {}),
                    raw_chars=len(text),
                    output=payload(text),
                )
                raise LLMError(
                    f"{self._mode} request failed twice for {schema.__name__}: {exc}"
                ) from exc

        raise LLMError(
            f"No structured-output strategy worked against {self.base_url_display}: "
            f"{last_error}"
        )

    def _modes(self) -> list[str]:
        return [self._mode] if self._mode else ["json_schema", "guided_json", "prompt"]

    def _call(
        self,
        mode: str,
        system: str,
        user: str,
        schema: type[T],
        json_schema: dict,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
        }
        self._last_usage = {}
        self._last_finish_reason = None
        if mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema,
                    "strict": True,
                },
            }
        elif mode == "guided_json":
            kwargs["extra_body"] = {"guided_json": json_schema}
        else:
            system = system + json_instructions(schema)

        kwargs["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response = self.client.chat.completions.create(**kwargs)
        self._last_usage = _usage(response)
        self._last_finish_reason = getattr(response.choices[0], "finish_reason", None)
        return response.choices[0].message.content or ""
