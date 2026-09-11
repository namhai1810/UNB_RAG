"""Provider-agnostic LLM interface used by every agent.

Agents never talk to a vendor SDK directly: they ask for either free text or a
validated Pydantic object, and the provider decides how to obtain it (native
structured outputs where available, constrained decoding on vLLM, JSON repair as
a last resort).
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# Reasoning models (Qwen3, DeepSeek-R1, ...) emit a visible scratchpad.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a usable response."""


class LLMProvider(ABC):
    """Minimal surface every backend must implement."""

    name: str

    @abstractmethod
    def complete(self, system: str, user: str, *, effort: str = "medium") -> str:
        """Return free-form text."""

    def health(self) -> tuple[bool, str]:
        """(reachable, human-readable detail). Cheap - must not cost a generation."""
        return True, self.name

    @abstractmethod
    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        effort: str = "medium",
    ) -> T:
        """Return an instance of `schema`, validated."""


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def strip_reasoning(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Handles bare JSON, fenced blocks, and prose wrapped around an object.
    """
    text = strip_reasoning(text)

    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Brace matching: tolerates trailing prose after the object.
    start = text.find("{")
    if start == -1:
        raise LLMError(f"No JSON object found in response: {text[:300]!r}")
    depth, in_str, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError as exc:
                    raise LLMError(f"Malformed JSON object: {exc}") from exc
    raise LLMError(f"Unterminated JSON object in response: {text[:300]!r}")


def validate(schema: type[T], payload: dict[str, Any]) -> T:
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise LLMError(f"Response did not match {schema.__name__}: {exc}") from exc


def json_instructions(schema: type[BaseModel]) -> str:
    """Appendix appended to the system prompt when decoding is not constrained."""
    return (
        "\n\nRespond with a single JSON object and nothing else - no prose, no code "
        "fences. It must validate against this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )
