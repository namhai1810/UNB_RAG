"""Small helpers for readable, bounded pipeline logs.

The application logs model prompts and outputs only when INFO logging is enabled
(``-v`` on either entry point). Payloads are kept on one line and truncated so
an evidence-heavy prompt cannot flood a terminal or log collector.
"""
from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from functools import wraps
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel

from src.config import settings

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    """Return a short correlation id suitable for human-readable logs."""
    return uuid4().hex[:12]


def set_request_id(request_id: str) -> Token:
    return _request_id.set(request_id)


def reset_request_id(token: Token) -> None:
    _request_id.reset(token)


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def compact(value: Any, *, max_chars: int | None = None) -> str:
    """Serialize a value on one line and cap its size."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        rendered = json.dumps(value, ensure_ascii=False)
    else:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            default=_json_default,
            separators=(",", ":"),
        )

    limit = max_chars or settings.log_max_chars
    if len(rendered) <= limit:
        return rendered
    omitted = len(rendered) - limit
    return f"{rendered[:limit]}…<truncated {omitted} chars>"


def payload(value: Any) -> Any:
    """Render potentially sensitive prompt/output data according to settings."""
    if settings.log_payloads:
        return value
    size = len(value) if isinstance(value, (str, list, tuple, dict)) else "unknown"
    return f"<payload logging disabled; size={size}>"


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit one structured-ish log line with the active request id."""
    parts = [f"request_id={_request_id.get()}"]
    parts.extend(f"{key}={compact(value)}" for key, value in fields.items())
    logger.info("%s | %s", event, " ".join(parts))


def state_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Return a useful state snapshot without duplicating passage text."""
    summary: dict[str, Any] = {}
    for key in (
        "query",
        "status",
        "round",
        "search_queries",
        "tried_queries",
        "rejected_chunk_ids",
        "triage",
        "verdict",
        "answer",
        "response",
        "error",
    ):
        value = state.get(key)
        if value not in (None, "", [], {}):
            summary[key] = payload(value) if key in {"query", "answer", "response"} else value

    if "evidence" in state:
        summary["evidence"] = []
        for item in state.get("evidence", []):
            summary["evidence"].append(
                {
                    "chunk_id": item.chunk_id,
                    "source": item.source,
                    "section": item.section,
                    "pages": [item.page_start, item.page_end],
                    "rerank_score": round(item.rerank_score, 4),
                    "round": item.round,
                }
            )
    return summary


def logged_node(logger: logging.Logger, name: str) -> Callable:
    """Decorate a graph node with state input/output and duration logging."""
    def decorate(func: Callable) -> Callable:
        @wraps(func)
        def wrapped(state: dict[str, Any]) -> dict:
            started = perf_counter()
            log_event(logger, "state.node.start", node=name, state=state_summary(state))
            try:
                update = func(state)
            except Exception:
                logger.exception("state.node.error | node=%s", name)
                raise
            log_event(
                logger,
                "state.node.end",
                node=name,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                update=state_summary(update),
            )
            return update
        return wrapped
    return decorate
