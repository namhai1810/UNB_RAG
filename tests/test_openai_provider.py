"""Structured-output negotiation against an OpenAI-compatible server.

The three tiers exist because servers disagree: some reject an unknown parameter
with a 400, and some accept the request and quietly ignore it. Both have to end
up at a validated object.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.llm.base import LLMError
from src.llm.openai_provider import OpenAIProvider


class Answer(BaseModel):
    label: str
    score: int


def _response(text: str):
    class Message:
        content = text

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    return Response()


class FakeCompletions:
    """Records calls; behaviour is driven by `handler`."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.handler(kwargs)


@pytest.fixture
def provider(monkeypatch):
    def build(handler) -> OpenAIProvider:
        monkeypatch.setattr(OpenAIProvider, "__init__", lambda self: None)
        p = OpenAIProvider()
        p.model = "test-model"
        p.base_url_display = "http://fake/v1"
        p._mode = None
        completions = FakeCompletions(handler)

        class Client:
            chat = type("Chat", (), {"completions": completions})()

        p.client = Client()
        p.completions = completions
        return p

    return build


def _mode_of(call: dict) -> str:
    if "response_format" in call:
        return "json_schema"
    if "extra_body" in call:
        return "guided_json"
    return "prompt"


def test_json_schema_mode_used_when_the_server_supports_it(provider):
    p = provider(lambda kw: _response('{"label": "a", "score": 1}'))
    assert p.structured("s", "u", Answer) == Answer(label="a", score=1)
    assert _mode_of(p.completions.calls[0]) == "json_schema"
    assert len(p.completions.calls) == 1


def test_falls_back_when_the_server_rejects_response_format(provider):
    def handler(kw):
        if "response_format" in kw:
            raise ValueError("400 unknown parameter: response_format")
        return _response('{"label": "b", "score": 2}')

    p = provider(handler)
    assert p.structured("s", "u", Answer).label == "b"
    assert [_mode_of(c) for c in p.completions.calls] == ["json_schema", "guided_json"]


def test_falls_all_the_way_to_prompt_mode(provider):
    def handler(kw):
        if "response_format" in kw or "extra_body" in kw:
            raise ValueError("unsupported")
        return _response('{"label": "c", "score": 3}')

    p = provider(handler)
    assert p.structured("s", "u", Answer).label == "c"
    assert [_mode_of(c) for c in p.completions.calls] == [
        "json_schema", "guided_json", "prompt"
    ]
    # The schema has to reach the model somehow in prompt mode.
    assert "JSON Schema" in p.completions.calls[-1]["messages"][0]["content"]


def test_server_that_silently_ignores_the_parameter_still_succeeds(provider):
    """The failure this chain exists for: HTTP 200, prose body, no error."""

    def handler(kw):
        if _mode_of(kw) != "prompt":
            return _response("Sure! The label is 'd'.")   # ignored the constraint
        return _response('{"label": "d", "score": 4}')

    p = provider(handler)
    assert p.structured("s", "u", Answer).label == "d"
    assert _mode_of(p.completions.calls[-1]) == "prompt"


def test_working_mode_is_cached_across_calls(provider):
    p = provider(lambda kw: _response('{"label": "e", "score": 5}'))
    p.structured("s", "u", Answer)
    p.structured("s", "u", Answer)
    assert len(p.completions.calls) == 2      # no re-probing
    assert p._mode == "json_schema"


def test_proven_mode_retries_once_then_raises(provider):
    calls = {"n": 0}

    def handler(kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _response('{"label": "f", "score": 6}')
        raise ValueError("connection reset")

    p = provider(handler)
    p.structured("s", "u", Answer)            # establishes json_schema mode
    with pytest.raises(LLMError, match="twice"):
        p.structured("s", "u", Answer)
    assert calls["n"] == 3                    # 1 success + failure + 1 retry


def test_reasoning_wrapper_around_json_is_tolerated(provider):
    p = provider(
        lambda kw: _response('<think>hmm</think>\n```json\n{"label":"g","score":7}\n```')
    )
    assert p.structured("s", "u", Answer) == Answer(label="g", score=7)


def test_all_modes_failing_raises_with_the_endpoint(provider):
    def handler(kw):
        raise ValueError("nope")

    p = provider(handler)
    with pytest.raises(LLMError, match="http://fake/v1"):
        p.structured("s", "u", Answer)
