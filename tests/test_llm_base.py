"""JSON extraction has to survive whatever a local model emits around the object."""
from __future__ import annotations

import pytest

from src.llm.base import LLMError, extract_json, strip_reasoning


def test_bare_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_prose_around_object():
    assert extract_json('Sure, here it is:\n{"a": 1}\nHope that helps!') == {"a": 1}


def test_reasoning_block_is_stripped():
    raw = '<think>The user wants {"trap": true}</think>\n{"a": 1}'
    assert extract_json(raw) == {"a": 1}


def test_braces_inside_strings_do_not_confuse_the_matcher():
    assert extract_json('{"a": "a } brace", "b": 2}') == {"a": "a } brace", "b": 2}


def test_escaped_quote_inside_string():
    assert extract_json(r'{"a": "say \"hi\" }", "b": 1}') == {"a": 'say "hi" }', "b": 1}


def test_nested_object():
    assert extract_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}


def test_no_object_raises():
    with pytest.raises(LLMError):
        extract_json("I refuse to answer.")


def test_unterminated_object_raises():
    with pytest.raises(LLMError):
        extract_json('{"a": 1')


def test_strip_reasoning_leaves_plain_text():
    assert strip_reasoning("plain answer") == "plain answer"
