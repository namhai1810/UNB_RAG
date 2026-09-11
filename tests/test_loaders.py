"""Header/footer detection and text cleaning are pure functions - test them directly."""
from __future__ import annotations

from src.ingestion.loaders import _chrome_lines, _clean, _strip_chrome


def test_repeated_header_is_detected_as_chrome():
    pages = [f"NIST SP 800-61r3\nBody text for page {i}\n{i}" for i in range(1, 11)]
    chrome = _chrome_lines(pages)
    assert "NIST SP 800-61r3" in chrome
    assert "Body text for page 1" not in chrome


def test_unique_lines_are_never_chrome():
    pages = [f"Unique heading {i}\nsome body\nmore body" for i in range(6)]
    chrome = _chrome_lines(pages)
    assert not any(line.startswith("Unique heading") for line in chrome)


def test_strip_chrome_removes_headers_and_bare_page_numbers():
    text = "NIST SP 800-61r3\nReal content here\n42"
    assert _strip_chrome(text, {"NIST SP 800-61r3"}).strip() == "Real content here"


def test_strip_chrome_keeps_numbers_that_are_content():
    text = "The framework defines 6 functions.\n7"
    out = _strip_chrome(text, set())
    assert "6 functions" in out
    assert out.strip().endswith("functions.")


def test_clean_rejoins_hyphenated_line_breaks():
    assert "cybersecurity" in _clean("cyber-\nsecurity posture")


def test_clean_collapses_whitespace_and_ligatures():
    assert _clean("identiﬁed    risk\n\n\n\nnext") == "identified risk\n\nnext"
