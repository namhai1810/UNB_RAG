from __future__ import annotations

from types import SimpleNamespace

from evaluation.evaluate_end_to_end import (
    SECTION_END,
    SECTION_START,
    _percentile,
    score_citations,
    summarise,
    upsert_report_section,
)


def _case():
    return {
        "primary_gold_chunk_id": "primary",
        "acceptable_gold_chunk_ids": ["primary", "overlap"],
    }


def test_score_citations_uses_acceptable_gold_chunk_ids():
    citations = [
        SimpleNamespace(chunk_id="overlap"),
        SimpleNamespace(chunk_id="noise"),
    ]
    scores = score_citations(_case(), citations)
    assert scores["correct_citation_count"] == 1
    assert scores["citation_precision"] == 0.5
    assert scores["citation_hit"] == 1
    assert scores["primary_citation_hit"] == 0


def test_score_citations_marks_an_uncited_answer_as_no_hit():
    scores = score_citations(_case(), [])
    assert scores["citation_precision"] is None
    assert scores["citation_hit"] == 0


def test_summary_reports_micro_precision_and_grounded_success():
    common = {
        "actual_category": "in_scope",
        "answer_produced": 1,
        "citation_hit": 1,
        "citation_integrity": 1,
        "error": "",
        "source": "guide.pdf",
    }
    rows = [
        {
            **common,
            "status": "answered",
            "fully_answered": 1,
            "grounded_success": 1,
            "rounds": 1,
            "citation_count": 2,
            "correct_citation_count": 1,
            "citation_precision": 0.5,
            "primary_citation_hit": 0,
            "latency_s": 2.0,
            "paraphrase_type": "lexical",
            "chunk_type": "paragraph",
        },
        {
            **common,
            "status": "answered_partial",
            "fully_answered": 0,
            "grounded_success": 0,
            "rounds": 2,
            "citation_count": 1,
            "correct_citation_count": 1,
            "citation_precision": 1.0,
            "primary_citation_hit": 1,
            "latency_s": 4.0,
            "paraphrase_type": "natural",
            "chunk_type": "table",
        },
    ]
    summary = summarise(rows)
    assert summary["grounded_success_rate"] == 0.5
    assert summary["citation"]["precision_micro"] == 2 / 3
    assert summary["latency"]["median_s"] == 3.0
    assert summary["latency"]["p95_s"] == 4.0


def test_nearest_rank_percentile():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_report_section_is_inserted_and_replaced_idempotently():
    report = "# Report\n\n## Interpretation notes\n\nNotes.\n"
    first = upsert_report_section(report, f"{SECTION_START}\nfirst\n{SECTION_END}")
    second = upsert_report_section(first, f"{SECTION_START}\nsecond\n{SECTION_END}")
    assert first.index(SECTION_START) < first.index("## Interpretation notes")
    assert second.count(SECTION_START) == 1
    assert "second" in second
    assert "first" not in second
