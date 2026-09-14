from __future__ import annotations

from types import SimpleNamespace

from evaluation.evaluate_end_to_end import _percentile, score_citations, summarise


def _case():
    return {
        "primary_gold_chunk_id": "primary",
        "acceptable_gold_chunk_ids": ["primary", "alternative"],
    }


def test_any_full_gold_citation_is_correct():
    scores = score_citations(
        _case(),
        [SimpleNamespace(chunk_id="alternative"), SimpleNamespace(chunk_id="noise")],
    )
    assert scores["correct_citation_count"] == 1
    assert scores["citation_precision"] == 0.5
    assert scores["citation_hit"] == 1
    assert scores["primary_citation_hit"] == 0


def test_uncited_answer_has_no_hit_and_undefined_precision():
    scores = score_citations(_case(), [])
    assert scores["citation_precision"] is None
    assert scores["citation_hit"] == 0


def test_summary_splits_single_and_multi_gold():
    common = {
        "status": "answered",
        "actual_category": "in_scope",
        "answer_produced": 1,
        "fully_answered": 1,
        "grounded_success": 1,
        "rounds": 1,
        "citation_count": 1,
        "correct_citation_count": 1,
        "citation_precision": 1.0,
        "citation_hit": 1,
        "primary_citation_hit": 1,
        "citation_integrity": 1,
        "error": "",
        "paraphrase_type": "semantic",
        "source": "guide.pdf",
        "chunk_type": "paragraph",
    }
    rows = [
        {**common, "gold_multiplicity": "single_gold", "duplication_scope": "unique", "latency_s": 2.0},
        {**common, "gold_multiplicity": "multi_gold", "duplication_scope": "cross_document", "latency_s": 4.0},
    ]
    summary = summarise(rows)
    assert summary["by_gold_multiplicity"]["single_gold"]["n"] == 1
    assert summary["by_gold_multiplicity"]["multi_gold"]["n"] == 1
    assert summary["by_retrieval_rounds"]["1"]["n"] == 2
    assert summary["latency"]["median_s"] == 3.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
