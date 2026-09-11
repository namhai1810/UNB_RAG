"""Citation integrity: a marker the reader cannot follow must never survive."""
from __future__ import annotations

from src.agents import answer_agent
from src.agents.answer_agent import GeneratedAnswer, generate


def test_citations_resolve_to_the_right_passages(fake_llm, evidence_factory, answer_ok):
    fake_llm.queue(answer_ok)
    payload = generate("q", evidence_factory(3))
    assert [c.marker for c in payload.citations] == [1, 2]
    assert payload.citations[0].pages == "p.11"
    assert payload.citations[0].source == "NIST.SP.800-61r3.pdf"
    assert payload.confidence == "high"


def test_out_of_range_marker_is_stripped_from_the_text(fake_llm, evidence_factory):
    fake_llm.queue(
        GeneratedAnswer(
            answer="Containment limits damage [1], and recovery follows [9].",
            used_indices=[1, 9],
            confidence="high",
            caveats="",
        )
    )
    payload = generate("q", evidence_factory(2))
    assert "[9]" not in payload.answer
    assert "[1]" in payload.answer
    assert payload.dropped_markers == [9]
    assert [c.marker for c in payload.citations] == [1]


def test_inline_markers_win_over_the_models_self_report(fake_llm, evidence_factory):
    fake_llm.queue(
        GeneratedAnswer(
            answer="Only the first passage is actually cited [1].",
            used_indices=[1, 2, 3],   # model over-reports
            confidence="high",
            caveats="",
        )
    )
    payload = generate("q", evidence_factory(3))
    assert [c.marker for c in payload.citations] == [1]


def test_uncited_answer_is_downgraded(fake_llm, evidence_factory):
    fake_llm.queue(
        GeneratedAnswer(
            answer="Containment limits damage.", used_indices=[], confidence="high",
            caveats="",
        )
    )
    payload = generate("q", evidence_factory(2))
    assert payload.confidence == "low"
    assert payload.citations == []
    assert "could not be tied" in payload.caveats


def test_no_evidence_short_circuits_without_calling_the_model(fake_llm):
    payload = generate("q", [])
    assert payload.citations == []
    assert payload.confidence == "low"
    assert fake_llm.calls == []


def test_caveat_prefix_is_prepended(fake_llm, evidence_factory, answer_ok):
    fake_llm.queue(answer_ok)
    payload = generate("q", evidence_factory(2), caveat_prefix="Budget exhausted.")
    assert payload.caveats.startswith("Budget exhausted.")


def test_page_range_citation_format(fake_llm, evidence_factory, answer_ok):
    fake_llm.queue(answer_ok)
    evidence = evidence_factory(2)
    evidence[0].page_end = 12
    payload = generate("q", evidence)
    assert payload.citations[0].pages == "pp.11-12"
