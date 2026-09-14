from __future__ import annotations

from test_case_generation.generate_retrieval import (
    GeneratedChunkCases,
    GeneratedQuestion,
    _build_cases,
    _multiplicity_scope,
    question_counts,
)


def _chunk(chunk_id: str, doc_id: str = "doc", source: str = "guide.pdf") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source": source,
        "section": "Incident response",
        "page_start": 1,
        "page_end": 1,
        "chunk_type": "paragraph",
        "text": "Organizations preserve incident logs before recovery begins.",
    }


def _generated(chunk_id: str) -> GeneratedChunkCases:
    return GeneratedChunkCases(
        chunk_id=chunk_id,
        reference_answer="Organizations preserve incident logs before recovery.",
        evidence_spans=["Organizations preserve incident logs before recovery begins."],
        questions=[
            GeneratedQuestion(paraphrase_type=style, query=f"How are logs handled ({style})?")
            for style in ("lexical", "semantic", "natural")
        ],
    )


def test_question_counts_make_exact_groups_of_two_or_three():
    assert question_counts(50) == [3] * 16 + [2]
    assert sum(question_counts(50)) == 50


def test_multiplicity_scope_distinguishes_same_cross_and_mixed():
    primary = _chunk("a-00001", "a")
    by_id = {
        "a-00002": _chunk("a-00002", "a"),
        "b-00001": _chunk("b-00001", "b"),
    }
    assert _multiplicity_scope(primary, [], by_id) == "unique"
    assert _multiplicity_scope(primary, [{"chunk_id": "a-00002"}], by_id) == "same_document"
    assert _multiplicity_scope(primary, [{"chunk_id": "b-00001"}], by_id) == "cross_document"
    assert _multiplicity_scope(
        primary, [{"chunk_id": "a-00002"}, {"chunk_id": "b-00001"}], by_id
    ) == "mixed"


def test_build_cases_keeps_full_and_partial_support_separate():
    primary = _chunk("a-00001", "a")
    full = _chunk("b-00001", "b")
    partial = _chunk("c-00001", "c")
    entry = {
        "primary_chunk": primary,
        "generated": _generated(primary["chunk_id"]),
        "duplication_scope": "cross_document",
        "audit": {
            "full": [{
                "chunk_id": full["chunk_id"],
                "match_kind": "semantic_equivalent",
                "evidence_spans": [full["text"]],
            }],
            "partial": [{
                "chunk_id": partial["chunk_id"],
                "match_kind": "semantic_equivalent",
                "evidence_spans": [partial["text"]],
            }],
        },
    }
    cases = _build_cases(
        {"single_gold": [], "multi_gold": [entry]},
        {"single_gold": 0, "multi_gold": 2},
        {chunk["chunk_id"]: chunk for chunk in (primary, full, partial)},
    )
    assert len(cases) == 2
    assert cases[0]["gold_multiplicity"] == "multi_gold"
    assert cases[0]["acceptable_gold_chunk_ids"] == ["a-00001", "b-00001"]
    assert cases[0]["partial_support_chunk_ids"] == ["c-00001"]
