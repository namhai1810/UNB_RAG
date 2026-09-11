"""Multi-query reciprocal rank fusion."""
from __future__ import annotations

from src.agents.retrieval_agent import _fuse
from src.retrieval import Evidence


def _ev(chunk_id: str, score: float = 0.5) -> Evidence:
    return Evidence(chunk_id=chunk_id, source="s.pdf", text="t", rerank_score=score)


def test_chunk_found_by_every_query_ranks_first():
    lists = [
        [_ev("a"), _ev("b"), _ev("c")],
        [_ev("c"), _ev("a"), _ev("d")],
        [_ev("a"), _ev("e"), _ev("c")],
    ]
    assert _fuse(lists, top_k=3)[0].chunk_id == "a"


def test_results_are_deduplicated():
    lists = [[_ev("a"), _ev("b")], [_ev("a"), _ev("b")]]
    assert [e.chunk_id for e in _fuse(lists, top_k=10)] == ["a", "b"]


def test_best_rerank_score_is_kept_for_a_duplicate():
    lists = [[_ev("a", 0.2)], [_ev("a", 0.9)]]
    assert _fuse(lists, top_k=1)[0].rerank_score == 0.9


def test_top_k_is_honoured():
    lists = [[_ev(c) for c in "abcdefgh"]]
    assert len(_fuse(lists, top_k=3)) == 3


def test_single_list_preserves_its_order():
    lists = [[_ev("a"), _ev("b"), _ev("c")]]
    assert [e.chunk_id for e in _fuse(lists, top_k=3)] == ["a", "b", "c"]


def test_empty_input_is_safe():
    assert _fuse([], top_k=5) == []
