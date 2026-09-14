from __future__ import annotations

import math

from evaluation.evaluate_retrieval import (
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_ranking,
)


def test_multi_gold_exposes_any_hit_and_coverage_separately():
    ranked = ["noise", "gold-a", "noise-2"]
    relevant = {"gold-a", "gold-b"}
    assert hit_at_k(ranked, relevant, 2) == 1.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant) == 0.5


def test_all_full_gold_chunks_have_equal_ndcg_relevance():
    relevance = {"primary": 2, "alternative": 2}
    assert ndcg_at_k(["primary", "alternative"], relevance, 2) == 1.0
    assert ndcg_at_k(["alternative", "primary"], relevance, 2) == 1.0


def test_partial_support_contributes_only_to_ndcg():
    case = {
        "acceptable_gold_chunk_ids": ["primary", "alternative"],
        "partial_support_chunk_ids": ["partial"],
    }
    scores = score_ranking(case, ["partial", "primary"], [1, 2])
    assert scores["hit@1"] == 0.0
    assert scores["recall@1"] == 0.0
    assert scores["mrr"] == 0.5
    assert 0.0 < scores["ndcg@1"] < 1.0
    assert math.isclose(scores["hit@2"], 1.0)
