from __future__ import annotations

import math

from evaluation.evaluate_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank, score_ranking


def test_retrieval_metrics_with_multiple_relevant_chunks():
    ranked = ["noise", "primary", "alternative"]
    relevant = {"primary", "alternative"}
    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert reciprocal_rank(ranked, relevant) == 0.5


def test_ndcg_uses_graded_primary_relevance():
    relevance = {"primary": 2, "alternative": 1}
    ideal = ndcg_at_k(["primary", "alternative"], relevance, 2)
    swapped = ndcg_at_k(["alternative", "primary"], relevance, 2)
    assert ideal == 1.0
    assert 0.0 < swapped < ideal


def test_score_ranking_exposes_requested_metrics():
    case = {
        "primary_gold_chunk_id": "primary",
        "acceptable_gold_chunk_ids": ["primary", "alternative"],
    }
    scores = score_ranking(case, ["primary", "noise"], [1, 2])
    assert scores["recall@1"] == 0.5
    assert scores["mrr"] == 1.0
    assert math.isclose(scores["ndcg@1"], 1.0)
