from __future__ import annotations

import json

from evaluation.compare_baselines import _same_benchmark, render_report
from evaluation.run_baselines import load_manifest, retrieval_env


def _retrieval_payload(model: str, reranker: str, hit: float) -> dict:
    return {
        "created_at": "2026-01-01T00:00:00+00:00",
        "config": {
            "case_count": 100,
            "dataset_schema_version": "2.0",
            "corpus_sha256": "same",
            "embedding_model": model,
            "reranker_model": reranker,
        },
        "summary": {
            "overall": {"hit@1": hit, "hit@10": hit, "mrr": hit, "ndcg@10": hit},
            "mean_latency_s": 0.1,
            "errors": 0,
        },
    }


def _e2e_payload(model: str) -> dict:
    return {
        "created_at": "2026-01-01T00:00:00+00:00",
        "config": {
            "case_count": 100,
            "dataset_schema_version": "2.0",
            "corpus_sha256": "same",
            "llm_model": model,
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": "BAAI/bge-reranker-v2-m3",
        },
        "summary": {
            "fully_answered_rate": 0.7,
            "grounded_success_rate": 0.6,
            "mean_rounds": 1.5,
            "errors": 0,
            "citation": {"hit_rate": 0.9, "precision_micro": 0.5},
            "latency": {"mean_s": 2.0},
        },
    }


def test_manifest_and_retrieval_environment_are_valid(tmp_path):
    manifest_path = tmp_path / "baselines.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "llm_baselines": [],
                "retrieval_baselines": [],
            }
        ),
        encoding="utf-8",
    )
    assert load_manifest(manifest_path)["schema_version"] == "1.0"

    env = retrieval_env(
        {
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_query_prompt_name": "query",
            "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
            "qdrant_path": "data/processed/test-index",
        }
    )
    assert env["EMBEDDING_QUERY_PROMPT_NAME"] == "query"
    from evaluation.run_baselines import CPU_THREAD_ENV
    assert set(CPU_THREAD_ENV.values()) == {"6"}
    assert env["QDRANT_PATH"].endswith("data/processed/test-index")


def test_report_marks_unmeasured_rows_without_inventing_scores():
    manifest = json.loads(
        (load_manifest.__globals__["DEFAULT_MANIFEST"]).read_text(encoding="utf-8")
    )
    retrieval = _retrieval_payload(
        "BAAI/bge-m3", "BAAI/bge-reranker-v2-m3", 0.98
    )
    e2e = _e2e_payload("QuixiAI/Qwen3-30B-A3B-AWQ")

    report = render_report(manifest, [retrieval], [e2e])

    assert "Qwen/Qwen3-8B | 8.2B | planned | —" in report
    assert "measured (n=100)" in report
    assert "98.00%" in report
    assert "69.00%" not in report


def test_delta_requires_same_schema_corpus_and_case_count():
    reference = {
        "config": {
            "case_count": 100,
            "dataset_schema_version": "2.0",
            "corpus_sha256": "abc",
        }
    }
    assert _same_benchmark(reference, reference)
    assert not _same_benchmark(
        {"config": {**reference["config"], "case_count": 10}}, reference
    )

