"""Evaluate chunk retrieval with Recall@K, MRR, and nDCG@K.

This evaluates reranked seed chunks only. Neighbor expansion is intentionally
excluded so a nearby semantic miss cannot be counted as a direct retrieval hit.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from src.config import settings
from src.retrieval import HybridRetriever
from test_case_generation.generate_retrieval import DEFAULT_OUTPUT

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DETAILS = Path(__file__).resolve().parent / "retrieval_results.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "retrieval_evaluation.md"


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(ranked_ids, 1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    def gain(grade: int, rank: int) -> float:
        return (2**grade - 1) / math.log2(rank + 1)

    dcg = sum(gain(relevance.get(chunk_id, 0), rank) for rank, chunk_id in enumerate(ranked_ids[:k], 1))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(gain(grade, rank) for rank, grade in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def score_ranking(case: dict, ranked_ids: list[str], ks: list[int]) -> dict[str, float]:
    relevant = set(case["acceptable_gold_chunk_ids"])
    primary = case["primary_gold_chunk_id"]
    relevance = {chunk_id: (2 if chunk_id == primary else 1) for chunk_id in relevant}
    scores = {f"recall@{k}": recall_at_k(ranked_ids, relevant, k) for k in ks}
    scores.update({f"ndcg@{k}": ndcg_at_k(ranked_ids, relevance, k) for k in ks})
    scores["mrr"] = reciprocal_rank(ranked_ids, relevant)
    return scores


def evaluate_case(case: dict, retriever: HybridRetriever, ks: list[int]) -> dict:
    started = time.perf_counter()
    error = ""
    evidence = []
    try:
        result = retriever.retrieve(case["query"], top_k=max(ks))
        evidence = result.evidence
    except Exception as exc:
        logging.exception("Retrieval failed for %s", case["id"])
        error = str(exc)

    ranked_ids = [item.chunk_id for item in evidence]
    scores = score_ranking(case, ranked_ids, ks)
    return {
        "id": case["id"],
        "group_id": case["group_id"],
        "query": case["query"],
        "paraphrase_type": case["paraphrase_type"],
        "source": case["gold"]["source"],
        "section": case["gold"].get("section", ""),
        "chunk_type": case["gold"].get("chunk_type", "paragraph"),
        "primary_gold_chunk_id": case["primary_gold_chunk_id"],
        "acceptable_gold_chunk_ids": case["acceptable_gold_chunk_ids"],
        "retrieved_chunk_ids": ranked_ids,
        "retrieved_scores": [round(item.rerank_score, 6) for item in evidence],
        **scores,
        "latency_s": round(time.perf_counter() - started, 3),
        "error": error,
    }


def aggregate(rows: list[dict], ks: list[int]) -> dict[str, float]:
    names = [*[f"recall@{k}" for k in ks], "mrr", *[f"ndcg@{k}" for k in ks]]
    return {name: mean(row[name] for row in rows) if rows else 0.0 for name in names}


def grouped_aggregates(rows: list[dict], field: str, ks: list[int]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        key: {"n": len(values), **aggregate(values, ks)}
        for key, values in sorted(groups.items())
    }


def _metric_table(summary: dict, ks: list[int]) -> str:
    lines = ["| Metric | Score |", "|---|---:|"]
    for k in ks:
        lines.append(f"| Recall@{k} | {summary[f'recall@{k}']:.4f} |")
    lines.append(f"| MRR | {summary['mrr']:.4f} |")
    for k in ks:
        lines.append(f"| nDCG@{k} | {summary[f'ndcg@{k}']:.4f} |")
    return "\n".join(lines)


def _breakdown_table(groups: dict[str, dict], ks: list[int]) -> str:
    display_k = max(ks)
    lines = [
        f"| Group | n | Recall@{display_k} | MRR | nDCG@{display_k} |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in groups.items():
        lines.append(
            f"| {name} | {values['n']} | {values[f'recall@{display_k}']:.4f} | "
            f"{values['mrr']:.4f} | {values[f'ndcg@{display_k}']:.4f} |"
        )
    return "\n".join(lines)


def render_markdown(payload: dict) -> str:
    config = payload["config"]
    ks = config["k_values"]
    summary = payload["summary"]
    rows = payload["results"]
    failures = sorted(rows, key=lambda row: (row["mrr"], row[f"ndcg@{max(ks)}"]))[:10]
    failure_lines = [
        "| ID | Style | Gold chunk | Retrieved top 3 | Query |",
        "|---|---|---|---|---|",
    ]
    for row in failures:
        query = row["query"].replace("|", "\\|")
        retrieved = ", ".join(row["retrieved_chunk_ids"][:3]) or "—"
        failure_lines.append(
            f"| {row['id']} | {row['paraphrase_type']} | "
            f"{row['primary_gold_chunk_id']} | {retrieved} | {query} |"
        )

    return f"""# Retrieval evaluation report

Generated at: `{payload['created_at']}`

## Evaluation setup

- Test cases: **{config['case_count']}** ({config['group_count']} source chunks)
- Retrieval: hybrid BGE-M3 + BM25/RRF, followed by `{config['reranker_model']}`
- Evaluated list: reranked seed chunks only; neighbor expansion is excluded
- K values: `{', '.join(map(str, ks))}`
- Device: `{config['device']}`
- Corpus SHA-256: `{config['corpus_sha256']}`
- Dataset schema: `{config['dataset_schema_version']}`

`primary_gold_chunk_id` has relevance grade 2 and overlap/alternative gold chunks
have grade 1 for nDCG. Recall uses all acceptable gold chunk IDs. MRR uses the
rank of the first acceptable chunk. Failed retrievals receive zero.

## Overall metrics

{_metric_table(summary['overall'], ks)}

Mean latency per query: **{summary['mean_latency_s']:.3f}s**. Errors: **{summary['errors']}**.

## By paraphrase type

{_breakdown_table(summary['by_paraphrase_type'], ks)}

## By source

{_breakdown_table(summary['by_source'], ks)}

## By chunk type

{_breakdown_table(summary['by_chunk_type'], ks)}

## Ten weakest cases

{chr(10).join(failure_lines)}

## Interpretation notes

- Recall@K measures the fraction of acceptable gold chunks present in the first K results.
- MRR rewards putting the first acceptable supporting chunk near the top.
- nDCG@K additionally rewards ranking the primary source chunk above overlap alternatives.
- These metrics evaluate retrieval only; they do not grade the generated answer text.
"""


def evaluate_dataset(dataset: dict, retriever: HybridRetriever, ks: list[int]) -> dict:
    cases = dataset["cases"]
    rows = [evaluate_case(case, retriever, ks) for case in cases]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "case_count": len(cases),
            "group_count": len({case["group_id"] for case in cases}),
            "k_values": ks,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "device": settings.device,
            "corpus_sha256": dataset["metadata"].get("corpus_sha256", "unknown"),
            "dataset_schema_version": dataset["metadata"].get("schema_version", "unknown"),
        },
        "summary": {
            "overall": aggregate(rows, ks),
            "by_paraphrase_type": grouped_aggregates(rows, "paraphrase_type", ks),
            "by_source": grouped_aggregates(rows, "source", ks),
            "by_chunk_type": grouped_aggregates(rows, "chunk_type", ks),
            "mean_latency_s": mean(row["latency_s"] for row in rows) if rows else 0.0,
            "errors": sum(bool(row["error"]) for row in rows),
        },
        "results": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate chunk retrieval metrics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--limit", type=int, help="evaluate only the first N cases")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if any(k < 1 or k > settings.top_k_fused for k in args.k):
        raise SystemExit(f"Every K must be between 1 and {settings.top_k_fused}")
    ks = sorted(set(args.k))
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    if args.limit:
        dataset = {**dataset, "cases": dataset["cases"][: args.limit]}
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    retriever = HybridRetriever()
    try:
        payload = evaluate_dataset(dataset, retriever, ks)
    finally:
        retriever.close()

    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.details.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(render_markdown(payload), encoding="utf-8")
    print(_metric_table(payload["summary"]["overall"], ks))
    print(f"Wrote details to {args.details}")
    print(f"Wrote Markdown report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
