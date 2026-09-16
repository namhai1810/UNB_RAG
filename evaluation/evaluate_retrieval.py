"""Evaluate schema-2 single- and multi-gold retrieval cases."""
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

HERE = Path(__file__).resolve().parent
DEFAULT_DETAILS = HERE / "retrieval_results.json"
DEFAULT_REPORT = HERE / "retrieval_evaluation.md"


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return float(bool(set(ranked_ids[:k]) & relevant_ids))


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(ranked_ids, 1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    def gain(grade: int, rank: int) -> float:
        return (2**grade - 1) / math.log2(rank + 1)

    dcg = sum(
        gain(relevance.get(chunk_id, 0), rank)
        for rank, chunk_id in enumerate(ranked_ids[:k], 1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(gain(grade, rank) for rank, grade in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def score_ranking(case: dict, ranked_ids: list[str], ks: list[int]) -> dict[str, float]:
    full = set(case["acceptable_gold_chunk_ids"])
    partial = set(case.get("partial_support_chunk_ids", [])) - full
    relevance = {chunk_id: 2 for chunk_id in full}
    relevance.update({chunk_id: 1 for chunk_id in partial})
    scores = {f"hit@{k}": hit_at_k(ranked_ids, full, k) for k in ks}
    scores.update({f"recall@{k}": recall_at_k(ranked_ids, full, k) for k in ks})
    scores.update({f"ndcg@{k}": ndcg_at_k(ranked_ids, relevance, k) for k in ks})
    scores["mrr"] = reciprocal_rank(ranked_ids, full)
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
    return {
        "id": case["id"],
        "group_id": case["group_id"],
        "query": case["query"],
        "gold_multiplicity": case["gold_multiplicity"],
        "duplication_scope": case["duplication_scope"],
        "paraphrase_type": case["paraphrase_type"],
        "source": case["gold"]["source"],
        "chunk_type": case["gold"].get("chunk_type", "paragraph"),
        "primary_gold_chunk_id": case["primary_gold_chunk_id"],
        "acceptable_gold_chunk_ids": case["acceptable_gold_chunk_ids"],
        "partial_support_chunk_ids": case.get("partial_support_chunk_ids", []),
        "retrieved_chunk_ids": ranked_ids,
        "retrieved_scores": [round(item.rerank_score, 6) for item in evidence],
        **score_ranking(case, ranked_ids, ks),
        "latency_s": round(time.perf_counter() - started, 3),
        "error": error,
    }


def aggregate(rows: list[dict], ks: list[int]) -> dict[str, float]:
    names = [
        *[f"hit@{k}" for k in ks],
        *[f"recall@{k}" for k in ks],
        "mrr",
        *[f"ndcg@{k}" for k in ks],
    ]
    return {name: mean(row[name] for row in rows) if rows else 0.0 for name in names}


def grouped_aggregates(rows: list[dict], field: str, ks: list[int]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        name: {"n": len(values), **aggregate(values, ks)}
        for name, values in sorted(groups.items())
    }


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
            "embedding_query_prompt_name": settings.embedding_query_prompt_name,
            "reranker_model": settings.reranker_model,
            "device": settings.device,
            "corpus_sha256": dataset["metadata"].get("corpus_sha256", "unknown"),
            "dataset_schema_version": dataset["metadata"].get("schema_version", "unknown"),
        },
        "summary": {
            "overall": aggregate(rows, ks),
            "by_gold_multiplicity": grouped_aggregates(rows, "gold_multiplicity", ks),
            "by_duplication_scope": grouped_aggregates(rows, "duplication_scope", ks),
            "by_paraphrase_type": grouped_aggregates(rows, "paraphrase_type", ks),
            "by_source": grouped_aggregates(rows, "source", ks),
            "by_chunk_type": grouped_aggregates(rows, "chunk_type", ks),
            "mean_latency_s": mean(row["latency_s"] for row in rows) if rows else 0.0,
            "errors": sum(bool(row["error"]) for row in rows),
        },
        "results": rows,
    }


def _metric_table(summary: dict, ks: list[int]) -> str:
    lines = ["| Metric | Score |", "|---|---:|"]
    for k in ks:
        lines.append(f"| Any-Gold Hit@{k} | {summary[f'hit@{k}']:.4f} |")
    for k in ks:
        lines.append(f"| Gold Coverage/Recall@{k} | {summary[f'recall@{k}']:.4f} |")
    lines.append(f"| MRR | {summary['mrr']:.4f} |")
    for k in ks:
        lines.append(f"| nDCG@{k} | {summary[f'ndcg@{k}']:.4f} |")
    return "\n".join(lines)


def _breakdown_table(groups: dict[str, dict], ks: list[int]) -> str:
    k = max(ks)
    lines = [
        f"| Group | n | Any-Gold Hit@{k} | Gold Coverage@{k} | MRR | nDCG@{k} |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in groups.items():
        lines.append(
            f"| {name} | {values['n']} | {values[f'hit@{k}']:.4f} | "
            f"{values[f'recall@{k}']:.4f} | {values['mrr']:.4f} | "
            f"{values[f'ndcg@{k}']:.4f} |"
        )
    return "\n".join(lines)


def _weakest_table(rows: list[dict], ks: list[int], multiplicity: str) -> str:
    k = max(ks)
    selected = [row for row in rows if row["gold_multiplicity"] == multiplicity]
    weakest = sorted(selected, key=lambda row: (row[f"hit@{k}"], row["mrr"], row[f"ndcg@{k}"]))[:5]
    lines = [
        "| ID | Scope | Gold count | Retrieved top 3 | Query |",
        "|---|---|---:|---|---|",
    ]
    for row in weakest:
        query = row["query"].replace("|", "\\|")
        retrieved = ", ".join(row["retrieved_chunk_ids"][:3]) or "—"
        lines.append(
            f"| {row['id']} | {row['duplication_scope']} | "
            f"{len(row['acceptable_gold_chunk_ids'])} | {retrieved} | {query} |"
        )
    return "\n".join(lines)


def render_markdown(payload: dict) -> str:
    config = payload["config"]
    summary = payload["summary"]
    ks = config["k_values"]
    single = summary["by_gold_multiplicity"]["single_gold"]
    multi = summary["by_gold_multiplicity"]["multi_gold"]
    return f"""# Retrieval and end-to-end evaluation report

Generated at: `{payload['created_at']}`

## Evaluation setup

- Test cases: **{config['case_count']}** from **{config['group_count']}** atomic facts
- Dataset schema: **{config['dataset_schema_version']}**
- Retrieval: hybrid `{config['embedding_model']}` + BM25/RRF, followed by `{config['reranker_model']}`
- Embedding query prompt: `{config.get('embedding_query_prompt_name') or 'none'}`
- Evaluated list: reranked seed chunks only; neighbor expansion is excluded
- K values: `{', '.join(map(str, ks))}`
- Device: `{config['device']}`
- Corpus SHA-256: `{config['corpus_sha256']}`

Every full-support gold chunk has relevance grade 2. Independently validated
partial-support chunks have grade 1. The primary gold records provenance only
and receives no ranking preference over another full gold.

`Any-Gold Hit@K` asks whether at least one answer-bearing chunk was found.
`Gold Coverage@K` asks what fraction of all repeated full-gold occurrences was
found. For single-gold cases these two metrics are identical.

## Overall retrieval metrics

{_metric_table(summary['overall'], ks)}

Mean retrieval latency per query: **{summary['mean_latency_s']:.3f}s**. Errors: **{summary['errors']}**.

## Single-gold retrieval

Cases: **{single['n']}**.

{_metric_table(single, ks)}

## Multi-gold retrieval

Cases: **{multi['n']}**.

{_metric_table(multi, ks)}

## Retrieval by duplication scope

{_breakdown_table(summary['by_duplication_scope'], ks)}

## Retrieval by paraphrase type

{_breakdown_table(summary['by_paraphrase_type'], ks)}

## Retrieval by source

{_breakdown_table(summary['by_source'], ks)}

## Retrieval by primary chunk type

{_breakdown_table(summary['by_chunk_type'], ks)}

## Weakest single-gold cases

{_weakest_table(payload['results'], ks, 'single_gold')}

## Weakest multi-gold cases

{_weakest_table(payload['results'], ks, 'multi_gold')}

## Retrieval interpretation

- Retrieval metrics evaluate reranked seed chunks, not generated answer text.
- Multi-gold Any-Gold Hit measures answerability; Gold Coverage separately
  measures whether the retriever found all repeated supporting occurrences.
- Scores from schema 2.0 should not be compared directly with the old schema 1.0
  baseline because the gold labels and relevance grades changed.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate schema-2 retrieval metrics")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--limit", type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if any(k < 1 or k > settings.top_k_fused for k in args.k):
        raise SystemExit(f"Every K must be between 1 and {settings.top_k_fused}")
    ks = sorted(set(args.k))
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    if dataset.get("metadata", {}).get("schema_version") != "2.0":
        raise SystemExit("Schema 2.0 dataset required; regenerate the grounded test set")
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
    args.details.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render_markdown(payload)
    end_to_end_details = HERE / "end_to_end_results.json"
    if end_to_end_details.exists():
        from evaluation.evaluate_end_to_end import render_markdown_section, upsert_report_section

        end_to_end_payload = json.loads(end_to_end_details.read_text(encoding="utf-8"))
        if (
            end_to_end_payload.get("config", {}).get("dataset_schema_version") == "2.0"
            and end_to_end_payload.get("config", {}).get("corpus_sha256")
            == payload["config"].get("corpus_sha256")
            and {row["id"] for row in end_to_end_payload.get("results", [])}
            == {row["id"] for row in payload["results"]}
        ):
            report = upsert_report_section(report, render_markdown_section(end_to_end_payload))
    args.report.write_text(report, encoding="utf-8")
    print(_metric_table(payload["summary"]["overall"], ks))
    print(f"Wrote details to {args.details}")
    print(f"Wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
