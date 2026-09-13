"""Evaluate the complete RAG graph and final gold-chunk citations.

The retrieval-only benchmark answers whether gold chunks can be found. This
module continues through triage, multi-query retrieval, neighbor expansion,
verification/rewrite, and answer generation. Citation correctness is
deterministic: a citation is correct when its chunk ID belongs to the case's
``acceptable_gold_chunk_ids``. This does not prove claim-level entailment.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from src.config import settings
from src.graph.workflow import run
from test_case_generation.generate_retrieval import DEFAULT_OUTPUT

HERE = Path(__file__).resolve().parent
DEFAULT_DETAILS = HERE / "end_to_end_results.json"
DEFAULT_REPORT = HERE / "retrieval_evaluation.md"

SECTION_START = "<!-- END_TO_END_EVALUATION_START -->"
SECTION_END = "<!-- END_TO_END_EVALUATION_END -->"
ANSWER_STATUSES = {"answered", "answered_partial"}


def _percentile(values: list[float], percentile: float) -> float:
    """Return the nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def score_citations(case: dict, citations: list) -> dict:
    acceptable = set(case["acceptable_gold_chunk_ids"])
    primary = case["primary_gold_chunk_id"]
    cited_ids = [citation.chunk_id for citation in citations]
    correct_ids = [chunk_id for chunk_id in cited_ids if chunk_id in acceptable]

    return {
        "citation_count": len(cited_ids),
        "correct_citation_count": len(correct_ids),
        "citation_precision": len(correct_ids) / len(cited_ids) if cited_ids else None,
        "citation_hit": int(bool(correct_ids)),
        "primary_citation_hit": int(primary in cited_ids),
        "cited_chunk_ids": cited_ids,
        "correct_cited_chunk_ids": correct_ids,
    }


def evaluate_case(case: dict) -> dict:
    started = time.perf_counter()
    try:
        state = run(case["query"])
        error = ""
    except Exception as exc:
        logging.exception("End-to-end evaluation failed for %s", case["id"])
        state = {"status": "error", "response": ""}
        error = str(exc)

    elapsed = time.perf_counter() - started
    answer = state.get("answer")
    citations = answer.citations if answer else []
    citation_scores = score_citations(case, citations)
    status = state.get("status", "error")
    triage = state.get("triage")

    return {
        "id": case["id"],
        "group_id": case["group_id"],
        "query": case["query"],
        "paraphrase_type": case["paraphrase_type"],
        "source": case["gold"]["source"],
        "chunk_type": case["gold"].get("chunk_type", "paragraph"),
        "reference_answer": case["reference_answer"],
        "primary_gold_chunk_id": case["primary_gold_chunk_id"],
        "acceptable_gold_chunk_ids": case["acceptable_gold_chunk_ids"],
        "actual_category": triage.category if triage else "",
        "status": status,
        "rounds": state.get("round", 0),
        "fully_answered": int(status == "answered"),
        "answer_produced": int(status in ANSWER_STATUSES),
        "grounded_success": int(status == "answered" and bool(citation_scores["citation_hit"])),
        **citation_scores,
        "citation_integrity": int(not answer.dropped_markers) if answer is not None else None,
        "dropped_markers": answer.dropped_markers if answer else [],
        "confidence": answer.confidence if answer else "",
        "latency_s": round(elapsed, 3),
        "error": error,
        "response": state.get("response", ""),
    }


def _group_summary(rows: list[dict]) -> dict:
    citations = sum(row["citation_count"] for row in rows)
    correct = sum(row["correct_citation_count"] for row in rows)
    return {
        "n": len(rows),
        "fully_answered_rate": mean(row["fully_answered"] for row in rows),
        "grounded_success_rate": mean(row["grounded_success"] for row in rows),
        "citation_hit_rate": mean(row["citation_hit"] for row in rows),
        "citation_precision_micro": correct / citations if citations else 0.0,
        "mean_latency_s": mean(row["latency_s"] for row in rows),
    }


def _grouped(rows: list[dict], field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: _group_summary(values) for name, values in sorted(groups.items())}


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}

    answered = [row for row in rows if row["answer_produced"]]
    cited = [row for row in rows if row["citation_count"]]
    latencies = [row["latency_s"] for row in rows]
    citation_count = sum(row["citation_count"] for row in rows)
    correct_count = sum(row["correct_citation_count"] for row in rows)
    integrity = [
        row["citation_integrity"]
        for row in rows
        if row["citation_integrity"] is not None
    ]

    return {
        "case_count": len(rows),
        "statuses": dict(sorted(Counter(row["status"] for row in rows).items())),
        "triage_in_scope_rate": mean(row["actual_category"] == "in_scope" for row in rows),
        "answer_produced_rate": mean(row["answer_produced"] for row in rows),
        "fully_answered_rate": mean(row["fully_answered"] for row in rows),
        "grounded_success_rate": mean(row["grounded_success"] for row in rows),
        "mean_rounds": mean(row["rounds"] for row in rows),
        "citation": {
            "total": citation_count,
            "correct": correct_count,
            "precision_micro": correct_count / citation_count if citation_count else 0.0,
            "precision_macro_cited": mean(row["citation_precision"] for row in cited) if cited else 0.0,
            "hit_rate": mean(row["citation_hit"] for row in rows),
            "primary_hit_rate": mean(row["primary_citation_hit"] for row in rows),
            "integrity_rate": mean(integrity) if integrity else 0.0,
            "uncited_answer_rate": mean(row["citation_count"] == 0 for row in answered) if answered else 0.0,
        },
        "latency": {
            "mean_s": mean(latencies),
            "median_s": median(latencies),
            "p95_s": _percentile(latencies, 0.95),
            "max_s": max(latencies),
            "mean_excluding_first_s": mean(latencies[1:]) if len(latencies) > 1 else latencies[0],
        },
        "errors": sum(bool(row["error"]) for row in rows),
        "by_paraphrase_type": _grouped(rows, "paraphrase_type"),
        "by_source": _grouped(rows, "source"),
        "by_chunk_type": _grouped(rows, "chunk_type"),
    }


def build_payload(dataset: dict, rows: list[dict]) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "case_count": len(rows),
            "corpus_sha256": dataset["metadata"].get("corpus_sha256", "unknown"),
            "dataset_schema_version": dataset["metadata"].get("schema_version", "unknown"),
            "llm_provider": settings.llm_provider,
            "llm_model": settings.anthropic_model if settings.is_anthropic else settings.openai_model,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "device": settings.device,
            "max_retrieval_rounds": settings.max_retrieval_rounds,
        },
        "summary": summarise(rows),
        "results": rows,
    }


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _breakdown_table(groups: dict[str, dict]) -> str:
    lines = [
        "| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in groups.items():
        lines.append(
            f"| {name} | {values['n']} | {_pct(values['fully_answered_rate'])} | "
            f"{_pct(values['grounded_success_rate'])} | {_pct(values['citation_hit_rate'])} | "
            f"{_pct(values['citation_precision_micro'])} | {values['mean_latency_s']:.3f}s |"
        )
    return "\n".join(lines)


def render_markdown_section(payload: dict) -> str:
    summary = payload["summary"]
    config = payload["config"]
    citation = summary["citation"]
    latency = summary["latency"]
    statuses = ", ".join(
        f"`{status}`: {count}" for status, count in summary["statuses"].items()
    )
    failed = sorted(
        payload["results"],
        key=lambda row: (
            row["grounded_success"],
            row["citation_hit"],
            row["citation_precision"] if row["citation_precision"] is not None else -1,
        ),
    )[:10]
    failure_lines = [
        "| ID | Status | Citation precision | Gold citation | Latency | Query |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in failed:
        precision = f"{row['citation_precision']:.2%}" if row["citation_precision"] is not None else "—"
        query = row["query"].replace("|", "\\|")
        failure_lines.append(
            f"| {row['id']} | {row['status']} | {precision} | "
            f"{'yes' if row['citation_hit'] else 'no'} | {row['latency_s']:.3f}s | {query} |"
        )

    return f"""{SECTION_START}
## End-to-end and citation evaluation

Generated at: `{payload['created_at']}`

- Test cases: **{config['case_count']}**, using the same grounded questions and gold chunks as the retrieval evaluation
- Full path: triage → multi-query retrieval and neighbor expansion → verifier/rewrite → answer generation
- LLM: `{config['llm_model']}`; retrieval device: `{config['device']}`
- Maximum retrieval rounds: **{config['max_retrieval_rounds']}**

A citation is counted as correct when its `chunk_id` is one of the case's
`acceptable_gold_chunk_ids`. This is exact gold-chunk correctness, not merely a
source-name match. It does not by itself prove that every generated claim is
entailed by the cited passage.

### End-to-end outcomes

| Metric | Score |
|---|---:|
| Triage in-scope rate | {_pct(summary['triage_in_scope_rate'])} |
| Answer produced (`answered` or `answered_partial`) | {_pct(summary['answer_produced_rate'])} |
| Fully answered | {_pct(summary['fully_answered_rate'])} |
| Grounded end-to-end success | {_pct(summary['grounded_success_rate'])} |
| Mean retrieval rounds | {summary['mean_rounds']:.3f} |
| Errors | {summary['errors']} |

Statuses: {statuses}.

`Grounded end-to-end success` requires both `status == answered` and at least
one final citation to an acceptable gold chunk.

### Citation correctness

| Metric | Score |
|---|---:|
| Correct citations / all citations (micro precision) | {citation['correct']} / {citation['total']} = {_pct(citation['precision_micro'])} |
| Mean precision among cited answers (macro) | {_pct(citation['precision_macro_cited'])} |
| Gold-citation hit rate | {_pct(citation['hit_rate'])} |
| Primary-gold citation hit rate | {_pct(citation['primary_hit_rate'])} |
| Citation-marker integrity | {_pct(citation['integrity_rate'])} |
| Uncited-answer rate | {_pct(citation['uncited_answer_rate'])} |

### End-to-end latency

| Metric | Time |
|---|---:|
| Mean | {latency['mean_s']:.3f}s |
| Median | {latency['median_s']:.3f}s |
| P95 | {latency['p95_s']:.3f}s |
| Maximum | {latency['max_s']:.3f}s |
| Mean excluding first query | {latency['mean_excluding_first_s']:.3f}s |

This latency covers the complete graph and is therefore not comparable to the
retrieval-only latency as if they measured the same work. The first query is
reported separately because it may include lazy model and CUDA initialization.

### End-to-end by paraphrase type

{_breakdown_table(summary['by_paraphrase_type'])}

### End-to-end by source

{_breakdown_table(summary['by_source'])}

### End-to-end by chunk type

{_breakdown_table(summary['by_chunk_type'])}

### Ten weakest end-to-end cases

{chr(10).join(failure_lines)}
{SECTION_END}"""


def upsert_report_section(report: str, section: str) -> str:
    """Insert or replace the generated end-to-end section in a Markdown report."""
    if SECTION_START in report and SECTION_END in report:
        before, remainder = report.split(SECTION_START, 1)
        _, after = remainder.split(SECTION_END, 1)
        return f"{before.rstrip()}\n\n{section}\n{after.lstrip()}"

    anchor = "## Interpretation notes"
    if anchor in report:
        before, after = report.split(anchor, 1)
        return f"{before.rstrip()}\n\n{section}\n\n{anchor}{after}"
    return f"{report.rstrip()}\n\n{section}\n"


def write_report(report_path: Path, payload: dict) -> None:
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    updated = upsert_report_section(existing, render_markdown_section(payload))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(updated, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the full RAG graph and final citation correctness."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ids", nargs="*", help="evaluate only these case IDs")
    parser.add_argument("--limit", type=int, help="evaluate only the first N selected cases")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed rows in --details and continue the remaining cases",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if args.ids:
        selected = set(args.ids)
        cases = [case for case in cases if case["id"] in selected]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No matching end-to-end evaluation cases.")

    rows_by_id: dict[str, dict] = {}
    if args.resume and args.details.exists():
        previous = json.loads(args.details.read_text(encoding="utf-8"))
        rows_by_id = {row["id"]: row for row in previous.get("results", [])}

    args.details.parent.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases, 1):
        if case["id"] in rows_by_id:
            print(f"({index}/{len(cases)}) {case['id']}: reused")
            continue
        print(f"({index}/{len(cases)}) {case['id']}: {case['query'][:70]}", flush=True)
        row = evaluate_case(case)
        rows_by_id[case["id"]] = row
        ordered_rows = [rows_by_id[item["id"]] for item in cases if item["id"] in rows_by_id]
        checkpoint = build_payload(dataset, ordered_rows)
        args.details.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"    -> {row['status']} rounds={row['rounds']} "
            f"citations={row['correct_citation_count']}/{row['citation_count']} "
            f"latency={row['latency_s']:.3f}s",
            flush=True,
        )

    rows = [rows_by_id[case["id"]] for case in cases]
    payload = build_payload(dataset, rows)
    args.details.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.report, payload)
    print(f"Wrote details to {args.details}")
    print(f"Updated Markdown report at {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
