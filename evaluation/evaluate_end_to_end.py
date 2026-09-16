"""Evaluate the complete RAG graph on schema-2 single/multi-gold cases."""
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
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


def score_citations(case: dict, citations: list) -> dict:
    full = set(case["acceptable_gold_chunk_ids"])
    cited_ids = [citation.chunk_id for citation in citations]
    correct_ids = [chunk_id for chunk_id in cited_ids if chunk_id in full]
    return {
        "citation_count": len(cited_ids),
        "correct_citation_count": len(correct_ids),
        "citation_precision": len(correct_ids) / len(cited_ids) if cited_ids else None,
        "citation_hit": int(bool(correct_ids)),
        "primary_citation_hit": int(case["primary_gold_chunk_id"] in cited_ids),
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
        "gold_multiplicity": case["gold_multiplicity"],
        "duplication_scope": case["duplication_scope"],
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
    citation_count = sum(row["citation_count"] for row in rows)
    correct_count = sum(row["correct_citation_count"] for row in rows)
    cited = [row for row in rows if row["citation_count"]]
    return {
        "n": len(rows),
        "fully_answered_rate": mean(row["fully_answered"] for row in rows),
        "grounded_success_rate": mean(row["grounded_success"] for row in rows),
        "citation_hit_rate": mean(row["citation_hit"] for row in rows),
        "citation_precision_micro": correct_count / citation_count if citation_count else 0.0,
        "citation_precision_macro": mean(row["citation_precision"] for row in cited) if cited else 0.0,
        "mean_rounds": mean(row["rounds"] for row in rows),
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
    integrity = [row["citation_integrity"] for row in rows if row["citation_integrity"] is not None]
    latencies = [row["latency_s"] for row in rows]
    citation_count = sum(row["citation_count"] for row in rows)
    correct_count = sum(row["correct_citation_count"] for row in rows)
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
        "by_gold_multiplicity": _grouped(rows, "gold_multiplicity"),
        "by_duplication_scope": _grouped(rows, "duplication_scope"),
        "by_retrieval_rounds": _grouped(rows, "rounds"),
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
            "openai_chat_template_kwargs": (
                settings.openai_chat_template_kwargs if not settings.is_anthropic else {}
            ),
            "embedding_model": settings.embedding_model,
            "embedding_query_prompt_name": settings.embedding_query_prompt_name,
            "reranker_model": settings.reranker_model,
            "device": settings.device,
            "max_retrieval_rounds": settings.max_retrieval_rounds,
        },
        "summary": summarise(rows),
        "results": rows,
    }


_RESUME_CONFIG_KEYS = (
    "dataset_schema_version",
    "corpus_sha256",
    "llm_provider",
    "llm_model",
    "openai_chat_template_kwargs",
    "embedding_model",
    "embedding_query_prompt_name",
    "reranker_model",
    "max_retrieval_rounds",
)


def resume_config_matches(previous: dict, current: dict) -> bool:
    """Only reuse rows produced by the exact model and retrieval configuration."""
    return all(previous.get(key) == current.get(key) for key in _RESUME_CONFIG_KEYS)


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _breakdown_table(groups: dict[str, dict]) -> str:
    lines = [
        "| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in groups.items():
        lines.append(
            f"| {name} | {values['n']} | {_pct(values['fully_answered_rate'])} | "
            f"{_pct(values['grounded_success_rate'])} | {_pct(values['citation_hit_rate'])} | "
            f"{_pct(values['citation_precision_micro'])} | {values['mean_rounds']:.3f} | "
            f"{values['mean_latency_s']:.3f}s |"
        )
    return "\n".join(lines)


def _weakest_table(rows: list[dict], multiplicity: str) -> str:
    selected = [row for row in rows if row["gold_multiplicity"] == multiplicity]
    failed = sorted(
        selected,
        key=lambda row: (
            row["grounded_success"], row["citation_hit"],
            row["citation_precision"] if row["citation_precision"] is not None else -1,
        ),
    )[:5]
    lines = [
        "| ID | Status | Scope | Citation precision | Gold citation | Latency | Query |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in failed:
        precision = f"{row['citation_precision']:.2%}" if row["citation_precision"] is not None else "—"
        query = row["query"].replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {row['status']} | {row['duplication_scope']} | {precision} | "
            f"{'yes' if row['citation_hit'] else 'no'} | {row['latency_s']:.3f}s | {query} |"
        )
    return "\n".join(lines)


def render_markdown_section(payload: dict) -> str:
    summary = payload["summary"]
    config = payload["config"]
    citation = summary["citation"]
    latency = summary["latency"]
    statuses = ", ".join(f"`{key}`: {value}" for key, value in summary["statuses"].items())
    return f"""{SECTION_START}
## End-to-end and citation evaluation

Generated at: `{payload['created_at']}`

- Full path: triage → multi-query retrieval and neighbor expansion → verifier/rewrite → answer generation
- LLM: `{config['llm_model']}`; retrieval device: `{config['device']}`
- Maximum retrieval rounds: **{config['max_retrieval_rounds']}**

A final citation is correct when it points to any independently full-supporting
gold chunk. Multi-gold answers need not cite every repeated occurrence.

### Overall end-to-end outcomes

| Metric | Score |
|---|---:|
| Triage in-scope rate | {_pct(summary['triage_in_scope_rate'])} |
| Answer produced | {_pct(summary['answer_produced_rate'])} |
| Fully answered | {_pct(summary['fully_answered_rate'])} |
| Grounded end-to-end success | {_pct(summary['grounded_success_rate'])} |
| Mean retrieval rounds | {summary['mean_rounds']:.3f} |
| Errors | {summary['errors']} |

Statuses: {statuses}.

Grounded success requires `status == answered` and at least one full-gold final citation.

### Single- versus multi-gold end-to-end results

{_breakdown_table(summary['by_gold_multiplicity'])}

### End-to-end by duplication scope

{_breakdown_table(summary['by_duplication_scope'])}

### Citation correctness

| Metric | Score |
|---|---:|
| Correct citations / all citations | {citation['correct']} / {citation['total']} = {_pct(citation['precision_micro'])} |
| Mean precision among cited answers | {_pct(citation['precision_macro_cited'])} |
| Full-gold citation hit rate | {_pct(citation['hit_rate'])} |
| Primary-gold citation hit rate | {_pct(citation['primary_hit_rate'])} |
| Citation-marker integrity | {_pct(citation['integrity_rate'])} |
| Uncited-answer rate | {_pct(citation['uncited_answer_rate'])} |

Primary-gold hit is diagnostic only: every validated full alternative is equally correct.
Gold-chunk precision is stricter than source matching but is not a claim-entailment judge.

### End-to-end latency

| Metric | Time |
|---|---:|
| Mean | {latency['mean_s']:.3f}s |
| Median | {latency['median_s']:.3f}s |
| P95 | {latency['p95_s']:.3f}s |
| Maximum | {latency['max_s']:.3f}s |
| Mean excluding first query | {latency['mean_excluding_first_s']:.3f}s |

Latency covers the complete graph. The first query is separated because it may
include lazy model and CUDA initialization. The main repeated cost is the number
of verifier/retrieval rounds:

{_breakdown_table(summary['by_retrieval_rounds'])}

The maximum is reported separately so a transient model stall cannot be mistaken
for typical latency; compare it with the median and P95.

### End-to-end by paraphrase type

{_breakdown_table(summary['by_paraphrase_type'])}

### End-to-end by source

{_breakdown_table(summary['by_source'])}

### End-to-end by primary chunk type

{_breakdown_table(summary['by_chunk_type'])}

### Weakest single-gold end-to-end cases

{_weakest_table(payload['results'], 'single_gold')}

### Weakest multi-gold end-to-end cases

{_weakest_table(payload['results'], 'multi_gold')}
{SECTION_END}"""


def upsert_report_section(report: str, section: str) -> str:
    if SECTION_START in report and SECTION_END in report:
        before, remainder = report.split(SECTION_START, 1)
        _, after = remainder.split(SECTION_END, 1)
        return f"{before.rstrip()}\n\n{section}\n{after.lstrip()}"
    return f"{report.rstrip()}\n\n{section}\n"


def write_report(report_path: Path, payload: dict) -> None:
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        upsert_report_section(report, render_markdown_section(payload)), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate full schema-2 RAG flow")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    if dataset.get("metadata", {}).get("schema_version") != "2.0":
        raise SystemExit("Schema 2.0 dataset required; regenerate the grounded test set")
    cases = dataset["cases"]
    if args.ids:
        selected = set(args.ids)
        cases = [case for case in cases if case["id"] in selected]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No matching end-to-end cases")

    rows_by_id: dict[str, dict] = {}
    if args.resume and args.details.exists():
        previous = json.loads(args.details.read_text(encoding="utf-8"))
        previous_config = previous.get("config", {})
        current_config = build_payload(dataset, [])["config"]
        if resume_config_matches(previous_config, current_config):
            rows_by_id = {row["id"]: row for row in previous.get("results", [])}

    args.details.parent.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases, 1):
        if case["id"] in rows_by_id:
            print(f"({index}/{len(cases)}) {case['id']}: reused")
            continue
        print(f"({index}/{len(cases)}) {case['id']}: {case['query'][:70]}", flush=True)
        row = evaluate_case(case)
        rows_by_id[case["id"]] = row
        ordered = [rows_by_id[item["id"]] for item in cases if item["id"] in rows_by_id]
        args.details.write_text(
            json.dumps(build_payload(dataset, ordered), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"    -> {row['status']} rounds={row['rounds']} "
            f"citations={row['correct_citation_count']}/{row['citation_count']} "
            f"latency={row['latency_s']:.3f}s",
            flush=True,
        )

    rows = [rows_by_id[case["id"]] for case in cases]
    payload = build_payload(dataset, rows)
    args.details.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.report, payload)
    print(f"Wrote details to {args.details}")
    print(f"Updated report at {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
