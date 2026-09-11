"""Offline evaluation harness.

Runs every query in `test_queries.json` through the graph and scores four things
the pipeline can actually be held to without human-labelled gold answers:

  1. Triage accuracy   - did the gatekeeper route correctly (incl. over-refusal)?
  2. Retrieval quality - did citations land in the documents that hold the answer?
  3. Citation integrity- is every inline marker resolvable to a real passage?
  4. Answer coverage   - do the expected key terms appear in the answer?

Deliberately not an LLM-judge: the point is a fast, deterministic regression
signal you can run after changing a chunk size or a prompt. Judge scoring can be
layered on top of `results.csv` later.

    python -m evaluation.evaluate                # all queries
    python -m evaluation.evaluate --ids q01 q15  # a subset
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.graph.workflow import run

HERE = Path(__file__).resolve().parent
QUERIES = HERE / "test_queries.json"
RESULTS = HERE / "results.csv"

console = Console()

# Statuses that mean "the system declined to answer", which is the correct
# outcome for the absent-information probe.
_DECLINED = {"insufficient", "rejected", "needs_clarification"}


def _keyword_recall(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    lowered = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in lowered)
    return hits / len(keywords)


def _source_metrics(citations, expected: list[str]) -> tuple[float | None, float | None]:
    """(hit, precision) against the documents expected to hold the answer."""
    if not expected:
        return None, None
    if not citations:
        return 0.0, 0.0
    sources = [c.source for c in citations]
    hit = 1.0 if any(s in expected for s in sources) else 0.0
    precision = sum(1 for s in sources if s in expected) / len(sources)
    return hit, precision


def evaluate_one(case: dict) -> dict:
    started = time.perf_counter()
    try:
        state = run(case["query"])
        error = ""
    except Exception as exc:  # a crash is a result, not a reason to stop the suite
        logging.exception("Query %s failed", case["id"])
        state, error = {"status": "error", "response": str(exc)}, str(exc)

    elapsed = time.perf_counter() - started

    triage = state.get("triage")
    answer = state.get("answer")
    citations = answer.citations if answer else []
    status = state.get("status", "error")

    actual_category = triage.category if triage else ""
    expected_category = case["expected_category"]

    source_hit, source_precision = _source_metrics(citations, case.get("expected_sources", []))
    recall = _keyword_recall(state.get("response", ""), case.get("must_include", []))

    expect_insufficient = case.get("expect_insufficient", False)
    if expect_insufficient:
        # Correct behaviour is refusing, or answering with an explicit caveat.
        abstained = status in _DECLINED or status == "answered_partial"
    else:
        abstained = None

    return {
        "id": case["id"],
        "query": case["query"],
        "expected_category": expected_category,
        "actual_category": actual_category,
        "triage_correct": int(actual_category == expected_category),
        "status": status,
        "rounds": state.get("round", 0),
        "n_citations": len(citations),
        "cited_sources": "|".join(sorted({c.source for c in citations})),
        "source_hit": source_hit,
        "source_precision": source_precision,
        "keyword_recall": recall,
        "confidence": answer.confidence if answer else "",
        "citation_integrity": int(not (answer.dropped_markers if answer else [])),
        "abstained_correctly": None if abstained is None else int(abstained),
        "latency_s": round(elapsed, 2),
        "error": error,
        "response": state.get("response", "")[:2000],
    }


def summarise(df: pd.DataFrame) -> None:
    table = Table(title="Summary", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("n", justify="right")

    def row(label: str, series: pd.Series, pct: bool = True) -> None:
        clean = series.dropna()
        if clean.empty:
            table.add_row(label, "-", "0")
            return
        value = clean.mean()
        table.add_row(label, f"{value:.1%}" if pct else f"{value:.2f}", str(len(clean)))

    row("Triage accuracy", df["triage_correct"])
    answered = df[df["status"].isin(["answered", "answered_partial"])]
    row("Retrieval hit@sources", df["source_hit"])
    row("Source precision", df["source_precision"])
    row("Keyword recall", df["keyword_recall"])
    row("Citation integrity", df["citation_integrity"])
    row("Abstained when it should", df["abstained_correctly"])
    row("Mean citations / answer", answered["n_citations"], pct=False)
    row("Mean retrieval rounds", df["rounds"], pct=False)
    row("Mean latency (s)", df["latency_s"], pct=False)
    console.print(table)

    # Where triage went wrong is the single most actionable breakdown.
    wrong = df[df["triage_correct"] == 0]
    if not wrong.empty:
        detail = Table(title="Triage misroutes", header_style="bold red")
        detail.add_column("id")
        detail.add_column("expected")
        detail.add_column("actual")
        detail.add_column("query", max_width=60)
        for _, r in wrong.iterrows():
            detail.add_row(r["id"], r["expected_category"], r["actual_category"], r["query"])
        console.print(detail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", help="run only these query ids")
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    cases = json.loads(QUERIES.read_text(encoding="utf-8"))
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids)]
    if not cases:
        console.print("[red]No matching queries[/red]")
        return 1

    rows = []
    for i, case in enumerate(cases, 1):
        console.print(f"[dim]({i}/{len(cases)})[/dim] {case['id']}: {case['query'][:70]}")
        result = evaluate_one(case)
        rows.append(result)
        console.print(
            f"    -> {result['status']}  triage={result['actual_category']}  "
            f"cites={result['n_citations']}  {result['latency_s']}s"
        )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    console.print(f"\nWrote [cyan]{args.out}[/cyan]")
    summarise(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
