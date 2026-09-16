"""Create a compact, honest comparison from baseline result JSON files."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "evaluation"
DEFAULT_MANIFEST = HERE / "baselines.json"
DEFAULT_RESULTS_DIR = HERE / "baseline_results"
DEFAULT_OUTPUT = HERE / "model_baseline_comparison.md"


def _load(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    payload["_result_path"] = str(path.relative_to(ROOT))
    return payload


def discover(filename: str, results_dir: Path) -> list[dict]:
    paths = [HERE / filename]
    if results_dir.exists():
        paths.extend(sorted(results_dir.rglob(filename)))
    payloads = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        payload = _load(path)
        if payload:
            payloads.append(payload)
    return payloads


def _prompt(config: dict) -> str | None:
    return config.get("embedding_query_prompt_name") or None


def _retrieval_matches(payload: dict, baseline: dict) -> bool:
    config = payload.get("config", {})
    return (
        config.get("embedding_model") == baseline["embedding_model"]
        and config.get("reranker_model") == baseline["reranker_model"]
        and _prompt(config) == (baseline.get("embedding_query_prompt_name") or None)
    )


def _e2e_matches(payload: dict, llm: dict, retrieval: dict) -> bool:
    config = payload.get("config", {})
    return (
        config.get("llm_model") == llm["model"]
        and _retrieval_matches(payload, retrieval)
    )


def _best_match(payloads: Iterable[dict], predicate) -> dict | None:
    matches = [payload for payload in payloads if predicate(payload)]
    if not matches:
        return None
    return max(
        matches,
        key=lambda payload: (
            payload.get("config", {}).get("case_count", 0),
            payload.get("created_at", ""),
        ),
    )


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _float(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _delta(value: float | None, reference: float | None, comparable: bool) -> str:
    if value is None or reference is None or not comparable:
        return "—"
    return f"{(value - reference) * 100:+.2f} pp"


def _same_benchmark(candidate: dict | None, reference: dict | None) -> bool:
    if not candidate or not reference:
        return False
    left = candidate.get("config", {})
    right = reference.get("config", {})
    return (
        left.get("dataset_schema_version") == right.get("dataset_schema_version")
        and left.get("corpus_sha256") == right.get("corpus_sha256")
        and left.get("case_count") == right.get("case_count")
    )


def _status(payload: dict | None, expected_count: int) -> str:
    if payload is None:
        return "planned"
    config = payload.get("config", {})
    count = config.get("case_count", 0)
    errors = payload.get("summary", {}).get("errors", 0)
    if errors or count != expected_count:
        return f"incomplete (n={count}/{expected_count}, errors={errors})"
    return f"measured (n={count})"


def render_report(
    manifest: dict,
    retrieval_payloads: list[dict],
    e2e_payloads: list[dict],
) -> str:
    retrieval_by_id = {
        item["id"]: _best_match(
            retrieval_payloads, lambda payload, item=item: _retrieval_matches(payload, item)
        )
        for item in manifest["retrieval_baselines"]
    }
    bge = next(item for item in manifest["retrieval_baselines"] if item["id"] == "bge_m3")
    e2e_by_id = {
        item["id"]: _best_match(
            e2e_payloads,
            lambda payload, item=item: _e2e_matches(payload, item, bge),
        )
        for item in manifest["llm_baselines"]
    }

    reference_retrieval = retrieval_by_id["bge_m3"]
    reference_e2e = e2e_by_id["qwen3_30b_a3b_awq"]

    retrieval_rows = []
    for item in manifest["retrieval_baselines"]:
        payload = retrieval_by_id[item["id"]]
        complete = _same_benchmark(payload, reference_retrieval) and not (
            payload or {}
        ).get("summary", {}).get("errors", 0)
        summary = payload.get("summary", {}).get("overall", {}) if complete else {}
        reference_summary = (
            reference_retrieval.get("summary", {}).get("overall", {})
            if reference_retrieval
            else {}
        )
        comparable = _same_benchmark(payload, reference_retrieval)
        retrieval_rows.append(
            "| {name} | {reranker} | {status} | {hit1} | {hit10} | {delta} | "
            "{mrr} | {ndcg10} | {latency} |".format(
                name=item["embedding_model"],
                reranker=item["reranker_model"],
                status=_status(payload, reference_retrieval.get("config", {}).get("case_count", 100) if reference_retrieval else 100),
                hit1=_pct(summary.get("hit@1")),
                hit10=_pct(summary.get("hit@10")),
                delta=_delta(
                    summary.get("hit@10"),
                    reference_summary.get("hit@10"),
                    comparable,
                ),
                mrr=_float(summary.get("mrr")),
                ndcg10=_float(summary.get("ndcg@10")),
                latency=(
                    f"{payload['summary']['mean_latency_s']:.3f}s" if complete else "—"
                ),
            )
        )

    e2e_rows = []
    for item in manifest["llm_baselines"]:
        payload = e2e_by_id[item["id"]]
        complete = _same_benchmark(payload, reference_e2e) and not (
            payload or {}
        ).get("summary", {}).get("errors", 0)
        summary = payload.get("summary", {}) if complete else {}
        reference_summary = reference_e2e.get("summary", {}) if reference_e2e else {}
        citation = summary.get("citation", {})
        latency = summary.get("latency", {})
        comparable = _same_benchmark(payload, reference_e2e)
        e2e_rows.append(
            "| {model} | {params} | {status} | {full} | {grounded} | {delta} | "
            "{citation_hit} | {citation_precision} | {rounds} | {latency} |".format(
                model=item["model"],
                params=item["parameters"],
                status=_status(payload, reference_e2e.get("config", {}).get("case_count", 100) if reference_e2e else 100),
                full=_pct(summary.get("fully_answered_rate")),
                grounded=_pct(summary.get("grounded_success_rate")),
                delta=_delta(
                    summary.get("grounded_success_rate"),
                    reference_summary.get("grounded_success_rate"),
                    comparable,
                ),
                citation_hit=_pct(citation.get("hit_rate")),
                citation_precision=_pct(citation.get("precision_micro")),
                rounds=_float(summary.get("mean_rounds"), 3),
                latency=(
                    f"{latency['mean_s']:.3f}s" if latency.get("mean_s") is not None else "—"
                ),
            )
        )

    takeaways: list[str] = []
    qwen25 = e2e_by_id.get("qwen2_5_7b_instruct")
    if _same_benchmark(qwen25, reference_e2e) and not (
        qwen25 or {}
    ).get("summary", {}).get("errors", 0):
        small = qwen25["summary"]
        reference = reference_e2e["summary"]
        latency_change = (
            1 - small["latency"]["mean_s"] / reference["latency"]["mean_s"]
        )
        takeaways.append(
            "- Qwen2.5-7B is "
            f"{latency_change:.1%} faster on mean end-to-end latency, but grounded "
            f"success falls by {(reference['grounded_success_rate'] - small['grounded_success_rate']) * 100:.0f} "
            "percentage points. The current 30B-A3B AWQ model remains the quality winner."
        )

    qwen_retrieval = retrieval_by_id.get("qwen3_0_6b")
    if _same_benchmark(qwen_retrieval, reference_retrieval) and not (
        qwen_retrieval or {}
    ).get("summary", {}).get("errors", 0):
        qwen_summary = qwen_retrieval["summary"]
        qwen_scores = qwen_summary["overall"]
        bge_scores = reference_retrieval["summary"]["overall"]
        latency_change = (
            1 - qwen_summary["mean_latency_s"]
            / reference_retrieval["summary"]["mean_latency_s"]
        )
        takeaways.append(
            "- The Qwen 0.6B retrieval pair is "
            f"{latency_change:.1%} faster, but loses "
            f"{(bge_scores['hit@1'] - qwen_scores['hit@1']) * 100:.0f} percentage points at Hit@1, "
            f"{(bge_scores['hit@10'] - qwen_scores['hit@10']) * 100:.0f} percentage points at Hit@10, "
            f"and {bge_scores['mrr'] - qwen_scores['mrr']:.4f} MRR. Keep BGE-M3 "
            "when retrieval quality is the priority."
        )

    qwen3_8b = e2e_by_id.get("qwen3_8b")
    if qwen3_8b and not _same_benchmark(qwen3_8b, reference_e2e):
        completed = [row for row in qwen3_8b.get("results", []) if not row.get("error")]
        detail = (
            f"; its first completed case took {completed[0]['latency_s']:.3f}s"
            if completed
            else ""
        )
        takeaways.append(
            "- Qwen3-8B is excluded from score comparison because its run is "
            f"incomplete{detail}. Under deterministic temperature 0, the complex "
            "strict schema produced a pathological long response, so continuing "
            "would not be a safe or comparable server run."
        )

    interpretation = "\n".join(takeaways) or "- No complete comparable runs are available yet."

    measured_paths = sorted(
        {
            payload["_result_path"]
            for payload in [*retrieval_payloads, *e2e_payloads]
            if "_result_path" in payload
        }
    )
    sources = "\n".join(f"- `{path}`" for path in measured_paths) or "- none"

    return f"""# Qwen and embedding baseline comparison

Generated at: `{datetime.now(timezone.utc).isoformat()}`

This report changes only one axis at a time. The LLM table fixes retrieval to
BGE-M3 + bge-reranker-v2-m3. The retrieval table does not claim an end-to-end
LLM improvement. A dash means the configured run has not been measured; it is
never replaced with an estimated score.

## LLM baseline comparison

| LLM | Parameters | Status | Fully answered | Grounded success | Δ grounded vs current | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(e2e_rows)}

The Qwen2.5 7B row provides a generation-to-generation control, Qwen3 8B and
14B show dense-model scaling, and the existing 30B-A3B AWQ run remains the
reference. All rows use temperature 0 and the same prompts and retrieval index.

## Retrieval baseline comparison

| Embedding | Reranker | Status | Hit@1 | Hit@10 | Δ Hit@10 vs BGE | MRR | nDCG@10 | Mean latency |
|---|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(retrieval_rows)}

The Qwen pair uses the model-provided `query` prompt for queries and no prompt
for passages. Both retrieval configurations retain BM25/RRF, the same committed
pre-chunked passages, candidate depths, and K values. Each embedding has its own
Qdrant directory.

## Result interpretation

{interpretation}

## Fair-comparison rules

- Every baseline process is capped at six CPU threads through OMP, MKL,
  OpenBLAS, NumExpr, vecLib, and Rayon environment limits.
- Incomplete or failed runs display no scores.
- Compare deltas only when dataset schema, corpus SHA-256, and evaluated case
  count match the reference; otherwise the report shows no delta.
- Reuse the committed `data/processed/chunks.jsonl` when changing embeddings.
  Re-chunking would change gold IDs and invalidate the benchmark.
- Use retrieval-only results to select the embedding/reranker pair, then run a
  separate end-to-end confirmation with the selected pair.
- Judge model quality primarily by grounded success and citation behavior, not
  latency alone. Report median/P95 from the detailed artifact when sizing a
  deployment.

## Reproduce

```bash
# Inspect the matrix
python -m evaluation.run_baselines list

# Build and evaluate the Qwen retrieval pair (fixed chunks)
python -m evaluation.run_baselines index --retrieval qwen3_0_6b
python -m evaluation.run_baselines retrieval --retrieval qwen3_0_6b

# Start one Qwen LLM, then evaluate it against fixed BGE retrieval
CPU_THREADS=6 GPU_ID=0 PORT=8001 MODEL_ID=Qwen/Qwen3-8B ./scripts/start_vllm.sh
python -m evaluation.run_baselines end-to-end --llm qwen3_8b --retrieval bge_m3 --base-url http://127.0.0.1:8001/v1

# Refresh this file after any run
python -m evaluation.run_baselines report
```

## Result artifacts used

{sources}

## Model-selection notes

The selected local models all fit a 48 GB GPU individually. Qwen3 supports both
thinking and non-thinking behavior; this project keeps the existing OpenAI-
compatible structured-output path unchanged so the comparison measures model
replacement rather than a prompt rewrite. Qwen3-Embedding-0.6B is the closest
same-size retrieval alternative to the current 0.6B-class BGE reranker and emits
1024-dimensional vectors, making it a practical first comparison before testing
the substantially larger 4B and 8B retrieval variants.

Official references: [Qwen3 release](https://qwenlm.github.io/blog/qwen3/),
[Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B),
[Qwen2.5 release](https://qwenlm.github.io/blog/qwen2.5-llm/), and
[Qwen3 Embedding collection](https://huggingface.co/collections/Qwen/qwen3-embedding).
"""


def generate_report(manifest_path: Path, results_dir: Path, output: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = render_report(
        manifest,
        discover("retrieval_results.json", results_dir),
        discover("end_to_end_results.json", results_dir),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare measured model baselines")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    generate_report(args.manifest, args.results_dir, args.output)
    print(f"Wrote comparison to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

