# Qwen and embedding baseline comparison

Generated at: `2026-09-15T16:48:43.738601+00:00`

This report changes only one axis at a time. The LLM table fixes retrieval to
BGE-M3 + bge-reranker-v2-m3. The retrieval table does not claim an end-to-end
LLM improvement. A dash means the configured run has not been measured; it is
never replaced with an estimated score.

## LLM baseline comparison

| LLM | Parameters | Status | Fully answered | Grounded success | Δ grounded vs current | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen2.5-7B-Instruct | 7.6B | measured (n=100) | 49.00% | 47.00% | -22.00 pp | 47.00% | 46.55% | 2.010 | 11.558s |
| Qwen/Qwen3-8B | 8.2B | incomplete (n=2/100, errors=1) | — | — | — | — | — | — | — |
| Qwen/Qwen3-14B | 14.8B | planned | — | — | — | — | — | — | — |
| QuixiAI/Qwen3-30B-A3B-AWQ | 30.5B total / 3.3B active, AWQ | measured (n=100) | 72.00% | 69.00% | +0.00 pp | 96.00% | 47.84% | 1.770 | 16.493s |

The Qwen2.5 7B row provides a generation-to-generation control, Qwen3 8B and
14B show dense-model scaling, and the existing 30B-A3B AWQ run remains the
reference. All rows use temperature 0 and the same prompts and retrieval index.

## Retrieval baseline comparison

| Embedding | Reranker | Status | Hit@1 | Hit@10 | Δ Hit@10 vs BGE | MRR | nDCG@10 | Mean latency |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BAAI/bge-m3 | BAAI/bge-reranker-v2-m3 | measured (n=100) | 68.00% | 98.00% | +0.00 pp | 0.7986 | 0.7730 | 0.467s |
| Qwen/Qwen3-Embedding-0.6B | Qwen/Qwen3-Reranker-0.6B | measured (n=100) | 56.00% | 95.00% | -3.00 pp | 0.7086 | 0.6996 | 0.262s |

The Qwen pair uses the model-provided `query` prompt for queries and no prompt
for passages. Both retrieval configurations retain BM25/RRF, the same committed
pre-chunked passages, candidate depths, and K values. Each embedding has its own
Qdrant directory.

## Result interpretation

- Qwen2.5-7B is 29.9% faster on mean end-to-end latency, but grounded success falls by 22 percentage points. The current 30B-A3B AWQ model remains the quality winner.
- The Qwen 0.6B retrieval pair is 44.0% faster, but loses 12 percentage points at Hit@1, 3 percentage points at Hit@10, and 0.0900 MRR. Keep BGE-M3 when retrieval quality is the priority.
- Qwen3-8B is excluded from score comparison because its run is incomplete; its first completed case took 143.253s. Under deterministic temperature 0, the complex strict schema produced a pathological long response, so continuing would not be a safe or comparable server run.

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

- `evaluation/baseline_results/qwen2_5_7b_instruct__bge_m3/end_to_end_results.json`
- `evaluation/baseline_results/qwen3_8b__bge_m3/end_to_end_results.json`
- `evaluation/baseline_results/retrieval__qwen3_0_6b/retrieval_results.json`
- `evaluation/end_to_end_results.json`
- `evaluation/retrieval_results.json`

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
