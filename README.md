# multi-agent-cyber-rag

An English-language four-agent retrieval pipeline over authoritative
cyber-security publications
(NIST CSF 2.0, NIST SP 800-61r3, the CISA/FBI #StopRansomware Guide). Every
answer is grounded in retrieved passages and carries citations down to the
document, section, and page.

The design principle: **the system is allowed to say no.** A triage agent refuses
what the corpus cannot answer, and a verifier agent refuses to hand thin evidence
to the answer generator. Both are more important than the generation step - an
ungrounded answer with a plausible citation is the failure mode a cited RAG
system exists to prevent.

---

## Architecture

```
                            User query
                                |
                    ┌───────────▼───────────┐
                    │ Agent 1: Triage       │   scope + retrieval plan
                    └───────────┬───────────┘
              ┌─────────────────┼─────────────────┐
        out_of_scope        ambiguous          in_scope
              │                 │                 │
           Reject           Clarify   ┌───────────▼───────────┐
                                      │ Agent 2: Retrieval    │◄──────┐
                                      └───────────┬───────────┘       │
                                    BGE-M3 dense ─┴─ BM25 sparse      │
                                          RRF fusion in Qdrant        │
                                       bge-reranker-v2-m3 rerank      │
                                      adjacent chunks from doc/section │
                                                  │                   │
                                      ┌───────────▼───────────┐       │
                                      │ Agent 3: Verifier     │       │
                                      └───────────┬───────────┘       │
                                   insufficient ──┴── enough      rewritten
                                          │              │         queries
                                          └──────────────┼─────────────┘
                                                         │  (budget: 3 rounds)
                                             ┌───────────▼───────────┐
                                             │ Agent 4: Answer       │
                                             └───────────┬───────────┘
                                            grounded answer + citations
```

| Agent | Role | LLM? |
|---|---|---|
| 1. Triage | Route in-scope / ambiguous / out-of-scope; rewrite the query into 2-3 retrieval queries | yes |
| 2. Retrieval | Hybrid search per query, reciprocal rank fusion, reranking, and adjacent-chunk expansion | no - deterministic |
| 3. Verifier | Decide whether the evidence can ground an answer; name what is missing and what to search instead | yes |
| 4. Answer | Write the answer from verified passages only, with inline `[n]` markers | yes |

### Retrieval

Retrieval combines BGE-M3 dense search with BM25 sparse search, fuses the
rankings with reciprocal rank fusion, reranks the candidates, and adds adjacent
chunks from the same document section for context. Retrieval is deterministic;
query formulation happens in triage or verifier rewrites. Tune retrieval and
neighbor expansion in `.env`.

---

## System requirements

- Python 3.11 (the version used for the verified test run) and `pip`; a Conda
  environment is recommended.
- Linux or another environment capable of running the Python CLI. The supplied
  launch scripts use Bash.
- Enough local disk space for the source PDFs, the Qdrant index, and downloaded
  Docling, embedding, and reranker model weights. The first model load requires
  network access unless those weights are already cached.
- An LLM backend: either an OpenAI-compatible endpoint such as vLLM, or an
  Anthropic API key. Running the documented local Qwen AWQ model requires a
  CUDA-capable GPU with sufficient VRAM.
- CUDA is optional for ingestion/retrieval: set `DEVICE=cpu` in `.env` for a
  slower CPU-only run. The offline test suite itself needs no GPU, API key, or
  network access.

## Setup

```bash
conda create -n hainn_rag python=3.11 -y
conda activate hainn_rag
python -m pip install -r requirements.txt
cp .env.example .env          # then edit - see "LLM backend" below
```

Put PDFs in `data/raw/`, then build the index:

```bash
python -m src.main ingest          # add --dump to also write chunks.jsonl
python -m src.main status          # config + how many chunks are indexed
```

Ingestion uses Docling to preserve prose, tables, sections, and page provenance.
It writes the local Qdrant index under `data/processed/qdrant` and an auditable
Markdown export under `data/processed/markdown`. The first run downloads the
required parsing and retrieval models. OCR, table mode, chunking, and retrieval
settings are documented in `.env.example`.

### LLM backend

Choose the backend with `LLM_PROVIDER` in `.env`.

For the documented local OpenAI-compatible setup:

```bash
python -m pip install -U vllm
MODEL_ID=QuixiAI/Qwen3-30B-A3B-AWQ ./scripts/start_vllm.sh
# In a second terminal:
./scripts/start_web.sh
```

```dotenv
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_MODEL=QuixiAI/Qwen3-30B-A3B-AWQ
```

Use `GPU_ID`, `PORT`, `MODEL_ID`, `MAX_MODEL_LEN`, and
`GPU_MEMORY_UTILIZATION` to override the local server defaults. Verify the
endpoint with `curl http://127.0.0.1:8000/v1/models`.

For Anthropic:

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-5
```

---

## Usage

```bash
python -m src.main ask "What should an organisation do during incident containment?"
python -m src.main ask "..." --trace     # show each agent's decision
python -m src.main chat                  # interactive
python -m src.main -v ask "..."          # detailed pipeline + LLM logs
```

Commands return an answer, source/section/page citations, confidence, and caveats.
Logs are written to `logs/cyber-rag.log`; use `-v` for console diagnostics and
set `LOG_PAYLOADS=false` when prompts or responses may contain sensitive data.

---

## Evaluation

### Test queries and system outputs

> [!IMPORTANT]
> **The complete question-output pairs are in
> [`evaluation/end_to_end_results.json`](evaluation/end_to_end_results.json).**
> Every object in `results[]` contains both fields:
>
> - **`query`**: the original English test question.
> - **`response`**: the final answer produced by the complete RAG pipeline.
>
> Because both fields are in the same object, no cross-file join is needed to
> read a question and its corresponding system output. Use `id` to trace that
> record back to the benchmark and retrieval results.

Each end-to-end record also includes `status`, citations, confidence, latency,
and evaluation measurements. Related artifacts:

| Artifact | Purpose |
|---|---|
| [`generated_retrieval_test_cases.json`](test_case_generation/generated_retrieval_test_cases.json) | The 100 benchmark queries and audited retrieval/citation labels. |
| [`retrieval_results.json`](evaluation/retrieval_results.json) | Ranked chunks and retrieval metrics for every query. |
| [`retrieval_evaluation.md`](evaluation/retrieval_evaluation.md) | Human-readable aggregate retrieval and end-to-end report. |

```bash
python -m test_case_generation.generate_retrieval  # rebuild the 100-case benchmark
python -m evaluation.evaluate_retrieval            # retrieval-only metrics
python -m evaluation.evaluate_end_to_end            # full four-agent flow
python -m evaluation.evaluate_end_to_end --resume   # resume an interrupted full run
```

The schema-2 benchmark contains 100 English questions from 34 atomic facts:
50 have one fully supporting chunk and 50 have multiple acceptable gold chunks.
See [`test_case_generation/README.md`](test_case_generation/README.md) for the
construction and audit method.

Detailed results are in
[`evaluation/retrieval_evaluation.md`](evaluation/retrieval_evaluation.md), and
model/retriever comparisons are in
[`evaluation/model_baseline_comparison.md`](evaluation/model_baseline_comparison.md).
Regenerate results after changing the parser, corpus, chunk boundaries, model,
or retrieval configuration.

---

## Testing

```bash
pytest tests/ -q
```

The test suite is offline: it mocks LLM and PDF-conversion dependencies, so it
requires no GPU, API key, or network connection. It covers ingestion, retrieval,
citations, evaluation scoring, provider behavior, the web API, and all workflow
routes.

---

## Layout

```
src/
  config.py              all settings, overridable from .env
  llm/                   provider abstraction: base, anthropic, openai-compatible
  ingestion/
    loaders.py           Docling -> Markdown + typed prose/table structures
    chunking.py          semantic prose + header:value table-row chunks
    indexing.py          Qdrant collection (dense + sparse) and hybrid query
  retrieval/
    dense.py             BGE-M3 bi-encoder
    sparse.py            BM25 sparse vectors (FastEmbed)
    reranker.py          bge-reranker-v2-m3 cross-encoder
    __init__.py          HybridRetriever: search -> rerank -> adjacent context
  agents/                the four agents, each a plain typed function
  graph/
    state.py             the state passed between nodes
    workflow.py          LangGraph wiring and routing
  main.py                CLI: ingest / ask / chat / status
test_case_generation/    schema-2 benchmark generator + generated JSON dataset
evaluation/              evaluators, JSON checkpoints, combined result report
tests/                   offline test suite
```

---

## Known limits

- Qdrant runs in local mode under `data/processed/qdrant` and permits one writer
  at a time. Use a Qdrant server for concurrent workloads.
- The supported scope is English: the corpus, benchmark queries, reference
  answers, prompts, and reported system outputs are English. Non-English queries
  and answers have not been evaluated and should be treated as unsupported, even
  though the embedding model itself is multilingual.
- Docling can represent a table spanning a page boundary as multiple consecutive
  tables. They remain separate table chunks with correct page citations.
- The schema-2 audit uses deterministic evidence checks plus an LLM support
  classifier. The methodology report records this model-assisted step; manually
  review the benchmark before using it as a high-stakes leaderboard.
- End-to-end citation correctness verifies whether cited chunks belong to the
  audited full-gold set. It does not by itself grade prose completeness or every
  factual claim in the generated answer.
