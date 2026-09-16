# multi-agent-cyber-rag

A four-agent retrieval pipeline over authoritative cyber-security publications
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

Agent 2 has no LLM call on purpose: query formulation already happened in triage
(or in the verifier's rewrite), so generating again here would only add latency
and a second place for the wording to drift.

### Why hybrid retrieval

Dense retrieval alone loses exact artifacts - `SP 800-61r3`, CVE identifiers,
registry paths, tool names - which are precisely what a security question hinges
on. BM25 alone loses paraphrase. Both branches live in one Qdrant collection and
are fused with reciprocal rank fusion, then a cross-encoder reranks the survivors:
fusion buys recall, the cross-encoder buys precision.

### Adjacent-chunk context

After multi-query fusion selects the reranked seed chunks, retrieval loads nearby
chunks directly from Qdrant by their stable IDs. This adds surrounding context
without another embedding search and keeps every neighbor's own page and section
metadata available for citations. The default policy takes one chunk before and
one after each seed, stays within the same document and section, removes
duplicates, and caps the complete evidence set at 12 chunks.

```dotenv
NEIGHBOR_CHUNK_WINDOW=1
NEIGHBOR_MAX_TOTAL_CHUNKS=12
NEIGHBOR_SAME_SECTION_ONLY=true
```

Set `NEIGHBOR_CHUNK_WINDOW=0` to disable expansion. Set
`NEIGHBOR_SAME_SECTION_ONLY=false` when context is allowed to cross section
boundaries. `TOP_K_RERANK` still controls the number of relevance-ranked seed
chunks; `NEIGHBOR_MAX_TOTAL_CHUNKS` controls seeds plus adjacent context.

---

## Setup

```bash
conda activate hainn_rag
pip install -r requirements.txt
cp .env.example .env          # then edit - see "LLM backend" below
```

Put PDFs in `data/raw/`, then build the index:

```bash
python -m src.main ingest          # add --dump to also write chunks.jsonl
python -m src.main status          # config + how many chunks are indexed
```

Ingestion now follows this structure-preserving path:

```text
PDF -> Docling -> structured document -> loader
                  |-> prose -> ContentBlock -> semantic paragraph chunks --|
                  |-> table -> StructuredTableBlock                         |
                              (caption, headers, rows/cells, page, section)  |
                              -> header:value row chunks --------------------|-> Qdrant
```

Every ingest always saves Docling's full Markdown export under
`data/processed/markdown/<pdf-name>.md` for visual inspection. Markdown is the audit representation, not the retrieval
representation. Each structured table row is serialized as `header: value`
pairs with section, caption, and row identity. Oversized cells are split while
the row identifier is repeated. Qdrant also retains `chunk_index`,
`table_headers`, the original `table_rows`, row range, `chunk_type`, section, and
Docling page provenance. The stable per-document `chunk_index` is used to load
adjacent context after reranking.

Docling uses TableFormer `accurate` mode by default. Set `DOCLING_TABLE_MODE=fast`
when throughput matters more than table fidelity, or `DOCLING_DO_OCR=true` for
scanned/image-only PDFs. The first run downloads Docling and embedding models to
the local Hugging Face cache.

### LLM backend

Both backends are implemented behind one interface (`src/llm/`); switch with
`LLM_PROVIDER` in `.env`. Nothing else changes.

**Local, OpenAI-compatible (default)** - start a server, then point at it:

```bash
pip install -U vllm

CUDA_VISIBLE_DEVICES=2 VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve QuixiAI/Qwen3-30B-A3B-AWQ \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --quantization awq \
  --dtype half \
  --max-model-len 16384 \
  --enforce-eager
```

This starts the 4-bit AWQ build of Qwen3-30B-A3B instead of loading the full
BF16 checkpoint. The command uses GPU 2 and reserves 30% of its VRAM for the
embedding and reranker models. Change `CUDA_VISIBLE_DEVICES=2` if another GPU
should be used. FlashInfer sampling is disabled because FlashInfer 0.6.18 is
incompatible with the CUDA 12.0 toolchain on this host. Wait until vLLM prints
`Application startup complete`, then verify it with
`curl http://127.0.0.1:8000/v1/models`.

In a second terminal, with the same conda environment active, run:

```bash
./scripts/start_web.sh
```

```dotenv
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_MODEL=QuixiAI/Qwen3-30B-A3B-AWQ
```

Structured output is negotiated automatically: `response_format=json_schema`
first, then vLLM's `guided_json`, then schema-in-prompt with JSON extraction.
Whichever tier the server accepts is cached for the process.

**Claude API:**

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-5
```

Uses native structured outputs (`client.messages.parse`). `ANTHROPIC_FALLBACKS=true`
retries on a `refusal` stop reason with a sibling model - worth keeping on, because
legitimate defensive questions about ransomware occasionally trip a safety
classifier, and a refused triage call would otherwise stall the graph.

---

## Usage

```bash
python -m src.main ask "What should an organisation do during incident containment?"
python -m src.main ask "..." --trace     # show each agent's decision
python -m src.main chat                  # interactive
python -m src.main -v ask "..."          # detailed pipeline + LLM logs
```

Every request receives a correlation ID and logs pipeline start/end, state
input/output for each graph node, routing decisions, retrieval candidates,
question rewrites, LLM prompt/output, token usage, latency, and provider
fallbacks. File logging is always INFO-level; `-v` also displays it in the
terminal. Long values are truncated to `LOG_MAX_CHARS` (default `4000`). Set
`LOG_PAYLOADS=false` when questions or model responses may contain sensitive
data; metadata, timings, state transitions, and counts remain logged.

Logs are also persisted to `logs/cyber-rag.log`, rotated at 10 MiB with five
backups by default. Configure this with `LOG_FILE`, `LOG_MAX_BYTES`, and
`LOG_BACKUP_COUNT`.

Output is the answer, a citation table (`source / section / pages`), a confidence
level, and any caveats.

---

## Evaluation

```bash
python -m test_case_generation.generate_retrieval  # rebuild the 100-case benchmark
python -m evaluation.evaluate_retrieval            # retrieval-only metrics
python -m evaluation.evaluate_end_to_end            # full four-agent flow
python -m evaluation.evaluate_end_to_end --resume   # resume an interrupted full run
```

The schema-2 benchmark contains 100 questions in 34 paraphrase groups, balanced
between two retrieval conditions:

- `single_gold`: the atomic answer is supported by exactly one chunk in the
  corpus. This isolates ordinary exact retrieval.
- `multi_gold`: equivalent answer evidence occurs in two or more chunks, within
  one document, across documents, or both. Every fully supporting chunk is an
  acceptable gold; partially supporting chunks are tracked separately.

The generator first proposes atomic facts from source chunks, then audits every
corpus chunk with exact-span checks, lexical/semantic candidate discovery, and
an evidence validator. A primary chunk records provenance only; it receives no
special credit during scoring. See
[`test_case_generation/README.md`](test_case_generation/README.md)
for the reproducible construction procedure, schema, distributions, validation
rules, and limitations.

Retrieval is reported with Any-Gold Hit@K, full-gold Coverage/Recall@K, first-full-
gold MRR, and graded nDCG (full gold = 2, partial support = 1). Metrics are split
by `single_gold`/`multi_gold` and by duplicate scope so Hit@K cannot hide missed
equivalent chunks. The end-to-end run executes triage, multi-query hybrid
retrieval, neighbor expansion, verifier/rewrite retries, and answer generation;
it reports answer outcomes, latency, retrieval metrics, and citation correctness.
Citation correctness accepts a citation to any fully supporting gold chunk, not
only the chunk from which the question was originally generated.

The maintained Markdown artifacts are:

- [`evaluation/retrieval_evaluation.md`](evaluation/retrieval_evaluation.md):
  retrieval and end-to-end results.
- [`test_case_generation/README.md`](test_case_generation/README.md):
  how the test dataset was constructed and audited.

The baseline matrix is defined in
[`evaluation/baselines.json`](evaluation/baselines.json). It compares
Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3-14B, and the current
Qwen3-30B-A3B-AWQ while holding retrieval fixed. A second axis compares the
current BGE-M3 pair with Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B on the same
pre-chunked corpus. Run and refresh it with:

```bash
python -m evaluation.run_baselines list
python -m evaluation.run_baselines index --retrieval qwen3_0_6b
python -m evaluation.run_baselines retrieval --retrieval qwen3_0_6b
python -m evaluation.run_baselines report
```

Results are collected in the separate
[`evaluation/model_baseline_comparison.md`](evaluation/model_baseline_comparison.md).
All benchmark launchers hard-cap OMP, MKL, OpenBLAS, NumExpr, vecLib, and Rayon
to six CPU threads.
Unexecuted configurations stay marked as `planned`; the report never fills
missing measurements with estimates.

The older 20-query harness remains available as `python -m evaluation.evaluate`
for routing and abstention regressions, but it is not the retrieval benchmark.

### Verified after the Docling migration

| What | Result |
|---|---|
| Real corpus conversion | 3 Markdown files; 34 tables; 167 prose + 196 table-row chunks |
| Temporary Qdrant payload smoke test | text plus structured table metadata persisted |
| Schema-2 retrieval benchmark | 100 questions; 50 single-gold + 50 multi-gold; 34 fact groups |
| Offline unit + integration tests | 91 passed |

The committed benchmark and reports are tied to the recorded corpus fingerprint
and chunk IDs. Rebuild both after changing the parser, corpus, or chunk boundaries.
Retrieval evaluation needs the local index and embedding/reranker models;
end-to-end evaluation additionally needs the configured LLM backend.

---

## Testing

```bash
pytest tests/ -q
```

No GPU, API key, or network required: the LLM and PDF conversion are faked where
needed, and chunking uses a word tokenizer. The suite covers Docling block
extraction, persisted Markdown, structured headers/rows/cells, `header: value`
row chunks, oversized-cell splitting, chunk budgets, rank fusion, adjacent-chunk
boundaries/deduplication, citation integrity, and every path through
the graph (rejection, clarification, the rewrite-retry loop, budget exhaustion,
and refusal to answer). It also checks schema-2 dataset multiplicity, any-gold
retrieval scoring, full-gold coverage, graded nDCG, and any-full-gold citation
correctness.

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

## Design notes

**Document structure survives extraction.** The loader walks Docling's reading-order
document tree instead of flattening each PDF page. Section headers update the
active section, tables are exported as Markdown `TableItem`s, and every block
retains its Docling page provenance. Repeated page chrome is filtered only from
retrieval blocks; the saved Markdown remains the complete Docling export so it
can be audited directly.

**Citations are validated, not trusted.** The answer agent reports which passages
it used, but only the `[n]` markers actually present in the answer text are
turned into citations, and any marker outside the evidence range is stripped from
the text and recorded in `dropped_markers`. An answer with no resolvable marker is
downgraded to low confidence.

**The retry loop cannot spin.** The verifier must supply new queries for a retry
to happen; chunks it has already dismissed are excluded from the next round; the
budget is `MAX_RETRIEVAL_ROUNDS`; and when it runs out the answer agent still
answers from what was found, flagged `answered_partial` with an explicit caveat.

**Failures surface at the edge.** `ask` and `chat` preflight the index and the LLM
endpoint (`models.list` / `models.retrieve` - never a billed generation) and print
what to fix. Connection, auth and rate-limit errors are never treated as "this
constraint style is unsupported", so a server that is simply down fails once
instead of being probed three times.

**Qdrant runs in local mode**, storing to `data/processed/qdrant` with no server.
This takes a single-writer file lock, so one pipeline at a time - move to a
Qdrant server if you need concurrent readers.

## Known limits

- The corpus is English-only; BGE-M3 is multilingual, so Vietnamese queries will
  retrieve, but the answer agent is prompted in English and the sources are English.
- Docling can represent a table spanning a page boundary as multiple consecutive
  tables. They remain separate table chunks with correct page citations.
- The schema-2 audit uses deterministic evidence checks plus an LLM support
  classifier. The methodology report records this model-assisted step; manually
  review the benchmark before using it as a high-stakes leaderboard.
- End-to-end citation correctness verifies whether cited chunks belong to the
  audited full-gold set. It does not by itself grade prose completeness or every
  factual claim in the generated answer.
