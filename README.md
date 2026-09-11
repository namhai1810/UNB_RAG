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
| 2. Retrieval | Hybrid search per query, reciprocal rank fusion across queries | no - deterministic |
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

First run downloads BGE-M3 (~2.3 GB) and bge-reranker-v2-m3 (~2.2 GB) to
`~/.cache/huggingface`. Indexing 3 NIST/CISA PDFs takes well under a minute on a
GPU and produces ~125 chunks.

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
python -m evaluation.evaluate                 # all 20 queries -> results.csv
python -m evaluation.evaluate --ids q15 q19   # a subset
```

`evaluation/test_queries.json` covers in-scope lookups, cross-document synthesis,
and four deliberate traps:

- **q15** asks for a statistic that does not exist in the corpus - the verifier must
  refuse rather than invent one.
- **q16, q17** are ambiguous and must be routed to clarification, not retrieval.
- **q19** requests operational attack capability and must be rejected.
- **q11, q20** are attacker-behaviour questions with defensive intent and must
  **not** be rejected - they are the over-refusal probe.

Scored metrics: triage accuracy, retrieval hit rate against the documents that
hold the answer, source precision, citation integrity, keyword recall, and
abstention correctness. It is deliberately not an LLM judge - the point is a
fast deterministic regression signal after changing a chunk size or a prompt.

### Measured so far

| What | Result |
|---|---|
| Retrieval hit@6 against expected source documents | **15/15 (100%)** |
| Section labels resolved from PDF bookmarks | 100% / 100% / 96% per document |
| Unit + integration tests | 46 passed |

The end-to-end agent metrics need a configured LLM backend; the graph itself is
covered offline by `tests/test_workflow.py` using a scripted fake model.

---

## Testing

```bash
pytest tests/ -q
```

No GPU, API key, or network required: the LLM is faked and chunking uses a word
tokenizer. The suite covers JSON extraction from messy model output, header/footer
stripping, chunk budgets and section boundaries, rank fusion, citation integrity,
and every path through the graph (rejection, clarification, the rewrite-retry
loop, budget exhaustion, and refusal to answer).

---

## Layout

```
src/
  config.py              all settings, overridable from .env
  llm/                   provider abstraction: base, anthropic, openai-compatible
  ingestion/
    loaders.py           PyMuPDF -> pages with section + page metadata, chrome stripped
    chunking.py          token-aware chunking, section-aware boundaries
    indexing.py          Qdrant collection (dense + sparse) and hybrid query
  retrieval/
    dense.py             BGE-M3 bi-encoder
    sparse.py            BM25 sparse vectors (FastEmbed)
    reranker.py          bge-reranker-v2-m3 cross-encoder
    __init__.py          HybridRetriever: search -> rerank -> Evidence
  agents/                the four agents, each a plain typed function
  graph/
    state.py             the state passed between nodes
    workflow.py          LangGraph wiring and routing
  main.py                CLI: ingest / ask / chat / status
evaluation/              test queries, harness, results.csv
tests/                   offline test suite
```

---

## Design notes

**Section labels come from the text, not the page.** PyMuPDF's bookmark map is
page-granular, so a section starting halfway down a page mislabels everything
above it. The chunker instead detects bookmark titles line by line in the
extracted text (100% / 100% / 96% of bookmarks recovered across the three
documents), and falls back to the page map only after it has disagreed for two
consecutive pages - which means a heading was genuinely missed.

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
- One long heading in the #StopRansomware Guide wraps across two lines and is not
  recovered as a section label; the page number in its citations is still correct.
- Evaluation scores keyword presence, not factual correctness. Add an LLM judge on
  top of `results.csv` if you need semantic scoring.
