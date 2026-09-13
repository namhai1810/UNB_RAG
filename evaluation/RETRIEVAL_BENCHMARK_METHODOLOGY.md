# Grounded Retrieval Benchmark: Construction and Evaluation Methodology

## 1. Purpose

This benchmark was created to evaluate whether the cyber-security RAG system
retrieves the source chunks that actually contain the information needed to
answer a question.

It complements the existing Type 1 test suite:

- **Type 1 — Triage classification:** evaluates whether a question is routed as
  `in_scope`, `ambiguous`, or `out_of_scope`.
- **Type 2 — Grounded retrieval:** evaluates whether the retrieval pipeline
  returns the expected supporting chunk for an answerable question.

The Type 2 benchmark evaluates retrieval independently from answer generation.
This separation makes it possible to determine whether a failure comes from
retrieval or from later RAG stages such as evidence verification, answer
generation, or citation rendering.

## 2. Source Corpus

The benchmark was generated from the same processed corpus used by the RAG
index. It contains the following English-language publications:

- `NIST.CSWP.29.pdf` — NIST Cybersecurity Framework 2.0
- `NIST.SP.800-61r3.pdf` — Incident Response Recommendations and
  Considerations for Cybersecurity Risk Management
- `StopRansomware-Guide 508.pdf` — CISA/FBI StopRansomware Guide

The processed corpus contains 363 paragraph and table chunks. Chunk identifiers
are stable within a fixed chunking configuration and corpus version.

The generated dataset records the SHA-256 hash of the source corpus:

```text
3033ace0581cc7581c762063fb7a4854aead91ae1b9992a5baf386a29d489f58
```

This hash prevents results from different corpus versions from being compared
as if they came from the same retrieval task.

## 3. Dataset Construction

### 3.1 Chunk selection

Candidate chunks were filtered before question generation. A chunk had to:

- contain at least 100 tokens for the final benchmark run;
- contain at least 300 normalized characters;
- contain a valid `chunk_id` and non-empty text;
- represent substantive cyber-security content rather than publication
  metadata.

The following kinds of sections were excluded:

- tables of contents and keyword pages;
- acknowledgements and references;
- citation instructions;
- revision histories and change records;
- publication-series descriptions;
- endorsement disclaimers;
- cover-page publication metadata.

Eligible chunks were sampled deterministically across `(source, chunk_type)`
strata. This avoided allowing one large publication or one representation type
to dominate the benchmark. The final generation used random seed `43`.

### 3.2 Question and answer generation

The configured local model, `QuixiAI/Qwen3-30B-A3B-AWQ`, generated the test
cases from selected English chunks.

For each chunk, the model was required to:

1. Identify one useful, self-contained fact or procedure explicitly stated in
   the chunk.
2. Produce one concise English reference answer.
3. Copy one to three short, verbatim English evidence spans from the chunk.
4. Produce two or three English questions that can all be answered by the same
   reference answer.

The questions cover three paraphrase styles:

- **Lexical:** preserves important terminology from the source while avoiding
  direct sentence copying.
- **Semantic:** changes wording and sentence structure substantially.
- **Natural:** resembles a realistic user question and may use common
  cyber-security terminology conversationally.

The final dataset contains exactly 100 questions from 34 source chunks:

| Paraphrase type | Questions |
|---|---:|
| Lexical | 32 |
| Semantic | 34 |
| Natural | 34 |
| **Total** | **100** |

Source and chunk-type coverage is:

| Source | Questions |
|---|---:|
| `NIST.CSWP.29.pdf` | 33 |
| `NIST.SP.800-61r3.pdf` | 44 |
| `StopRansomware-Guide 508.pdf` | 23 |

| Chunk type | Questions |
|---|---:|
| Paragraph | 67 |
| Table | 33 |

### 3.3 Test-case schema

The dataset is stored as a flat list so each question can be evaluated as an
independent retrieval request. Questions derived from the same source chunk
share a `group_id`.

Example:

```json
{
  "id": "ret_001",
  "group_id": "ret_group_001",
  "type": "retrieval_grounded",
  "query": "What action can be taken against a new firewall rule allowing open traffic?",
  "paraphrase_type": "lexical",
  "reference_answer": "Delete the new firewall rule after confirming it is malicious or unnecessary.",
  "primary_gold_chunk_id": "example-document-00042",
  "acceptable_gold_chunk_ids": [
    "example-document-00042"
  ],
  "gold": {
    "doc_id": "example-document",
    "source": "StopRansomware-Guide 508.pdf",
    "section": "For cloud environments",
    "page_start": 24,
    "page_end": 24,
    "chunk_type": "paragraph",
    "content_sha256": "...",
    "evidence_spans": [
      "Delete any new firewall rules that allow open traffic"
    ]
  }
}
```

The reference answer is included for future answer-quality evaluation. The
retrieval metrics in this report use the gold chunk labels, not exact string
matching against the reference answer.

## 4. Gold-Chunk Design

Using exactly one acceptable chunk would create false failures because the
corpus uses overlapping chunks. The same supporting sentence may therefore
appear in the source chunk and an adjacent chunk.

Each case distinguishes between:

- `primary_gold_chunk_id`: the chunk from which the question was generated;
- `acceptable_gold_chunk_ids`: the primary chunk plus any other chunk containing
  the same verbatim supporting evidence.

Alternative gold chunks were discovered deterministically by searching the
processed corpus for the validated evidence spans. In the final dataset, 21
questions have more than one acceptable gold chunk.

For graded ranking evaluation, the primary chunk receives relevance grade 2 and
overlap alternatives receive relevance grade 1.

## 5. Automated Quality Controls

A generated chunk group was accepted only if all of the following checks
passed:

- the returned `chunk_id` matched the requested source chunk;
- exactly two or three questions were returned as requested;
- all required paraphrase styles were present exactly once;
- every question was non-empty, unique, and ended with a question mark;
- questions did not refer to an unspecified “chunk,” “passage,” “text,” or
  “document”;
- the reference answer was non-empty;
- each evidence span had sufficient length;
- every evidence span appeared verbatim in the gold chunk;
- no question duplicated another question already accepted into the dataset.

Invalid groups were retried up to five times. The output file was written only
after all 100 cases passed validation, preventing partial datasets from being
mistaken for completed runs.

The final artifact was also audited for:

- exactly 100 unique case IDs;
- exactly 100 unique English questions;
- English-only questions and reference answers;
- 34 unique primary gold chunks;
- valid primary-to-acceptable gold membership;
- absence of excluded metadata sections;
- exact evidence-span grounding.

## 6. Retrieval Pipeline Under Evaluation

Each test question was passed directly to the production retrieval stack:

1. BGE-M3 dense query encoding.
2. BM25 sparse query encoding.
3. Dense and sparse candidate retrieval from Qdrant.
4. Reciprocal Rank Fusion (RRF).
5. Cross-encoder reranking with `BAAI/bge-reranker-v2-m3`.

Only reranked **seed chunks** were evaluated. Adjacent chunks added by neighbor
expansion were deliberately excluded. Otherwise, a semantically incorrect seed
could retrieve a nearby gold chunk and be counted as a successful direct hit.

The evaluated cutoffs were `K = 1, 3, 5, 10`.

## 7. Evaluation Metrics

### 7.1 Recall@K

For query `q`, let `G_q` be the set of acceptable gold chunks and `R_q^K` be
the first `K` retrieved chunks:

> **Recall@K(q)** = |G_q ∩ R_q^K| / |G_q|

The reported score is the mean Recall@K across all test questions.

### 7.2 Mean Reciprocal Rank

Let `r_q` be the rank of the first acceptable gold chunk for query `q`:

> **RR(q)** = 1 / r_q, if a gold chunk is retrieved; otherwise **RR(q) = 0**.

Mean Reciprocal Rank is:

> **MRR** = [Σ RR(q)] / |Q|, for all q ∈ Q

MRR emphasizes how quickly the first usable supporting chunk appears.

### 7.3 nDCG@K

Discounted Cumulative Gain uses graded relevance:

> **DCG@K** = Σ(i = 1 ... K) [(2^rel_i − 1) / log₂(i + 1)]

where the primary gold chunk has relevance 2, an acceptable overlap chunk has
relevance 1, and all other chunks have relevance 0.

The normalized score is:

> **nDCG@K** = DCG@K / IDCG@K

Unlike MRR, nDCG also rewards ranking the primary source chunk above weaker
overlap alternatives.

## 8. Final Results

The evaluation completed all 100 questions with zero runtime errors.

| Metric | Score |
|---|---:|
| Recall@1 | 0.6150 |
| Recall@3 | 0.8550 |
| Recall@5 | 0.8950 |
| Recall@10 | 0.9400 |
| MRR | 0.8105 |
| nDCG@1 | 0.7000 |
| nDCG@3 | 0.7999 |
| nDCG@5 | 0.8150 |
| nDCG@10 | 0.8291 |

Mean measured retrieval latency was 1.076 seconds per query on `cuda:1`.

### 8.1 Results by paraphrase type

| Type | n | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Lexical | 32 | 0.9219 | 0.8316 | 0.8394 |
| Natural | 34 | 0.9265 | 0.7721 | 0.7970 |
| Semantic | 34 | 0.9706 | 0.8290 | 0.8515 |

### 8.2 Results by source

| Source | n | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| `NIST.CSWP.29.pdf` | 33 | 0.8939 | 0.6049 | 0.6488 |
| `NIST.SP.800-61r3.pdf` | 44 | 0.9432 | 0.9072 | 0.9060 |
| `StopRansomware-Guide 508.pdf` | 23 | 1.0000 | 0.9203 | 0.9405 |

### 8.3 Results by chunk type

| Chunk type | n | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Paragraph | 67 | 0.9403 | 0.7868 | 0.8104 |
| Table | 33 | 0.9394 | 0.8586 | 0.8671 |

The system achieves high Recall@10, but Recall@1 is substantially lower. This
indicates that the correct evidence is usually present in the candidate set but
is not always ranked first. The largest source-level weakness is
`NIST.CSWP.29.pdf`, whose sections often contain similar CSF terminology and
closely related Tier/Core descriptions.

## 9. Challenges and Mitigations

### 9.1 Questions that were too close to the source

Synthetic generation can copy distinctive source wording, making retrieval
unrealistically easy.

**Mitigation:** Three explicit paraphrase styles were generated and reported
separately. Semantic and natural variants reduce dependence on lexical overlap.

### 9.2 Questions broader than their reference answers

An early generation attempt produced a short answer such as a Tier label while
asking broader questions about the Tier's risk-management properties.

**Mitigation:** The generation prompt now requires the reference answer to fully
answer every variant and requires evidence spans to support the complete answer,
not only a label or heading.

### 9.3 Non-question model outputs

Some early semantic variants were declarative sentences rather than questions.

**Mitigation:** A test case is now rejected unless every query ends with a
question mark and passes the remaining structural checks.

### 9.4 Low-value publication metadata

Random corpus sampling initially selected citation instructions, change logs,
endorsement disclaimers, and publication metadata. These are retrievable facts,
but they do not represent the target cyber-security use cases.

**Mitigation:** Section-level filters remove these categories before sampling.

### 9.5 Context-dependent wording

Questions such as “What does the text recommend?” cannot be retrieved reliably
because they are not self-contained.

**Mitigation:** The prompt and validator reject unspecified references to the
chunk, passage, text, or document.

### 9.6 Overlapping chunks

Chunk overlap means multiple chunk IDs may contain the same supporting sentence.
Exact matching against only the generation chunk would incorrectly penalize a
valid retrieval.

**Mitigation:** The benchmark records a primary gold chunk and automatically
discovers acceptable overlap chunks using validated evidence spans.

### 9.7 Neighbor-expansion leakage

The production pipeline adds adjacent context after seed retrieval. Counting a
gold chunk reached only through adjacency could hide a semantic retrieval miss.

**Mitigation:** The benchmark evaluates seed rankings before neighbor expansion.

### 9.8 Evaluation runtime and shared GPU resources

CPU evaluation was impractically slow for the cross-encoder reranker, while the
available GPUs were shared with other workloads.

**Mitigation:** A GPU with sufficient available memory was selected explicitly.
The final run completed without out-of-memory or retrieval errors.

## 10. Reproduction

Generate the final dataset:

```bash
python -m test_case_generation.generate_retrieval \
  --count 100 \
  --batch-size 4 \
  --max-attempts 5 \
  --min-tokens 100 \
  --seed 43
```

Evaluate retrieval:

```bash
DEVICE=cuda:1 python -m evaluation.evaluate_retrieval --k 1 3 5 10
```

Run all automated tests:

```bash
python -m pytest -q
```

The completed implementation passed all 82 project tests.

## 11. Artifacts

- Generator: `test_case_generation/generate_retrieval.py`
- Generated dataset: `test_case_generation/generated_retrieval_test_cases.json`
- Evaluator: `evaluation/evaluate_retrieval.py`
- Per-query rankings: `evaluation/retrieval_results.json`
- Generated metric report: `evaluation/retrieval_evaluation.md`
- Generator tests: `tests/test_retrieval_case_generation.py`
- Metric tests: `tests/test_retrieval_evaluation.py`

The repository currently ignores `*.json` files. The dataset and detailed
results therefore exist locally but require an explicit Git ignore exception or
forced add if they need to be version-controlled.

## 12. Limitations and Recommended Next Steps

This benchmark is suitable for deterministic retrieval regression testing, but
it has several limitations:

- Questions are synthetic and generated from the indexed corpus, so they may be
  cleaner than real user queries.
- Only 34 of the 363 chunks are represented.
- The benchmark evaluates retrieval, not reference-answer correctness,
  faithfulness, citation correctness, or verifier behavior.
- Evidence-span matching identifies duplicated overlap text but may not find a
  differently worded chunk that is also sufficient to answer the question.
- Results can be overfit if the same benchmark is repeatedly used to tune
  retrieval parameters.

Recommended follow-up work:

1. Freeze this dataset as a held-out regression test and create a separate
   development split for parameter tuning.
2. Add human-written queries from realistic usage logs after privacy review.
3. Add hard negatives from nearby chunks with similar CSF terminology.
4. Add unanswerable in-domain questions to measure evidence-verifier abstention.
5. Add end-to-end metrics for answer correctness, faithfulness, and citation
   accuracy while retaining the retrieval-only report for diagnosis.
