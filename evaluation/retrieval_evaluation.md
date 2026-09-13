# Retrieval evaluation report

Generated at: `2026-09-12T10:58:04.021567+00:00`

## Evaluation setup

- Test cases: **100** (34 source chunks)
- Retrieval: hybrid BGE-M3 + BM25/RRF, followed by `BAAI/bge-reranker-v2-m3`
- Evaluated list: reranked seed chunks only; neighbor expansion is excluded
- K values: `1, 3, 5, 10`
- Device: `cuda:1`
- Corpus SHA-256: `3033ace0581cc7581c762063fb7a4854aead91ae1b9992a5baf386a29d489f58`
- Dataset schema: `1.0`

`primary_gold_chunk_id` has relevance grade 2 and overlap/alternative gold chunks
have grade 1 for nDCG. Recall uses all acceptable gold chunk IDs. MRR uses the
rank of the first acceptable chunk. Failed retrievals receive zero.

## Overall metrics

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

Mean latency per query: **1.076s**. Errors: **0**.

## By paraphrase type

| Group | n | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| lexical | 32 | 0.9219 | 0.8316 | 0.8394 |
| natural | 34 | 0.9265 | 0.7721 | 0.7970 |
| semantic | 34 | 0.9706 | 0.8290 | 0.8515 |

## By source

| Group | n | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| NIST.CSWP.29.pdf | 33 | 0.8939 | 0.6049 | 0.6488 |
| NIST.SP.800-61r3.pdf | 44 | 0.9432 | 0.9072 | 0.9060 |
| StopRansomware-Guide 508.pdf | 23 | 1.0000 | 0.9203 | 0.9405 |

## By chunk type

| Group | n | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| paragraph | 67 | 0.9403 | 0.7868 | 0.8104 |
| table | 33 | 0.9394 | 0.8586 | 0.8671 |

## Ten weakest cases

| ID | Style | Gold chunk | Retrieved top 3 | Query |
|---|---|---|---|---|
| ret_022 | lexical | 17684aea55c3-00055 | 6ca34d2a8b1c-00161, 6ca34d2a8b1c-00020, 6ca34d2a8b1c-00176 | Which part of the NIST CSF covers coordinating response activities with stakeholders? |
| ret_024 | natural | 17684aea55c3-00055 | 6ca34d2a8b1c-00020, 6ca34d2a8b1c-00162, 17684aea55c3-00011 | What CSF function includes notifying stakeholders about cybersecurity incidents? |
| ret_007 | lexical | 17684aea55c3-00074 | 17684aea55c3-00017, 17684aea55c3-00009, 17684aea55c3-00057 | What does the CSF Tier describe? |
| ret_023 | semantic | 17684aea55c3-00055 | 6ca34d2a8b1c-00020, 6ca34d2a8b1c-00161, 17684aea55c3-00023 | What component of the NIST Cybersecurity Framework involves sharing incident information with stakeholders? |
| ret_008 | semantic | 17684aea55c3-00074 | 17684aea55c3-00060, 17684aea55c3-00061, 17684aea55c3-00059 | How is the level of an organization's cybersecurity risk management practices described? |
| ret_009 | natural | 17684aea55c3-00074 | 17684aea55c3-00017, 17684aea55c3-00057, 17684aea55c3-00009 | What does the CSF Tier indicate about an organization's cybersecurity practices? |
| ret_055 | lexical | 17684aea55c3-00059 | 17684aea55c3-00017, 17684aea55c3-00074, 17684aea55c3-00009 | What characterizes the Risk Informed tier of the CSF Tiers according to the NIST document? |
| ret_057 | natural | 17684aea55c3-00059 | 17684aea55c3-00061, 17684aea55c3-00017, 17684aea55c3-00060 | What does the Risk Informed tier indicate about an organization's approach to cybersecurity risk management? |
| ret_068 | semantic | 17684aea55c3-00025 | 6ca34d2a8b1c-00020, 17684aea55c3-00010, 17684aea55c3-00009 | Which part of the CSF outlines the Functions, Categories, and Subcategories? |
| ret_076 | lexical | 6ca34d2a8b1c-00025 | 6ca34d2a8b1c-00032, 6ca34d2a8b1c-00033, 6ca34d2a8b1c-00031 | What are the phases of the incident response life cycle model? |

<!-- END_TO_END_EVALUATION_START -->
## End-to-end and citation evaluation

Generated at: `2026-09-13T06:02:13.862712+00:00`

- Test cases: **100**, using the same grounded questions and gold chunks as the retrieval evaluation
- Full path: triage → multi-query retrieval and neighbor expansion → verifier/rewrite → answer generation
- LLM: `QuixiAI/Qwen3-30B-A3B-AWQ`; retrieval device: `cuda`
- Maximum retrieval rounds: **3**

A citation is counted as correct when its `chunk_id` is one of the case's
`acceptable_gold_chunk_ids`. This is exact gold-chunk correctness, not merely a
source-name match. It does not by itself prove that every generated claim is
entailed by the cited passage.

### End-to-end outcomes

| Metric | Score |
|---|---:|
| Triage in-scope rate | 98.00% |
| Answer produced (`answered` or `answered_partial`) | 98.00% |
| Fully answered | 72.00% |
| Grounded end-to-end success | 63.00% |
| Mean retrieval rounds | 1.710 |
| Errors | 0 |

Statuses: `answered`: 72, `answered_partial`: 26, `rejected`: 2.

`Grounded end-to-end success` requires both `status == answered` and at least
one final citation to an acceptable gold chunk.

### Citation correctness

| Metric | Score |
|---|---:|
| Correct citations / all citations (micro precision) | 89 / 192 = 46.35% |
| Mean precision among cited answers (macro) | 62.75% |
| Gold-citation hit rate | 85.00% |
| Primary-gold citation hit rate | 81.00% |
| Citation-marker integrity | 95.92% |
| Uncited-answer rate | 1.02% |

### End-to-end latency

| Metric | Time |
|---|---:|
| Mean | 39.888s |
| Median | 28.703s |
| P95 | 63.407s |
| Maximum | 269.921s |
| Mean excluding first query | 39.831s |

This latency covers the complete graph and is therefore not comparable to the
retrieval-only latency as if they measured the same work. The first query is
reported separately because it may include lazy model and CUDA initialization.

### End-to-end by paraphrase type

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| lexical | 32 | 68.75% | 59.38% | 87.50% | 57.14% | 31.635s |
| natural | 34 | 67.65% | 58.82% | 79.41% | 36.71% | 41.329s |
| semantic | 34 | 79.41% | 70.59% | 88.24% | 50.00% | 46.215s |

### End-to-end by source

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| NIST.CSWP.29.pdf | 33 | 84.85% | 66.67% | 78.79% | 45.00% | 47.932s |
| NIST.SP.800-61r3.pdf | 44 | 56.82% | 50.00% | 81.82% | 43.82% | 33.239s |
| StopRansomware-Guide 508.pdf | 23 | 82.61% | 82.61% | 100.00% | 53.49% | 41.068s |

### End-to-end by chunk type

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| paragraph | 67 | 74.63% | 62.69% | 80.60% | 40.44% | 36.515s |
| table | 33 | 66.67% | 63.64% | 93.94% | 60.71% | 46.737s |

### Ten weakest end-to-end cases

| ID | Status | Citation precision | Gold citation | Latency | Query |
|---|---|---:|---:|---:|---|
| ret_054 | answered | — | no | 25.063s | What must organizations do with suppliers and third parties regarding incident response as outlined in the GV.SC controls? |
| ret_060 | rejected | — | no | 6.001s | What does the PR.AA element say about asset access? |
| ret_099 | rejected | — | no | 8.717s | Who is the Community Profile designed for? |
| ret_004 | answered | 0.00% | no | 13.612s | How does the NIST Cybersecurity Framework (CSF) 2.0 assist organizations in managing cybersecurity incidents? |
| ret_006 | answered_partial | 0.00% | no | 67.670s | What are the key advantages of using the NIST Cybersecurity Framework 2.0 for handling cybersecurity incidents? |
| ret_009 | answered | 0.00% | no | 30.917s | What does the CSF Tier indicate about an organization's cybersecurity practices? |
| ret_022 | answered | 0.00% | no | 20.953s | Which part of the NIST CSF covers coordinating response activities with stakeholders? |
| ret_023 | answered | 0.00% | no | 22.577s | What component of the NIST Cybersecurity Framework involves sharing incident information with stakeholders? |
| ret_024 | answered | 0.00% | no | 21.820s | What CSF function includes notifying stakeholders about cybersecurity incidents? |
| ret_050 | answered | 0.00% | no | 44.303s | How does the NIST SP 800-61r3 document aim to improve an organization's incident response capabilities through the CSF 2.0 Functions? |
<!-- END_TO_END_EVALUATION_END -->

## Interpretation notes

- Recall@K measures the fraction of acceptable gold chunks present in the first K results.
- MRR rewards putting the first acceptable supporting chunk near the top.
- nDCG@K additionally rewards ranking the primary source chunk above overlap alternatives.
- These metrics evaluate retrieval only; they do not grade the generated answer text.
