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

## Interpretation notes

- Recall@K measures the fraction of acceptable gold chunks present in the first K results.
- MRR rewards putting the first acceptable supporting chunk near the top.
- nDCG@K additionally rewards ranking the primary source chunk above overlap alternatives.
- These metrics evaluate retrieval only; they do not grade the generated answer text.
