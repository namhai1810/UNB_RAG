# Retrieval and end-to-end evaluation report

Generated at: `2026-09-13T11:03:10.344329+00:00`

## Evaluation setup

- Test cases: **100** from **34** atomic facts
- Dataset schema: **2.0**
- Retrieval: hybrid BGE-M3 + BM25/RRF, followed by `BAAI/bge-reranker-v2-m3`
- Evaluated list: reranked seed chunks only; neighbor expansion is excluded
- K values: `1, 3, 5, 10`
- Device: `cuda`
- Corpus SHA-256: `3033ace0581cc7581c762063fb7a4854aead91ae1b9992a5baf386a29d489f58`

Every full-support gold chunk has relevance grade 2. Independently validated
partial-support chunks have grade 1. The primary gold records provenance only
and receives no ranking preference over another full gold.

`Any-Gold Hit@K` asks whether at least one answer-bearing chunk was found.
`Gold Coverage@K` asks what fraction of all repeated full-gold occurrences was
found. For single-gold cases these two metrics are identical.

## Overall retrieval metrics

| Metric | Score |
|---|---:|
| Any-Gold Hit@1 | 0.6800 |
| Any-Gold Hit@3 | 0.9200 |
| Any-Gold Hit@5 | 0.9400 |
| Any-Gold Hit@10 | 0.9800 |
| Gold Coverage/Recall@1 | 0.4925 |
| Gold Coverage/Recall@3 | 0.7608 |
| Gold Coverage/Recall@5 | 0.8108 |
| Gold Coverage/Recall@10 | 0.8975 |
| MRR | 0.7986 |
| nDCG@1 | 0.6833 |
| nDCG@3 | 0.7123 |
| nDCG@5 | 0.7322 |
| nDCG@10 | 0.7730 |

Mean retrieval latency per query: **0.467s**. Errors: **0**.

## Single-gold retrieval

Cases: **50**.

| Metric | Score |
|---|---:|
| Any-Gold Hit@1 | 0.7200 |
| Any-Gold Hit@3 | 0.9400 |
| Any-Gold Hit@5 | 0.9400 |
| Any-Gold Hit@10 | 0.9800 |
| Gold Coverage/Recall@1 | 0.7200 |
| Gold Coverage/Recall@3 | 0.9400 |
| Gold Coverage/Recall@5 | 0.9400 |
| Gold Coverage/Recall@10 | 0.9800 |
| MRR | 0.8200 |
| nDCG@1 | 0.7267 |
| nDCG@3 | 0.8339 |
| nDCG@5 | 0.8326 |
| nDCG@10 | 0.8556 |

## Multi-gold retrieval

Cases: **50**.

| Metric | Score |
|---|---:|
| Any-Gold Hit@1 | 0.6400 |
| Any-Gold Hit@3 | 0.9000 |
| Any-Gold Hit@5 | 0.9400 |
| Any-Gold Hit@10 | 0.9800 |
| Gold Coverage/Recall@1 | 0.2650 |
| Gold Coverage/Recall@3 | 0.5817 |
| Gold Coverage/Recall@5 | 0.6817 |
| Gold Coverage/Recall@10 | 0.8150 |
| MRR | 0.7772 |
| nDCG@1 | 0.6400 |
| nDCG@3 | 0.5907 |
| nDCG@5 | 0.6317 |
| nDCG@10 | 0.6904 |

## Retrieval by duplication scope

| Group | n | Any-Gold Hit@10 | Gold Coverage@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| cross_document | 14 | 1.0000 | 0.8214 | 0.7702 | 0.6546 |
| mixed | 15 | 1.0000 | 0.8389 | 0.9333 | 0.7841 |
| same_document | 21 | 0.9524 | 0.7937 | 0.6703 | 0.6473 |
| unique | 50 | 0.9800 | 0.9800 | 0.8200 | 0.8556 |

## Retrieval by paraphrase type

| Group | n | Any-Gold Hit@10 | Gold Coverage@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| lexical | 32 | 0.9688 | 0.9062 | 0.8021 | 0.7903 |
| natural | 34 | 1.0000 | 0.9338 | 0.8013 | 0.7856 |
| semantic | 34 | 0.9706 | 0.8529 | 0.7926 | 0.7441 |

## Retrieval by source

| Group | n | Any-Gold Hit@10 | Gold Coverage@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| NIST.CSWP.29.pdf | 33 | 0.9697 | 0.8813 | 0.8465 | 0.8011 |
| NIST.SP.800-61r3.pdf | 43 | 1.0000 | 0.9496 | 0.7547 | 0.7659 |
| StopRansomware-Guide 508.pdf | 24 | 0.9583 | 0.8264 | 0.8115 | 0.7471 |

## Retrieval by primary chunk type

| Group | n | Any-Gold Hit@10 | Gold Coverage@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| paragraph | 65 | 0.9692 | 0.8641 | 0.7986 | 0.7565 |
| table | 35 | 1.0000 | 0.9595 | 0.7986 | 0.8037 |

## Weakest single-gold cases

| ID | Scope | Gold count | Retrieved top 3 | Query |
|---|---|---:|---|---|
| ret_046 | unique | 1 | e5c72935263d-00030, e5c72935263d-00009, e5c72935263d-00037 | What is the recommended procedure for restoring data after a ransomware incident? |
| ret_007 | unique | 1 | e5c72935263d-00019, e5c72935263d-00036, e5c72935263d-00031 | What is a recommended practice for detecting malware through network traffic? |
| ret_012 | unique | 1 | e5c72935263d-00005, e5c72935263d-00031, e5c72935263d-00030 | What is a common method for ransomware to enter an organization? |
| ret_010 | unique | 1 | e5c72935263d-00007, e5c72935263d-00005, e5c72935263d-00023 | What are potential vectors for ransomware attacks? |
| ret_026 | unique | 1 | 6ca34d2a8b1c-00036, 6ca34d2a8b1c-00161, 6ca34d2a8b1c-00037 | How should an organization handle incident response responsibilities with external partners? |

## Weakest multi-gold cases

| ID | Scope | Gold count | Retrieved top 3 | Query |
|---|---|---:|---|---|
| ret_094 | same_document | 2 | 6ca34d2a8b1c-00075, 6ca34d2a8b1c-00070, 17684aea55c3-00023 | Which CSF component addresses supplier risk management? |
| ret_095 | same_document | 2 | 6ca34d2a8b1c-00020, 17684aea55c3-00007, 17684aea55c3-00009 | What NIST CSF guideline covers monitoring third-party risks? |
| ret_089 | same_document | 2 | 6ca34d2a8b1c-00136, 6ca34d2a8b1c-00141, 6ca34d2a8b1c-00137 | What should be monitored for signs of unauthorized activity? |
| ret_082 | cross_document | 2 | 6ca34d2a8b1c-00095, 6ca34d2a8b1c-00093, 6ca34d2a8b1c-00091 | What is the main goal of the ID.RA CSF element? |
| ret_091 | cross_document | 2 | 6ca34d2a8b1c-00020, 6ca34d2a8b1c-00150, 6ca34d2a8b1c-00039 | Which control in the incident response process focuses on classifying and prioritizing security events? |

## Retrieval interpretation

- Retrieval metrics evaluate reranked seed chunks, not generated answer text.
- Multi-gold Any-Gold Hit measures answerability; Gold Coverage separately
  measures whether the retriever found all repeated supporting occurrences.
- Scores from schema 2.0 should not be compared directly with the old schema 1.0
  baseline because the gold labels and relevance grades changed.

<!-- END_TO_END_EVALUATION_START -->
## End-to-end and citation evaluation

Generated at: `2026-09-13T11:33:08.474149+00:00`

- Full path: triage → multi-query retrieval and neighbor expansion → verifier/rewrite → answer generation
- LLM: `QuixiAI/Qwen3-30B-A3B-AWQ`; retrieval device: `cuda`
- Maximum retrieval rounds: **3**

A final citation is correct when it points to any independently full-supporting
gold chunk. Multi-gold answers need not cite every repeated occurrence.

### Overall end-to-end outcomes

| Metric | Score |
|---|---:|
| Triage in-scope rate | 100.00% |
| Answer produced | 99.00% |
| Fully answered | 72.00% |
| Grounded end-to-end success | 69.00% |
| Mean retrieval rounds | 1.770 |
| Errors | 0 |

Statuses: `answered`: 72, `answered_partial`: 27, `insufficient`: 1.

Grounded success requires `status == answered` and at least one full-gold final citation.

### Single- versus multi-gold end-to-end results

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| multi_gold | 50 | 76.00% | 74.00% | 98.00% | 54.70% | 1.700 | 16.827s |
| single_gold | 50 | 68.00% | 64.00% | 94.00% | 40.87% | 1.840 | 16.159s |

### End-to-end by duplication scope

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| cross_document | 14 | 92.86% | 85.71% | 92.86% | 48.15% | 1.429 | 12.349s |
| mixed | 15 | 86.67% | 86.67% | 100.00% | 57.14% | 1.600 | 14.499s |
| same_document | 21 | 57.14% | 57.14% | 100.00% | 56.25% | 1.952 | 21.474s |
| unique | 50 | 68.00% | 64.00% | 94.00% | 40.87% | 1.840 | 16.159s |

### Citation correctness

| Metric | Score |
|---|---:|
| Correct citations / all citations | 111 / 232 = 47.84% |
| Mean precision among cited answers | 63.64% |
| Full-gold citation hit rate | 96.00% |
| Primary-gold citation hit rate | 87.00% |
| Citation-marker integrity | 98.99% |
| Uncited-answer rate | 0.00% |

Primary-gold hit is diagnostic only: every validated full alternative is equally correct.
Gold-chunk precision is stricter than source matching but is not a claim-entailment judge.

### End-to-end latency

| Metric | Time |
|---|---:|
| Mean | 16.493s |
| Median | 12.878s |
| P95 | 26.859s |
| Maximum | 119.555s |
| Mean excluding first query | 16.318s |

Latency covers the complete graph. The first query is separated because it may
include lazy model and CUDA initialization. The main repeated cost is the number
of verifier/retrieval rounds:

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54 | 100.00% | 98.15% | 98.15% | 50.41% | 1.000 | 11.054s |
| 2 | 15 | 100.00% | 100.00% | 100.00% | 47.37% | 2.000 | 15.909s |
| 3 | 31 | 9.68% | 3.23% | 90.32% | 43.84% | 3.000 | 26.249s |

The maximum is reported separately so a transient model stall cannot be mistaken
for typical latency; compare it with the median and P95.

### End-to-end by paraphrase type

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| lexical | 32 | 71.88% | 68.75% | 96.88% | 55.71% | 1.781 | 18.912s |
| natural | 34 | 76.47% | 76.47% | 100.00% | 42.53% | 1.618 | 14.333s |
| semantic | 34 | 67.65% | 61.76% | 91.18% | 46.67% | 1.912 | 16.376s |

### End-to-end by source

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| NIST.CSWP.29.pdf | 33 | 69.70% | 69.70% | 100.00% | 52.44% | 1.727 | 16.196s |
| NIST.SP.800-61r3.pdf | 43 | 74.42% | 67.44% | 93.02% | 43.30% | 1.860 | 15.822s |
| StopRansomware-Guide 508.pdf | 24 | 70.83% | 70.83% | 95.83% | 49.06% | 1.667 | 18.103s |

### End-to-end by primary chunk type

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| paragraph | 65 | 69.23% | 66.15% | 95.38% | 45.51% | 1.738 | 17.284s |
| table | 35 | 77.14% | 74.29% | 97.14% | 52.63% | 1.829 | 15.024s |

### Weakest single-gold end-to-end cases

| ID | Status | Scope | Citation precision | Gold citation | Latency | Query |
|---|---|---|---:|---:|---:|---|
| ret_023 | insufficient | unique | — | no | 17.910s | Under what conditions is TLP:CLEAR applicable? |
| ret_007 | answered | unique | 0.00% | no | 19.206s | What is a recommended practice for detecting malware through network traffic? |
| ret_026 | answered | unique | 0.00% | no | 12.123s | How should an organization handle incident response responsibilities with external partners? |
| ret_029 | answered_partial | unique | 20.00% | yes | 26.586s | How do the CSF Tiers describe the evolution of risk management approaches? |
| ret_027 | answered_partial | unique | 25.00% | yes | 26.274s | What is required for third-party involvement in incident response activities? |

### Weakest multi-gold end-to-end cases

| ID | Status | Scope | Citation precision | Gold citation | Latency | Query |
|---|---|---|---:|---:|---:|---|
| ret_079 | answered | cross_document | 0.00% | no | 20.833s | How should companies handle cybersecurity incidents according to the guidelines? |
| ret_053 | answered_partial | mixed | 16.67% | yes | 24.403s | What is required for maintaining software and system inventories within an organization? |
| ret_063 | answered_partial | same_document | 33.33% | yes | 25.064s | What incident response framework should organizations choose based on their needs? |
| ret_077 | answered_partial | same_document | 33.33% | yes | 23.257s | What is the role of federal agencies in responding to cyber incidents affecting organizations? |
| ret_056 | answered_partial | mixed | 50.00% | yes | 23.198s | What steps does a company take to update its cybersecurity approach using what it has learned? |
<!-- END_TO_END_EVALUATION_END -->
