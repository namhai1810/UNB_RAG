# Retrieval and end-to-end evaluation report

Generated at: `2026-09-15T11:37:57.501879+00:00`

## Evaluation setup

- Test cases: **100** from **34** atomic facts
- Dataset schema: **2.0**
- Retrieval: hybrid `Qwen/Qwen3-Embedding-0.6B` + BM25/RRF, followed by `Qwen/Qwen3-Reranker-0.6B`
- Embedding query prompt: `query`
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
| Any-Gold Hit@1 | 0.5600 |
| Any-Gold Hit@3 | 0.8400 |
| Any-Gold Hit@5 | 0.8800 |
| Any-Gold Hit@10 | 0.9500 |
| Gold Coverage/Recall@1 | 0.3967 |
| Gold Coverage/Recall@3 | 0.6742 |
| Gold Coverage/Recall@5 | 0.7558 |
| Gold Coverage/Recall@10 | 0.8650 |
| MRR | 0.7086 |
| nDCG@1 | 0.5600 |
| nDCG@3 | 0.6163 |
| nDCG@5 | 0.6491 |
| nDCG@10 | 0.6996 |

Mean retrieval latency per query: **0.262s**. Errors: **0**.

## Single-gold retrieval

Cases: **50**.

| Metric | Score |
|---|---:|
| Any-Gold Hit@1 | 0.5600 |
| Any-Gold Hit@3 | 0.8200 |
| Any-Gold Hit@5 | 0.8600 |
| Any-Gold Hit@10 | 0.9600 |
| Gold Coverage/Recall@1 | 0.5600 |
| Gold Coverage/Recall@3 | 0.8200 |
| Gold Coverage/Recall@5 | 0.8600 |
| Gold Coverage/Recall@10 | 0.9600 |
| MRR | 0.7072 |
| nDCG@1 | 0.5600 |
| nDCG@3 | 0.6982 |
| nDCG@5 | 0.7113 |
| nDCG@10 | 0.7599 |

## Multi-gold retrieval

Cases: **50**.

| Metric | Score |
|---|---:|
| Any-Gold Hit@1 | 0.5600 |
| Any-Gold Hit@3 | 0.8600 |
| Any-Gold Hit@5 | 0.9000 |
| Any-Gold Hit@10 | 0.9400 |
| Gold Coverage/Recall@1 | 0.2333 |
| Gold Coverage/Recall@3 | 0.5283 |
| Gold Coverage/Recall@5 | 0.6517 |
| Gold Coverage/Recall@10 | 0.7700 |
| MRR | 0.7100 |
| nDCG@1 | 0.5600 |
| nDCG@3 | 0.5344 |
| nDCG@5 | 0.5869 |
| nDCG@10 | 0.6393 |

## Retrieval by duplication scope

| Group | n | Any-Gold Hit@10 | Gold Coverage@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| cross_document | 14 | 0.9286 | 0.6786 | 0.7024 | 0.5900 |
| mixed | 15 | 0.9333 | 0.8222 | 0.8333 | 0.7259 |
| same_document | 21 | 0.9524 | 0.7937 | 0.6270 | 0.6104 |
| unique | 50 | 0.9600 | 0.9600 | 0.7072 | 0.7599 |

## Retrieval by paraphrase type

| Group | n | Any-Gold Hit@10 | Gold Coverage@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| lexical | 32 | 0.9375 | 0.8594 | 0.7240 | 0.7043 |
| natural | 34 | 0.9706 | 0.8995 | 0.7263 | 0.7312 |
| semantic | 34 | 0.9412 | 0.8358 | 0.6765 | 0.6636 |

## Retrieval by source

| Group | n | Any-Gold Hit@10 | Gold Coverage@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| NIST.CSWP.29.pdf | 33 | 0.9697 | 0.8838 | 0.6717 | 0.6768 |
| NIST.SP.800-61r3.pdf | 43 | 0.9302 | 0.8566 | 0.6886 | 0.6901 |
| StopRansomware-Guide 508.pdf | 24 | 0.9583 | 0.8542 | 0.7951 | 0.7480 |

## Retrieval by primary chunk type

| Group | n | Any-Gold Hit@10 | Gold Coverage@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| paragraph | 65 | 0.9538 | 0.8641 | 0.7062 | 0.6876 |
| table | 35 | 0.9429 | 0.8667 | 0.7131 | 0.7218 |

## Weakest single-gold cases

| ID | Scope | Gold count | Retrieved top 3 | Query |
|---|---|---:|---|---|
| ret_039 | unique | 1 | 6ca34d2a8b1c-00031, 6ca34d2a8b1c-00033, 6ca34d2a8b1c-00032 | What are the key functions that make up the incident response life cycle based on CSF 2.0? |
| ret_046 | unique | 1 | e5c72935263d-00040, e5c72935263d-00009, e5c72935263d-00029 | What is the recommended procedure for restoring data after a ransomware incident? |
| ret_050 | unique | 1 | 6ca34d2a8b1c-00000, 6ca34d2a8b1c-00002, 6ca34d2a8b1c-00007 | What is the main goal of the NIST SP 800-61r3 publication? |
| ret_037 | unique | 1 | 17684aea55c3-00067, 6ca34d2a8b1c-00031, 6ca34d2a8b1c-00033 | What are the six CSF 2.0 Functions in the incident response life cycle model? |
| ret_043 | unique | 1 | 6ca34d2a8b1c-00040, 6ca34d2a8b1c-00132, 6ca34d2a8b1c-00130 | What does the CSF 2.0 Community Profile recommend regarding protection mechanisms? |

## Weakest multi-gold cases

| ID | Scope | Gold count | Retrieved top 3 | Query |
|---|---|---:|---|---|
| ret_067 | mixed | 3 | 6ca34d2a8b1c-00081, 6ca34d2a8b1c-00148, 6ca34d2a8b1c-00040 | How does the CSF 2.0 recommend identifying the cause of an incident? |
| ret_081 | cross_document | 2 | 6ca34d2a8b1c-00099, 6ca34d2a8b1c-00097, 6ca34d2a8b1c-00100 | What does the ID.RA CSF element focus on? |
| ret_094 | same_document | 2 | 6ca34d2a8b1c-00070, 6ca34d2a8b1c-00077, 6ca34d2a8b1c-00100 | Which CSF component addresses supplier risk management? |
| ret_082 | cross_document | 2 | 6ca34d2a8b1c-00099, 6ca34d2a8b1c-00097, 6ca34d2a8b1c-00100 | What is the main goal of the ID.RA CSF element? |
| ret_083 | cross_document | 2 | 6ca34d2a8b1c-00097, 17684aea55c3-00033, 6ca34d2a8b1c-00100 | What is the purpose of the ID.RA in the CSF? |

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
