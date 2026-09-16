

<!-- END_TO_END_EVALUATION_START -->
## End-to-end and citation evaluation

Generated at: `2026-09-15T16:40:52.554340+00:00`

- Full path: triage → multi-query retrieval and neighbor expansion → verifier/rewrite → answer generation
- LLM: `Qwen/Qwen2.5-7B-Instruct`; retrieval device: `cuda:2`
- Maximum retrieval rounds: **3**

A final citation is correct when it points to any independently full-supporting
gold chunk. Multi-gold answers need not cite every repeated occurrence.

### Overall end-to-end outcomes

| Metric | Score |
|---|---:|
| Triage in-scope rate | 97.00% |
| Answer produced | 50.00% |
| Fully answered | 49.00% |
| Grounded end-to-end success | 47.00% |
| Mean retrieval rounds | 2.010 |
| Errors | 0 |

Statuses: `answered`: 49, `answered_partial`: 1, `insufficient`: 47, `needs_clarification`: 3.

Grounded success requires `status == answered` and at least one full-gold final citation.

### Single- versus multi-gold end-to-end results

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| multi_gold | 50 | 54.00% | 52.00% | 52.00% | 50.00% | 1.920 | 10.599s |
| single_gold | 50 | 44.00% | 42.00% | 42.00% | 42.00% | 2.100 | 12.518s |

### End-to-end by duplication scope

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| cross_document | 14 | 42.86% | 42.86% | 42.86% | 40.00% | 1.929 | 10.530s |
| mixed | 15 | 86.67% | 86.67% | 86.67% | 52.78% | 1.267 | 8.699s |
| same_document | 21 | 38.10% | 33.33% | 33.33% | 53.33% | 2.381 | 12.001s |
| unique | 50 | 44.00% | 42.00% | 42.00% | 42.00% | 2.100 | 12.518s |

### Citation correctness

| Metric | Score |
|---|---:|
| Correct citations / all citations | 54 / 116 = 46.55% |
| Mean precision among cited answers | 63.00% |
| Full-gold citation hit rate | 47.00% |
| Primary-gold citation hit rate | 41.00% |
| Citation-marker integrity | 98.00% |
| Uncited-answer rate | 0.00% |

Primary-gold hit is diagnostic only: every validated full alternative is equally correct.
Gold-chunk precision is stricter than source matching but is not a claim-entailment judge.

### End-to-end latency

| Metric | Time |
|---|---:|
| Mean | 11.558s |
| Median | 12.649s |
| P95 | 17.277s |
| Maximum | 21.060s |
| Mean excluding first query | 11.513s |

Latency covers the complete graph. The first query is separated because it may
include lazy model and CUDA initialization. The main repeated cost is the number
of verifier/retrieval rounds:

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 2.038s |
| 1 | 42 | 100.00% | 95.24% | 95.24% | 46.94% | 1.000 | 7.747s |
| 2 | 6 | 100.00% | 100.00% | 100.00% | 58.33% | 2.000 | 12.689s |
| 3 | 49 | 2.04% | 2.04% | 2.04% | 16.67% | 3.000 | 15.270s |

The maximum is reported separately so a transient model stall cannot be mistaken
for typical latency; compare it with the median and P95.

### End-to-end by paraphrase type

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| lexical | 32 | 40.62% | 37.50% | 37.50% | 45.71% | 2.031 | 11.636s |
| natural | 34 | 58.82% | 55.88% | 55.88% | 44.44% | 1.971 | 11.461s |
| semantic | 34 | 47.06% | 47.06% | 47.06% | 50.00% | 2.029 | 11.582s |

### End-to-end by source

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| NIST.CSWP.29.pdf | 33 | 66.67% | 60.61% | 60.61% | 46.00% | 1.758 | 10.999s |
| NIST.SP.800-61r3.pdf | 43 | 44.19% | 44.19% | 44.19% | 43.14% | 2.233 | 12.441s |
| StopRansomware-Guide 508.pdf | 24 | 33.33% | 33.33% | 33.33% | 60.00% | 1.958 | 10.745s |

### End-to-end by primary chunk type

| Group | n | Fully answered | Grounded success | Citation hit | Citation precision | Mean rounds | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| paragraph | 65 | 53.85% | 50.77% | 50.77% | 48.05% | 1.877 | 11.139s |
| table | 35 | 40.00% | 40.00% | 40.00% | 43.59% | 2.257 | 12.338s |

### Weakest single-gold end-to-end cases

| ID | Status | Scope | Citation precision | Gold citation | Latency | Query |
|---|---|---|---:|---:|---:|---|
| ret_005 | insufficient | unique | — | no | 15.799s | What is required for risk management practices to be considered official? |
| ret_006 | insufficient | unique | — | no | 16.654s | What must be done to make risk management practices official? |
| ret_007 | insufficient | unique | — | no | 15.887s | What is a recommended practice for detecting malware through network traffic? |
| ret_010 | insufficient | unique | — | no | 16.590s | What are potential vectors for ransomware attacks? |
| ret_011 | insufficient | unique | — | no | 17.935s | How can malicious actors exploit trusted relationships with third parties? |

### Weakest multi-gold end-to-end cases

| ID | Status | Scope | Citation precision | Gold citation | Latency | Query |
|---|---|---|---:|---:|---:|---|
| ret_056 | insufficient | mixed | — | no | 14.930s | What steps does a company take to update its cybersecurity approach using what it has learned? |
| ret_057 | insufficient | same_document | — | no | 15.732s | What are the sub-elements of the GOVERN function in the CSF Core? |
| ret_058 | insufficient | same_document | — | no | 17.993s | Which components are part of the organizational risk management strategy according to the CSF Core? |
| ret_063 | insufficient | same_document | — | no | 15.471s | What incident response framework should organizations choose based on their needs? |
| ret_064 | insufficient | same_document | — | no | 13.853s | Which incident response model is recommended for organizations to select according to their specific requirements? |
<!-- END_TO_END_EVALUATION_END -->
