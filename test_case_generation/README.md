# Grounded retrieval dataset construction report

Generated at: `2026-09-13T11:01:37.590576+00:00`

## Dataset identity

- Schema: **2.0**
- Cases: **100** from **34** atomic facts
- Candidate chunks audited: **91**
- Corpus SHA-256: `3033ace0581cc7581c762063fb7a4854aead91ae1b9992a5baf386a29d489f58`
- Fact/support validator: `QuixiAI/Qwen3-30B-A3B-AWQ`
- Semantic candidate discovery: `BAAI/bge-m3`
- Random seed: **43**

## Reproduce the dataset

Prerequisites: the processed corpus must exist at `data/processed/chunks.jsonl`,
and the configured LLM plus BGE-M3 model must be available. From the repository
root, run:

```bash
python -m test_case_generation.generate_retrieval
```

By default this rewrites `generated_retrieval_test_cases.json` and this README
with the corpus fingerprint and measured distribution. Then regenerate the
combined result report:

```bash
python -m evaluation.evaluate_retrieval
python -m evaluation.evaluate_end_to_end
# If a full-flow run was interrupted:
python -m evaluation.evaluate_end_to_end --resume
```

Run `python -m test_case_generation.generate_retrieval --help` to change quotas,
seed, candidate count, input chunks, or output paths. Generation is model-assisted,
so review the resulting evidence labels before treating a regenerated set as a
benchmark baseline.

## Gold multiplicity definitions

- `single_gold`: exactly one chunk in the audited corpus independently supports
  the complete reference answer.
- `multi_gold`: two or more chunks independently support the complete reference
  answer.
- `same_document`: all full-support chunks belong to one document.
- `cross_document`: full-support chunks occur in different documents, with one
  occurrence per represented document.
- `mixed`: full-support chunks span documents and at least one document contains
  multiple occurrences.

`primary_gold_chunk_id` records which chunk generated the fact. All entries in
`acceptable_gold_chunk_ids` have equal full relevance; the primary is not more
correct than a validated alternative.

## Construction procedure

1. Load the complete processed corpus and reject metadata/front-matter fragments.
2. Deterministically sample eligible candidate chunks across `(source, chunk_type)` strata.
3. Generate one atomic reference answer, verbatim primary evidence, and lexical,
   semantic, and natural questions for every candidate chunk.
4. Search the complete corpus for alternative support using exact evidence-span
   matches, lexical fact-token coverage, and the top
   5 BGE-M3 semantic candidates.
5. Accept exact copies only when every primary evidence span occurs in the
   candidate. Send other candidates to the support validator as `full`, `partial`,
   or `none`; full/partial judgments must provide candidate-local verbatim evidence.
6. Classify facts by the number and document scope of independently full-supporting chunks.
7. Sample balanced single- and multi-gold groups, then assign two or three
   paraphrases per group to reach the requested question totals.

Candidate discovery is intentionally high-recall and validation is independent
of retrieval rank. “Unique” is therefore an operational label under this audit,
not a mathematical proof that no human could find another equivalent passage.

## Final distribution

### Questions by gold multiplicity

| Group | Count |
|---|---:|
| multi_gold | 50 |
| single_gold | 50 |

### Source groups

| Group | Count |
|---|---:|
| NIST.CSWP.29.pdf | 33 |
| NIST.SP.800-61r3.pdf | 43 |
| StopRansomware-Guide 508.pdf | 24 |

### Primary chunk types

| Group | Count |
|---|---:|
| paragraph | 65 |
| table | 35 |

### Paraphrase styles

| Group | Count |
|---|---:|
| lexical | 32 |
| natural | 34 |
| semantic | 34 |

### Duplication scope

| Group | Count |
|---|---:|
| cross_document | 14 |
| mixed | 15 |
| same_document | 21 |
| unique | 50 |

### Number of acceptable gold chunks per question

| Group | Count |
|---|---:|
| 1 | 50 |
| 2 | 32 |
| 3 | 12 |
| 4 | 3 |
| 6 | 3 |

## Schema and scoring contract

Every case stores `gold_multiplicity`, `duplication_scope`, a provenance-only
`primary_gold_chunk_id`, all full-support `gold_chunks`, and separately labelled
`partial_support_chunks`. The compatibility field `acceptable_gold_chunk_ids`
contains only full-support IDs.

For single-gold cases, Recall@K is equivalent to Hit@K. For multi-gold cases,
the evaluator reports both Any-Gold Hit@K (answerability) and Gold Coverage@K
(how many repeated occurrences were found). MRR uses the first full gold. nDCG
assigns grade 2 to every full gold, grade 1 to partial support, and grade 0 to all
other chunks.

Citation correctness accepts a citation to any full gold chunk. It does not
require citing every repeated occurrence and does not by itself establish
claim-level entailment.

## Quality controls and limitations

- Primary and alternative evidence spans must occur verbatim in their labelled chunks.
- Every multi-gold case has at least two full-support chunks; every single-gold
  case has exactly one after the audit.
- Two to three paraphrases from one fact are correlated, so reports include both
  question and group counts and keep multiplicity results separate.
- Synthetic questions may be cleaner than production traffic.
- Semantic candidate discovery uses an embedding model only to propose passages;
  it never automatically assigns semantic gold relevance.
- Support validation is model-assisted. High-impact benchmark releases should
  still receive human review, particularly all multi-gold alternatives and a
  sample of operationally unique cases.
