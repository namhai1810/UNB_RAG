# Triage test-case generation

This generator creates a balanced JSON data set for exactly the three categories
used by `src.agents.triage_agent`:

- `in_scope`
- `ambiguous`
- `out_of_scope`

`ambiguous` is the category name in the application; `ambitious` is not a valid
label.

Generate ten English questions per category:

```bash
python -m test_case_generation.generate
```

Generate twenty Vietnamese questions per category:

```bash
python -m test_case_generation.generate \
  --per-category 20 \
  --language vi \
  --output test_case_generation/generated_test_cases.json
```

The command uses the same LLM configuration as the application (`.env`). For a
suite of five or more, up to five slots per category are reserved for a manually
labelled English/Vietnamese bank; this prevents one model pattern from dominating
the suite. The same bank fills any shortfall if the model repeats questions. Each
JSON item only has the fields needed to evaluate classification:

```json
{
  "id": "amb_001",
  "query": "Chúng tôi vừa bị tấn công, giờ phải làm gì?",
  "expected_category": "ambiguous"
}
```

