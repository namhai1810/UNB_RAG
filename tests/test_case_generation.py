from __future__ import annotations

from test_case_generation.generate import CATEGORIES, QuestionBatch, generate_cases

FAKE_QUESTIONS = (
    "How does NIST define the GOVERN function?",
    "Which incident logs should responders preserve first?",
    "What controls protect offline ransomware backups?",
    "When should an organisation begin recovery planning?",
    "Why are current and target profiles different?",
)


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def structured(self, system, user, schema, *, effort="medium"):
        self.calls.append(user)
        category = next(category for category in CATEGORIES if f"Target category: {category}" in user)
        count = int(user.split("Number of new questions: ", 1)[1].splitlines()[0])
        return QuestionBatch(
            questions=[f"{category}: {FAKE_QUESTIONS[i]}" for i in range(count)]
        )


def test_generate_cases_is_balanced_and_uses_only_three_categories():
    cases = generate_cases(per_category=2, language="en", llm=FakeLLM())

    assert len(cases) == 6
    assert {case["expected_category"] for case in cases} == set(CATEGORIES)
    assert all(set(case) == {"id", "query", "expected_category"} for case in cases)
    assert all(
        sum(case["expected_category"] == category for case in cases) == 2
        for category in CATEGORIES
    )


def test_generate_cases_retries_duplicate_questions():
    class DuplicateThenUniqueLLM:
        def __init__(self) -> None:
            self.calls = 0

        def structured(self, system, user, schema, *, effort="medium"):
            self.calls += 1
            category = next(
                category for category in CATEGORIES if f"Target category: {category}" in user
            )
            if category == "in_scope" and self.calls == 1:
                return QuestionBatch(questions=["Same question?", "Same question?"])
            count = int(user.split("Number of new questions: ", 1)[1].splitlines()[0])
            return QuestionBatch(
                questions=[f"{category}: {FAKE_QUESTIONS[i]}" for i in range(count)]
            )

    cases = generate_cases(
        per_category=2,
        language="en",
        llm=DuplicateThenUniqueLLM(),
        max_attempts=3,
    )

    assert len(cases) == 6
    assert len({case["query"].casefold() for case in cases}) == 6


def test_generate_cases_uses_fallback_when_model_returns_nothing():
    class EmptyLLM:
        def structured(self, system, user, schema, *, effort="medium"):
            return QuestionBatch(questions=[])

    cases = generate_cases(
        per_category=2,
        language="vi",
        llm=EmptyLLM(),
        max_attempts=1,
    )

    assert len(cases) == 6
    assert {case["expected_category"] for case in cases} == set(CATEGORIES)


def test_generate_cases_rejects_non_positive_count():
    try:
        generate_cases(per_category=0, llm=FakeLLM())
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
