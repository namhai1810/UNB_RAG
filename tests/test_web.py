from src.agents.answer_agent import AnswerPayload, Citation
from src.web import _serialize_state


def test_serialize_state_returns_only_json_safe_public_fields():
    answer = AnswerPayload(
        answer="Contain the incident [1].",
        citations=[Citation(marker=1, source="NIST SP 800-61r3", section="Containment", pages="p.22", chunk_id="chunk-1", quote="Containment prevents expansion.")],
        confidence="high", caveats="",
    )
    result = _serialize_state({"status": "answered", "response": answer.answer, "round": 1, "answer": answer, "trace": [{"node": "answer"}], "evidence": [object()]})
    assert result["answer"]["citations"][0]["pages"] == "p.22"
    assert result["rounds"] == 1
    assert "evidence" not in result


def test_serialize_non_answer_route():
    result = _serialize_state({"status": "needs_clarification", "response": "Which system was affected?", "trace": []})
    assert result == {"status": "needs_clarification", "response": "Which system was affected?", "rounds": 0, "answer": None, "trace": []}


def test_serialize_state_includes_structured_table_citation():
    answer = AnswerPayload(
        answer="Use the Govern function [1].",
        citations=[
            Citation(
                marker=1,
                source="NIST.CSWP.29.pdf",
                section="CSF Core",
                pages="p.20",
                chunk_id="table-1",
                quote="Row: 1 Function: Govern Purpose: Establish strategy",
                chunk_type="table",
                table_caption="CSF 2.0 Functions",
                table_headers=["Function", "Purpose"],
                table_rows=[
                    {"Function": "Govern", "Purpose": "Establish strategy"}
                ],
                table_row_start=1,
                table_row_end=1,
            )
        ],
        confidence="high",
        caveats="",
    )

    result = _serialize_state({"status": "answered", "answer": answer})
    citation = result["answer"]["citations"][0]

    assert citation["chunk_type"] == "table"
    assert citation["table_headers"] == ["Function", "Purpose"]
    assert citation["table_rows"] == [
        {"Function": "Govern", "Purpose": "Establish strategy"}
    ]
