import json

from bankreg_trustrag.answer_generator import AnswerGenerator, GeneratedAnswer
from bankreg_trustrag.llm_client import StructuredLLMResult
from bankreg_trustrag.query_plan import CalculationResult, QueryPlan, RetrievalResult


def _plan() -> QueryPlan:
    return QueryPlan.model_validate({
        "original_query": "两类保费之和是否与保险业总数一致？",
        "user_goal": "依据三份统计表及计算结果回答一致性问题",
        "answer_requirements": [
            {"id": "ar1", "question": "两类保费之和是否与总数一致", "required_outputs": ["calc_sum", "r_total", "calc_diff"]}
        ],
        "retrieval_tasks": [
            {"id": "r_life", "query": "人身险保费", "expected_information": "人身险保费", "expected_value_type": "number"},
            {"id": "r_property", "query": "财产险保费", "expected_information": "财产险保费", "expected_value_type": "number"},
            {"id": "r_total", "query": "保险业总数", "expected_information": "保险业总数", "expected_value_type": "number"},
        ],
        "operations": [
            {"id": "op_sum", "type": "sum", "inputs": ["r_life", "r_property"], "output_id": "calc_sum"},
            {"id": "op_diff", "type": "subtract", "inputs": ["calc_sum", "r_total"], "output_id": "calc_diff", "parameters": {"absolute": True}},
        ],
        "requires_multiple_sources": True,
        "requires_table_retrieval": True,
        "requires_calculation": True,
    })


def _retrieval_results() -> dict[str, RetrievalResult]:
    rows = {
        "r_life": ("35378.91", "人身险表", "C6", "e_life"),
        "r_property": ("15867.79", "财产险表", "C6", "e_property"),
        "r_total": ("51246.71", "保险业表", "C8", "e_total"),
    }
    results: dict[str, RetrievalResult] = {}
    for task_id, (value, title, cell, evidence_id) in rows.items():
        results[task_id] = RetrievalResult.model_validate({
            "task_id": task_id,
            "status": "resolved",
            "expected_information": task_id,
            "selected": {
                "value": value,
                "unit": "亿元",
                "evidence_ids": [evidence_id],
                "document_id": task_id,
                "document_title": title,
                "document_type": "xlsx",
                "sheet_name": "Sheet1",
                "cell_address": cell,
                "indicator": "原保险保费收入",
                "period": "2023-12",
                "content": f"原保险保费收入 {value}亿元",
            },
            "candidates": [],
            "evidence_ids": [evidence_id],
        })
    return results


def _calculation_results() -> dict[str, CalculationResult]:
    return {
        "calc_sum": CalculationResult.model_validate({
            "id": "calc_sum",
            "operation": "sum",
            "input_refs": ["r_life", "r_property"],
            "inputs": [
                {"ref": "r_life", "value": "35378.91", "unit": "亿元", "evidence_ids": ["e_life"]},
                {"ref": "r_property", "value": "15867.79", "unit": "亿元", "evidence_ids": ["e_property"]},
            ],
            "result": "51246.70",
            "unit": "亿元",
            "trace": "35378.91 + 15867.79 = 51246.70",
            "evidence_ids": ["e_life", "e_property"],
        }),
        "calc_diff": CalculationResult.model_validate({
            "id": "calc_diff",
            "operation": "subtract",
            "input_refs": ["calc_sum", "r_total"],
            "inputs": [
                {"ref": "calc_sum", "value": "51246.70", "unit": "亿元", "evidence_ids": ["e_life", "e_property"]},
                {"ref": "r_total", "value": "51246.71", "unit": "亿元", "evidence_ids": ["e_total"]},
            ],
            "result": "0.01",
            "unit": "亿元",
            "trace": "abs(51246.70 - 51246.71) = 0.01",
            "evidence_ids": ["e_life", "e_property", "e_total"],
        }),
    }


class _CapturingClient:
    def __init__(self, result: StructuredLLMResult):
        self.result = result
        self.messages = None

    def structured(self, messages, *args, **kwargs):
        self.messages = messages
        return self.result


def test_answer_agent_receives_plan_all_evidence_calculations_and_sources():
    authored = GeneratedAnswer(
        answer="两类分项合计51246.70亿元，与总数51246.71亿元相差0.01亿元；结合这些数据，我认为整体接近，但并非严格相等。",
        answered_requirement_ids=["ar1"],
        output_refs_by_requirement={"ar1": ["calc_sum", "r_total", "calc_diff"]},
    )
    client = _CapturingClient(StructuredLLMResult("ok", value=authored, attempts=1))
    generator = AnswerGenerator(client)

    outcome = generator.generate(
        _plan().original_query,
        _plan(),
        _retrieval_results(),
        _calculation_results(),
        verification_feedback=[{"error_type": "unsupported_claim", "claim": "旧草稿"}],
    )

    assert outcome.status == "ok"
    assert outcome.generated.answer == authored.answer
    payload = json.loads(client.messages[1]["content"])
    assert payload["query_plan"]["user_goal"] == _plan().user_goal
    assert len(payload["provided_evidence"]) == 3
    assert {item["document_title"] for item in payload["source_ledger"]} == {"人身险表", "财产险表", "保险业表"}
    assert payload["calculation_results"]["calc_sum"]["trace"] == "35378.91 + 15867.79 = 51246.70"
    assert payload["calculation_results"]["calc_diff"]["trace"] == "abs(51246.70 - 51246.71) = 0.01"
    assert payload["verification_feedback"][0]["error_type"] == "unsupported_claim"
    assert "自行决定" in client.messages[0]["content"]


def test_fixed_formatter_is_used_only_when_answer_llm_fails():
    client = _CapturingClient(StructuredLLMResult("error", attempts=2, error="api_unavailable"))

    outcome = AnswerGenerator(client).generate(
        _plan().original_query,
        _plan(),
        _retrieval_results(),
        _calculation_results(),
    )

    assert outcome.status == "fallback"
    assert outcome.error == "api_unavailable"
    assert "35378.91 + 15867.79 = 51246.70" in outcome.generated.answer
