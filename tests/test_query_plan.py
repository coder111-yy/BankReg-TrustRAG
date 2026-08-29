import pytest
from pydantic import ValidationError

from bankreg_trustrag.query_plan import QueryPlan


def _cross_file_plan():
    return {
        "original_query": "两类公司保费合计是多少，和全国总数差多少？",
        "user_goal": "查询三项数据，计算两类公司合计及其与全国合计的差额",
        "answer_requirements": [
            {"id": "ar1", "question": "两类公司合计是多少", "required_outputs": ["calc_total"]},
            {"id": "ar2", "question": "与全国总数相差多少", "required_outputs": ["calc_diff"]},
        ],
        "entities": {"indicators": ["原保险保费收入"], "periods": ["2023-10"]},
        "retrieval_tasks": [
            {"id": "r1", "query": "人身险原保险保费收入", "expected_information": "人身险数值", "expected_value_type": "number"},
            {"id": "r2", "query": "财产险原保险保费收入", "expected_information": "财产险数值", "expected_value_type": "number"},
            {"id": "r3", "query": "全国合计", "expected_information": "全国合计数值", "expected_value_type": "number"},
        ],
        "operations": [
            {"id": "op1", "type": "sum", "inputs": ["r1", "r2"], "output_id": "calc_total"},
            {"id": "op2", "type": "subtract", "inputs": ["calc_total", "r3"], "output_id": "calc_diff", "parameters": {"absolute": True}},
        ],
        "requires_multiple_sources": True,
        "requires_table_retrieval": True,
        "requires_calculation": True,
    }


def test_query_plan_binds_every_answer_requirement_to_an_execution_output():
    plan = QueryPlan.model_validate(_cross_file_plan())

    assert plan.answer_requirements[0].required_outputs == ["calc_total"]
    assert plan.operations[1].input_refs() == ["calc_total", "r3"]


def test_query_plan_rejects_missing_required_output():
    payload = _cross_file_plan()
    payload["answer_requirements"][1]["required_outputs"] = ["missing"]

    with pytest.raises(ValidationError, match="unknown outputs"):
        QueryPlan.model_validate(payload)


def test_query_plan_requires_explicit_growth_direction():
    payload = _cross_file_plan()
    payload["operations"] = [
        {"id": "op1", "type": "growth_rate", "inputs": ["r1", "r2"], "output_id": "calc_total"}
    ]
    payload["answer_requirements"] = [
        {"id": "ar1", "question": "增长率是多少", "required_outputs": ["calc_total"]}
    ]

    with pytest.raises(ValidationError, match="old_ref and new_ref"):
        QueryPlan.model_validate(payload)


def test_query_plan_requires_explicit_subtraction_direction_or_absolute_semantics():
    payload = _cross_file_plan()
    payload["operations"][1]["parameters"] = {}

    with pytest.raises(ValidationError, match="explicit absolute flag"):
        QueryPlan.model_validate(payload)


def test_query_plan_rejects_retrieval_dependency_cycle():
    payload = _cross_file_plan()
    payload["retrieval_tasks"] = [
        {
            "id": "r1",
            "query": "first",
            "expected_information": "first",
            "dependencies": ["r2"],
        },
        {
            "id": "r2",
            "query": "second",
            "expected_information": "second",
            "dependencies": ["r1"],
        },
    ]
    payload["operations"] = []
    payload["answer_requirements"] = [
        {"id": "ar1", "question": "answer", "required_outputs": ["r1"]}
    ]

    with pytest.raises(ValidationError, match="retrieval dependencies are cyclic"):
        QueryPlan.model_validate(payload)
