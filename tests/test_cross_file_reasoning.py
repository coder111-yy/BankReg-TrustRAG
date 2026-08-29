from bankreg_trustrag.agentic_executor import BoundedAgentExecutor
from bankreg_trustrag.answer_generator import AnswerGenerator
from bankreg_trustrag.calculator import Calculator
from bankreg_trustrag.llm_client import LLMClient, LLMClientConfig
from bankreg_trustrag.query_plan import QueryPlan
from bankreg_trustrag.query_planner import PlannerOutcome
from bankreg_trustrag.retrieval.index import HybridIndex
from bankreg_trustrag.retrieval_tools import RetrievalTools


class _Planner:
    def __init__(self, plan):
        self.value = plan

    def plan(self, *args, **kwargs):
        return PlannerOutcome("ok", self.value, 1)


def _executor(plan, index):
    return BoundedAgentExecutor(
        _Planner(plan),
        RetrievalTools(index),
        Calculator(),
        AnswerGenerator(LLMClient(LLMClientConfig())),
    )


def _insurance_index():
    return HybridIndex(
        [
            {"doc_id": "life", "title": "2023年10月人身险公司经营情况表", "file_name": "life.xlsx"},
            {"doc_id": "property", "title": "2023年10月财产保险公司经营情况表", "file_name": "property.xlsx"},
            {"doc_id": "national", "title": "2023年10月全国各地区原保险保费收入情况表", "file_name": "national.xlsx"},
        ],
        [],
        [
            {"evidence_id": "cell:life:C6", "doc_id": "life", "indicator": "原保险保费收入", "period": "2023-10", "value_text": "31739.18", "unit": "亿元", "cell_address": "C6", "context": "人身险公司 原保险保费收入 31739.18"},
            {"evidence_id": "cell:property:C6", "doc_id": "property", "indicator": "原保险保费收入", "period": "2023-10", "value_text": "13428.79", "unit": "亿元", "cell_address": "C6", "context": "财产保险公司 原保险保费收入 13428.79"},
            {"evidence_id": "cell:national:C4", "doc_id": "national", "indicator": "全国合计", "row_header": "全国合计", "period": "2023-10", "value_text": "45167.98", "unit": "亿元", "cell_address": "C4", "context": "全国合计 原保险保费收入 45167.98"},
        ],
    )


def _premium_plan(include_difference):
    requirements = [{"id": "ar1", "question": "两类公司原保险保费收入合计是多少", "required_outputs": ["calc_total"]}]
    retrieval_tasks = [
        {"id": "r_life", "query": "2023年10月人身险公司原保险保费收入", "expected_information": "人身险原保险保费收入", "source_scope": {"document_title": "2023年10月人身险公司经营情况表", "year": 2023, "month": 10}, "semantic_constraints": {"indicator": "原保险保费收入", "institution": "人身险公司", "period": "2023-10"}, "expected_value_type": "number", "expected_unit": "亿元"},
        {"id": "r_property", "query": "2023年10月财产保险公司原保险保费收入", "expected_information": "财产险原保险保费收入", "source_scope": {"document_title": "2023年10月财产保险公司经营情况表", "year": 2023, "month": 10}, "semantic_constraints": {"indicator": "原保险保费收入", "institution": "财产保险公司", "period": "2023-10"}, "expected_value_type": "number", "expected_unit": "亿元"},
    ]
    operations = [{"id": "op_total", "type": "sum", "inputs": ["r_life", "r_property"], "output_id": "calc_total"}]
    if include_difference:
        requirements.append({"id": "ar2", "question": "与全国总数相差多少", "required_outputs": ["calc_diff"]})
        retrieval_tasks.append({"id": "r_national", "query": "2023年10月全国原保险保费收入全国合计", "expected_information": "全国原保险保费收入合计", "source_scope": {"document_title": "2023年10月全国各地区原保险保费收入情况表", "year": 2023, "month": 10}, "semantic_constraints": {"row_label": "全国合计", "period": "2023-10"}, "expected_value_type": "number", "expected_unit": "亿元"})
        operations.append({"id": "op_diff", "type": "subtract", "inputs": ["calc_total", "r_national"], "output_id": "calc_diff", "parameters": {"absolute": True}})
    return QueryPlan.model_validate({
        "original_query": "跨文件保险保费问题",
        "user_goal": "计算跨文件保费合计和差额",
        "answer_requirements": requirements,
        "entities": {"indicators": ["原保险保费收入"], "periods": ["2023-10"]},
        "retrieval_tasks": retrieval_tasks,
        "operations": operations,
        "requires_multiple_sources": True,
        "requires_table_retrieval": True,
        "requires_calculation": True,
        "requires_clarification": False,
    })


def test_two_file_sum_is_calculated_from_two_independent_cells():
    plan = _premium_plan(False)
    state = _executor(plan, _insurance_index()).run(plan.original_query)

    assert state.calculation_results["calc_total"].result == "45167.97"
    assert state.calculation_results["calc_total"].evidence_ids == ["cell:life:C6", "cell:property:C6"]
    assert "45167.97亿元" in state.final_answer


def test_three_file_sum_and_difference_answers_both_requirements():
    plan = _premium_plan(True)
    state = _executor(plan, _insurance_index()).run(plan.original_query)

    assert state.calculation_results["calc_total"].result == "45167.97"
    assert state.calculation_results["calc_diff"].result == "0.01"
    assert set(state.answer_outcome.generated.answered_requirement_ids) == {"ar1", "ar2"}
    assert "45167.97亿元" in state.final_answer
    assert "0.01亿元" in state.final_answer


def test_year_over_year_increase_and_growth_rate_use_explicit_direction():
    index = HybridIndex(
        [
            {"doc_id": "y2023", "title": "2023年9月保险业经营情况表", "file_name": "2023.xlsx"},
            {"doc_id": "y2024", "title": "2024年9月保险业经营情况表", "file_name": "2024.xlsx"},
        ], [],
        [
            {"evidence_id": "cell:2023:C5", "doc_id": "y2023", "indicator": "原保险保费收入", "period": "2023-09", "value_text": "40000", "unit": "亿元", "cell_address": "C5", "context": "2023年9月 原保险保费收入 40000"},
            {"evidence_id": "cell:2024:C5", "doc_id": "y2024", "indicator": "原保险保费收入", "period": "2024-09", "value_text": "44000", "unit": "亿元", "cell_address": "C5", "context": "2024年9月 原保险保费收入 44000"},
        ],
    )
    plan = QueryPlan.model_validate({
        "original_query": "2024年9月保险业原保险保费收入相比2023年同期增加多少，增长率是多少？",
        "user_goal": "计算同比增长额和增长率",
        "answer_requirements": [
            {"id": "ar1", "question": "增加多少", "required_outputs": ["calc_increase"]},
            {"id": "ar2", "question": "增长率是多少", "required_outputs": ["calc_growth"]},
        ],
        "entities": {"indicators": ["原保险保费收入"], "periods": ["2023-09", "2024-09"]},
        "retrieval_tasks": [
            {"id": "r_old", "query": "2023年9月原保险保费收入", "expected_information": "2023年同期值", "source_scope": {"year": 2023, "month": 9}, "semantic_constraints": {"indicator": "原保险保费收入", "period": "2023-09"}, "expected_value_type": "number", "expected_unit": "亿元"},
            {"id": "r_new", "query": "2024年9月原保险保费收入", "expected_information": "2024年当期值", "source_scope": {"year": 2024, "month": 9}, "semantic_constraints": {"indicator": "原保险保费收入", "period": "2024-09"}, "expected_value_type": "number", "expected_unit": "亿元"},
        ],
        "operations": [
            {"id": "op_inc", "type": "subtract", "left": "r_new", "right": "r_old", "output_id": "calc_increase", "parameters": {"absolute": False}},
            {"id": "op_growth", "type": "growth_rate", "old_ref": "r_old", "new_ref": "r_new", "output_id": "calc_growth"},
        ],
        "requires_multiple_sources": True,
        "requires_table_retrieval": True,
        "requires_calculation": True,
        "requires_clarification": False,
    })

    state = _executor(plan, index).run(plan.original_query)

    assert state.calculation_results["calc_increase"].result == "4000"
    assert state.calculation_results["calc_growth"].result == "10"
    assert "4000亿元" in state.final_answer
    assert "10%" in state.final_answer


def test_retrieval_ambiguity_requests_quarter_instead_of_defaulting_q1():
    index = HybridIndex(
        [{"doc_id": "bank", "title": "2023年大型商业银行指标", "file_name": "bank.xlsx"}], [],
        [
            {"evidence_id": "cell:q1", "doc_id": "bank", "indicator": "不良贷款余额", "period": "2023", "column_header": "一季度", "value_text": "10", "unit": "亿元", "cell_address": "B4", "context": "大型商业银行 一季度 不良贷款余额 10"},
            {"evidence_id": "cell:q2", "doc_id": "bank", "indicator": "不良贷款余额", "period": "2023", "column_header": "二季度", "value_text": "11", "unit": "亿元", "cell_address": "C4", "context": "大型商业银行 二季度 不良贷款余额 11"},
        ],
    )
    plan = QueryPlan.model_validate({
        "original_query": "2023年大型商业银行不良贷款余额是多少？",
        "user_goal": "查询不良贷款余额",
        "answer_requirements": [{"id": "ar1", "question": "不良贷款余额是多少", "required_outputs": ["r1"]}],
        "entities": {"indicators": ["不良贷款余额"], "periods": ["2023"], "institutions": ["大型商业银行"]},
        "retrieval_tasks": [{"id": "r1", "query": "2023年大型商业银行不良贷款余额", "expected_information": "不良贷款余额", "source_scope": {"year": 2023}, "semantic_constraints": {"indicator": "不良贷款余额", "institution": "大型商业银行", "period": "2023"}, "expected_value_type": "number", "expected_unit": "亿元"}],
        "operations": [],
        "requires_multiple_sources": False,
        "requires_table_retrieval": True,
        "requires_calculation": False,
        "requires_clarification": False,
    })

    state = _executor(plan, index).run(plan.original_query)

    assert state.final_answer is None
    assert state.clarification["stage"] == "retrieval"
    assert "多个季度候选" in state.clarification["reason"]
