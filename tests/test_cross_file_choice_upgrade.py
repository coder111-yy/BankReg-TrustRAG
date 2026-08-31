from types import SimpleNamespace

from bankreg_trustrag.query_plan import AnswerRequirement, PlannerOutput, RetrievalTask
from bankreg_trustrag.query_planner import _expand_planner_output, _source_grounded_choice_plan
from bankreg_trustrag.retrieval.index import Hit, HybridIndex
from bankreg_trustrag.retrieval_tools import RetrievalTools, _text_result
from bankreg_trustrag.service import TrustRAGService


def test_multiple_choice_uses_agentic_planner_and_preserves_separate_choices():
    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(agentic_planner_enabled=True)
    service.index = SimpleNamespace(begin_query=lambda: None)
    observed = {}
    sentinel = object()

    def ask_agentic(question, *args, **kwargs):
        observed["question"] = question
        return sentinel

    service._ask_agentic = ask_agentic
    service._ask_legacy = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("agentic multiple-choice request unexpectedly used legacy retrieval")
    )

    response = service.ask(
        "依据两个年度报表，哪项比较正确？",
        choices=["2024年较低", "2025年较低"],
    )

    assert response is sentinel
    assert "A. 2024年较低" in observed["question"]
    assert "B. 2025年较低" in observed["question"]


def test_source_scoped_indicator_and_quarter_resolve_without_redundant_column_label():
    documents = [{
        "doc_id": "annual-2024",
        "title": "2024年商业银行主要监管指标情况表（季度）",
        "file_name": "2024年商业银行主要监管指标情况表.xls",
        "document_type": "excel",
    }]
    rows = [
        {
            "evidence_id": f"cell:annual-2024:商业银行季度:{column}25",
            "doc_id": "annual-2024",
            "sheet_name": "商业银行季度",
            "indicator": "流动性覆盖率**",
            "row_header": "流动性覆盖率**",
            "column_header": quarter,
            "period": "2024",
            "year": 2024,
            "value_text": value,
            "cell_address": f"{column}25",
            "context": f"流动性覆盖率 | {quarter} | {value}",
            "cell_type": "data",
        }
        for column, quarter, value in (
            ("B", "一季度", "1.50838"),
            ("C", "二季度", "1.507"),
            ("D", "三季度", "1.53286"),
            ("E", "四季度", "1.54729"),
        )
    ]

    def provider(*, doc_ids=None, limit=20000, **kwargs):
        allowed = set(doc_ids or [])
        return [row for row in rows if not allowed or row["doc_id"] in allowed][:limit]

    index = HybridIndex(
        documents,
        [],
        [],
        table_provider=provider,
        table_count=len(rows),
    )
    task = RetrievalTask.model_validate({
        "id": "r_2024_q4",
        "query": "核对2024年四季度流动性覆盖率",
        "expected_information": "2024年四季度流动性覆盖率",
        "source_scope": {
            "document_title": "2024年商业银行主要监管指标情况表",
            "year": 2024,
            "quarter": 4,
        },
        "semantic_constraints": {
            "indicator": "流动性覆盖率",
            "period": "2024年四季度",
        },
        "expected_value_type": "number",
    })

    execution = RetrievalTools(index).execute(task)

    assert execution.result.status == "resolved"
    assert execution.result.selected is not None
    assert execution.result.selected.evidence_ids == [
        "cell:annual-2024:商业银行季度:E25"
    ]
    assert execution.result.selected.value == "1.54729"


def test_inline_multiple_choice_is_not_duplicated_for_agentic_planning():
    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(agentic_planner_enabled=True)
    service.index = SimpleNamespace(begin_query=lambda: None)
    observed = {}
    sentinel = object()

    def ask_agentic(question, *args, **kwargs):
        observed["question"] = question
        return sentinel

    service._ask_agentic = ask_agentic
    service._ask_legacy = lambda *args, **kwargs: None
    question = "哪项正确？A.甲 B.乙 C.丙 D.丁"

    assert service.ask(question) is sentinel
    assert observed["question"] == question


def test_rule_plus_two_annual_reports_is_never_collapsed_to_one_quoted_source():
    question = (
        "依据《商业银行资本管理办法》附件22的信息披露要求，并核对2024年、2025年"
        "商业银行主要监管指标表，哪项四季度流动性覆盖率的比较正确？"
        "A.上升3.263个百分点 B.下降3.263个百分点"
    )
    compact = PlannerOutput(
        user_goal="核对制度并比较两个年度报表",
        answer_requirements=[AnswerRequirement(
            id="ar1",
            question=question,
            required_outputs=["rule", "y2024", "y2025", "delta"],
        )],
        retrieval_tasks=[
            {
                "id": "rule",
                "query": "附件22 流动性覆盖率 信息披露",
                "expected_information": "制度披露口径",
                "source_hint": "商业银行资本管理办法_附件22",
            },
            {
                "id": "y2024",
                "query": "2024年四季度流动性覆盖率",
                "expected_information": "2024年第四季度值",
            },
            {
                "id": "y2025",
                "query": "2025年四季度流动性覆盖率",
                "expected_information": "2025年第四季度值",
            },
        ],
        operations=[],
        requires_clarification=False,
    )

    assert _source_grounded_choice_plan(question, compact) is None


def test_compare_operation_discards_supporting_text_and_keeps_two_numeric_years():
    question = "依据制度口径，比较2024年和2025年四季度流动性覆盖率。"
    compact = PlannerOutput.model_validate({
        "user_goal": "核对制度并比较两个年度数值",
        "answer_requirements": [{
            "id": "ar1",
            "question": question,
            "required_outputs": ["rule", "y2024", "y2025", "comparison"],
        }],
        "retrieval_tasks": [
            {
                "id": "rule",
                "query": "流动性覆盖率披露要求",
                "expected_information": "制度口径",
            },
            {
                "id": "y2024",
                "query": "2024年四季度流动性覆盖率",
                "expected_information": "2024年数值",
                "indicator": "流动性覆盖率",
                "period": "2024年四季度",
                "expected_value_type": "number",
            },
            {
                "id": "y2025",
                "query": "2025年四季度流动性覆盖率",
                "expected_information": "2025年数值",
                "indicator": "流动性覆盖率",
                "period": "2025年四季度",
                "expected_value_type": "number",
            },
        ],
        "operations": [{
            "type": "compare",
            "output_id": "comparison",
            "inputs": ["rule", "y2024", "y2025"],
        }],
        "requires_clarification": False,
    })

    plan = _expand_planner_output(question, compact)

    assert plan.retrieval_tasks[0].expected_value_type == "text"
    assert plan.operations[0].input_refs() == ["y2024", "y2025"]


def _two_number_plan(operation):
    return PlannerOutput.model_validate({
        "user_goal": "比较两个年度值",
        "answer_requirements": [{
            "id": "ar1",
            "question": "比较两个年度值",
            "required_outputs": [operation["output_id"]],
        }],
        "retrieval_tasks": [
            {
                "id": "y2024", "query": "2024年数值", "expected_information": "2024年数值",
                "period": "2024年", "expected_value_type": "number",
            },
            {
                "id": "y2025", "query": "2025年数值", "expected_information": "2025年数值",
                "period": "2025年", "expected_value_type": "number",
            },
        ],
        "operations": [operation],
        "requires_clarification": False,
    })


def test_missing_subtraction_direction_is_inferred_without_plan_failure():
    payload = {
        "type": "subtract",
        "output_id": "delta",
        "inputs": ["y2024", "y2025"],
    }
    assert _expand_planner_output(
        "2024年与2025年相差多少？",
        _two_number_plan(payload),
    ).operations[0].parameters["absolute"] is True


def test_source_scoped_long_regulatory_task_accepts_relevant_paragraph():
    task = RetrievalTask.model_validate({
        "id": "rule",
        "query": "附件22 流动性覆盖率 信息披露 监管口径",
        "expected_information": (
            "确认流动性覆盖率是否属于按季度披露的核心监管指标，"
            "并取得披露口径、期末值要求和相关说明"
        ),
        "source_scope": {"document_title": "商业银行资本管理办法_附件22"},
        "semantic_constraints": {"indicator": "流动性覆盖率"},
        "expected_value_type": "text",
    })
    hit = Hit(
        "text",
        {
            "evidence_id": "text:rule:p1",
            "doc_id": "rule-doc",
            "content": "商业银行应当披露流动性覆盖率，相关数值采用季度期末值。",
        },
        fused_score=0.35,
        rerank_score=0.45,
    )

    class TextOnlyIndex:
        doc_by_id = {
            "rule-doc": {"title": "商业银行资本管理办法_附件22", "document_type": "word"}
        }
        text = []

        def hybrid_search(self, query, qa_type, top_k=8, filters=None):
            assert qa_type == "regulatory_fact"
            return [hit]

        def search_tables(self, *args, **kwargs):
            raise AssertionError("regulatory text task was incorrectly routed to table search")

    execution = RetrievalTools(TextOnlyIndex()).execute(task)

    assert execution.result.status == "resolved"
    assert execution.result.selected is not None
    assert execution.result.selected.evidence_ids == ["text:rule:p1"]


def test_semantic_output_aliases_bind_to_real_retrieval_task_ids():
    compact = PlannerOutput.model_validate({
        "user_goal": "核对附件并比较两个年度",
        "answer_requirements": [
            {
                "id": "rq_rule",
                "question": "附件22要求是什么？",
                "required_outputs": ["attachment22_disclosure_requirement"],
            },
            {
                "id": "rq_values",
                "question": "两个年度的值和比较结果是什么？",
                "required_outputs": ["lc_ratio_2024_q4", "lc_ratio_2025_q4", "comparison_result"],
            },
        ],
        "retrieval_tasks": [
            {
                "id": "rt1",
                "query": "附件22 流动性覆盖率披露要求",
                "expected_information": "制度依据",
                "source_hint": "商业银行资本管理办法_附件22",
                "expected_value_type": "text",
            },
            {
                "id": "rt2",
                "query": "2024年四季度流动性覆盖率",
                "expected_information": "2024年Q4值",
                "indicator": "流动性覆盖率",
                "period": "2024年四季度",
                "expected_value_type": "number",
            },
            {
                "id": "rt3",
                "query": "2025年四季度流动性覆盖率",
                "expected_information": "2025年Q4值",
                "indicator": "流动性覆盖率",
                "period": "2025年四季度",
                "expected_value_type": "number",
            },
        ],
        "operations": [
            {
                "type": "subtract",
                "output_id": "delta_lc_ratio",
                "inputs": ["lc_ratio_2025_q4", "lc_ratio_2024_q4"],
                "absolute": False,
            },
            {
                "type": "compare",
                "output_id": "comparison_result",
                "inputs": ["lc_ratio_2024_q4", "lc_ratio_2025_q4"],
            },
        ],
        "requires_clarification": False,
    })

    plan = _expand_planner_output("依据附件22比较2024年和2025年四季度流动性覆盖率", compact)

    assert plan.operations[0].input_refs() == ["rt3", "rt2"]
    assert plan.operations[1].input_refs() == ["rt2", "rt3"]
    assert plan.answer_requirements[0].required_outputs == ["rt1"]
    assert plan.answer_requirements[1].required_outputs == ["rt2", "rt3", "comparison_result"]
