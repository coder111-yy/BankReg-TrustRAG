from bankreg_trustrag.agentic_executor import BoundedAgentExecutor
from bankreg_trustrag.answer_generator import AnswerGenerationOutcome, GeneratedAnswer
from bankreg_trustrag.calculator import Calculator
from bankreg_trustrag.completeness import CompletenessChecker
from bankreg_trustrag.query_plan import QueryPlan, RetrievalResult
from bankreg_trustrag.query_planner import PlannerOutcome
from bankreg_trustrag.retrieval.index import Hit, HybridIndex
from bankreg_trustrag.retrieval_tools import RetrievalExecution, RetrievalTools


def _plan():
    return QueryPlan.model_validate({
        "original_query": "两类公司保费合计是多少，和全国总数差多少？",
        "user_goal": "计算合计和差额",
        "answer_requirements": [
            {"id": "ar1", "question": "两类公司合计是多少", "required_outputs": ["calc_total"]},
            {"id": "ar2", "question": "与全国相差多少", "required_outputs": ["calc_diff"]},
        ],
        "entities": {"indicators": ["原保险保费收入"], "periods": ["2023-10"]},
        "retrieval_tasks": [
            {"id": "r1", "query": "人身险原保险保费收入", "expected_information": "人身险保费", "source_scope": {"document_title": "人身险表", "year": 2023, "month": 10}, "semantic_constraints": {"indicator": "原保险保费收入", "period": "2023-10"}, "expected_value_type": "number", "expected_unit": "亿元"},
            {"id": "r2", "query": "财产险原保险保费收入", "expected_information": "财产险保费", "source_scope": {"document_title": "财产险表", "year": 2023, "month": 10}, "semantic_constraints": {"indicator": "原保险保费收入", "period": "2023-10"}, "expected_value_type": "number", "expected_unit": "亿元"},
            {"id": "r3", "query": "全国合计", "expected_information": "全国合计", "source_scope": {"document_title": "全国表", "year": 2023, "month": 10}, "semantic_constraints": {"row_label": "全国合计", "period": "2023-10"}, "expected_value_type": "number", "expected_unit": "亿元"},
        ],
        "operations": [
            {"id": "op1", "type": "sum", "inputs": ["r1", "r2"], "output_id": "calc_total"},
            {"id": "op2", "type": "subtract", "inputs": ["calc_total", "r3"], "output_id": "calc_diff", "parameters": {"absolute": True}},
        ],
        "requires_multiple_sources": True,
        "requires_table_retrieval": True,
        "requires_calculation": True,
        "requires_clarification": False,
    })


class _Planner:
    def __init__(self, plan):
        self.value = plan

    def plan(self, *args, **kwargs):
        return PlannerOutcome("ok", self.value, 1)


class _Answerer:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return AnswerGenerationOutcome("ok", GeneratedAnswer(
            answer="人身险为31739.18亿元，财产险为13428.79亿元，合计45167.97亿元；全国合计45167.98亿元，相差0.01亿元。",
            answered_requirement_ids=["ar1", "ar2"],
            output_refs_by_requirement={"ar1": ["calc_total"], "ar2": ["calc_diff"]},
        ), 1)


def _index():
    return HybridIndex(
        [
            {"doc_id": "life", "title": "人身险表", "file_name": "life.xlsx"},
            {"doc_id": "property", "title": "财产险表", "file_name": "property.xlsx"},
            {"doc_id": "national", "title": "全国表", "file_name": "national.xlsx"},
        ],
        [],
        [
            {"evidence_id": "cell:life:C6", "doc_id": "life", "indicator": "原保险保费收入", "period": "2023-10", "value_text": "31739.18", "unit": "亿元", "cell_address": "C6", "context": "人身险 原保险保费收入 31739.18"},
            {"evidence_id": "cell:property:C6", "doc_id": "property", "indicator": "原保险保费收入", "period": "2023-10", "value_text": "13428.79", "unit": "亿元", "cell_address": "C6", "context": "财产险 原保险保费收入 13428.79"},
            {"evidence_id": "cell:national:C4", "doc_id": "national", "indicator": "全国合计", "period": "2023-10", "value_text": "45167.98", "unit": "亿元", "cell_address": "C4", "context": "全国合计 45167.98"},
        ],
    )


def test_bounded_executor_runs_three_retrievals_and_two_calculations():
    answerer = _Answerer()
    executor = BoundedAgentExecutor(
        _Planner(_plan()),
        RetrievalTools(_index()),
        Calculator(),
        answerer,
        CompletenessChecker(),
    )

    state = executor.run(_plan().original_query)

    assert state.clarification is None
    assert state.calculation_results["calc_total"].result == "45167.97"
    assert state.calculation_results["calc_diff"].result == "0.01"
    assert state.final_answer and "相差0.01亿元" in state.final_answer
    assert state.completeness.complete is True
    assert state.trace()["answered_requirements"] == ["ar1", "ar2"]
    assert answerer.calls == 1


def test_executor_emits_only_public_high_level_statuses():
    events = []
    executor = BoundedAgentExecutor(
        _Planner(_plan()),
        RetrievalTools(_index()),
        Calculator(),
        _Answerer(),
    )

    executor.run(
        _plan().original_query,
        observer=lambda stage, details: events.append((stage, details)),
    )

    stages = [stage for stage, _ in events]
    assert stages == [
        "planning",
        "tasks_planned",
        "retrieving_task",
        "retrieving_task",
        "retrieving_task",
        "retrieval_complete",
        "calculating",
        "generating",
    ]
    assert all(details.get("label") for _, details in events)
    assert all(set(details) == {"label"} for _, details in events)


def test_executor_stops_before_generation_when_retrieval_is_ambiguous():
    payload = _plan().model_dump()
    payload["answer_requirements"] = [{"id": "ar1", "question": "不良贷款余额是多少", "required_outputs": ["r1"]}]
    payload["retrieval_tasks"] = [{
        "id": "r1", "query": "2023年大型商业银行不良贷款余额", "expected_information": "不良贷款余额",
        "source_scope": {"year": 2023},
        "semantic_constraints": {"indicator": "不良贷款余额", "period": "2023"},
        "expected_value_type": "number", "expected_unit": "亿元",
    }]
    payload["operations"] = []
    payload["requires_multiple_sources"] = False
    payload["requires_calculation"] = False
    plan = QueryPlan.model_validate(payload)
    index = HybridIndex(
        [{"doc_id": "bank", "title": "大型商业银行指标", "file_name": "bank.xlsx"}], [],
        [
            {"evidence_id": "cell:q1", "doc_id": "bank", "indicator": "不良贷款余额", "period": "2023", "column_header": "一季度", "value_text": "10", "unit": "亿元", "cell_address": "B4", "context": "一季度 10"},
            {"evidence_id": "cell:q2", "doc_id": "bank", "indicator": "不良贷款余额", "period": "2023", "column_header": "二季度", "value_text": "11", "unit": "亿元", "cell_address": "C4", "context": "二季度 11"},
        ],
    )
    answerer = _Answerer()

    state = BoundedAgentExecutor(_Planner(plan), RetrievalTools(index), Calculator(), answerer).run(plan.original_query)

    assert state.clarification["stage"] == "retrieval"
    assert "季度" in state.clarification["reason"]
    assert answerer.calls == 0


def test_completeness_checker_rejects_unbound_answer_output():
    checker = CompletenessChecker()
    generated = GeneratedAnswer(
        answer="只回答了合计。",
        answered_requirement_ids=["ar1", "ar2"],
        output_refs_by_requirement={"ar1": ["calc_total"], "ar2": []},
    )

    result = checker.check_answer(_plan(), generated)

    assert result.complete is False
    assert result.missing_requirement_ids == ("ar2",)
    assert result.missing_outputs == ("calc_diff",)


def test_text_evidence_bundle_is_registered_as_resolved_output_without_scalar_value():
    plan = QueryPlan.model_validate({
        "original_query": "比较两个制度文件的要求",
        "user_goal": "综合两个文件中的文本事实",
        "answer_requirements": [
            {"id": "ar1", "question": "文件一规定什么", "required_outputs": ["r1"]},
            {"id": "ar2", "question": "文件二规定什么", "required_outputs": ["r2"]},
        ],
        "retrieval_tasks": [
            {"id": "r1", "query": "文件一要求", "expected_information": "文件一规则", "expected_value_type": "text"},
            {"id": "r2", "query": "文件二要求", "expected_information": "文件二规则", "expected_value_type": "text"},
        ],
        "operations": [],
        "requires_multiple_sources": True,
    })

    class TextTools:
        def execute(self, task):
            hit = Hit("text", {"evidence_id": f"e_{task.id}", "content": f"{task.expected_information}正文"})
            # Compatible text tools may return an evidence bundle without a
            # selected scalar/cell. It is still a valid RetrievalTask output.
            result = RetrievalResult(
                task_id=task.id,
                status="resolved",
                expected_information=task.expected_information,
                evidence_ids=[hit.evidence_id],
            )
            return RetrievalExecution(result, [hit])

    class TextAnswerer:
        def generate(self, *args, **kwargs):
            return AnswerGenerationOutcome("ok", GeneratedAnswer(
                answer="文件一和文件二的要求已依据各自证据作出综合说明。",
                answered_requirement_ids=["ar1", "ar2"],
                output_refs_by_requirement={"ar1": ["r1"], "ar2": ["r2"]},
            ), 1)

    state = BoundedAgentExecutor(
        _Planner(plan), TextTools(), Calculator(), TextAnswerer()
    ).run(plan.original_query)

    assert state.clarification is None
    assert state.execution_error is None
    assert set(state.resolved_outputs) == {"r1", "r2"}
    assert state.completeness.complete is True
    assert state.final_answer is not None


def test_executor_retries_direct_rule_fact_as_text_and_preserves_plan_id():
    plan = QueryPlan.model_validate({
        "original_query": "核心一级资本充足率阈值是多少",
        "user_goal": "查询制度文件中的阈值规则",
        "answer_requirements": [
            {"id": "ar1", "question": "阈值是多少", "required_outputs": ["r1"]}
        ],
        "retrieval_tasks": [{
            "id": "r1",
            "query": "持续经营触发事件 核心一级资本充足率阈值",
            "expected_information": "核心一级资本充足率阈值",
            "semantic_constraints": {"indicator": "核心一级资本充足率"},
            "expected_value_type": "number",
            "expected_unit": "%",
        }],
        "operations": [],
    })

    class RuleTools:
        def __init__(self):
            self.value_types = []

        def execute(self, task):
            self.value_types.append(task.expected_value_type)
            if task.expected_value_type == "number":
                return RetrievalExecution(RetrievalResult(
                    task_id=task.id,
                    status="not_found",
                    expected_information=task.expected_information,
                ), [])
            hit = Hit("text", {"evidence_id": "e_rule", "content": "核心一级资本充足率降至5.125%时触发。"})
            return RetrievalExecution(RetrievalResult(
                task_id=task.id,
                status="resolved",
                expected_information=task.expected_information,
                evidence_ids=[hit.evidence_id],
            ), [hit])

    class RuleAnswerer:
        def generate(self, *args, **kwargs):
            return AnswerGenerationOutcome("ok", GeneratedAnswer(
                answer="制度规定的触发阈值为5.125%。",
                answered_requirement_ids=["ar1"],
                output_refs_by_requirement={"ar1": ["r1"]},
            ), 1)

    tools = RuleTools()
    state = BoundedAgentExecutor(
        _Planner(plan), tools, Calculator(), RuleAnswerer()
    ).run(plan.original_query)

    assert tools.value_types == ["number", "text"]
    assert list(state.retrieval_results) == ["r1"]
    assert list(state.resolved_outputs) == ["r1"]
    assert state.retrieval_results["r1"].evidence_ids == ["e_rule"]
    assert state.unresolved_requirements == []
    assert state.execution_error is None
    assert state.final_answer == "制度规定的触发阈值为5.125%。"
