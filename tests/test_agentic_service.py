from types import SimpleNamespace
import time

from bankreg_trustrag.agentic_executor import BoundedAgentExecutor
from bankreg_trustrag.answer_generator import AnswerGenerationOutcome, GeneratedAnswer
from bankreg_trustrag.calculator import Calculator
from bankreg_trustrag.query_plan import QueryPlan
from bankreg_trustrag.query_planner import PlannerOutcome
from bankreg_trustrag.retrieval.index import HybridIndex
from bankreg_trustrag.retrieval_tools import RetrievalTools
from bankreg_trustrag.service import TrustRAGService


def _plan():
    return QueryPlan.model_validate({
        "original_query": "2023年10月两类公司保费合计是多少，和全国总数差多少？",
        "user_goal": "计算两类公司保费合计及其与全国总数的差额",
        "answer_requirements": [
            {"id": "ar1", "question": "两类公司保费合计是多少", "required_outputs": ["calc_total"]},
            {"id": "ar2", "question": "和全国总数差多少", "required_outputs": ["calc_diff"]},
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


def _failed_plan():
    return QueryPlan.model_validate({
        "original_query": "跨文件问题",
        "user_goal": "安全处理规划失败",
        "answer_requirements": [
            {"id": "ar1", "question": "跨文件问题", "required_outputs": ["planning_unavailable"]}
        ],
        "retrieval_tasks": [],
        "operations": [],
        "requires_clarification": True,
        "clarification_reason": "查询规划暂时不可用：schema_validation",
    })


class _Planner:
    def plan(self, *args, **kwargs):
        return PlannerOutcome("ok", _plan(), 1)


class _FailedPlanner:
    def __init__(self, status="error", delay=0.0):
        self.status = status
        self.delay = delay

    def plan(self, *args, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        return PlannerOutcome(
            self.status,
            _failed_plan(),
            2,
            "schema_validation" if self.status == "error" else "llm_disabled",
        )


class _Answerer:
    def generate(self, *args, **kwargs):
        return AnswerGenerationOutcome("ok", GeneratedAnswer(
            answer=(
                "2023年10月，人身险原保险保费收入为31739.18亿元，财产险为13428.79亿元，"
                "两者合计45167.97亿元。同期全国合计为45167.98亿元，两者相差0.01亿元。"
            ),
            answered_requirement_ids=["ar1", "ar2"],
            output_refs_by_requirement={"ar1": ["calc_total"], "ar2": ["calc_diff"]},
        ), 1)


class _RepairingAnswerer:
    def __init__(self):
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            answer = "两类公司合计999亿元，与全国总数相差0.01亿元。"
        else:
            answer = (
                "人身险原保险保费收入为31739.18亿元，财产险为13428.79亿元，"
                "合计45167.97亿元；全国合计45167.98亿元，实际差值为0.01亿元。"
            )
        return AnswerGenerationOutcome("ok", GeneratedAnswer(
            answer=answer,
            answered_requirement_ids=["ar1", "ar2"],
            output_refs_by_requirement={"ar1": ["calc_total"], "ar2": ["calc_diff"]},
        ), 1)


def _service():
    index = HybridIndex(
        [
            {"doc_id": "life", "title": "人身险表", "file_name": "life.xlsx", "status": "effective"},
            {"doc_id": "property", "title": "财产险表", "file_name": "property.xlsx", "status": "effective"},
            {"doc_id": "national", "title": "全国表", "file_name": "national.xlsx", "status": "effective"},
        ],
        [],
        [
            {"evidence_id": "cell:life:C6", "doc_id": "life", "indicator": "原保险保费收入", "period": "2023-10", "value_text": "31739.18", "unit": "亿元", "cell_address": "C6", "context": "人身险原保险保费收入 31739.18亿元"},
            {"evidence_id": "cell:property:C6", "doc_id": "property", "indicator": "原保险保费收入", "period": "2023-10", "value_text": "13428.79", "unit": "亿元", "cell_address": "C6", "context": "财产险原保险保费收入 13428.79亿元"},
            {"evidence_id": "cell:national:C4", "doc_id": "national", "indicator": "全国合计", "period": "2023-10", "value_text": "45167.98", "unit": "亿元", "cell_address": "C4", "context": "全国合计 45167.98亿元"},
        ],
    )
    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(
        agentic_planner_enabled=True,
        agentic_planner_failure_mode="clarify",
        min_trust=0.58,
        top_k=8,
    )
    service.index = index
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.semantic = SimpleNamespace(enabled=False)
    service.agentic_executor = BoundedAgentExecutor(
        _Planner(), RetrievalTools(index), Calculator(), _Answerer()
    )
    return service


def test_service_feature_flag_runs_agentic_plan_and_persists_full_trace():
    response = _service().ask(_plan().original_query)

    assert response.answer.endswith("两者相差0.01亿元。")
    assert response.trust["decision"] == "answer"
    assert len(response.evidence) == 3
    trace = response.query_plan["execution_trace"]
    assert len(trace["retrieval_tasks"]) == 3
    assert len(trace["retrieval_results"]) == 3
    assert [item["result"] for item in trace["calculation_results"]] == ["45167.97", "0.01"]
    assert trace["answered_requirements"] == ["ar1", "ar2"]
    assert set(trace["latency"]) >= {"planning_ms", "retrieval_ms", "calculation_ms", "generation_ms", "verification_ms", "total_ms"}


def test_service_asks_answer_agent_to_repair_fact_failure_without_retrieval_or_recalculation():
    service = _service()
    answerer = _RepairingAnswerer()
    service.agentic_executor = BoundedAgentExecutor(
        _Planner(), RetrievalTools(service.index), Calculator(), answerer
    )

    response = service.ask(_plan().original_query)

    assert len(answerer.calls) == 2
    assert answerer.calls[1]["verification_feedback"]
    assert "45167.97亿元" in response.answer
    assert "999亿元" not in response.answer
    assert response.verification["numeric_ok"] is True
    attempts = response.query_plan["execution_trace"]["verification_attempts"]
    assert len(attempts) == 2
    assert attempts[0]["numeric_ok"] is False
    assert attempts[1]["numeric_ok"] is True


def test_service_feature_flag_false_keeps_legacy_path():
    service = _service()
    service.settings.agentic_planner_enabled = False

    response = service.ask("2023年10月人身险表中的原保险保费收入是多少？")

    assert "execution_trace" not in response.query_plan
    assert response.query_plan["agent_workflow"]["tasks"]


def test_configured_planner_failure_does_not_fall_back_to_incomplete_legacy_answer():
    service = _service()
    service.settings.agentic_planner_failure_mode = "legacy"
    service.agentic_executor = BoundedAgentExecutor(
        _FailedPlanner("error"), RetrievalTools(service.index), Calculator(), _Answerer()
    )

    response = service.ask("根据三份文件计算合计并与全国总数比较")

    assert response.trust["decision"] == "clarify"
    assert response.answer.startswith("查询规划失败，未执行检索或生成答案")
    assert response.evidence == []
    assert response.query_plan["execution_trace"]["planner_status"] == "error"
    assert response.query_plan["execution_trace"]["retrieval_results"] == []


def test_total_latency_includes_disabled_planner_before_legacy_fallback():
    service = _service()
    service.settings.agentic_planner_failure_mode = "legacy"
    service.agentic_executor = BoundedAgentExecutor(
        _FailedPlanner("disabled", delay=0.02),
        RetrievalTools(service.index),
        Calculator(),
        _Answerer(),
    )

    response = service.ask("2023年10月人身险表中的原保险保费收入是多少？")

    assert response.latency_ms >= 20
    assert "execution_trace" not in response.query_plan
