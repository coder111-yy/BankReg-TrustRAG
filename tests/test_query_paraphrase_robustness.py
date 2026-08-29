from bankreg_trustrag.agentic_executor import BoundedAgentExecutor
from bankreg_trustrag.answer_generator import AnswerGenerator
from bankreg_trustrag.calculator import Calculator
from bankreg_trustrag.llm_client import LLMClient, LLMClientConfig
from bankreg_trustrag.query_plan import QueryPlan
from bankreg_trustrag.query_planner import PlannerOutcome
from bankreg_trustrag.retrieval.index import HybridIndex
from bankreg_trustrag.retrieval_tools import RetrievalTools


QUESTIONS = [
    "2024年9月保险业总资产是多少？",
    "帮我查一下24年9月份保险行业一共有多少资产。",
    "结合上文年份，去年9月份保险业资产总额是多少？",
    "2024-09保险业总资产数据。",
    "请问2024年9月末保险行业资产规模是多少？",
]


def _asset_plan(question):
    return QueryPlan.model_validate({
        "original_query": question,
        "user_goal": "查询2024年9月保险业总资产",
        "answer_requirements": [{"id": "ar1", "question": "2024年9月保险业总资产是多少", "required_outputs": ["r_asset"]}],
        "entities": {"indicators": ["总资产"], "periods": ["2024-09"], "institutions": ["保险业"]},
        "retrieval_tasks": [{
            "id": "r_asset",
            "query": "2024年9月保险业总资产",
            "expected_information": "2024年9月保险业总资产",
            "source_scope": {"year": 2024, "month": 9},
            "semantic_constraints": {"indicator": "总资产", "institution": "保险业", "period": "2024-09"},
            "expected_value_type": "number",
            "expected_unit": "亿元",
        }],
        "operations": [],
        "requires_multiple_sources": False,
        "requires_table_retrieval": True,
        "requires_calculation": False,
        "requires_clarification": False,
    })


class _Planner:
    def plan(self, question, *args, **kwargs):
        return PlannerOutcome("ok", _asset_plan(question), 1)


def test_five_paraphrases_retrieve_the_same_core_fact_without_a_filename():
    index = HybridIndex(
        [{"doc_id": "insurance202409", "title": "2024年9月保险业经营情况表", "file_name": "insurance.xlsx"}],
        [],
        [{
            "evidence_id": "cell:insurance202409:C13",
            "doc_id": "insurance202409",
            "indicator": "总资产",
            "period": "2024-09",
            "value_text": "350023.51",
            "unit": "亿元",
            "cell_address": "C13",
            "context": "保险业 总资产 2024-09 350023.51亿元",
        }],
    )
    executor = BoundedAgentExecutor(
        _Planner(),
        RetrievalTools(index),
        Calculator(),
        AnswerGenerator(LLMClient(LLMClientConfig())),
    )

    states = [executor.run(question, [{"role": "user", "content": "当前为2025年"}]) for question in QUESTIONS]

    assert all(state.final_answer and "350023.51亿元" in state.final_answer for state in states)
    assert all(state.retrieval_results["r_asset"].evidence_ids == ["cell:insurance202409:C13"] for state in states)
    assert all(state.completeness and state.completeness.complete for state in states)

