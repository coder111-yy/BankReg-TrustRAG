from types import SimpleNamespace

from bankreg_trustrag.agent import build_agent_workflow
from bankreg_trustrag.query import extract_inline_choices, parse_query
from bankreg_trustrag.retrieval.index import Hit
from bankreg_trustrag.service import _multi_file_search


def _task(workflow, task_id):
    return next(task for task in workflow["tasks"] if task["id"] == task_id)


def test_single_file_text_lookup_uses_plain_retrieval():
    question = "查询《商业银行资本管理办法.pdf》中的资本充足率规定"
    parsed = parse_query(question)
    workflow = build_agent_workflow(parsed, question, [])

    assert parsed.intent == "lookup"
    assert parsed.requirements["multi_file"] is False
    assert parsed.requirements["table"] is False
    assert _task(workflow, "retrieve_primary")["action"] == "hybrid_retrieval_and_rerank"
    assert "table_operation" not in {task["id"] for task in workflow["tasks"]}


def test_number_and_structured_answer_formats_are_supported():
    number_query = parse_query("计算银行资本充足率，只返回数字")
    structured_query = parse_query("请用 JSON 结构化列出银行风险指标")

    assert number_query.answer_format == "number"
    assert structured_query.answer_format == "structured"


def test_single_excel_cell_lookup_adds_only_table_lookup_capability():
    question = "根据《2025年3月商业银行主要监管指标情况表.xlsx》，不良贷款率是多少？"
    parsed = parse_query(question)
    workflow = build_agent_workflow(parsed, question, [])

    assert parsed.intent == "lookup"
    assert parsed.answer_format == "short_answer"
    assert parsed.requirements["table"] is True
    assert parsed.requirements["calculation"] is False
    assert _task(workflow, "table_operation")["action"] == "locate_indicator_period_cell"


def test_excel_maximum_choice_combines_compare_calculation_and_choice():
    question = "根据 Excel 表，在截至当期-账面余额口径下，哪项数值最高？"
    choices = ["贷款", "债券", "股票", "存款"]
    parsed = parse_query(question, choices)
    workflow = build_agent_workflow(parsed, question, choices)

    assert parsed.intent == "compare"
    assert parsed.answer_format == "multiple_choice"
    assert parsed.requirements == {
        "retrieval": True,
        "multi_file": False,
        "table": True,
        "calculation": True,
        "comparison": True,
        "multi_hop": False,
        "option_evaluation": True,
    }
    assert _task(workflow, "table_operation")["action"] == "locate_cells_and_compare_deterministically"
    assert _task(workflow, "evaluate_options")["option_count"] == 4


def test_service_opens_table_retrieval_from_requirements_not_legacy_qa_type():
    class RecordingIndex:
        doc_by_id = {}
        model_status = {"mode": "disabled"}

        def __init__(self):
            self.qa_types = []

        def hybrid_search(self, query, qa_type, top_k=8, filters=None):
            self.qa_types.append(qa_type)
            return []

    from bankreg_trustrag.service import TrustRAGService

    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(min_trust=0.58, top_k=8)
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.index = RecordingIndex()
    service.semantic = SimpleNamespace(enabled=False)

    response = service.ask(
        "根据 Excel 表，在截至当期-账面余额口径下，哪项数值最高？"
        "A.贷款 B.债券 C.股票 D.存款"
    )

    assert response.qa_type == "regulatory_fact"
    assert response.query_plan["retrieval_qa_type"] == "table_lookup"
    assert set(service.index.qa_types) == {"table_lookup"}


def test_inline_choices_feed_the_same_answer_format_path():
    question = "银行资本要求中哪项正确？A.甲 B.乙 C.丙 D.丁"
    stem, choices = extract_inline_choices(question)
    parsed = parse_query(question)
    workflow = build_agent_workflow(parsed, stem, choices)

    assert choices == ["甲", "乙", "丙", "丁"]
    assert parsed.answer_format == "multiple_choice"
    assert parsed.requirements["option_evaluation"] is True
    assert _task(workflow, "evaluate_options")["option_count"] == 4


def test_two_file_lookup_uses_one_multi_file_retrieval_agent():
    question = "查询《文件一.pdf》和《文件二.pdf》中对银行风险分类的规定"
    parsed = parse_query(question)
    workflow = build_agent_workflow(parsed, question, [])
    retrieval_tasks = [task for task in workflow["tasks"] if task["agent"] == "Retrieval Agent"]

    assert parsed.intent == "lookup"
    assert parsed.requirements["multi_file"] is True
    assert len(retrieval_tasks) == 1
    assert retrieval_tasks[0]["action"] == "hybrid_multi_file_retrieval_and_rerank"


def test_multi_file_retrieval_searches_each_source_and_keeps_provenance():
    class RecordingIndex:
        def __init__(self):
            self.calls = []

        def hybrid_search(self, query, qa_type, top_k=8, filters=None):
            self.calls.append(dict(filters or {}))
            titles = (filters or {}).get("title") or []
            if titles == ["甲规则.pdf"]:
                return [Hit("text", {"evidence_id": "text:a", "doc_id": "a", "content": "甲规则"}, fused_score=0.3)]
            if titles == ["乙规则.pdf"]:
                return [Hit("text", {"evidence_id": "text:b", "doc_id": "b", "content": "乙规则"}, fused_score=0.2)]
            return []

    index = RecordingIndex()
    hits = _multi_file_search(
        index,
        "比较银行监管规则",
        "regulatory_fact",
        8,
        {"title": ["甲规则.pdf", "乙规则.pdf"]},
        {"title_hints": ["甲规则.pdf", "乙规则.pdf"]},
    )

    assert {hit.item["doc_id"] for hit in hits} == {"a", "b"}
    assert {tuple(call.get("title", [])) for call in index.calls} >= {("甲规则.pdf",), ("乙规则.pdf",)}


def test_pdf_rule_plus_excel_data_judgment_composes_capabilities():
    question = (
        "根据《监管规定.pdf》中的最低比例要求，以及《2023年保险资金运用表.xlsx》中的实际数据，"
        "判断该指标是否达标。"
    )
    parsed = parse_query(question)
    workflow = build_agent_workflow(parsed, question, [])

    assert parsed.intent == "judge"
    assert parsed.answer_format == "free_text"
    assert parsed.requirements == {
        "retrieval": True,
        "multi_file": True,
        "table": True,
        "calculation": True,
        "comparison": True,
        "multi_hop": True,
        "option_evaluation": False,
    }
    assert _task(workflow, "retrieve_primary")["scope"] == "multi_file"
    assert _task(workflow, "table_operation")["action"] == "locate_cells_and_compare_deterministically"


def test_multi_pdf_comparison_does_not_invent_table_requirement():
    question = "比较《甲规则.pdf》和《乙规则.pdf》中资本充足率要求的差异"
    parsed = parse_query(question)
    workflow = build_agent_workflow(parsed, question, [])

    assert parsed.intent == "compare"
    assert parsed.requirements["multi_file"] is True
    assert parsed.requirements["comparison"] is True
    assert parsed.requirements["table"] is False
    assert "table_operation" not in {task["id"] for task in workflow["tasks"]}


def test_multi_file_summary_keeps_summary_intent_and_comparison_requirement():
    question = "总结《甲规定.pdf》《乙规定.pdf》《丙规定.pdf》中对资本充足率的规定有什么不同"
    parsed = parse_query(question)
    workflow = build_agent_workflow(parsed, question, [])

    assert parsed.intent == "summarize"
    assert parsed.requirements["multi_file"] is True
    assert parsed.requirements["comparison"] is True
    assert parsed.requirements["multi_hop"] is True
    assert workflow["intent_decision"]["intent"] == "summarize"


def test_multi_file_table_calculation_is_composed_without_new_qa_type():
    question = "根据《2023年指标表.xlsx》和《2024年指标表.xlsx》计算银行资本充足率差值"
    parsed = parse_query(question)
    workflow = build_agent_workflow(parsed, question, [])

    assert parsed.intent == "calculate"
    assert parsed.requirements["multi_file"] is True
    assert parsed.requirements["table"] is True
    assert parsed.requirements["calculation"] is True
    assert _task(workflow, "table_operation")["action"] == "locate_cells_and_calculate_deterministically"
