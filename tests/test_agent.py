from types import SimpleNamespace

from bankreg_trustrag.agent import build_agent_workflow, identify_intent, organize_evidence, run_choice_agent
from bankreg_trustrag.query import extract_inline_choices, parse_query
from bankreg_trustrag.retrieval.index import HybridIndex
from bankreg_trustrag.retrieval.index import Hit
from bankreg_trustrag.schemas import ParsedQuery
from bankreg_trustrag.service import TrustRAGService


def test_inline_multiple_choice_is_split_from_retrieval_stem():
    stem, choices = extract_inline_choices(
        "关于资金使用，下列哪项正确？A. 商业银行不得挪用客户资金；B. 商业银行可以挪用客户资金；C. 商业银行应当挪用客户资金；D. 商业银行可以随意使用客户资金"
    )

    assert stem == "关于资金使用，下列哪项正确？"
    assert choices[0] == "商业银行不得挪用客户资金"
    assert len(choices) == 4


def test_inline_multiple_choice_accepts_comma_labels_used_by_frontend_paste():
    stem, choices = extract_inline_choices(
        "根据《消费金融公司管理办法》，下列哪项表述正确？A,消费金融公司是经国家金融监管管理总局批准设立的非银行金融机构。B,核心数据遭到泄露。C,核心数据遭到破坏。D,重要数据遭到泄露。"
    )

    assert stem.startswith("根据《消费金融公司管理办法》")
    assert len(choices) == 4
    assert choices[0].startswith("消费金融公司")


def test_quoted_material_name_becomes_a_source_filter_hint():
    parsed = parse_query("根据《数据安全事件分级》，下列哪项正确？")

    assert parsed.entities["title_hints"] == ["数据安全事件分级"]


def test_agent_workflow_plans_cross_file_rule_table_and_verification_tasks():
    parsed = parse_query("根据监管制度和2025年商业银行主要监管指标情况表，判断不良贷款率是否满足监管要求")

    workflow = build_agent_workflow(parsed, parsed.original_query, [], {"title": ["商业银行主要监管指标情况表"]})

    task_ids = {task["id"] for task in workflow["tasks"]}
    assert {"understand", "retrieve_primary", "table_operation", "generate", "verify"}.issubset(task_ids)
    assert "retrieve_rule" not in task_ids
    retrieval = next(task for task in workflow["tasks"] if task["id"] == "retrieve_primary")
    assert retrieval["action"] == "hybrid_multi_file_retrieval_and_rerank"
    assert retrieval["preserve_source_boundaries"] is True
    assert workflow["intent"] == "judge"
    assert workflow["answer_format"] == "free_text"
    assert isinstance(workflow["intent"], str)
    assert workflow["question_understanding"]["requires_multi_hop"] is True


def test_intent_answer_format_and_valid_choice_count_are_independent():
    decision = identify_intent(
        "根据表格比较哪一项数值最高？",
        ["甲", " ", "", "乙"],
        "table_lookup",
    )

    assert decision == {
        "intent": "compare",
        "answer_format": "multiple_choice",
        "qa_type": "table_lookup",
        "choice_count": 2,
        "source": "explicit_options",
    }


def test_same_qa_type_can_build_different_workflows_from_requirements():
    lookup = ParsedQuery(
        "查询资本充足率规定",
        "regulatory_fact",
        requirements={
            "retrieval": True,
            "multi_file": False,
            "table": False,
            "calculation": False,
            "comparison": False,
            "multi_hop": False,
            "option_evaluation": False,
        },
    )
    comparison = ParsedQuery(
        "比较两个表格中的资本充足率",
        "regulatory_fact",
        intent="compare",
        requirements={
            "retrieval": True,
            "multi_file": True,
            "table": True,
            "calculation": True,
            "comparison": True,
            "multi_hop": True,
            "option_evaluation": False,
        },
    )

    lookup_tasks = {task["id"] for task in build_agent_workflow(lookup, lookup.original_query, [])["tasks"]}
    comparison_workflow = build_agent_workflow(comparison, comparison.original_query, [])
    comparison_tasks = {task["id"] for task in comparison_workflow["tasks"]}

    assert "table_operation" not in lookup_tasks
    assert "table_operation" in comparison_tasks
    assert next(task for task in comparison_workflow["tasks"] if task["id"] == "retrieve_primary")["scope"] == "multi_file"


def test_evidence_organization_keeps_statistical_role_when_cross_file_refuses():
    hit = Hit("table", {"evidence_id": "cell:stat:E14", "source_title": "监管指标表", "cell_address": "E14"})

    items = organize_evidence(
        [hit],
        ["cell:stat:E14"],
        [{"type": "refusal", "table_evidence_ids": ["cell:stat:E14"], "reason": "缺少监管阈值"}],
    )

    assert items[0]["role"] == "statistical_value"


def test_choice_agent_retrieves_options_independently_and_selects_supported_option():
    index = HybridIndex(
        [{"doc_id": "d1", "title": "资金管理办法", "file_name": "rule.docx", "status": "effective"}],
        [{"evidence_id": "text:d1:p1", "doc_id": "d1", "content": "商业银行不得挪用客户资金。"}],
        [],
    )
    result = run_choice_agent(
        index,
        "关于资金使用，下列哪项正确？",
        [
            "商业银行不得挪用客户资金",
            "商业银行可以挪用客户资金",
            "商业银行应当挪用客户资金",
            "商业银行可以随意使用客户资金",
        ],
        "regulatory_fact",
    )

    assert result.selected_label == "A"
    assert len(result.option_hits) == 4
    assert result.assessments[0]["evidence_ids"] == ["text:d1:p1"]
    assert result.human_in_loop is None


def test_choice_agent_reranks_shared_stem_only_once():
    class RecordingIndex:
        def __init__(self):
            self.calls = []

        def hybrid_search(self, query, qa_type, top_k=8, filters=None, *, rerank=True, dense=True):
            self.calls.append((query, rerank, dense))
            return [Hit("text", {"evidence_id": "text:d1:p1", "content": "商业银行不得挪用客户资金。"}, fused_score=0.2)]

    index = RecordingIndex()
    run_choice_agent(index, "关于资金使用，下列哪项正确？", ["商业银行不得挪用客户资金", "商业银行可以挪用客户资金"], "regulatory_fact")

    assert [(rerank, dense) for _, rerank, dense in index.calls] == [(True, True), (False, False), (False, False)]


def test_choice_agent_compares_table_column_values_and_accepts_total_alias():
    index = HybridIndex(
        [{"doc_id": "report", "title": "2024年9月全国各地区原保险保费收入情况表", "file_name": "report.xlsx", "status": "effective"}],
        [],
        [
            {"evidence_id": "cell:report:G4", "doc_id": "report", "indicator": "全国合计", "row_header": "全国合计", "column_header": "健康险", "period": "2024-09", "value_text": "8225.18", "cell_address": "G4", "context": "全国合计 | 健康险 | 8225.18"},
            {"evidence_id": "cell:report:G5", "doc_id": "report", "indicator": "公司本级", "row_header": "公司本级", "column_header": "健康险", "period": "2024-09", "value_text": "2.96", "cell_address": "G5", "context": "公司本级 | 健康险 | 2.96"},
            {"evidence_id": "cell:report:G6", "doc_id": "report", "indicator": "北 京", "row_header": "北 京", "column_header": "健康险", "period": "2024-09", "value_text": "495.59", "cell_address": "G6", "context": "北 京 | 健康险 | 495.59"},
            {"evidence_id": "cell:report:G7", "doc_id": "report", "indicator": "天 津", "row_header": "天 津", "column_header": "健康险", "period": "2024-09", "value_text": "97.37", "cell_address": "G7", "context": "天 津 | 健康险 | 97.37"},
        ],
    )

    result = run_choice_agent(
        index,
        "根据《2024年9月全国各地区原保险保费收入情况表》，在“健康险”口径下，以下哪一项数值最高？",
        ["全国总计", "北京", "天津", "公司本级"],
        "table_lookup",
        filters={"title": ["全国各地区原保险保费收入情况表"]},
    )

    assert result.selected_label == "A"
    assert result.human_in_loop is None
    assert result.assessments[0]["table_comparison"]["value"] == 8225.18
    assert result.assessments[0]["evidence_ids"] == [
        "cell:report:G4",
        "cell:report:G6",
        "cell:report:G7",
        "cell:report:G5",
    ]

    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(min_trust=0.58, top_k=8)
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.index = index
    service.semantic = SimpleNamespace(enabled=False)
    response = service.ask(
        "根据 Excel 附件《2024年9月全国各地区原保险保费收入情况表》，"
        "在“健康险”口径下，以下哪一项数值最高？"
        "A.全国总计 B.北京 C.天津 D.公司本级"
    )

    assert response.answer.startswith("明确答案：选项 A（全国总计）")
    assert "A=8225.18，B=495.59，C=97.37，D=2.96" in response.answer
    assert response.answer.endswith("最终结论：选择 A。")
    assert response.trust["decision"] == "answer"
    assert response.query_plan["agent"]["selected_option"] == "A"
    assert response.query_plan["agent_workflow"]["answer_generation"]["strategy"] == "deterministic_table_comparison_explanation"
    assert [item["cell_address"] for item in response.evidence] == ["G4", "G6", "G7", "G5"]


def test_table_comparison_matches_generic_hierarchy_prefix_to_plain_option():
    index = HybridIndex(
        [{"doc_id": "report", "title": "年度机构统计表", "file_name": "report.xlsx"}],
        [],
        [
            {"evidence_id": "cell:report:B5", "doc_id": "report", "indicator": "机构合计", "row_header": "机构合计", "column_header": "一季度", "period": "2023", "value_text": "300", "cell_address": "B5", "context": "机构合计 | 一季度 | 300"},
            {"evidence_id": "cell:report:B6", "doc_id": "report", "indicator": "其中:甲类机构", "row_header": "其中:甲类机构", "column_header": "一季度", "period": "2023", "value_text": "200", "cell_address": "B6", "context": "其中:甲类机构 | 一季度 | 200"},
            {"evidence_id": "cell:report:B7", "doc_id": "report", "indicator": "乙类机构", "row_header": "乙类机构", "column_header": "一季度", "period": "2023", "value_text": "100", "cell_address": "B7", "context": "乙类机构 | 一季度 | 100"},
        ],
    )

    result = run_choice_agent(
        index,
        "根据年度机构统计表，在“一季度”口径下，以下哪一项数值最高？",
        ["甲类机构", "乙类机构", "机构合计"],
        "table_lookup",
    )

    assert result.selected_label == "C"
    assert result.human_in_loop is None
    assert result.assessments[0]["table_comparison"]["cell_address"] == "B6"
    assert result.assessments[2]["table_comparison"]["cell_address"] == "B5"


def test_table_comparison_uses_structured_row_queries_for_every_option():
    rows = {
        "全国合计": ("G4", 8225.18),
        "北京": ("G6", 495.59),
        "天津": ("G7", 97.37),
        "公司本级": ("G5", 2.96),
    }

    class StructuredOnlyIndex:
        def __init__(self):
            self.calls = []

        def hybrid_search(self, query, qa_type, top_k=8, filters=None, *, rerank=True, dense=True):
            self.calls.append(query)
            hits = []
            for row, (cell, value) in rows.items():
                if f"“{row}”在“健康险”口径下" not in query:
                    continue
                hits.append(Hit("table", {
                    "evidence_id": f"cell:report:{cell}",
                    "indicator": row,
                    "row_header": row,
                    "column_header": "健康险",
                    "period": "2024-09",
                    "value_text": str(value),
                    "cell_address": cell,
                    "context": f"{row} | 健康险 | {value}",
                }, fused_score=0.2))
            return hits

    index = StructuredOnlyIndex()
    result = run_choice_agent(
        index,
        "根据2024年9月统计表，在“健康险”口径下，以下哪一项数值最高？",
        list(rows),
        "table_lookup",
    )

    assert result.selected_label == "A"
    assert all(
        any(f"“{row}”在“健康险”口径下" in query for query in index.calls)
        for row in rows
    )


def test_table_comparison_accepts_numbered_repeated_rows_when_winner_is_unambiguous():
    index = HybridIndex(
        [{"doc_id": "report", "title": "2025年9月保险业经营情况表", "file_name": "report.xlsx", "status": "effective"}],
        [],
        [
            {"evidence_id": "cell:report:C7", "doc_id": "report", "indicator": "原保险保费收入", "row_header": "原保险保费收入", "column_header": "单位:亿元 / 本年累计/截至当期", "period": "2025-09", "value_text": "52145.77", "cell_address": "C7", "context": "原保险保费收入 | 本年累计/截至当期 | 52145.77"},
            {"evidence_id": "cell:report:C8", "doc_id": "report", "indicator": "1、财产险", "row_header": "1、财产险", "column_header": "单位:亿元 / 本年累计/截至当期", "period": "2025-09", "value_text": "11250.32", "cell_address": "C8", "context": "1、财产险 | 本年累计/截至当期 | 11250.32"},
            {"evidence_id": "cell:report:C9", "doc_id": "report", "indicator": "2、人身险", "row_header": "2、人身险", "column_header": "单位:亿元 / 本年累计/截至当期", "period": "2025-09", "value_text": "40895.45", "cell_address": "C9", "context": "2、人身险 | 本年累计/截至当期 | 40895.45"},
            {"evidence_id": "cell:report:C11", "doc_id": "report", "indicator": "1、财产险", "row_header": "1、财产险", "column_header": "单位:亿元 / 本年累计/截至当期", "period": "2025-09", "value_text": "6981.4", "cell_address": "C11", "context": "1、财产险 | 本年累计/截至当期 | 6981.4"},
            {"evidence_id": "cell:report:C12", "doc_id": "report", "indicator": "2、人身险", "row_header": "2、人身险", "column_header": "单位:亿元 / 本年累计/截至当期", "period": "2025-09", "value_text": "11725.37", "cell_address": "C12", "context": "2、人身险 | 本年累计/截至当期 | 11725.37"},
            {"evidence_id": "cell:report:C13", "doc_id": "report", "indicator": "总资产", "row_header": "总资产", "column_header": "单位:亿元 / 本年累计/截至当期", "period": "2025-09", "value_text": "404005.89", "cell_address": "C13", "context": "总资产 | 本年累计/截至当期 | 404005.89"},
        ],
    )
    question = (
        "根据 Excel 附件《2025年9月保险业经营情况表》（工作表：保险业经营数据（月度）），"
        "在“本年累计/截至当期”口径下，以下哪一项数值最高？"
    )

    result = run_choice_agent(
        index,
        question,
        ["人身险", "原保险保费收入", "总资产", "财产险"],
        "table_lookup",
        filters={"title": ["2025年9月保险业经营情况表"]},
    )

    assert result.selected_label == "C"
    assert result.human_in_loop is None
    assert result.assessments[2]["table_comparison"]["value"] == 404005.89
    assert result.assessments[2]["evidence_ids"] == [
        "cell:report:C9",
        "cell:report:C12",
        "cell:report:C7",
        "cell:report:C13",
        "cell:report:C8",
        "cell:report:C11",
    ]

    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(min_trust=0.58, top_k=8)
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.index = index
    service.semantic = SimpleNamespace(enabled=False)
    response = service.ask(question + "A.人身险 B.原保险保费收入 C.总资产 D.财产险")

    assert response.answer.startswith("明确答案：选项 C（总资产）")
    assert "A=40895.45，B=52145.77，C=404005.89，D=11250.32" in response.answer
    assert response.answer.endswith("最终结论：选择 C。")
    assert response.trust["decision"] == "answer"
    assert response.query_plan["agent"]["selected_option"] == "C"


def test_choice_agent_hands_off_when_no_option_has_evidence():
    index = HybridIndex(
        [{"doc_id": "d1", "title": "无关资料", "file_name": "other.docx", "status": "effective"}],
        [],
        [],
    )
    result = run_choice_agent(
        index,
        "关于监管要求，下列哪项正确？",
        ["选项一", "选项二", "选项三", "选项四"],
        "regulatory_fact",
    )

    assert result.selected_index is None
    assert result.human_in_loop is not None
    assert result.human_in_loop["status"] == "pending"


def test_choice_agent_keeps_fifth_option_and_supports_legacy_search_fixture():
    class LegacyIndex:
        """Deliberately omits rerank/dense keyword parameters."""

        def __init__(self):
            self.calls = []

        def hybrid_search(self, query, qa_type, top_k=8, filters=None):
            self.calls.append(query)
            if "选项E" in query:
                return [Hit(
                    "text",
                    {"evidence_id": "text:d1:e", "content": "第五个正确答案"},
                    lexical_score=2.0,
                    fused_score=0.3,
                )]
            return []

    index = LegacyIndex()
    result = run_choice_agent(
        index,
        "关于银行监管要求，下列哪项正确？",
        ["错误一", "错误二", "错误三", "错误四", "第五个正确答案"],
        "regulatory_fact",
    )

    assert len(result.choices) == 5
    assert len(result.option_hits) == 5
    assert result.assessments[4]["label"] == "E"
    assert result.selected_label == "E"
    assert any("选项E" in query for query in index.calls)


def test_choice_agent_accepts_two_supported_claims_at_boundary_score():
    index = HybridIndex(
        [{"doc_id": "d1", "title": "行政许可目录", "file_name": "rule.pdf", "status": "effective"}],
        [
            {"evidence_id": "text:d1:p1", "doc_id": "d1", "content": "行政许可目录包括机构设立、机构变更和机构终止。"},
            {"evidence_id": "text:d1:p2", "doc_id": "d1", "content": "法人机构筹建审批属于机构设立类行政许可事项。"},
        ],
        [],
    )

    result = run_choice_agent(
        index,
        "关于行政许可目录，下列哪项正确？",
        ["行政许可目录包括机构设立、机构变更和机构终止；法人机构筹建审批属于机构设立类行政许可事项。", "行政许可目录包括机构设立；寿险折现率由基础利率曲线加综合溢价形成。"],
        "business_process",
    )

    assert result.selected_label == "A"


def test_service_uses_choice_agent_for_inline_options():
    class FakeIndex:
        doc_by_id = {"d1": {"title": "资金管理办法", "file_name": "rule.docx", "status": "effective"}}
        model_status = {"mode": "disabled"}

        def hybrid_search(self, query, qa_type, top_k=8, filters=None):
            if "不得挪用客户资金" in query:
                return [Hit("text", {"evidence_id": "text:d1:p1", "doc_id": "d1", "content": "商业银行不得挪用客户资金。"}, lexical_score=2.0, fused_score=0.2)]
            return []

    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(min_trust=0.58, top_k=8)
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.index = FakeIndex()
    service.semantic = SimpleNamespace(enabled=False)

    response = service.ask(
        "关于资金使用，下列哪项正确？A. 商业银行不得挪用客户资金；B. 商业银行可以挪用客户资金；C. 商业银行应当挪用客户资金；D. 商业银行可以随意使用客户资金"
    )

    assert response.answer.startswith("明确答案：选项 A")
    assert "核心证据" in response.answer
    assert "商业银行不得挪用客户资金" in response.answer
    assert response.answer.endswith("最终结论：选择 A。")
    assert "选项 B" not in response.answer
    assert 80 <= len(response.answer) <= 220
    assert response.query_plan["agent"]["route"] == "choice_agent"
    assert response.query_plan["agent"]["selected_option"] == "A"
    assert response.query_plan["agent_workflow"]["answer_generation"]["strategy"] == "evidence_grounded_choice_explanation"
    assert response.query_plan["agent_workflow"]["evidence_organization"]["items"][0]["role"] == "option_support"
    assert response.query_plan["operations"][0]["display_evidence_ids"] == ["text:d1:p1"]
    assert response.query_plan["operations"][0]["evidence_explanations"][0]["evidence_id"] == "text:d1:p1"


def test_service_records_verification_retry_for_unconfirmed_current_version():
    class RetryIndex:
        doc_by_id = {"d1": {"title": "资本管理办法", "file_name": "rule.docx", "status": "unknown"}}
        model_status = {"mode": "disabled"}

        def __init__(self):
            self.calls = 0

        def hybrid_search(self, *args, **kwargs):
            self.calls += 1
            return [Hit("text", {"evidence_id": "text:d1:p1", "doc_id": "d1", "content": "商业银行资本充足率为10%。"}, lexical_score=2.0, fused_score=0.2)]

    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(min_trust=0.58, top_k=8)
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.index = RetryIndex()
    service.semantic = SimpleNamespace(enabled=False)

    response = service.ask("当前商业银行资本充足率是多少？")

    retry = response.query_plan["agent_workflow"]["assisted_verification"]["retry"]
    assert service.index.calls == 2
    assert retry is not None
    assert retry["type"] == "verification_retry"


def test_service_refuses_out_of_scope_question_before_retrieval():
    class FakeIndex:
        model_status = {"mode": "disabled"}

        def hybrid_search(self, *args, **kwargs):
            raise AssertionError("域外问题不应进入检索")

    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(min_trust=0.58, top_k=8)
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.index = FakeIndex()
    service.semantic = SimpleNamespace(enabled=False)

    response = service.ask("你好")

    assert response.trust["decision"] == "refuse"
    assert response.trust["score"] == 0.2
    assert response.evidence == []
    assert response.query_plan["scope"]["in_scope"] is False
    assert "天气" not in response.answer
