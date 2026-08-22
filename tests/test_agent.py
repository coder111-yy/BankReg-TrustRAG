from types import SimpleNamespace

from bankreg_trustrag.agent import build_agent_workflow, organize_evidence, run_choice_agent
from bankreg_trustrag.query import extract_inline_choices, parse_query
from bankreg_trustrag.retrieval.index import HybridIndex
from bankreg_trustrag.retrieval.index import Hit
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
    assert {"understand", "retrieve_primary", "retrieve_rule", "table_calculation", "generate", "verify"}.issubset(task_ids)
    assert workflow["question_understanding"]["requires_multi_hop"] is True


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

    assert response.answer.startswith("选项 A")
    assert response.query_plan["agent"]["route"] == "choice_agent"
    assert response.query_plan["agent"]["selected_option"] == "A"
    assert response.query_plan["agent_workflow"]["evidence_organization"]["items"][0]["role"] == "option_support"
    assert response.query_plan["operations"][0]["display_evidence_ids"] == ["text:d1:p1"]


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
