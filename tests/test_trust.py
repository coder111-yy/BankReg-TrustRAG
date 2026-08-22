from types import SimpleNamespace

from bankreg_trustrag.retrieval.index import Hit
from bankreg_trustrag.service import TrustRAGService, _requested_year_missing
from bankreg_trustrag.verification import trust_decision, verify_claims
from bankreg_trustrag.reasoning import cross_file_answer, table_answer, text_answer


def test_unverified_number_is_not_answered():
    hit = Hit("text", {"evidence_id": "e1", "content": "阈值为 5%"}, fused_score=0.04)
    verification = verify_claims("阈值为 10%", "阈值是多少", [hit], ["阈值为 10%"])
    assert not verification.numeric_ok
    decision = trust_decision([hit], verification, "clause_threshold")
    assert decision["decision"] in {"clarify", "refuse"}


def test_verification_links_each_claim_to_supporting_evidence():
    hit = Hit("text", {"evidence_id": "text:d1:p1", "content": "商业银行不得挪用客户资金。"})

    verification = verify_claims("商业银行不得挪用客户资金。", "资金使用要求是什么", [hit], ["商业银行不得挪用客户资金。"])

    assert verification.passed
    assert verification.claim_results == [{
        "claim_id": "claim_1",
        "text": "商业银行不得挪用客户资金。",
        "evidence_ids": ["text:d1:p1"],
        "supported": True,
    }]


def test_verification_uses_bounded_clause_context_window():
    hit = Hit(
        "text",
        {
            "evidence_id": "text:d1:p6",
            "content": "(一)直接发行且实缴的。",
            "context_window": "一、核心一级资本工具的合格标准 (一)直接发行且实缴的。",
        },
    )

    verification = verify_claims("核心一级资本工具直接发行且实缴。", "资本工具要求", [hit], ["核心一级资本工具直接发行且实缴。"])

    assert verification.passed


def test_verification_does_not_treat_generic_ke_as_normative_word():
    hit = Hit("text", {"evidence_id": "text:d1:p1", "content": "商业银行监管指标说明。"})

    verification = verify_claims("当前未检索到可引用的监管阈值。", "监管阈值是多少", [hit], [])

    assert verification.normative_strength_ok


def test_broad_table_question_requests_indicator():
    hit = Hit("table", {"evidence_id": "cell:d1:Sheet1:B1", "source_title": "2023年12月保险业经营情况表", "period": "2023-12", "indicator": "保险业经营数据", "value_text": '"保险业经营数据"', "context": "保险业经营数据"})
    draft = table_answer("请查询2023年12月保险业经营情况表中的经营数据", None, [hit])
    assert draft.operations[0]["type"] == "clarification"
    assert "补充指标" in draft.answer


def test_ambiguous_period_question_does_not_claim_a_source():
    hit = Hit("table", {"evidence_id": "cell:qa:Sheet1:F20", "source_title": "QA数据", "period": "2023-10", "indicator": "Q019", "value_text": '"题目文本"', "context": "题目文本"})

    draft = table_answer("2023年10月的指标是多少", None, [hit])

    assert draft.operations[0]["source"] is None
    assert "2023-10" in draft.answer
    assert "未能唯一确定统计表" in draft.answer


def test_table_answer_selects_numeric_cell_for_requested_indicator():
    hits = [
        Hit("table", {"evidence_id": "cell:d1:Sheet1:B5", "source_title": "2023年10月保险业经营情况表", "period": "2023-10", "indicator": "原保险保费收入", "value_text": '"原保险保费收入"', "context": "原保险保费收入"}),
        Hit("table", {"evidence_id": "cell:d1:Sheet1:C5", "source_title": "2023年10月保险业经营情况表", "period": "2023-10", "indicator": "原保险保费收入", "value_text": "45167.98", "context": "原保险保费收入 | 45167.98"}),
    ]

    draft = table_answer("请查询《2023年10月保险业经营情况表.xls》中的原保险保费收入", None, hits)

    assert draft.operations[0]["type"] == "table_lookup"
    assert "45167.98" in draft.answer


def test_table_answer_formats_fractional_ratio_as_percentage_and_selects_march_quarter():
    hits = [
        Hit("table", {"evidence_id": "cell:d1:Sheet1:B14", "source_title": "2025年商业银行主要监管指标情况表", "period": "2025", "indicator": "不良贷款率", "value_text": "0.01513", "column_header": "一季度", "cell_address": "B14", "context": "不良贷款率 | 一季度 | 0.01513"}),
        Hit("table", {"evidence_id": "cell:d1:Sheet1:C14", "source_title": "2025年商业银行主要监管指标情况表", "period": "2025", "indicator": "不良贷款率", "value_text": "0.01491", "column_header": "二季度", "cell_address": "C14", "context": "不良贷款率 | 二季度 | 0.01491"}),
    ]

    draft = table_answer("2025年3月商业银行主要监管指标情况表中的不良贷款率是多少？", None, hits)

    assert "2025年一季度" in draft.answer
    assert "1.513%" in draft.answer
    assert draft.operations[0]["cell"] == "B14"
    assert draft.operations[0]["raw_value"] == 0.01513


def test_verification_accepts_fractional_source_value_rendered_as_percentage():
    hit = Hit("table", {"evidence_id": "cell:d1:Sheet1:B14", "source_title": "2025年商业银行主要监管指标情况表", "period": "2025", "indicator": "不良贷款率", "value_text": "0.01513", "column_header": "一季度", "cell_address": "B14", "context": "不良贷款率 | 一季度 | 0.01513"})
    answer = "不良贷款率在2025年一季度的值为 1.513%。"

    verification = verify_claims(answer, "2025年3月商业银行主要监管指标情况表中的不良贷款率是多少？", [hit], [answer])

    assert verification.passed


def test_table_answer_keeps_monthly_period_instead_of_inventing_quarter():
    hit = Hit("table", {"evidence_id": "cell:d1:Sheet1:C6", "source_title": "2023年10月财产保险公司经营情况表", "period": "2023-10", "indicator": "原保险保费收入", "value_text": "13428.79", "column_header": "单位:亿元、万件", "cell_address": "C6", "context": "原保险保费收入 | 单位:亿元、万件 | 13428.79"})

    draft = table_answer("《2023年10月财产保险公司经营情况表.xlsx》中“原保险保费收入”的统计值是多少？", None, [hit])
    verification = verify_claims(draft.answer, "2023年10月财产保险公司经营情况表中的原保险保费收入是多少", [hit], draft.claims)

    assert "2023年10月" in draft.answer
    assert "四季度" not in draft.answer
    assert "13428.79" in draft.answer
    assert verification.passed


def test_table_answer_selects_requested_row_and_column_dimension():
    hit = Hit("table", {"evidence_id": "cell:d1:Sheet1:C4", "source_title": "2023年10月全国各地区原保险保费收入情况表", "period": "合计 / 45167.98", "indicator": "全国合计", "value_text": "45167.98", "column_header": "合计 / 45167.98", "cell_address": "C4", "context": "全国合计 | 合计 / 45167.98 | 45167.98"})

    draft = table_answer("2023年10月全国各地区原保险保费收入情况表.xlsx中“全国合计”在“合计”口径下的数值是多少", None, [hit])
    verification = verify_claims(draft.answer, "2023年10月全国各地区原保险保费收入情况表中的数据", [hit], draft.claims)

    assert "全国合计" in draft.answer
    assert "合计" in draft.answer
    assert "45167.98" in draft.answer
    assert draft.operations[0]["cell"] == "C4"
    assert verification.passed


def test_cross_file_judgment_uses_latest_quarter_and_refuses_without_threshold():
    hits = [
        Hit("table", {"evidence_id": "cell:stat:Sheet1:B14", "source_title": "2025年商业银行主要监管指标情况表", "period": "2025", "indicator": "不良贷款率", "value_text": "0.01513", "column_header": "一季度", "cell_address": "B14", "context": "不良贷款率 | 一季度 | 0.01513"}),
        Hit("table", {"evidence_id": "cell:stat:Sheet1:E14", "source_title": "2025年商业银行主要监管指标情况表", "period": "2025", "indicator": "不良贷款率", "value_text": "0.01496", "column_header": "四季度", "cell_address": "E14", "context": "不良贷款率 | 四季度 | 0.01496"}),
        Hit("table", {"evidence_id": "cell:rule:指标解释:C6", "source_title": "2025年国家金融监督管理总局监管统计信息发布日程表", "sheet_name": "指标解释", "period": "2025", "indicator": "4.0", "value_text": '"不良贷款余额 / 各项贷款余额 × 100%"', "cell_address": "C6", "context": "不良贷款率 | 不良贷款余额 / 各项贷款余额 × 100%"}),
    ]

    draft = cross_file_answer("根据监管制度和2025年商业银行主要监管指标情况表，判断当前不良贷款率是否满足监管要求，并说明计算依据。", hits)

    assert draft.operations[0]["type"] == "refusal"
    assert draft.operations[0]["cell"] == "E14"
    assert "2025年四季度" in draft.answer
    assert "1.496%" in draft.answer
    assert "不良贷款余额 / 各项贷款余额 × 100%" in draft.answer
    assert "监管阈值" in draft.answer
    assert "cell:rule:指标解释:C6" in draft.operations[0]["display_evidence_ids"]


def test_cross_file_judgment_compares_value_only_when_threshold_is_evidenced():
    hits = [
        Hit("table", {"evidence_id": "cell:stat:Sheet1:E14", "source_title": "2025年商业银行主要监管指标情况表", "period": "2025", "indicator": "不良贷款率", "value_text": "0.01496", "column_header": "四季度", "cell_address": "E14", "context": "不良贷款率 | 四季度 | 0.01496"}),
        Hit("text", {"evidence_id": "text:rule:p1", "source_title": "商业银行风险监管核心指标", "content": "商业银行不良贷款率不得超过5%。"}),
    ]

    draft = cross_file_answer("根据监管制度和2025年商业银行主要监管指标情况表，判断当前不良贷款率是否满足监管要求。", hits)

    assert draft.operations[0]["type"] == "cross_file_judgment"
    assert "满足监管要求" in draft.answer
    assert "1.496%" in draft.answer
    assert "5%" in draft.answer


def test_absolute_future_risk_question_does_not_return_unrelated_fragment():
    hit = Hit("text", {"evidence_id": "text:d1:p847", "content": "商业银行已计提的信用风险损失准备是否能够有效覆"})

    draft = text_answer("某银行明年是否一定不会发生风险？", None, [hit])

    assert draft.operations[0]["type"] == "clarification"
    assert "不能根据当前问题断定" in draft.answer
    assert draft.answer != hit.item["content"]


def test_unbounded_latest_regulation_question_is_refused():
    hit = Hit("text", {"evidence_id": "text:d1:p1", "content": "监管部门规定的标准。"})

    draft = text_answer("请告诉我监管部门最新规定。", None, [hit])

    assert draft.operations[0]["type"] == "refusal"
    assert "拒绝直接给出结论" in draft.answer


def test_missing_requested_year_is_detected():
    hit = Hit("text", {"evidence_id": "text:d1:p1", "content": "2015年保险市场年报"})

    assert _requested_year_missing(["2028"], [hit]) == "2028"


def test_invalid_short_year_is_rejected_even_if_evidence_exists():
    hit = Hit("text", {"evidence_id": "text:d1:p1", "content": "209年相关资料"})

    assert _requested_year_missing(["209"], [hit]) == "209"


def test_service_keeps_source_specific_clarification():
    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(min_trust=0.58, top_k=10)
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.index = SimpleNamespace(
        doc_by_id={"d1": {"title": "2023年12月保险业经营情况表", "file_name": "2023年12月保险业经营情况表.xlsx"}},
        hybrid_search=lambda *args, **kwargs: [
            Hit(
                "table",
                {
                    "evidence_id": "cell:d1:Sheet1:B1",
                    "doc_id": "d1",
                    "source_title": "2023年12月保险业经营情况表",
                    "period": "2023-12",
                    "indicator": "保险业经营数据",
                    "value_text": '"保险业经营数据"',
                    "context": "保险业经营数据",
                },
                fused_score=0.04,
            )
        ],
    )

    response = service.ask("请查询《2023年12月保险业经营情况表.xlsx》中的保险业经营数据")

    assert response.trust["decision"] == "clarify"
    assert "已定位到" in response.answer
    assert "补充指标" in response.answer
