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


def test_percentage_from_calculation_trace_is_verified():
    hit = Hit("table", {"evidence_id": "e1", "value_text": "47945.35", "unit": "亿元"})
    answer = "同比增长率为12.74%。"
    operation = {
        "type": "calculation",
        "operation": "growth_rate",
        "result": "12.74",
        "unit": "%",
        "evidence_ids": ["e1"],
    }

    verification = verify_claims(answer, "同比增长率是多少", [hit], [answer], [operation])

    assert verification.numeric_ok is True
    assert verification.unit_ok is True


def test_verification_links_each_claim_to_supporting_evidence():
    hit = Hit("text", {"evidence_id": "text:d1:p1", "content": "商业银行不得挪用客户资金。"})

    verification = verify_claims("商业银行不得挪用客户资金。", "资金使用要求是什么", [hit], ["商业银行不得挪用客户资金。"])

    assert verification.passed
    assert verification.claim_results == [{
        "claim_id": "claim_1",
        "text": "商业银行不得挪用客户资金。",
        "evidence_ids": ["text:d1:p1"],
        "calculation_ids": [],
        "supported": True,
    }]


def test_verification_accepts_calculated_spread_and_links_calculation_trace():
    hits = [
        Hit("table", {"evidence_id": "e_life", "value_text": "35378.91", "unit": "亿元", "context": "人身险原保险保费收入 35378.91"}),
        Hit("table", {"evidence_id": "e_property", "value_text": "15867.79", "unit": "亿元", "context": "财产险原保险保费收入 15867.79"}),
        Hit("table", {"evidence_id": "e_industry", "value_text": "51246.71", "unit": "亿元", "context": "保险业原保险保费收入 51246.71"}),
    ]
    operations = [
        {
            "type": "calculation", "id": "calc1", "operation": "sum",
            "input_refs": ["r1", "r2"], "result": "51246.7", "unit": "亿元",
            "trace": "35378.91 + 15867.79 = 51246.7",
            "evidence_ids": ["e_life", "e_property"], "details": {},
        },
        {
            "type": "calculation", "id": "calc2", "operation": "verify_consistency",
            "input_refs": ["calc1", "r3"], "result": "false", "unit": None,
            "trace": "spread 0.01 <= tolerance 0 = false",
            "evidence_ids": ["e_life", "e_property", "e_industry"],
            "details": {"spread": "0.01", "tolerance": "0"},
        },
    ]
    answer = (
        "人身险与财产险原保险保费收入之和为51246.70亿元，"
        "保险业原保险保费收入为51246.71亿元；两者相差0.01亿元，不一致。"
    )
    claims = [
        "人身险与财产险原保险保费收入之和为51246.70亿元，保险业原保险保费收入为51246.71亿元",
        "两者相差0.01亿元，不一致",
    ]

    verification = verify_claims(answer, "两者是否一致", hits, claims, operations)

    assert verification.passed
    assert verification.numeric_ok is True
    assert verification.failure_details == []
    assert verification.claim_results[1]["calculation_ids"] == ["calc2"]
    assert verification.claim_results[1]["evidence_ids"] == ["e_life", "e_property", "e_industry"]


def test_verification_does_not_decide_consistency_language():
    hit = Hit("table", {"evidence_id": "e1", "value_text": "51246.71", "unit": "亿元"})
    operation = {
        "type": "calculation", "id": "calc2", "operation": "verify_consistency",
        "input_refs": ["calc1", "r3"], "result": "false", "unit": None,
        "trace": "spread 0.01 <= tolerance 0 = false", "evidence_ids": ["e1"],
        "details": {"spread": "0.01", "tolerance": "0"},
    }
    basic = "两者存在0.01亿元差异，数值基本一致。"
    exact = "两者相差0.01亿元，但完全一致。"

    basic_verification = verify_claims(basic, "是否一致", [hit], [basic], [operation])
    exact_verification = verify_claims(exact, "是否一致", [hit], [exact], [operation])

    assert basic_verification.passed
    assert exact_verification.passed
    assert not any(
        item["error_type"] == "consistency_mismatch"
        for item in [*basic_verification.failure_details, *exact_verification.failure_details]
    )


def test_qualitative_business_conclusion_uses_declared_calculation_refs():
    hit = Hit("table", {"evidence_id": "e1", "value_text": "51246.71", "unit": "亿元"})
    operation = {
        "type": "calculation", "id": "calc2", "operation": "compare",
        "input_refs": ["calc1", "r3"], "result": "false", "unit": None,
        "trace": "51246.7 == 51246.71 = false", "evidence_ids": ["e1"],
        "details": {"operator": "=="},
    }
    claim = "两者数值不同，因此整体上不一致。"

    verification = verify_claims(
        claim,
        "两者是否一致",
        [hit],
        [claim],
        [operation],
        grounding_refs={"ar1": ["calc2"]},
    )

    assert verification.passed
    assert verification.claim_results[0]["calculation_ids"] == ["calc2"]
    assert verification.claim_results[0]["evidence_ids"] == ["e1"]


def test_verification_failure_details_include_numeric_provenance():
    hit = Hit("table", {"evidence_id": "e1", "value_text": "10", "unit": "亿元"})

    verification = verify_claims("结果为11亿元。", "结果是多少", [hit], ["结果为11亿元。"], [])

    assert not verification.numeric_ok
    failure = next(item for item in verification.failure_details if item["error_type"] == "unsupported_number")
    assert failure["claim"] == "结果为11亿元。"
    assert failure["actual"] == "11"
    assert failure["evidence_ids"] == ["e1"]
    assert failure["calculation_ids"] == []


def test_verification_reports_answer_completeness_without_judging_style():
    hit = Hit("table", {"evidence_id": "e1", "value_text": "10", "unit": "亿元", "context": "总资产10亿元"})
    completeness = SimpleNamespace(complete=False, missing_requirement_ids=("ar2",))

    verification = verify_claims(
        "总资产为10亿元。",
        "总资产是多少，与去年相差多少？",
        [hit],
        ["总资产为10亿元。"],
        [],
        completeness=completeness,
    )

    assert verification.completeness_ok is False
    assert any(
        item["error_type"] == "incomplete_answer"
        and item["actual"] == {"missing_requirement_ids": ["ar2"]}
        for item in verification.failure_details
    )


def test_verification_does_not_derive_unrecorded_spread_from_compare_inputs():
    hit = Hit("table", {"evidence_id": "e1", "value_text": "51246.71", "unit": "亿元"})
    operation = {
        "type": "calculation", "id": "calc2", "operation": "compare",
        "input_refs": ["calc1", "r3"],
        "inputs": [
            {"ref": "calc1", "value": "51246.70", "unit": "亿元", "evidence_ids": ["e1"]},
            {"ref": "r3", "value": "51246.71", "unit": "亿元", "evidence_ids": ["e1"]},
        ],
        "result": "false", "unit": None,
        "trace": "51246.70 == 51246.71 = false", "evidence_ids": ["e1"],
        "details": {"operator": "=="},
    }
    answer = "两者相差0.01亿元，数值基本一致，但并非完全相等。"

    verification = verify_claims(answer, "是否一致", [hit], [answer], [operation])

    assert not verification.passed
    assert any(
        item["error_type"] == "unsupported_number" and item["actual"] == "0.01"
        for item in verification.failure_details
    )


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
    assert "规则文件A（商业银行风险监管核心指标）提供规则" in draft.answer
    assert "数据文件B（2025年商业银行主要监管指标情况表）提供数据" in draft.answer
    assert "比较过程：实际值1.496%≤5%" in draft.answer
    assert "最终结论：满足监管要求" in draft.answer
    assert "满足监管要求" in draft.answer
    assert "1.496%" in draft.answer
    assert "5%" in draft.answer
    assert draft.operations[0]["rule_source"] == "商业银行风险监管核心指标"
    assert draft.operations[0]["data_source"] == "2025年商业银行主要监管指标情况表"


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
