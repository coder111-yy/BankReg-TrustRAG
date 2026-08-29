import pytest

from bankreg_trustrag.calculator import CalculationError, Calculator
from bankreg_trustrag.query_plan import CalculationTask, RetrievalCandidate, RetrievalResult, RetrievalTask
from bankreg_trustrag.retrieval.index import Hit, HybridIndex
from bankreg_trustrag.retrieval_tools import RetrievalTools


def _resolved(task_id, value, unit="亿元", evidence_id=None):
    evidence_id = evidence_id or f"cell:{task_id}"
    candidate = RetrievalCandidate(value=str(value), unit=unit, evidence_ids=[evidence_id])
    return RetrievalResult(
        task_id=task_id,
        status="resolved",
        expected_information=task_id,
        selected=candidate,
        candidates=[candidate],
        evidence_ids=[evidence_id],
    )


def test_retrieval_tasks_keep_human_property_and_national_values_isolated():
    documents = [
        {"doc_id": "life", "title": "2023年10月人身险公司经营情况表", "file_name": "life.xlsx", "document_type": "xlsx"},
        {"doc_id": "property", "title": "2023年10月财产保险公司经营情况表", "file_name": "property.xlsx", "document_type": "xlsx"},
        {"doc_id": "national", "title": "2023年10月全国各地区原保险保费收入情况表", "file_name": "national.xlsx", "document_type": "xlsx"},
    ]
    tables = [
        {"evidence_id": "cell:life:C6", "doc_id": "life", "indicator": "原保险保费收入", "period": "2023-10", "value_text": "31739.18", "unit": "亿元", "cell_address": "C6", "context": "人身险公司 原保险保费收入 31739.18"},
        {"evidence_id": "cell:property:C6", "doc_id": "property", "indicator": "原保险保费收入", "period": "2023-10", "value_text": "13428.79", "unit": "亿元", "cell_address": "C6", "context": "财产保险公司 原保险保费收入 13428.79"},
        {"evidence_id": "cell:national:C4", "doc_id": "national", "indicator": "全国合计", "row_header": "全国合计", "column_header": "合计", "period": "2023-10", "value_text": "45167.98", "unit": "亿元", "cell_address": "C4", "context": "全国合计 合计 45167.98"},
    ]
    tools = RetrievalTools(HybridIndex(documents, [], tables))
    tasks = [
        RetrievalTask(id="r1", query="人身险原保险保费收入", expected_information="人身险", source_scope={"document_title": documents[0]["title"], "year": 2023, "month": 10}, semantic_constraints={"indicator": "原保险保费收入", "institution": "人身险公司", "period": "2023-10"}, expected_value_type="number", expected_unit="亿元"),
        RetrievalTask(id="r2", query="财产险原保险保费收入", expected_information="财产险", source_scope={"document_title": documents[1]["title"], "year": 2023, "month": 10}, semantic_constraints={"indicator": "原保险保费收入", "institution": "财产保险公司", "period": "2023-10"}, expected_value_type="number", expected_unit="亿元"),
        RetrievalTask(id="r3", query="全国合计", expected_information="全国", source_scope={"document_title": documents[2]["title"], "year": 2023, "month": 10}, semantic_constraints={"row_label": "全国合计", "column_label": "合计", "period": "2023-10"}, expected_value_type="number", expected_unit="亿元"),
    ]

    results = [tools.execute(task).result for task in tasks]

    assert [item.status for item in results] == ["resolved", "resolved", "resolved"]
    assert [item.selected.value for item in results] == ["31739.18", "13428.79", "45167.98"]
    assert [item.evidence_ids[0] for item in results] == ["cell:life:C6", "cell:property:C6", "cell:national:C4"]


def test_aggregate_column_does_not_match_the_same_word_in_row_context():
    documents = [{
        "doc_id": "national", "title": "2023年10月全国各地区原保险保费收入情况表",
        "file_name": "national.xlsx", "document_type": "xlsx",
    }]
    tables = [
        {"evidence_id": "cell:national:C4", "doc_id": "national", "indicator": "全国合计", "row_header": "全国合计", "column_header": "合计", "period": "2023-10", "value_text": "45167.98", "cell_address": "C4", "context": "全国合计 | 合计 | 45167.98"},
        {"evidence_id": "cell:national:D4", "doc_id": "national", "indicator": "全国合计", "row_header": "全国合计", "column_header": "财产保险", "period": "2023-10", "value_text": "11366.02", "cell_address": "D4", "context": "全国合计 | 财产保险 | 11366.02"},
        {"evidence_id": "cell:national:E4", "doc_id": "national", "indicator": "全国合计", "row_header": "全国合计", "column_header": "寿险", "period": "2023-10", "value_text": "24912.74", "cell_address": "E4", "context": "全国合计 | 寿险 | 24912.74"},
    ]
    task = RetrievalTask(
        id="r3", query="全国合计原保险保费收入", expected_information="全国合计",
        source_scope={"document_title": documents[0]["title"], "year": 2023, "month": 10},
        semantic_constraints={"row_label": "全国合计", "column_label": "合计", "period": "2023-10"},
        expected_value_type="number",
    )

    result = RetrievalTools(HybridIndex(documents, [], tables)).execute(task).result

    assert result.status == "resolved"
    assert result.selected.value == "45167.98"
    assert result.selected.cell_address == "C4"
    assert len(result.candidates) == 1


def test_retrieval_accepts_canonical_insurance_and_national_scopes():
    documents = [
        {"doc_id": "property", "title": "2023年10月财产保险公司经营情况表", "file_name": "property.xlsx"},
        {"doc_id": "industry", "title": "2023年10月保险业经营情况表", "file_name": "industry.xlsx"},
    ]
    tables = [
        {"evidence_id": "cell:property:C6", "doc_id": "property", "indicator": "原保险保费收入", "period": "2023-10", "value_text": "13428.79", "unit": "亿元", "cell_address": "C6", "context": "原保险保费收入 13428.79"},
        {"evidence_id": "cell:industry:C5", "doc_id": "industry", "indicator": "原保险保费收入", "period": "2023-10", "value_text": "45167.98", "unit": "亿元", "cell_address": "C5", "context": "原保险保费收入 45167.98"},
    ]
    tools = RetrievalTools(HybridIndex(documents, [], tables))
    property_task = RetrievalTask(
        id="property",
        query="财产险公司原保险保费收入",
        expected_information="财产险数值",
        source_scope={"year": 2023, "month": 10},
        semantic_constraints={"indicator": "原保险保费收入", "institution": "财产险公司", "period": "2023-10"},
        expected_value_type="number",
    )
    national_task = RetrievalTask(
        id="national",
        query="全国原保险保费收入",
        expected_information="全国总数",
        source_scope={"year": 2023, "month": 10},
        semantic_constraints={"indicator": "原保险保费收入", "statistical_scope": "全国", "period": "2023-10"},
        expected_value_type="number",
    )

    property = tools.execute(property_task).result
    national = tools.execute(national_task).result

    assert property.status == "resolved"
    assert property.selected.value == "13428.79"
    assert national.status == "resolved"
    assert national.selected.value == "45167.98"


@pytest.mark.parametrize("indicator, statistical_scope", [
    ("保险业总资产", None),
    ("资产", "保险行业"),
    ("资产规模", "保险业"),
])
def test_total_asset_indicator_and_industry_scope_aliases(indicator, statistical_scope):
    index = HybridIndex(
        [{"doc_id": "industry", "title": "2024年9月保险业经营情况表", "file_name": "industry.xlsx"}],
        [],
        [{"evidence_id": "cell:industry:C12", "doc_id": "industry", "indicator": "总资产", "row_header": "总资产", "period": "2024-09", "value_text": "350023.51", "unit": "亿元", "cell_address": "C12", "context": "总资产 350023.51"}],
    )
    task = RetrievalTask(
        id="assets",
        query="2024年9月保险行业资产",
        expected_information="保险业总资产",
        source_scope={"year": 2024, "month": 9},
        semantic_constraints={"indicator": indicator, "statistical_scope": statistical_scope, "period": "2024-09"},
        expected_value_type="number",
    )

    result = RetrievalTools(index).execute(task).result

    assert result.status == "resolved"
    assert result.selected.value == "350023.51"


def test_retrieval_detects_quarter_ambiguity_after_search():
    index = HybridIndex(
        [{"doc_id": "bank", "title": "2023年大型商业银行监管指标", "file_name": "bank.xlsx"}],
        [],
        [
            {"evidence_id": "cell:bank:B4", "doc_id": "bank", "indicator": "不良贷款余额", "period": "2023", "column_header": "一季度", "value_text": "10", "unit": "亿元", "cell_address": "B4", "context": "大型商业银行 不良贷款余额 一季度 10"},
            {"evidence_id": "cell:bank:C4", "doc_id": "bank", "indicator": "不良贷款余额", "period": "2023", "column_header": "二季度", "value_text": "11", "unit": "亿元", "cell_address": "C4", "context": "大型商业银行 不良贷款余额 二季度 11"},
        ],
    )
    task = RetrievalTask(
        id="r1",
        query="2023年大型商业银行不良贷款余额",
        expected_information="不良贷款余额",
        source_scope={"year": 2023},
        semantic_constraints={"indicator": "不良贷款余额", "institution": "大型商业银行", "period": "2023"},
        expected_value_type="number",
        expected_unit="亿元",
    )

    result = RetrievalTools(index).execute(task).result

    assert result.status == "ambiguous"
    assert "季度" in result.ambiguity_reason
    assert len(result.candidates) == 2


def test_text_retrieval_expands_following_chunks_after_structural_lead_in():
    rows = [
        {"evidence_id": "text:rule:p58", "doc_id": "rule", "paragraph_no": 58, "content": "（一）压力测试情景"},
        {"evidence_id": "text:rule:p59", "doc_id": "rule", "paragraph_no": 59, "content": "描述恢复计划压力测试的情景设置和主要情景指标。"},
        {"evidence_id": "text:rule:p60", "doc_id": "rule", "paragraph_no": 60, "content": "情景设置至少包括系统性压力情景、自身压力情景和混合压力情景。"},
        {"evidence_id": "text:rule:p61", "doc_id": "rule", "paragraph_no": 61, "content": "（二）压力测试结果"},
    ]

    class LeadOnlyIndex:
        text = rows
        doc_by_id = {"rule": {"title": "恢复计划示例", "document_type": "word"}}

        def search_text(self, *args, **kwargs):
            return [Hit("text", self.text[1], fused_score=1.0)]

    task = RetrievalTask(
        id="r2",
        query="恢复计划压力测试至少包括哪些情景",
        expected_information="压力测试情景列表",
        expected_value_type="text",
    )

    result = RetrievalTools(LeadOnlyIndex()).execute(task).result

    assert result.status == "resolved"
    assert result.evidence_ids == ["text:rule:p59", "text:rule:p60", "text:rule:p61"]
    assert any("系统性压力情景、自身压力情景和混合压力情景" in item.content for item in result.candidates)


def test_decimal_calculator_traces_sum_absolute_difference_and_growth_rate():
    retrievals = {
        "r1": _resolved("r1", "31739.18", evidence_id="cell:life"),
        "r2": _resolved("r2", "13428.79", evidence_id="cell:property"),
        "r3": _resolved("r3", "45167.98", evidence_id="cell:national"),
        "old": _resolved("old", "100"),
        "new": _resolved("new", "110"),
    }
    calculator = Calculator()
    total = calculator.execute(
        CalculationTask(id="op1", type="sum", inputs=["r1", "r2"], output_id="calc_total"),
        retrievals,
        {},
    )
    difference = calculator.execute(
        CalculationTask(id="op2", type="subtract", inputs=["calc_total", "r3"], output_id="calc_diff", parameters={"absolute": True}),
        retrievals,
        {"calc_total": total},
    )
    growth = calculator.execute(
        CalculationTask(id="op3", type="growth_rate", old_ref="old", new_ref="new", output_id="calc_growth"),
        retrievals,
        {},
    )

    assert total.result == "45167.97"
    assert total.trace == "31739.18 + 13428.79 = 45167.97"
    assert difference.result == "0.01"
    assert difference.evidence_ids == ["cell:life", "cell:property", "cell:national"]
    assert growth.result == "10"
    assert growth.unit == "%"
    assert growth.trace == "(110 - 100) / 100 = 10%"

    precise_growth = calculator.execute(
        CalculationTask(id="op4", type="growth_rate", old_ref="old", new_ref="new", output_id="precise_growth"),
        {"old": _resolved("old", "42526.85"), "new": _resolved("new", "47945.35")},
        {},
    )
    assert precise_growth.result == "12.74"
    assert precise_growth.details["decimal_places"] == 2


def test_decimal_calculator_rejects_incompatible_units():
    with pytest.raises(CalculationError, match="incompatible units"):
        Calculator().execute(
            CalculationTask(id="op", type="sum", inputs=["a", "b"], output_id="total"),
            {"a": _resolved("a", "1", "亿元"), "b": _resolved("b", "2", "万元")},
            {},
        )
