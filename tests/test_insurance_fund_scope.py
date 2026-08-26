from types import SimpleNamespace

import pytest

from bankreg_trustrag.ingestion.parsers import _cell_records
from bankreg_trustrag.query import parse_query
from bankreg_trustrag.retrieval.index import HybridIndex
from bankreg_trustrag.schemas import Document
from bankreg_trustrag.service import TrustRAGService
from bankreg_trustrag.utils import canonical_dimension_label


REPORT_TITLE = "2023年4季度保险业资金运用情况表"
SHEET_NAME = "2023年4季度保险资金运用情况表"
QUESTION_PREFIX = f"根据 Excel 附件《{REPORT_TITLE}》（工作表：{SHEET_NAME}），"


def _document() -> Document:
    return Document(
        doc_id="insurance_funds",
        title=REPORT_TITLE,
        authority=None,
        document_no=None,
        publish_date=None,
        effective_date=None,
        expire_date=None,
        document_type="excel",
        topic=[],
        version=None,
        status="unknown",
        source_url=None,
        local_path="report.xlsx",
        sha256="hash",
        file_name="report.xlsx",
    )


def _source_rows():
    return [
        [REPORT_TITLE, None, None, None],
        [None, None, None, None],
        [None, "单位：亿元", None, None],
        ["项目", "截至当期", None, None],
        [None, "账面余额", "规模占比", "同比增长"],
        ["资金运用余额", 281573.609414449, 1, 0.110549403288851],
        ["年化财务收益率", 0.0223489054088857, None, None],
        ["年化综合收益率", 0.0321660147974141, None, None],
        ["其中：人身险公司", None, None, None],
        ["项目", "截至当期", None, None],
        [None, "账面余额", "规模占比", "同比增长"],
        ["资金运用余额", 251914.165162238, 1, 0.115195576760055],
        ["其中：银行存款", 21558.1319226107, 0.085, -0.034],
        ["债券", 115775.997486169, 0.459, 0.230],
        ["股票", 18160.2530356672, 0.072, 0.030],
        ["证券投资基金", 13549.1502564767, 0.053, 0.121],
        ["长期股权投资", 23047.1741545042, 0.091, -0.006],
        ["年化财务收益率", 0.0229060905779469, None, None],
        ["年化综合收益率", 0.0337423885317111, None, None],
        ["其中：财产保险公司", None, None, None],
        ["项目", "截至当期", None, None],
        [None, "账面余额", "规模占比", "同比增长"],
        ["资金运用余额", 20200.4695774817, 1, 0.0459107274121537],
    ]


def test_parser_carries_insurance_company_section_into_cell_context():
    records = _cell_records(_document(), SHEET_NAME, _source_rows(), SHEET_NAME)
    by_cell = {record.cell_address: record for record in records}

    assert "保险业总体" in by_cell["B6"].context
    assert "人身险公司" in by_cell["B12"].context
    assert "财产保险公司" in by_cell["B23"].context


def test_query_extracts_fund_metric_column_path_and_company_scope():
    question = QUESTION_PREFIX + "财产保险公司的“资金运用余额”在“截至当期-账面余额”口径下的数值是多少？"
    parsed = parse_query(question)

    assert parsed.entities["table_name"] == "保险业资金运用情况表"
    assert parsed.entities["indicator"] == "资金运用余额"
    assert parsed.entities["row_label"] == "资金运用余额"
    assert parsed.entities["column_label"] == "截至当期-账面余额"
    assert parsed.entities["insurance_company_scope"] == "财产保险公司"
    assert parsed.entities["period"] == "2023年4季度"
    assert parsed.entities["period_normalized"] == "2023-Q4"
    assert parsed.entities["quarter"] == "四季度"
    assert canonical_dimension_label("截至当期-账面余额") == canonical_dimension_label("截至当期 / 账面余额")


def test_retrieval_does_not_cross_quarters_when_cell_period_only_contains_year():
    documents = [
        {
            "doc_id": "q2",
            "title": "2023年2季度保险业资金运用情况表_2023年二季度保险业资金运用情况表",
            "file_name": "161_2023年2季度保险业资金运用情况表.xlsx",
        },
        {
            "doc_id": "q4",
            "title": "2023年4季度保险业资金运用情况表_2023年四季度保险业资金运用情况表",
            "file_name": "134_2023年4季度保险业资金运用情况表.xlsx",
        },
    ]
    tables = [
        {"evidence_id": "cell:q2:A20", "doc_id": "q2", "sheet_name": "2023年1季度保险资金运用情况表", "table_name": "2023年1季度保险资金运用情况表", "indicator": "其中:财产保险公司", "row_header": "其中:财产保险公司", "value_text": "其中:财产保险公司", "period": "2023", "cell_address": "A20", "context": "其中:财产保险公司"},
        {"evidence_id": "cell:q2:B23", "doc_id": "q2", "sheet_name": "2023年1季度保险资金运用情况表", "table_name": "2023年1季度保险资金运用情况表", "indicator": "资金运用余额", "row_header": "资金运用余额", "column_header": "单位:亿元 / 截至当期 / 账面余额", "value_text": "20153.01", "unit": "亿元", "period": "2023", "cell_address": "B23", "context": "资金运用余额 | 截至当期 / 账面余额 | 20153.01"},
        {"evidence_id": "cell:q4:A20", "doc_id": "q4", "sheet_name": SHEET_NAME, "table_name": SHEET_NAME, "indicator": "其中:财产保险公司", "row_header": "其中:财产保险公司", "value_text": "其中:财产保险公司", "period": "2023", "cell_address": "A20", "context": "其中:财产保险公司"},
        {"evidence_id": "cell:q4:B23", "doc_id": "q4", "sheet_name": SHEET_NAME, "table_name": SHEET_NAME, "indicator": "资金运用余额", "row_header": "资金运用余额", "column_header": "单位:亿元 / 截至当期 / 账面余额", "value_text": "20200.4695774817", "unit": "亿元", "period": "2023", "cell_address": "B23", "context": "资金运用余额 | 截至当期 / 账面余额 | 20200.4695774817"},
    ]
    index = HybridIndex(documents, [], tables)
    question = QUESTION_PREFIX + "财产保险公司的“资金运用余额”在“截至当期-账面余额”口径下的数值是多少？"

    hits = index.search_tables(question, top_k=8)

    assert hits
    assert {hit.item["doc_id"] for hit in hits} == {"q4"}
    assert hits[0].item["value_text"] == "20200.4695774817"


def test_service_prefers_exact_quoted_quarter_title_over_generic_table_name():
    documents = [
        {"doc_id": "q2", "title": "2023年2季度保险业资金运用情况表_2023年二季度保险业资金运用情况表", "file_name": "q2.xlsx", "status": "effective"},
        {"doc_id": "q4", "title": "2023年4季度保险业资金运用情况表_2023年四季度保险业资金运用情况表", "file_name": "q4.xlsx", "status": "effective"},
    ]
    tables = [
        {"evidence_id": "cell:q2:A20", "doc_id": "q2", "sheet_name": "错误的一季度工作表名", "table_name": "保险业资金运用情况表", "indicator": "其中:财产保险公司", "row_header": "其中:财产保险公司", "value_text": "其中:财产保险公司", "period": "2023", "cell_address": "A20", "context": "其中:财产保险公司"},
        {"evidence_id": "cell:q2:B23", "doc_id": "q2", "sheet_name": "错误的一季度工作表名", "table_name": "保险业资金运用情况表", "indicator": "资金运用余额", "row_header": "资金运用余额", "column_header": "截至当期 / 账面余额", "value_text": "20153.01", "unit": "亿元", "period": "2023", "cell_address": "B23", "context": "资金运用余额 | 截至当期 / 账面余额 | 20153.01"},
        {"evidence_id": "cell:q4:A20", "doc_id": "q4", "sheet_name": SHEET_NAME, "table_name": "保险业资金运用情况表", "indicator": "其中:财产保险公司", "row_header": "其中:财产保险公司", "value_text": "其中:财产保险公司", "period": "2023", "cell_address": "A20", "context": "其中:财产保险公司"},
        {"evidence_id": "cell:q4:B23", "doc_id": "q4", "sheet_name": SHEET_NAME, "table_name": "保险业资金运用情况表", "indicator": "资金运用余额", "row_header": "资金运用余额", "column_header": "截至当期 / 账面余额", "value_text": "20200.4695774817", "unit": "亿元", "period": "2023", "cell_address": "B23", "context": "资金运用余额 | 截至当期 / 账面余额 | 20200.4695774817"},
    ]
    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(min_trust=0.58, top_k=8)
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.index = HybridIndex(documents, [], tables)
    service.semantic = SimpleNamespace(enabled=False)
    question = QUESTION_PREFIX + "财产保险公司的“资金运用余额”在“截至当期-账面余额”口径下的数值是多少？"

    response = service.ask(question)

    assert response.trust["decision"] == "answer"
    assert response.query_plan["operations"][0]["raw_value"] == 20200.4695774817
    assert response.evidence[0]["doc_id"] == "q4"


@pytest.mark.parametrize(
    ("scope_text", "expected_cell", "expected_value", "expected_scope"),
    [
        ("", "B6", 281573.609414449, "保险业总体"),
        ("人身险公司的", "B12", 251914.165162238, "人身险公司"),
        ("财产保险公司的", "B23", 20200.4695774817, "财产保险公司"),
    ],
)
def test_service_resolves_repeated_fund_balance_by_company_section(
    scope_text,
    expected_cell,
    expected_value,
    expected_scope,
):
    document = {
        "doc_id": "insurance_funds",
        "title": REPORT_TITLE,
        "file_name": "report.xlsx",
        "status": "effective",
    }
    # These rows intentionally mirror the old persisted evidence: repeated
    # indicators have no section in context. HybridIndex must reconstruct the
    # section from the preceding worksheet headings at service startup.
    tables = [
        {"evidence_id": "cell:insurance_funds:B6", "doc_id": "insurance_funds", "sheet_name": SHEET_NAME, "table_name": SHEET_NAME, "indicator": "资金运用余额", "row_header": "资金运用余额", "column_header": "单位:亿元 / 截至当期 / 账面余额", "period": "2023", "value_text": "281573.609414449", "unit": "亿元", "cell_address": "B6", "context": "资金运用余额 | 截至当期 / 账面余额 | 281573.609414449"},
        {"evidence_id": "cell:insurance_funds:A9", "doc_id": "insurance_funds", "sheet_name": SHEET_NAME, "table_name": SHEET_NAME, "indicator": "其中:人身险公司", "row_header": "其中:人身险公司", "column_header": "项目", "period": "2023", "value_text": "其中:人身险公司", "cell_address": "A9", "context": "其中:人身险公司"},
        {"evidence_id": "cell:insurance_funds:B12", "doc_id": "insurance_funds", "sheet_name": SHEET_NAME, "table_name": SHEET_NAME, "indicator": "资金运用余额", "row_header": "资金运用余额", "column_header": "单位:亿元 / 截至当期 / 账面余额", "period": "2023", "value_text": "251914.165162238", "unit": "亿元", "cell_address": "B12", "context": "资金运用余额 | 截至当期 / 账面余额 | 251914.165162238"},
        {"evidence_id": "cell:insurance_funds:A20", "doc_id": "insurance_funds", "sheet_name": SHEET_NAME, "table_name": SHEET_NAME, "indicator": "其中:财产保险公司", "row_header": "其中:财产保险公司", "column_header": "项目", "period": "2023", "value_text": "其中:财产保险公司", "cell_address": "A20", "context": "其中:财产保险公司"},
        {"evidence_id": "cell:insurance_funds:B23", "doc_id": "insurance_funds", "sheet_name": SHEET_NAME, "table_name": SHEET_NAME, "indicator": "资金运用余额", "row_header": "资金运用余额", "column_header": "单位:亿元 / 截至当期 / 账面余额", "period": "2023", "value_text": "20200.4695774817", "unit": "亿元", "cell_address": "B23", "context": "资金运用余额 | 截至当期 / 账面余额 | 20200.4695774817"},
    ]
    index = HybridIndex([document], [], tables)
    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(min_trust=0.58, top_k=8)
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.index = index
    service.semantic = SimpleNamespace(enabled=False)
    question = QUESTION_PREFIX + scope_text + "“资金运用余额”在“截至当期-账面余额”口径下的数值是多少？"

    response = service.ask(question)

    assert response.trust["decision"] == "answer"
    assert response.query_plan["operations"][0]["cell"] == expected_cell
    assert response.query_plan["operations"][0]["raw_value"] == expected_value
    assert response.query_plan["operations"][0]["section_scope"] == expected_scope
    assert response.evidence[0]["cell_address"] == expected_cell
