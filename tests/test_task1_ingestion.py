from bankreg_trustrag.ingestion.manifest import _cell_type, _enrich_document, _enrich_table_cell, _reporting_period
from bankreg_trustrag.ingestion.parsers import _date_from_name, _join_pdf_lines, _metadata


def test_coverage_range_is_not_publish_date():
    assert _date_from_name("2013年至2017年6月国内21家主要银行绿色信贷数据.pdf") is None


def test_month_is_reporting_period():
    assert _reporting_period("2025年9月保险业经营情况表") == "2025-09"


def test_table_cell_classification():
    assert _cell_type({"value": "单位：亿元、%", "indicator": None}) == "unit"
    assert _cell_type({"value": 52145.77, "indicator": "原保险保费收入"}) == "data"
    assert _cell_type({"value": "注：本表数据为截至当期", "indicator": "注"}) == "note"


def test_pdf_line_reconstruction():
    blocks = _join_pdf_lines("2013年，银监会印发《关于报送绿色信\n贷统计表的通知》。\n一、制度简介")
    assert blocks[0] == "2013年，银监会印发《关于报送绿色信贷统计表的通知》。"
    assert blocks[1] == "一、制度简介"


def test_percentage_semantics_use_display_scale():
    row = _enrich_table_cell({
        "value": 0.08927,
        "unit": "%",
        "indicator": "比上年同期增长率",
        "column_header": "1月",
        "row_header": "比上年同期增长率",
        "context": "比上年同期增长率 | 1月 | 0.08927",
        "period": "2026-01",
    })
    assert row["raw_value"] == 0.08927
    assert abs(row["numeric_value"] - 8.927) < 1e-9
    assert abs(row["value"] - 8.927) < 1e-9
    assert row["display_value"] == "8.927%"
    assert row["value_scale"] == 100.0


def test_explicit_institution_dimension():
    row = _enrich_table_cell({
        "value": 4502.47,
        "unit": "亿元",
        "indicator": "可疑类贷款余额",
        "column_header": "一季度 / 大型商业银行",
        "row_header": "可疑类贷款余额",
        "context": "可疑类贷款余额 | 一季度 | 大型商业银行 | 4502.47",
        "period": "2023年一季度",
    })
    assert row["institution"] == "大型商业银行"


def test_explicit_region_dimension():
    row = _enrich_table_cell({
        "value": 8426.99,
        "unit": "亿元",
        "indicator": "健康险",
        "column_header": "原保险保费收入",
        "row_header": "全国合计",
        "context": "全国合计 | 健康险 | 8426.99",
        "period": "2025-09",
    })
    assert row["region"] == "全国"


def test_doc_id_is_unique_per_physical_file(tmp_path):
    a = tmp_path / "A.pdf"
    b = tmp_path / "B.pdf"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    doc_a = _metadata(a, tmp_path)
    doc_b = _metadata(b, tmp_path)
    assert doc_a.sha256 == doc_b.sha256
    assert doc_a.doc_id != doc_b.doc_id


def test_own_document_number_and_authority_from_header():
    document = {
        "title": "商业银行资本管理办法",
        "authority": None,
        "document_no": None,
        "publish_date": None,
        "document_type": "word",
        "topic": [],
        "source_url": None,
        "local_path": "x.docx",
        "sha256": "abc",
    }
    text_rows = [
        {"content": "国家金融监督管理总局令2023年第4号"},
        {"content": "商业银行资本管理办法"},
        {"content": "第一章 总则"},
    ]
    enriched = _enrich_document(document, text_rows)
    assert enriched["document_no"] == "国家金融监督管理总局令2023年第4号"
    assert enriched["authority"] == "国家金融监督管理总局"
