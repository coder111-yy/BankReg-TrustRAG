from bankreg_trustrag.utils import canonical_table_label, normalized_number, tokens
from bankreg_trustrag.query import extract_dimension_labels, extract_indicator, parse_query


def test_tokens_keep_chinese_and_numbers():
    assert "监" in tokens("监管比例 10.5%")
    assert "10.5%" in tokens("监管比例 10.5%")


def test_normalized_number_percent():
    assert normalized_number("10.5%") == 0.105
    assert normalized_number("1,200") == 1200.0


def test_query_extracts_filename_without_question_prefix():
    parsed = parse_query("请查询《2023年12月保险业经营情况表.xlsx》中的经营数据")
    assert parsed.entities["filenames"] == ["2023年12月保险业经营情况表.xlsx"]


def test_query_extracts_requested_year():
    parsed = parse_query("请查询2028年银行监管统计数据")

    assert parsed.entities["years"] == ["2028"]


def test_query_extracts_short_invalid_year_for_rejection():
    parsed = parse_query("请查询209年银行监管统计数据")

    assert parsed.entities["years"] == ["209"]


def test_query_extracts_table_indicator_and_quarter_from_month_question():
    parsed = parse_query("2025年3月商业银行主要监管指标情况表中的不良贷款率是多少？")

    assert parsed.entities["table_name"] == "商业银行主要监管指标情况表"
    assert parsed.entities["indicator"] == "不良贷款率"
    assert parsed.entities["period_normalized"] == "2025-03"
    assert parsed.entities["quarter"] == "一季度"


def test_query_does_not_treat_table_title_as_indicator_and_extracts_row_column_labels():
    question = "2023年10月全国各地区原保险保费收入情况表.xlsx中“全国合计”在“合计”口径下的数值是多少"
    parsed = parse_query(question)

    assert extract_indicator(question) is None
    assert extract_dimension_labels(question) == ("全国合计", "合计")
    assert "indicator" not in parsed.entities
    assert parsed.entities["row_label"] == "全国合计"
    assert parsed.entities["column_label"] == "合计"


def test_single_quoted_metric_before_scope_marker_is_a_column_not_a_row():
    question = "根据2024年9月统计表，在“健康险”口径下，以下哪一项数值最高？"

    assert extract_dimension_labels(question) == (None, "健康险")
    parsed = parse_query(question)
    assert "row_label" not in parsed.entities
    assert parsed.entities["column_label"] == "健康险"


def test_table_label_normalization_removes_outline_prefix_but_keeps_decimal():
    assert canonical_table_label("1、财产险") == "财产险"
    assert canonical_table_label("（二） 人身险") == "人身险"
    assert canonical_table_label("1.21") == "1.21"


def test_table_label_normalization_removes_generic_hierarchy_prefixes():
    assert canonical_table_label("其中：任意机构") == canonical_table_label("任意机构")
    assert canonical_table_label("其中包括，任意指标") == canonical_table_label("任意指标")
