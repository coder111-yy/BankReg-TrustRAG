from pathlib import Path

import pytest

from bankreg_trustrag.ingestion.manifest import build_document_relations
from bankreg_trustrag.ingestion.parsers import _header_row_count, parse_file


def test_excel_header_window_stops_before_first_numeric_data_row():
    rows = [
        [None, "2023年10月全国各地区原保险保费收入情况表", None],
        [None, None, "单位:亿元"],
        [None, "地区", "合计"],
        [None, "全国合计", 45167.98],
    ]

    assert _header_row_count(rows) == 3


def test_wps_legacy_doc_fallback_recovers_text_when_attachment_exists():
    roots = list(Path(".").glob("03-*"))
    if not roots:
        pytest.skip("contest corpus is not present")
    matches = list((roots[0] / "nfra_page_attachments_500").glob("*数据安全事件分级.doc"))
    if not matches:
        pytest.skip("legacy .doc fixture is not present")

    result = parse_file(matches[0], roots[0])

    assert len(result.text_evidence) >= 10
    assert any("数据安全事件" in item.content for item in result.text_evidence)


def test_manifest_builds_explicit_attachment_relation_only_when_parent_exists():
    relations = build_document_relations([
        {"doc_id": "parent", "title": "监管办法", "sha256": "one"},
        {"doc_id": "attachment", "title": "监管办法_附件:指标解释", "sha256": "two"},
        {"doc_id": "unrelated", "title": "其他办法_附件:表格", "sha256": "three"},
    ])

    assert relations == [{
        "source_doc_id": "attachment", "target_doc_id": "parent", "relation_type": "attachment_of",
        "confidence": 1.0, "rationale": "filename_attachment_marker",
    }]
