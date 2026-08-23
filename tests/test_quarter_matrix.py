from bankreg_trustrag.ingestion.parsers import _cell_records
from bankreg_trustrag.reasoning import table_answer
from bankreg_trustrag.retrieval.index import Hit
from bankreg_trustrag.schemas import Document
from bankreg_trustrag.verification import verify_claims


def _document() -> Document:
    return Document(
        doc_id="DOC_quarter",
        title="2023年商业银行主要指标分机构类情况表(季度)",
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
        local_path="source.xlsx",
        sha256="hash",
        file_name="source.xlsx",
    )


def test_quarter_matrix_parser_keeps_indicator_quarter_and_institution_dimensions():
    rows = [
        ["商业银行主要指标分机构类情况表(季度)(2023年)", None, None],
        [None, None, None],
        ["机构", None, "大型商业银行"],
        ["时间/指标", None, None],
        ["一季度", "不良贷款余额", 100.1],
        [None, "不良贷款率", 0.01],
        ["二季度", "不良贷款余额", 101.2],
        ["三季度", "不良贷款余额", 102.3],
        ["四季度", "不良贷款余额", 103.4],
    ]

    records = _cell_records(_document(), "商业银行分机构类情况表", rows, "商业银行分机构类情况表")
    target = [
        record for record in records
        if record.indicator == "不良贷款余额" and record.column_header == "大型商业银行"
    ]

    assert [(record.row_header, record.value) for record in target] == [
        ("一季度", 100.1),
        ("二季度", 101.2),
        ("三季度", 102.3),
        ("四季度", 103.4),
    ]


def test_table_answer_returns_all_quarters_for_year_only_matrix_query():
    hits = [
        Hit(
            "table",
            {
                "evidence_id": f"cell:quarter:{cell}",
                "indicator": "不良贷款余额",
                "period": "2023",
                "row_header": quarter,
                "column_header": "大型商业银行",
                "value_text": value,
                "cell_address": cell,
                "context": f"不良贷款余额 | {quarter} | 大型商业银行 | {value}",
            },
        )
        for quarter, value, cell in [
            ("一季度", "100.1", "C5"),
            ("二季度", "101.2", "C16"),
            ("三季度", "102.3", "C27"),
            ("四季度", "103.4", "C38"),
        ]
    ]

    draft = table_answer(
        "根据《2023年商业银行主要指标分机构类情况表（季度）》，“不良贷款余额”在“大型商业银行”口径下的数值是多少",
        None,
        hits,
    )

    assert "一季度100.1" in draft.answer
    assert "二季度101.2" in draft.answer
    assert "三季度102.3" in draft.answer
    assert "四季度103.4" in draft.answer
    assert draft.operations[0]["display_evidence_ids"] == [hit.evidence_id for hit in hits]
    verification = verify_claims(draft.answer, "2023年商业银行季度表中的不良贷款余额", hits, draft.claims)
    assert verification.passed
