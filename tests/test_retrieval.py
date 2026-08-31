import json

from bankreg_trustrag.retrieval.index import HybridIndex
from bankreg_trustrag.query import parse_query
from bankreg_trustrag.query_plan import RetrievalTask
from bankreg_trustrag.retrieval_tools import RetrievalTools
from bankreg_trustrag.storage import Store


def test_hybrid_retrieval_returns_exact_clause():
    index = HybridIndex(
        [{"doc_id": "d1", "title": "监管办法", "file_name": "a.docx", "status": "effective"}],
        [{"evidence_id": "text:d1:p1", "doc_id": "d1", "content": "商业银行不得挪用客户资金。", "chapter": "第三章", "article_no": "第二十条"}],
        [],
    )
    hits = index.search_text("不得挪用客户资金", top_k=3)
    assert hits and hits[0].evidence_id == "text:d1:p1"


def test_table_retrieval_preserves_cell_location():
    index = HybridIndex(
        [{"doc_id": "d1", "title": "2025年3月监管指标", "file_name": "a.xlsx", "status": "effective"}],
        [],
        [{"evidence_id": "cell:d1:Sheet1:D8", "doc_id": "d1", "sheet_name": "Sheet1", "indicator": "不良贷款率", "period": "2025-03", "value_text": "1.21", "unit": "%", "cell_address": "D8", "context": "不良贷款率 2025-03 1.21"}],
    )
    hits = index.search_tables("2025年3月 不良贷款率", top_k=3)
    assert hits and hits[0].item["cell_address"] == "D8"


def test_store_backed_table_retrieval_is_lazy(tmp_path):
    store = Store(tmp_path / "bankreg.sqlite3")
    store.connection.execute(
        "INSERT INTO documents(doc_id,title,file_name,status,sha256) VALUES (?,?,?,?,?)",
        ("d1", "2025年3月监管指标", "a.xlsx", "effective", "sha"),
    )
    store.connection.execute(
        "INSERT INTO table_evidence(evidence_id,doc_id,sheet_name,indicator,period,value_text,unit,cell_address,context) VALUES (?,?,?,?,?,?,?,?,?)",
        ("cell:d1:Sheet1:D8", "d1", "Sheet1", "不良贷款率", "2025-03", json.dumps("1.21", ensure_ascii=False), "%", "D8", "不良贷款率 2025-03 1.21"),
    )
    store.connection.commit()
    store.all_tables = lambda: (_ for _ in ()).throw(AssertionError("全表加载会造成内存峰值"))

    index = HybridIndex.from_store(store)
    hits = index.search_tables("2025年3月 不良贷款率", top_k=3)

    assert hits and hits[0].item["cell_address"] == "D8"


def test_filename_filter_accepts_question_wrapped_filename():
    parsed = parse_query("请查询《2023年12月保险业经营情况表.xlsx》中的数据")
    index = HybridIndex(
        [{"doc_id": "d1", "title": "2023年12月保险业经营情况表", "file_name": "139_2023年12月保险业经营情况表_2023年12月保险业经营情况表.xlsx", "status": "effective"}],
        [],
        [{"evidence_id": "cell:d1:Sheet1:C5", "doc_id": "d1", "sheet_name": "Sheet1", "indicator": "原保险保费收入", "period": "2023-12", "value_text": "51246.71", "unit": "亿元", "cell_address": "C5", "context": "原保险保费收入 2023-12 51246.71"}],
    )
    hits = index.hybrid_search(parsed.original_query, parsed.qa_type, filters={"file_name": parsed.entities["filenames"]})
    assert hits and hits[0].item["cell_address"] == "C5"


def test_metadata_filter_matches_filename_punctuation_variants():
    index = HybridIndex(
        [{"doc_id": "d1", "title": "中资商业银行行政许可事项申请材料_目录及格式要求(2023年版)", "file_name": "430_附件：申请材料目录.pdf", "status": "effective"}],
        [{"evidence_id": "text:d1:p1", "doc_id": "d1", "paragraph_no": 1, "content": "中资商业银行法人机构筹建审批属于机构设立类行政许可事项。"}],
        [],
    )

    hits = index.search_text("行政许可事项", filters={"title": ["中资商业银行行政许可事项申请材料目录及格式要求（2023年版）"]})

    assert hits and hits[0].evidence_id == "text:d1:p1"


def test_text_retrieval_attaches_adjacent_clause_context():
    index = HybridIndex(
        [{"doc_id": "d1", "title": "资本工具标准", "file_name": "rule.docx", "status": "effective"}],
        [
            {"evidence_id": "text:d1:p5", "doc_id": "d1", "paragraph_no": 5, "content": "一、核心一级资本工具的合格标准"},
            {"evidence_id": "text:d1:p6", "doc_id": "d1", "paragraph_no": 6, "content": "(一)直接发行且实缴的。"},
        ],
        [],
    )

    hit = index.search_text("核心一级资本工具直接发行且实缴", top_k=1)[0]

    assert "核心一级资本工具" in hit.item["context_window"]
    assert "直接发行且实缴" in hit.item["context_window"]


def test_text_retrieval_context_keeps_complete_numbered_item_after_pdf_line_split():
    index = HybridIndex(
        [{"doc_id": "guide", "title": "银行函证工作操作指引", "file_name": "guide.pdf"}],
        [
            {"evidence_id": "text:guide:p55", "doc_id": "guide", "paragraph_no": 55, "content": "3.函证范围和回函用章。在实现集约化或数字化的情况"},
            {"evidence_id": "text:guide:p56", "doc_id": "guide", "paragraph_no": 56, "content": "下,银行业金融机构应当就询证函的函证范围"},
            {"evidence_id": "text:guide:p57", "doc_id": "guide", "paragraph_no": 57, "content": "以及所采用的回函用章的适用范围进行公示"},
            {"evidence_id": "text:guide:p58", "doc_id": "guide", "paragraph_no": 58, "content": "说明可一并查询具体业务的最高机构层"},
            {"evidence_id": "text:guide:p59", "doc_id": "guide", "paragraph_no": 59, "content": "级及回函用章。"},
            {"evidence_id": "text:guide:p60", "doc_id": "guide", "paragraph_no": 60, "content": "4.回函服务的收费标准。"},
        ],
        [],
    )

    hit = index.search_text("函证范围和回函用章", top_k=1)[0]

    assert hit.evidence_id == "text:guide:p55"
    assert "最高机构层" in hit.item["context_window"]
    assert "级及回函用章" in hit.item["context_window"]


def test_benchmark_document_is_not_default_evidence():
    index = HybridIndex(
        [
            {"doc_id": "qa", "title": "QA数据", "file_name": "QA数据.xlsx"},
            {"doc_id": "source", "title": "2023年10月保险业经营情况表", "file_name": "source.xlsx"},
        ],
        [],
        [
            {"evidence_id": "cell:qa:Sheet1:F20", "doc_id": "qa", "indicator": "Q019", "period": "2023-10", "value_text": "题目文本", "context": "题目文本"},
            {"evidence_id": "cell:source:Sheet1:C5", "doc_id": "source", "indicator": "原保险保费收入", "period": "2023-10", "value_text": "51246.71", "context": "原保险保费收入 2023-10 51246.71"},
        ],
    )

    hits = index.search_tables("2023年10月的指标是多少", top_k=8)

    assert hits
    assert all(hit.item["doc_id"] != "qa" for hit in hits)


def test_annual_bank_table_maps_march_to_first_quarter():
    index = HybridIndex(
        [{"doc_id": "bank2025", "title": "2025年商业银行主要监管指标情况表", "file_name": "bank.xls", "status": "effective"}],
        [],
        [
            {"evidence_id": "cell:bank2025:商业银行季度:A14", "doc_id": "bank2025", "indicator": "不良贷款率", "period": "2025", "value_text": '"不良贷款率"', "column_header": "时间 / 项目", "cell_address": "A14", "context": "不良贷款率 | 时间 / 项目"},
            {"evidence_id": "cell:bank2025:商业银行季度:B14", "doc_id": "bank2025", "indicator": "不良贷款率", "period": "2025", "value_text": "0.01513", "column_header": "一季度", "cell_address": "B14", "context": "不良贷款率 | 一季度 | 0.01513"},
            {"evidence_id": "cell:bank2025:商业银行季度:C14", "doc_id": "bank2025", "indicator": "不良贷款率", "period": "2025", "value_text": "0.01491", "column_header": "二季度", "cell_address": "C14", "context": "不良贷款率 | 二季度 | 0.01491"},
        ],
    )

    hits = index.search_tables("2025年3月商业银行主要监管指标情况表中的不良贷款率是多少", top_k=3, filters={"title": ["商业银行主要监管指标情况表"]})

    assert hits and hits[0].item["cell_address"] == "B14"


def test_task_retrieval_matches_chinese_quarter_context_with_slash_separator():
    index = HybridIndex(
        [{"doc_id": "quarter", "title": "2023年银行业金融机构保障性安居工程贷款情况表(季度)", "file_name": "quarter.xlsx", "status": "effective"}],
        [],
        [{
            "evidence_id": "cell:quarter:B6",
            "doc_id": "quarter",
            "indicator": "保障性安居工程贷款",
            "period": "2023",
            "row_header": "商业银行合计",
            "column_header": "一季度",
            "value_text": "123.45",
            "cell_address": "B6",
            "context": "商业银行合计 | 2023年 / 一季度 | 123.45",
        }],
    )
    task = RetrievalTask.model_validate({
        "id": "r1",
        "query": "2023年一季度保障性安居工程贷款",
        "expected_information": "商业银行合计一季度数值",
        "source_scope": {"year": 2023, "quarter": 1},
        "semantic_constraints": {
            "indicator": "保障性安居工程贷款",
            "institution": "商业银行合计",
            "period": "2023年一季度",
            "row_label": "商业银行合计",
            "column_label": "一季度",
        },
        "expected_value_type": "number",
    })

    execution = RetrievalTools(index).execute(task)

    assert execution.result.status == "resolved"
    assert execution.result.selected is not None
    assert execution.result.selected.value == "123.45"


def test_formula_evidence_is_retrievable_when_indicator_and_formula_are_separate_cells():
    index = HybridIndex(
        [{"doc_id": "rule", "title": "2025年监管统计信息发布日程表", "file_name": "rule.xls"}],
        [],
        [{"evidence_id": "cell:rule:指标解释:C6", "doc_id": "rule", "table_name": "指标解释", "indicator": "4.0", "period": "指标范围及计算公式", "value_text": '"不良贷款余额 / 各项贷款余额 × 100%"', "context": "不良贷款余额 / 各项贷款余额 × 100%"}],
    )

    hits = index.search_formula_evidence("不良贷款率", year="2025")

    assert len(hits) == 1
    assert hits[0].item["evidence_id"] == "cell:rule:指标解释:C6"


def test_year_only_table_query_does_not_mix_same_indicator_from_other_year():
    index = HybridIndex(
        [
            {"doc_id": "bank2024", "title": "2024年商业银行主要监管指标情况表", "file_name": "2024.xls"},
            {"doc_id": "bank2025", "title": "2025年商业银行主要监管指标情况表", "file_name": "2025.xls"},
        ],
        [],
        [
            {"evidence_id": "cell:2024:Sheet1:E14", "doc_id": "bank2024", "indicator": "不良贷款率", "period": "2024", "value_text": "0.02", "column_header": "四季度", "cell_address": "E14", "context": "不良贷款率 | 四季度 | 0.02"},
            {"evidence_id": "cell:2025:Sheet1:E14", "doc_id": "bank2025", "indicator": "不良贷款率", "period": "2025", "value_text": "0.01496", "column_header": "四季度", "cell_address": "E14", "context": "不良贷款率 | 四季度 | 0.01496"},
        ],
    )

    hits = index.search_tables("2025年不良贷款率", top_k=3)

    assert hits and all(hit.item["period"] == "2025" for hit in hits)


def test_regional_table_retrieval_matches_row_and_column_dimension():
    question = "2023年10月全国各地区原保险保费收入情况表.xlsx中“全国合计”在“合计”口径下的数值是多少"
    parsed = parse_query(question)
    index = HybridIndex(
        [{"doc_id": "regional2023", "title": "2023年10月全国各地区原保险保费收入情况表", "file_name": "144_2023年10月全国各地区原保险保费收入情况表_2023年10月全国各地区原保险保费收入情况表.xlsx", "status": "unknown"}],
        [],
        [
            {"evidence_id": "cell:regional2023:Sheet1:C4", "doc_id": "regional2023", "indicator": "全国合计", "period": "合计 / 45167.98", "value_text": "45167.98", "column_header": "合计 / 45167.98", "cell_address": "C4", "context": "全国合计 | 合计 / 45167.98 | 45167.98"},
            {"evidence_id": "cell:regional2023:Sheet1:D4", "doc_id": "regional2023", "indicator": "全国合计", "period": "财产保险 / 11366.02", "value_text": "11366.02", "column_header": "财产保险 / 11366.02", "cell_address": "D4", "context": "全国合计 | 财产保险 / 11366.02 | 11366.02"},
        ],
    )

    hits = index.hybrid_search(question, parsed.qa_type, top_k=3, filters={"file_name": parsed.entities["filenames"]})

    assert hits and hits[0].item["cell_address"] == "C4"

def test_metadata_filter_ignores_minor_title_word_variants():
    index = HybridIndex(
        [{
            "doc_id": "d1",
            "title": "附件1：寿险合同负债评估的折现率曲线",
            "file_name": "附件1：寿险合同负债评估的折现率曲线.pdf",
        }],
        [{
            "evidence_id": "text:d1:p1",
            "doc_id": "d1",
            "paragraph_no": 1,
            "content": "折现率曲线由基础利率曲线加综合溢价形成。",
        }],
        [],
    )

    hits = index.search_text(
        "折现率曲线",
        filters={"title": ["寿险合同负债评估折现率曲线"]},
    )

    assert hits
