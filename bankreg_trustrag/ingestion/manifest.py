from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .parsers import ParseResult, parse_file


SUPPORTED = {".doc", ".docx", ".pdf", ".xls", ".xlsx"}
_ATTACHMENT_RE = re.compile(r"^(?P<parent>.+?)(?:[_\s-]*附件\s*(?P<number>\d+)?)\s*[:：_-]?\s*(?P<title>.*)$")


def iter_source_files(data_dir: Path) -> Iterator[Path]:
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _coverage(records: list[dict], key: str) -> dict[str, Any]:
    total = len(records)
    present = sum(record.get(key) not in (None, "", [], {}) for record in records)
    return {
        "present": present,
        "total": total,
        "rate": round(present / total, 4) if total else 1.0,
    }


def _explicit_authority(text: str) -> str | None:
    aliases = [
        ("国家金融监督管理总局", "国家金融监督管理总局"),
        ("中国银行保险监督管理委员会", "中国银行保险监督管理委员会"),
        ("中国银保监会", "中国银行保险监督管理委员会"),
        ("中国银行业监督管理委员会", "中国银行业监督管理委员会"),
        ("中国银监会", "中国银行业监督管理委员会"),
        ("银监会", "中国银行业监督管理委员会"),
        ("中国保险监督管理委员会", "中国保险监督管理委员会"),
        ("中国保监会", "中国保险监督管理委员会"),
        ("保监会", "中国保险监督管理委员会"),
        ("中国人民银行", "中国人民银行"),
    ]
    for alias, canonical in aliases:
        if alias in text:
            return canonical
    return None


def _reporting_period(title: str) -> str | None:
    value = str(title or "")
    range_match = re.search(r"(20\d{2})年(?:末)?\s*(?:至|到|—|-)\s*(20\d{2})年(?:(\d{1,2})月)?", value)
    if range_match:
        end = range_match.group(2) + (f"-{int(range_match.group(3)):02d}" if range_match.group(3) else "")
        return f"{range_match.group(1)}~{end}"
    quarter = re.search(r"(20\d{2})年.*?([一二三四1-4])季度", value)
    if quarter:
        q = {"一": 1, "二": 2, "三": 3, "四": 4}.get(quarter.group(2), int(quarter.group(2)) if quarter.group(2).isdigit() else 0)
        return f"{quarter.group(1)}-Q{q}"
    month = re.search(r"(20\d{2})年(\d{1,2})月", value)
    if month:
        return f"{month.group(1)}-{int(month.group(2)):02d}"
    year = re.search(r"(20\d{2})年", value)
    return year.group(1) if year else None


_REGULATORY_TITLE_RE = re.compile(
    r"(办法|规定|规则|指引|通知|意见|条例|细则|公告|决定|命令|通告|规范|暂行办法|实施办法|管理办法)"
)


def _document_numbers(text: str) -> list[str]:
    """Extract Chinese regulatory document numbers without treating every year as one."""
    normalized = re.sub(r"\s+", "", str(text or ""))
    patterns = [
        r"[\u4e00-\u9fffA-Za-z]{1,24}〔20\d{2}〕\d+号",
        r"[\u4e00-\u9fffA-Za-z]{1,24}\[20\d{2}\]\d+号",
        r"[\u4e00-\u9fffA-Za-z]{1,24}\(20\d{2}\)\d+号",
        r"[\u4e00-\u9fffA-Za-z]{1,24}（20\d{2}）\d+号",
        r"(?:国家金融监督管理总局|中国人民银行|中国银保监会|中国银监会|中国保监会)令20\d{2}年第\d+号",
        r"(?:国家金融监督管理总局|中国人民银行|中国银保监会|中国银监会|中国保监会)令第\d+号",
    ]
    values: list[str] = []
    for pattern in patterns:
        values.extend(re.findall(pattern, normalized))
    return list(dict.fromkeys(values))[:30]


def _looks_like_regulatory_document(title: str) -> bool:
    return bool(_REGULATORY_TITLE_RE.search(str(title or "")))


def _own_document_number(title: str, text_rows: list[dict]) -> tuple[str | None, str | None]:
    """Find the current document's own number conservatively.

    Referenced regulations in the body remain in referenced_document_nos and
    are not promoted to document_no unless the source strongly resembles an
    official regulation and the number appears as a short header line.
    """
    title_numbers = _document_numbers(title)
    if len(title_numbers) == 1:
        return title_numbers[0], "title"

    if not _looks_like_regulatory_document(title):
        return None, None

    early = [str(item.get("content") or "") for item in text_rows[:10]]
    for line in early[:6]:
        nums = _document_numbers(line)
        compact = re.sub(r"\s+", "", line)
        if len(nums) == 1 and len(compact) <= 70:
            return nums[0], "header"

    combined = " ".join(early[:5])
    nums = _document_numbers(combined)
    if len(nums) == 1:
        return nums[0], "early_content"
    return None, None


def _authority_from_document_no(document_no: str | None) -> str | None:
    value = str(document_no or "")
    if not value:
        return None
    if value.startswith(("金规", "金办发", "金复", "国家金融监督管理总局令")):
        return "国家金融监督管理总局"
    if value.startswith(("银保监", "中国银保监会")):
        return "中国银行保险监督管理委员会"
    if value.startswith(("银监", "中国银监会")):
        return "中国银行业监督管理委员会"
    if value.startswith(("保监", "中国保监会")):
        return "中国保险监督管理委员会"
    if value.startswith(("银发", "银办发", "中国人民银行")):
        return "中国人民银行"
    return None

def _topics(text: str) -> tuple[list[str], list[str]]:
    value = text or ""
    topic_rules = [
        ("资本监管", ("资本管理", "资本充足率", "核心一级资本")),
        ("绿色金融", ("绿色信贷", "绿色金融")),
        ("保险监管", ("保险业", "保险公司", "原保险保费")),
        ("银行监管", ("银行业", "商业银行", "贷款")),
        ("数据安全", ("数据安全", "网络安全", "信息科技")),
        ("风险管理", ("风险管理", "压力测试", "恢复和处置")),
    ]
    domain_rules = [
        ("资本充足率", ("资本充足率", "一级资本", "资本工具")),
        ("绿色信贷", ("绿色信贷",)),
        ("资产负债", ("总资产", "总负债", "资产负债")),
        ("贷款质量", ("不良贷款", "可疑类贷款", "关注类贷款")),
        ("保险经营", ("原保险保费", "保险业经营", "人身险", "财产险")),
    ]
    topics = [name for name, words in topic_rules if any(word in value for word in words)]
    domains = [name for name, words in domain_rules if any(word in value for word in words)]
    return topics, domains


def _enrich_document(document: dict, text_rows: list[dict]) -> dict:
    result = dict(document)
    title = str(result.get("title") or "")
    first_text = " ".join(str(item.get("content") or "") for item in text_rows[:40])
    corpus_text = f"{title} {first_text}"

    attachment = _ATTACHMENT_RE.match(title)
    if attachment:
        family_title = attachment.group("parent").strip(" _-：:")
        result["family_title"] = family_title
        result["attachment_no"] = attachment.group("number") or None
        result["attachment_title"] = (attachment.group("title") or "").strip(" _-：:") or None
        result["parent_title"] = family_title
    else:
        result["family_title"] = title
        result["attachment_no"] = None
        result["attachment_title"] = None
        result["parent_title"] = None

    own_number, own_number_source = _own_document_number(title, text_rows)
    if not result.get("document_no") and own_number:
        result["document_no"] = own_number
    result["document_no_source"] = own_number_source

    referenced = _document_numbers(first_text)
    if result.get("document_no"):
        referenced = [value for value in referenced if value != result.get("document_no")]
    result["referenced_document_nos"] = referenced

    if not result.get("authority"):
        result["authority"] = _explicit_authority(corpus_text)
        if result.get("authority"):
            result["authority_source"] = "content_explicit"
        else:
            inferred = _authority_from_document_no(result.get("document_no"))
            if inferred:
                result["authority"] = inferred
                result["authority_source"] = "document_no_prefix"
            else:
                result["authority_source"] = None
    else:
        result["authority_source"] = "title_explicit"

    result["reporting_period"] = _reporting_period(title)

    topics, domains = _topics(corpus_text)
    result["topic"] = list(dict.fromkeys([*(result.get("topic") or []), *topics]))
    result["regulatory_topic"] = topics
    result["business_domain"] = domains
    result["source_collection"] = str(result.get("local_path") or "").replace("\\", "/").split("/", 1)[0] or None
    result["is_regulatory_document"] = _looks_like_regulatory_document(title)

    core = ["title", "document_type", "local_path", "sha256", "family_title"]
    optional = ["authority", "publish_date", "document_no", "source_url", "reporting_period"]
    score = sum(bool(result.get(key)) for key in core) / len(core) * 0.75
    score += sum(bool(result.get(key)) for key in optional) / len(optional) * 0.25
    result["metadata_quality"] = round(score, 3)
    return result

def _raw_numeric_value(value: Any) -> tuple[float | None, bool]:
    """Return numeric content and whether a percent sign was explicit in text."""
    if isinstance(value, bool) or value is None:
        return None, False
    if isinstance(value, (int, float)):
        return float(value), False

    text = str(value).strip().replace(",", "")
    explicit_percent = text.endswith(("%", "％"))
    text = re.sub(r"[%％]$", "", text).strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return float(text), explicit_percent
    return None, explicit_percent


def _semantic_numeric_value(value: Any, unit: str | None) -> tuple[float | None, float, str | None]:
    """Normalize numeric_value to the displayed business unit.

    Excel commonly stores 8.927% as raw 0.08927.  The evidence keeps raw_value
    for audit, while numeric_value/value become 8.927 so downstream retrieval
    cannot silently answer a rate 100x too small.
    """
    raw, explicit_percent = _raw_numeric_value(value)
    if raw is None:
        return None, 1.0, None

    normalized_unit = str(unit or "").strip().replace("％", "%")
    if not normalized_unit and explicit_percent:
        normalized_unit = "%"
    scale = 1.0
    numeric = raw
    if normalized_unit == "%":
        if explicit_percent:
            numeric = raw
        elif abs(raw) <= 1.0:
            scale = 100.0
            numeric = raw * scale
    elif normalized_unit == "‰":
        if abs(raw) <= 1.0:
            scale = 1000.0
            numeric = raw * scale

    rendered = format(numeric, ".15g")
    display = f"{rendered}{normalized_unit}" if normalized_unit else rendered
    return numeric, scale, display


def _numeric_value(value: Any) -> float | None:
    # Kept for cell classification and backwards-compatible tests.
    raw, explicit_percent = _raw_numeric_value(value)
    if raw is None:
        return None
    return raw if not explicit_percent else raw


def _cell_type(record: dict) -> str:
    value = record.get("value")
    text = str(value or "").strip()
    indicator = str(record.get("indicator") or "").strip()
    if indicator == "注" or text.startswith(("注：", "注:", "说明：", "说明:")):
        return "note"
    if re.search(r"^单位[:：]", text):
        return "unit"
    if _raw_numeric_value(value)[0] is not None:
        return "data"
    if text in {"时间", "项目", "年份", "年度"} or re.fullmatch(r"(?:20\d{2}年?|(?:1[0-2]|0?[1-9])月|[一二三四]季度)", text):
        return "header"
    if text and text == indicator:
        return "label"
    return "text_data" if indicator and text else "label"


def _statistical_scope(record: dict) -> str | None:
    blob = " ".join(str(record.get(key) or "") for key in ["column_header", "row_header", "context"])
    candidates = [
        "本年累计/截至当期", "截至当期-账面余额", "截至当期", "本年累计",
        "期末余额", "账面余额", "当期", "同比", "比上年同期增长率",
    ]
    return next((value for value in candidates if value in blob), None)


_REGION_ALIASES = {
    "全国": "全国", "全国合计": "全国",
    "北京": "北京", "北京市": "北京", "天津": "天津", "天津市": "天津",
    "河北": "河北", "河北省": "河北", "山西": "山西", "山西省": "山西",
    "内蒙古": "内蒙古", "辽宁": "辽宁", "辽宁省": "辽宁", "吉林": "吉林", "吉林省": "吉林",
    "黑龙江": "黑龙江", "黑龙江省": "黑龙江", "上海": "上海", "上海市": "上海",
    "江苏": "江苏", "江苏省": "江苏", "浙江": "浙江", "浙江省": "浙江",
    "安徽": "安徽", "安徽省": "安徽", "福建": "福建", "福建省": "福建",
    "江西": "江西", "江西省": "江西", "山东": "山东", "山东省": "山东",
    "河南": "河南", "河南省": "河南", "湖北": "湖北", "湖北省": "湖北",
    "湖南": "湖南", "湖南省": "湖南", "广东": "广东", "广东省": "广东",
    "广西": "广西", "海南": "海南", "海南省": "海南", "重庆": "重庆", "重庆市": "重庆",
    "四川": "四川", "四川省": "四川", "贵州": "贵州", "贵州省": "贵州",
    "云南": "云南", "云南省": "云南", "西藏": "西藏", "陕西": "陕西", "陕西省": "陕西",
    "甘肃": "甘肃", "甘肃省": "甘肃", "青海": "青海", "青海省": "青海",
    "宁夏": "宁夏", "新疆": "新疆", "大连": "大连", "宁波": "宁波",
    "厦门": "厦门", "青岛": "青岛", "深圳": "深圳",
}


def _dimension_parts(record: dict) -> list[str]:
    values = [
        str(record.get("column_header") or ""),
        str(record.get("row_header") or ""),
        str(record.get("context") or ""),
    ]
    parts: list[str] = []
    for value in values:
        for part in re.split(r"\s*(?:/|\||>|→)\s*", value):
            cleaned = re.sub(r"^\s*\d+(?:\.\d+)*[、.\s]*", "", part).strip()
            if cleaned and cleaned not in parts:
                parts.append(cleaned)
    return parts


def _infer_region(record: dict) -> str | None:
    for part in _dimension_parts(record):
        compact = re.sub(r"\s+", "", part)
        if compact in _REGION_ALIASES:
            return _REGION_ALIASES[compact]
        for alias, canonical in _REGION_ALIASES.items():
            if compact.startswith(alias) and compact.endswith(("合计", "地区")):
                return canonical
    return None


def _infer_institution(record: dict) -> str | None:
    indicator = str(record.get("indicator") or "")
    for part in _dimension_parts(record):
        compact = re.sub(r"\s+", "", part)
        if not compact or compact == indicator:
            continue
        if re.search(r"(月|季度|年)$", compact):
            continue
        if any(noise in compact for noise in ("贷款余额", "保费收入", "总资产", "总负债", "增长率", "占比")):
            continue
        if re.search(r"(银行业金融机构|商业银行|政策性银行|开发性金融机构|邮政储蓄银行|城市商业银行|农村商业银行|农村合作银行|外资银行|外资法人银行|信用社|财务公司|信托公司|保险公司|人身险公司|财产保险公司|保险机构)$", compact):
            return compact
    return None


def _infer_insurance_type(record: dict) -> str | None:
    blob = " ".join(_dimension_parts(record))
    for value in ("人身险", "财产险", "健康险", "寿险", "意外险", "车险"):
        if value in blob:
            return value
    return None


def _enrich_table_cell(record: dict) -> dict:
    result = dict(record)
    result["cell_type"] = _cell_type(result)

    raw_value = result.get("value")
    result["raw_value"] = raw_value
    numeric, scale, display_value = _semantic_numeric_value(raw_value, result.get("unit"))
    result["numeric_value"] = numeric
    result["value_scale"] = scale
    result["display_value"] = display_value

    # Production retrieval reads "value".  For numeric evidence store the
    # business/display numeric value while raw_value preserves the source cell.
    if result["cell_type"] == "data" and numeric is not None:
        result["value"] = numeric

    result["statistical_scope"] = _statistical_scope(result)
    result["institution"] = _infer_institution(result)
    result["region"] = _infer_region(result)
    result["insurance_type"] = _infer_insurance_type(result)
    result["dimension_labels"] = _dimension_parts(result)

    period = str(result.get("period") or "")
    year = re.search(r"(20\d{2})", period)
    month = re.search(r"(?:-|/|年)(1[0-2]|0?[1-9])(?:月)?$", period)
    quarter = re.search(r"(?:Q([1-4])|([一二三四])季度)", period)
    result["year"] = int(year.group(1)) if year else None
    result["month"] = int(month.group(1)) if month else None
    if quarter:
        result["quarter"] = int(quarter.group(1)) if quarter.group(1) else {"一": 1, "二": 2, "三": 3, "四": 4}[quarter.group(2)]
    else:
        result["quarter"] = None
    return result

def _enrich_text(record: dict) -> dict:
    result = dict(record)
    content = str(result.get("content") or "")
    if result.get("article_no"):
        result["content_type"] = "article"
    elif re.match(r"^第[一二三四五六七八九十百千万零〇0-9]+章", content):
        result["content_type"] = "chapter_heading"
    elif result.get("section") and content == result.get("section"):
        result["content_type"] = "section_heading"
    elif content.startswith("表") and "行" in content[:20]:
        result["content_type"] = "table_row_text"
    else:
        result["content_type"] = "paragraph"
    return result


def build_document_relations(documents: list[dict]) -> list[dict]:
    """Build duplicate and attachment relations only from explicit local evidence."""
    relations: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    by_sha: dict[str, list[dict]] = {}
    by_title = {str(item.get("title") or ""): item for item in documents}

    for document in documents:
        by_sha.setdefault(str(document.get("sha256") or ""), []).append(document)
    for group in by_sha.values():
        if len(group) < 2:
            continue
        primary = group[0]
        for duplicate in group[1:]:
            key = (str(duplicate["doc_id"]), str(primary["doc_id"]), "duplicate_of")
            if key not in seen and key[0] != key[1]:
                seen.add(key)
                relations.append({
                    "source_doc_id": key[0], "target_doc_id": key[1],
                    "relation_type": key[2], "confidence": 1.0,
                    "rationale": "identical_sha256",
                })

    for document in documents:
        parent_title = str(document.get("parent_title") or "")
        if not parent_title:
            match = _ATTACHMENT_RE.match(str(document.get("title") or ""))
            parent_title = match.group("parent").strip(" _-：:") if match else ""
        parent = by_title.get(parent_title)
        if parent and parent["doc_id"] != document["doc_id"]:
            key = (str(document["doc_id"]), str(parent["doc_id"]), "attachment_of")
            if key not in seen:
                seen.add(key)
                relations.append({
                    "source_doc_id": key[0], "target_doc_id": key[1],
                    "relation_type": key[2], "confidence": 1.0,
                    "rationale": "explicit_attachment_title",
                })
    return relations


def _subset_coverage(records: list[dict], key: str) -> dict[str, Any]:
    return _coverage(records, key)


def _quality_report(
    documents: list[dict],
    text: list[dict],
    raw_tables: list[dict],
    production_tables: list[dict],
    errors: list[dict],
) -> dict:
    data_cells = [item for item in production_tables if item.get("cell_type") == "data"]
    structure_cells = [
        item for item in raw_tables
        if item.get("cell_type") in {"header", "label", "unit"}
    ]
    traceable_text = sum(bool(item.get("source_location")) for item in text)
    traceable_table = sum(bool(item.get("cell_address") and item.get("sheet_name")) for item in production_tables)

    by_doc_type = {str(item.get("doc_id")): str(item.get("document_type") or "") for item in documents}
    pdf_text = [item for item in text if by_doc_type.get(str(item.get("doc_id"))) == "pdf"]

    regulatory_documents = [item for item in documents if item.get("is_regulatory_document")]
    regulation_ids = {str(item.get("doc_id")) for item in regulatory_documents}
    article_doc_ids = {
        str(item.get("doc_id"))
        for item in text
        if item.get("article_no") and str(item.get("doc_id")) in regulation_ids
    }

    percentage_cells = [
        item for item in data_cells
        if str(item.get("unit") or "").replace("％", "%") == "%"
    ]
    normalized_percent: list[dict] = []
    for item in percentage_cells:
        raw, explicit_percent = _raw_numeric_value(item.get("raw_value"))
        numeric = item.get("numeric_value")
        scale = float(item.get("value_scale") or 1.0)
        if raw is None or numeric is None:
            continue
        expected_scale = 1.0 if explicit_percent or abs(raw) > 1.0 else 100.0
        expected = raw * expected_scale
        if abs(scale - expected_scale) < 1e-9 and abs(float(numeric) - expected) < 1e-8:
            normalized_percent.append(item)

    duplicate_content_groups = Counter(str(item.get("sha256") or "") for item in documents)
    duplicate_content_groups = {
        key: count for key, count in duplicate_content_groups.items()
        if key and count > 1
    }

    report = {
        "documents": {
            "count": len(documents),
            "unique_doc_ids": len({item.get("doc_id") for item in documents}),
            "duplicate_content_groups": len(duplicate_content_groups),
            "duplicate_content_files": sum(duplicate_content_groups.values()),
            "title": _coverage(documents, "title"),
            "document_type": _coverage(documents, "document_type"),
            "authority": _coverage(documents, "authority"),
            "publish_date": _coverage(documents, "publish_date"),
            "document_no": _coverage(documents, "document_no"),
            "reporting_period": _coverage(documents, "reporting_period"),
            "source_url": _coverage(documents, "source_url"),
            "family_title": _coverage(documents, "family_title"),
            "regulatory_documents": len(regulatory_documents),
            "regulatory_authority": _coverage(regulatory_documents, "authority"),
            "regulatory_document_no": _coverage(regulatory_documents, "document_no"),
        },
        "text_evidence": {
            "count": len(text),
            "page_all": _coverage(text, "page"),
            "page_pdf_only": _coverage(pdf_text, "page"),
            "chapter_all": _coverage(text, "chapter"),
            "section_all": _coverage(text, "section"),
            "article_no_all": _coverage(text, "article_no"),
            "regulatory_documents_with_article_structure": {
                "present": len(article_doc_ids),
                "total": len(regulatory_documents),
                "rate": round(len(article_doc_ids) / len(regulatory_documents), 4)
                if regulatory_documents else 1.0,
            },
            "traceable_rate": round(traceable_text / len(text), 4) if text else 1.0,
        },
        "table_evidence": {
            "raw_cell_count": len(raw_tables),
            "production_count": len(production_tables),
            "raw_cell_types": dict(Counter(str(item.get("cell_type") or "unknown") for item in raw_tables)),
            "production_cell_types": dict(Counter(str(item.get("cell_type") or "unknown") for item in production_tables)),
            "structure_cells": len(structure_cells),
            "data_cells": len(data_cells),
            "data_indicator": _coverage(data_cells, "indicator"),
            "data_period": _coverage(data_cells, "period"),
            "data_numeric_value": _coverage(data_cells, "numeric_value"),
            "data_unit": _coverage(data_cells, "unit"),
            "data_institution": _coverage(data_cells, "institution"),
            "data_region": _coverage(data_cells, "region"),
            "percentage_cells": len(percentage_cells),
            "percentage_semantics": {
                "present": len(normalized_percent),
                "total": len(percentage_cells),
                "rate": round(len(normalized_percent) / len(percentage_cells), 4)
                if percentage_cells else 1.0,
            },
            "traceable_rate": round(traceable_table / len(production_tables), 4)
            if production_tables else 1.0,
        },
        "issues": {
            "errors": sum(item.get("severity") == "error" for item in errors),
            "warnings": sum(item.get("severity") != "error" for item in errors),
            "samples": errors[:20],
        },
    }

    production_types = set(report["table_evidence"]["production_cell_types"])
    report["acceptance"] = {
        "doc_id_unique": report["documents"]["unique_doc_ids"] == len(documents),
        "title_rate_ge_99": report["documents"]["title"]["rate"] >= 0.99,
        "file_type_rate_100": report["documents"]["document_type"]["rate"] == 1.0,
        "text_traceability_100": report["text_evidence"]["traceable_rate"] == 1.0,
        "table_traceability_100": report["table_evidence"]["traceable_rate"] == 1.0,
        "data_indicator_rate_ge_95": report["table_evidence"]["data_indicator"]["rate"] >= 0.95,
        "data_period_rate_ge_95": report["table_evidence"]["data_period"]["rate"] >= 0.95,
        "data_numeric_value_100": report["table_evidence"]["data_numeric_value"]["rate"] == 1.0,
        "percentage_semantics_100": report["table_evidence"]["percentage_semantics"]["rate"] == 1.0,
        "production_table_types_clean": production_types.issubset({"data", "note", "text_data"}),
    }
    return report

def _issue_severity(message: str) -> str:
    value = str(message or "").lower()
    hard = ("failed", "missing", "no usable", "unsupported", "timeout", "unhandled")
    return "error" if any(token in value for token in hard) else "warning"


def build_manifest(data_dir: Path, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    documents: list[dict] = []
    text_evidence: list[dict] = []
    raw_table_cells: list[dict] = []
    errors: list[dict] = []

    for path in iter_source_files(data_dir):
        try:
            result: ParseResult = parse_file(path, data_dir)
            text_rows = [_enrich_text(item.to_dict()) for item in result.text_evidence]
            table_rows = [_enrich_table_cell(item.to_dict()) for item in result.table_evidence]
            document = _enrich_document(result.document.to_dict(), text_rows)
            documents.append(document)
            text_evidence.extend(text_rows)
            raw_table_cells.extend(table_rows)
            errors.extend(
                {
                    "local_path": result.document.local_path,
                    "warning": warning,
                    "severity": _issue_severity(warning),
                }
                for warning in result.warnings
            )
        except Exception as exc:
            errors.append({
                "local_path": str(path),
                "warning": f"unhandled parse failure: {type(exc).__name__}: {exc}",
                "severity": "error",
            })

    # Keep all classified cells for audit. Production evidence excludes
    # structural title/header/unit/label cells so they cannot compete in Top-K.
    production_types = {"data", "note", "text_data"}
    table_evidence = [
        item for item in raw_table_cells
        if item.get("cell_type") in production_types
    ]
    table_structure = [
        item for item in raw_table_cells
        if item.get("cell_type") not in production_types
    ]

    write_jsonl(artifact_dir / "documents.jsonl", documents)
    write_jsonl(artifact_dir / "text_evidence.jsonl", text_evidence)
    write_jsonl(artifact_dir / "table_cells_raw.jsonl", raw_table_cells)
    write_jsonl(artifact_dir / "table_structure.jsonl", table_structure)
    write_jsonl(artifact_dir / "table_evidence.jsonl", table_evidence)
    relations = build_document_relations(documents)
    write_jsonl(artifact_dir / "document_relations.jsonl", relations)
    write_jsonl(artifact_dir / "ingestion_errors.jsonl", errors)

    quality = _quality_report(documents, text_evidence, raw_table_cells, table_evidence, errors)
    (artifact_dir / "ingestion_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "data_dir": str(data_dir.resolve()),
        "documents": len(documents),
        "unique_documents": len({item["doc_id"] for item in documents}),
        "duplicate_documents": len(documents) - len({item["doc_id"] for item in documents}),
        "text_evidence": len(text_evidence),
        "table_cells_raw": len(raw_table_cells),
        "table_structure": len(table_structure),
        "table_evidence": len(table_evidence),
        "duplicate_content_groups": quality["documents"]["duplicate_content_groups"],
        "document_relations": len(relations),
        "errors": sum(item.get("severity") == "error" for item in errors),
        "warnings": sum(item.get("severity") != "error" for item in errors),
        "quality_report": "ingestion_quality_report.json",
        "error_samples": errors[:20],
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
