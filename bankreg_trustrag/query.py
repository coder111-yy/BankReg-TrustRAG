from __future__ import annotations

import re
from typing import Any

from .schemas import ParsedQuery
from .utils import insurance_company_scope, normalize_text, reporting_period_details


# These are high-value structured-table indicators.  Keeping the longest terms
# first prevents a generic suffix such as ``率`` from winning over the actual
# indicator requested by the user.
KNOWN_INDICATORS = (
    "不良贷款率",
    "拨备覆盖率",
    "贷款拨备率",
    "原保险保费收入",
    "资金运用余额",
    "正常类贷款占比",
    "关注类贷款占比",
    "净息差",
    "资本充足率",
    "核心一级资本充足率",
    "一级资本充足率",
    "资产利润率",
    "净利润",
    "贷款余额",
    "不良贷款余额",
    "银行业总资产",
    "银行业总负债",
)

KNOWN_TABLE_NAMES = (
    "商业银行主要监管指标情况表",
    "商业银行主要指标分机构类情况表",
    "银行业金融机构资产负债情况表",
    "银行业总资产、总负债",
    "保险业经营情况表",
    "保险业资金运用情况表",
    "财产保险公司经营情况表",
    "人身险公司经营情况表",
    "全国各地区原保险保费收入情况表",
)


# The service is intentionally a local bank-regulatory knowledge base.  A
# domain gate prevents generic dense retrieval from turning a short overlap
# such as ``天气`` / ``天的`` into a confident answer.
REGULATORY_DOMAIN_TERMS = (
    "银行", "保险", "金融", "监管", "贷款", "存款", "资本", "保费", "消费金融",
    "合规", "条款", "办法", "通知", "指标", "报表", "数据安全", "风险", "征信",
    "利率", "资产", "负债", "不良", "拨备", "审计", "机构", "业务",
)
OUT_OF_SCOPE_TERMS = (
    "天气", "气温", "温度", "下雨", "晴天", "阴天", "天气预报", "股票行情", "彩票",
    "电影", "音乐", "旅游攻略", "菜谱", "减肥", "写诗", "翻译成英语",
)


CHOICE_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])([A-ZＡ-Ｚ])\s*[\.．、,，:：\)）]\s*", re.IGNORECASE)

REQUIREMENT_KEYS = (
    "retrieval",
    "multi_file",
    "table",
    "calculation",
    "comparison",
    "multi_hop",
    "option_evaluation",
)


def valid_choices(choices: list[str] | None) -> list[str]:
    """Return the single normalized option list used by routing and agents."""
    return [normalize_text(item) for item in (choices or []) if normalize_text(item)]


def choice_label(index: int) -> str:
    """Return spreadsheet-style labels (A..Z, AA..) without silent truncation."""
    if index < 0:
        raise ValueError("choice index must be non-negative")
    label = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def choice_index(label: str) -> int:
    normalized = normalize_text(label).upper()
    if not normalized or not re.fullmatch(r"[A-Z]+", normalized):
        raise ValueError(f"invalid choice label: {label}")
    value = 0
    for char in normalized:
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def classify_question_scope(question: str) -> dict[str, Any]:
    """Classify whether a question belongs to this service's knowledge scope.

    This is a routing guard, not a semantic answer.  Explicit source hints
    (document names, files, or statistical tables) are accepted even when the
    question uses unusual wording; otherwise at least one banking/regulatory
    anchor is required before retrieval is allowed.
    """
    normalized = normalize_text(question)
    source_hint = bool(
        re.search(r"\.(?:xlsx|xls|docx|doc|pdf)", normalized, re.IGNORECASE)
        or re.search(r"《[^》]{2,100}》", normalized)
        or any(term in normalized for term in ("统计表", "报表", "工作表", "Excel", "表中"))
    )
    domain_terms = [term for term in REGULATORY_DOMAIN_TERMS if term in normalized]
    out_of_scope_terms = [term for term in OUT_OF_SCOPE_TERMS if term in normalized]
    if out_of_scope_terms and not source_hint:
        return {
            "in_scope": False,
            "reason": "问题不在银行业监管知识库范围内",
            "domain_terms": domain_terms,
            "out_of_scope_terms": out_of_scope_terms,
            "source_hint": source_hint,
        }
    if not source_hint and not domain_terms:
        return {
            "in_scope": False,
            "reason": "问题不在银行业监管知识库范围内",
            "domain_terms": domain_terms,
            "out_of_scope_terms": out_of_scope_terms,
            "source_hint": source_hint,
        }
    return {
        "in_scope": True,
        "reason": None,
        "domain_terms": domain_terms,
        "out_of_scope_terms": out_of_scope_terms,
        "source_hint": source_hint,
    }


def extract_inline_choices(question: str) -> tuple[str, list[str]]:
    """Extract ordered A-Z options pasted into the question box.

    The web UI intentionally accepts a single free-text question.  Evaluation
    rows, however, keep the stem and options in separate columns, so the
    service must support both request shapes.  We only treat a label sequence
    as choices when at least two labels are present; this avoids splitting
    ordinary prose such as a document title containing ``A.``.
    """
    text = str(question or "").strip()
    matches = list(CHOICE_LABEL_RE.finditer(text))
    if len(matches) < 2:
        return text, []
    # Only accept an ordered option block.  This prevents a stray ``A.`` in
    # the stem followed by a later ``B.`` from destroying the query.
    labels = [normalize_text(match.group(1)).upper() for match in matches]
    indices = [choice_index(label) for label in labels]
    if indices != list(range(indices[0], indices[0] + len(indices))) or indices[0] != 0:
        return text, []
    stem = text[: matches[0].start()].strip(" ：:，,；;\n\t")
    choices: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : end].strip(" ：:，,；;\n\t")
        if value:
            choices.append(value)
    if len(choices) < 2:
        return text, []
    return stem, valid_choices(choices)


def infer_answer_format(question: str, choices: list[str] | None = None) -> str:
    """Infer the requested presentation independently from semantic intent."""
    normalized = normalize_text(question)
    options = valid_choices(choices)
    if len(options) >= 2:
        return "multiple_choice"
    if any(term in normalized for term in ("JSON", "结构化", "字段", "表格形式", "列表形式")):
        return "structured"
    if any(term in normalized for term in ("只返回数字", "仅返回数字", "数值结果", "计算结果")):
        return "number"
    if any(term in normalized for term in ("是多少", "哪一项", "哪个", "查询", "查找", "列出")):
        return "short_answer"
    return "free_text"


def infer_intent(question: str, qa_type: str, choices: list[str] | None = None) -> str:
    """Classify the user's goal into a small, composable intent vocabulary."""
    normalized = normalize_text(question)
    # Summary takes precedence over comparison because questions such as
    # “总结三份文件有什么不同” request a synthesis, not a single winner.
    if any(term in normalized for term in ("总结", "汇总", "归纳", "概括", "综述")):
        return "summarize"
    if any(term in normalized for term in ("核验", "验证", "查证", "确认是否真实", "确认是否一致")):
        return "verify"
    if any(term in normalized for term in ("判断", "是否达标", "是否合规", "是否符合", "是否满足", "满足监管要求")):
        return "judge"
    if any(term in normalized for term in ("最高", "最低", "最大", "最小", "最多", "最少", "比较", "对比", "相比", "差异", "有什么不同")):
        return "compare"
    if any(term in normalized for term in ("计算", "求和", "合计", "平均", "差值", "差额", "增长率", "增幅", "加总")):
        return "calculate"
    if any(term in normalized for term in ("解释", "说明", "为什么", "原因", "含义", "口径是什么")):
        return "explain"
    # The legacy category is a fallback hint only; it never determines the
    # whole workflow.  It preserves sensible behavior for terse benchmark rows.
    if qa_type == "cross_file_judgment":
        return "judge"
    return "lookup"


def infer_requirements(
    question: str,
    qa_type: str,
    entities: dict[str, Any],
    intent: str,
    answer_format: str,
) -> dict[str, bool]:
    """Infer capabilities from language, extracted structure and source scope."""
    normalized = normalize_text(question)
    filenames = list(entities.get("filenames") or [])
    title_hints = list(entities.get("title_hints") or [])
    source_count = len(dict.fromkeys([*filenames, *title_hints]))
    plural_source_language = bool(re.search(
        r"(?:两|二|三|四|五|多|各)\s*(?:份|个)?\s*(?:文件|文档|材料|报表|表格)|跨文件|多个文件|多份文件|分别检索",
        normalized,
    ))
    has_rule_language = any(term in normalized for term in (
        "监管规定", "监管制度", "监管要求", "最低比例", "最高比例", "阈值", "规则", "办法", "条款",
    ))
    has_data_language = any(term in normalized for term in (
        "Excel", "xlsx", "xls", "统计表", "报表", "实际数据", "表中", "单元格", "Sheet",
    ))
    multi_file = bool(
        source_count >= 2
        or plural_source_language
        or (has_rule_language and has_data_language)
        or qa_type == "cross_file_judgment"
    )
    table = bool(
        qa_type == "table_lookup"
        or any(entities.get(key) for key in ("table_name", "row_label", "column_label"))
        or has_data_language
    )
    comparison = bool(
        intent in {"compare", "judge", "verify"}
        or any(term in normalized for term in ("超过", "低于", "高于", "不低于", "不高于", "达标", "相比", "不同", "差异", "异同", "区别"))
    )
    calculation = bool(
        intent == "calculate"
        or any(term in normalized for term in ("计算", "差值", "差额", "增长率", "增幅", "求和", "平均", "最高", "最低", "最大", "最小"))
        or (table and comparison and intent in {"compare", "judge", "verify"})
    )
    option_evaluation = answer_format == "multiple_choice"
    multi_hop = bool(
        multi_file
        or qa_type == "cross_file_judgment"
        or (has_rule_language and has_data_language)
    )
    return {
        "retrieval": True,
        "multi_file": multi_file,
        "table": table,
        "calculation": calculation,
        "comparison": comparison,
        "multi_hop": multi_hop,
        "option_evaluation": option_evaluation,
    }


def enrich_parsed_query(parsed: ParsedQuery, question: str, choices: list[str] | None = None) -> ParsedQuery:
    """Populate the new understanding fields while preserving legacy fields."""
    options = valid_choices(choices)
    parsed.intent = infer_intent(question, parsed.qa_type, options)
    parsed.answer_format = infer_answer_format(question, options)
    parsed.requirements = infer_requirements(
        question,
        parsed.qa_type,
        parsed.entities,
        parsed.intent,
        parsed.answer_format,
    )
    parsed.requires_table = parsed.requirements["table"]
    parsed.requires_multi_hop = parsed.requirements["multi_hop"]
    return parsed


def _contains(text: str, values: tuple[str, ...]) -> bool:
    return any(value in text for value in values)


def _extract_filenames(text: str) -> list[str]:
    """Extract document names without surrounding Chinese prose or brackets."""
    matches = re.findall(
        r"[^\s《》“”\"'，。！？?!；;（）()<>]+?\.(?:xlsx|xls|docx|doc|pdf)",
        text,
        re.IGNORECASE,
    )
    result: list[str] = []
    for match in matches:
        filename = match.strip("《》“”\"'，。！？?!；;（）()<>")
        if filename and filename not in result:
            result.append(filename)
    return result


def extract_indicator(text: str) -> str | None:
    """Extract an explicit structured-table indicator from a question."""
    normalized = normalize_text(text)
    # A metric appearing inside a workbook/table title is not necessarily the
    # requested row indicator.  For example, in
    # ``全国各地区原保险保费收入情况表`` the phrase is the report subject,
    # while the requested row may be ``全国合计``.
    title_free = normalized
    for filename in _extract_filenames(normalized):
        title_free = title_free.replace(filename, "")
    for table_name in KNOWN_TABLE_NAMES:
        title_free = title_free.replace(table_name, "")
    for indicator in sorted(KNOWN_INDICATORS, key=len, reverse=True):
        if indicator in title_free:
            return indicator
    # Handle indicators not in the small high-value vocabulary when the user
    # writes the common ``表中的XXX是多少`` form.
    match = re.search(r"表(?:中|内|里的|中的)的?([^，。！？?!；;]{2,24})(?:[？?。！!]|$)", normalized)
    if match:
        candidate = match.group(1).strip(" ：:，,、")
        candidate = re.sub(r"(?:是多少|数值|值|数据|为多少|有多少|是多少呢)$", "", candidate).strip()
        candidate = re.sub(r"^(请问|查询|查看)", "", candidate)
        if candidate and candidate not in {"经营", "数据", "情况", "指标", "内容", "结果"} and not any(token in candidate for token in ("数据", "结果", "情况")):
            return candidate
    return None


def extract_dimension_labels(text: str) -> tuple[str | None, str | None]:
    """Extract row and column/口径 labels from a structured-table question."""
    normalized = normalize_text(text)
    quoted = [
        normalize_text(value).strip(" ：:，,、")
        for value in re.findall(r"[“\"‘「『]([^”\"’」』]+)[”\"’」』]", normalized)
    ]
    quoted = [value for value in quoted if value]

    # In ``在“健康险”口径下`` the only quoted label is a column, not a row.
    # Treat the grammatical marker as authoritative before falling back to
    # positional quoted labels.  This also preserves the common form
    # ``“全国合计”在“合计”口径下`` as row=全国合计, column=合计.
    column_match = re.search(
        r"(?:在|按|以)\s*[“\"‘「『]?([^”\"’」』，。？！?]{1,24})[”\"’」』]?\s*(?:口径|列|栏目)(?:下|中)?",
        normalized,
    )
    column_label = normalize_text(column_match.group(1)).strip(" ：:，,、") if column_match else None
    row_candidates = [value for value in quoted if value != column_label]
    row_label = row_candidates[0] if row_candidates else None
    if column_label is None and len(quoted) > 1:
        row_label, column_label = quoted[0], quoted[1]
    elif column_label is None and quoted:
        row_label = quoted[0]
    if row_label is None and "全国合计" in normalized:
        row_label = "全国合计"
    if column_label is None and re.search(r"(?:在|按|以)[“\"「『]?合计[”\"」』]?(?:口径|列|栏目)?(?:下|中)?", normalized):
        column_label = "合计"
    return row_label, column_label


def extract_table_name(text: str) -> str | None:
    """Extract a table name without its year/month prefix or file suffix."""
    normalized = normalize_text(text)
    for table_name in sorted(KNOWN_TABLE_NAMES, key=len, reverse=True):
        if table_name in normalized:
            return table_name
    match = re.search(
        r"(?:20\d{2}年(?:\s*\d{1,2}月)?)([\u4e00-\u9fffA-Za-z0-9（）()、，,\-]+?表)(?:中|内|里的|中的|文件|Excel|xlsx|xls|[。！？?!]|$)",
        normalized,
    )
    return match.group(1).strip() if match else None


def extract_title_hints(text: str) -> list[str]:
    """Extract quoted document/material names for metadata filtering."""
    normalized = normalize_text(text)
    hints: list[str] = []
    for value in re.findall(r"《([^》]{2,100})》", normalized):
        hint = value.strip(" ：:，,、")
        if hint and hint not in hints:
            hints.append(hint)
    return hints


def period_details(text: str) -> tuple[str | None, str | None, str | None]:
    """Return display period, normalized month, and quarter label."""
    return reporting_period_details(text)


def parse_query(question: str, choices: list[str] | None = None) -> ParsedQuery:
    inline_stem, inline_choices = extract_inline_choices(question)
    effective_choices = valid_choices(choices or inline_choices)
    text = normalize_text(inline_stem if inline_choices else question)
    qa_type = "regulatory_fact"
    if _contains(text, ("Excel", "表", "指标", "数值", "多少", "余额", "收入", "资产", "负债", "比例", "率")) and re.search(r"20\d{2}|季度|月份|月度", text):
        qa_type = "table_lookup"
    if _contains(text, ("是否合规", "是否符合", "达到", "满足", "判断", "比较", "超过", "低于", "根据.*表")):
        qa_type = "cross_file_judgment" if _contains(text, ("根据", "监管要求", "指标", "表中", "数据")) else "clause_threshold"
    if _contains(text, ("流程", "步骤", "材料", "办理", "申请", "应当包括", "需要哪些")):
        qa_type = "business_process"
    if _contains(text, ("第", "条", "比例", "不得", "应当", "可以", "期限", "阈值", "上限", "下限")) and qa_type == "regulatory_fact":
        qa_type = "clause_threshold"
    entities: dict[str, Any] = {}
    display_period, normalized_period, quarter = period_details(text)
    if display_period:
        entities["period"] = display_period
    if normalized_period:
        entities["period_normalized"] = normalized_period
    if quarter:
        entities["quarter"] = quarter
    years = sorted(set(re.findall(r"(?<!\d)\d{1,4}(?=年)", text)), key=lambda value: int(value))
    if years:
        entities["years"] = years
    article = re.search(r"第[一二三四五六七八九十百零0-9]+[条章节款]", text)
    if article:
        entities["article_no"] = article.group(0)
    indicator = extract_indicator(text)
    if indicator:
        entities["indicator"] = indicator
    table_name = extract_table_name(text)
    if table_name:
        entities["table_name"] = table_name
    title_hints = extract_title_hints(text)
    if title_hints:
        entities["title_hints"] = title_hints
    institution_type = next((value for value in ("商业银行", "银行业金融机构", "保险公司", "保险业") if value in text), None)
    if institution_type:
        entities["institution_type"] = institution_type
    company_scope = insurance_company_scope(text)
    if company_scope:
        entities["insurance_company_scope"] = company_scope
    row_label, column_label = extract_dimension_labels(text)
    if row_label:
        entities["row_label"] = row_label
    if column_label:
        entities["column_label"] = column_label
    entities["filenames"] = _extract_filenames(text)
    rewritten = [text]
    if qa_type in {"table_lookup", "cross_file_judgment"}:
        rewritten.append(text.replace("Excel", "").replace("表中", "表"))
    parsed = ParsedQuery(
        original_query=text,
        qa_type=qa_type,
        entities=entities,
        rewritten_queries=rewritten,
    )
    return enrich_parsed_query(parsed, text, effective_choices)
