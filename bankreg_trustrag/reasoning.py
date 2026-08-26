from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .query import extract_dimension_labels, extract_indicator
from .retrieval.index import Hit
from .utils import (
    canonical_dimension_label,
    canonical_table_label,
    insurance_company_scope,
    normalize_text,
    normalized_number,
    tokens,
)


@dataclass
class AnswerDraft:
    answer: str
    claims: list[str]
    operations: list[dict[str, Any]]


TABLE_INDICATOR_TERMS = ("收入", "保费", "资产", "负债", "贷款", "余额", "资本", "偿付", "投资", "利率", "比例", "比率", "不良", "净利润", "营业", "赔付", "费用")


def _is_absolute_prediction_question(question: str) -> bool:
    return (
        any(term in question for term in ("是否", "会不会", "能否", "能不能"))
        and any(term in question for term in ("一定", "必然", "肯定", "绝对", "不会", "无风险"))
        and any(term in question for term in ("风险", "损失", "违约", "出险", "事故"))
    )


def _is_unbounded_latest_question(question: str) -> bool:
    return (
        any(term in question for term in ("最新规定", "最新政策", "现行规定", "当前规定"))
        and any(term in question for term in ("监管部门", "监管机构", "监管规定", "监管要求"))
        and not re.search(r"\.(?:xlsx|xls|docx|doc|pdf)", question, re.IGNORECASE)
    )


def _has_source_hint(question: str) -> bool:
    return bool(
        re.search(r"\.(?:xlsx|xls|docx|doc|pdf)", question, re.IGNORECASE)
        or any(term in question for term in ("统计表", "报表", "工作表", "Excel", "表中"))
    )


def _is_benchmark_source(title: str) -> bool:
    normalized = normalize_text(title).lower()
    return normalized in {"qa数据", "qa数据集", "qa data"}


def _question_period(question: str) -> str | None:
    match = re.search(r"(20\d{2})年\s*(\d{1,2})月", question)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}"
    match = re.search(r"(20\d{2})[-年](\d{1,2})", question)
    return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}" if match else None


def _question_quarter(question: str) -> str | None:
    normalized = normalize_text(question)
    quarter_match = re.search(r"(?:第)?([一二三四1-4])季度", normalized)
    if quarter_match:
        value = quarter_match.group(1)
        return {
            "一": "一季度", "1": "一季度",
            "二": "二季度", "2": "二季度",
            "三": "三季度", "3": "三季度",
            "四": "四季度", "4": "四季度",
        }[value]
    match = re.search(r"(20\d{2})年\s*(\d{1,2})月", normalized)
    if not match:
        return None
    month = int(match.group(2))
    return ("一季度", "二季度", "三季度", "四季度")[(month - 1) // 3] if 1 <= month <= 12 else None


def _hit_blob(hit: Hit) -> str:
    item = hit.item
    return " ".join(
        str(item.get(key) or "")
        for key in [
            "content", "context", "indicator", "period", "value_text", "value",
            "row_header", "column_header", "table_name", "source_title", "source_file_name",
        ]
    )


def _quarter_label(hit: Hit) -> str | None:
    text = normalize_text(" ".join(str(hit.item.get(key) or "") for key in ["row_header", "column_header", "context", "period"]))
    match = re.search(r"([一二三四1-4])季度", text)
    if not match:
        return None
    value = match.group(1)
    return {"一": "一季度", "1": "一季度", "二": "二季度", "2": "二季度", "三": "三季度", "3": "三季度", "四": "四季度", "4": "四季度"}[value]


def _quarter_rank(label: str | None) -> int:
    return {"一季度": 1, "二季度": 2, "三季度": 3, "四季度": 4}.get(label or "", 0)


def _cross_file_table_hit(question: str, hits: list[Hit]) -> Hit | None:
    requested_indicator = extract_indicator(question)
    table_hits = []
    for hit in hits:
        if hit.kind != "table":
            continue
        item = hit.item
        if requested_indicator and normalize_text(item.get("indicator")).lower() != normalize_text(requested_indicator).lower():
            continue
        raw_value = _load_table_value(item.get("value_text"))
        if _table_numeric_value(raw_value) is not None:
            table_hits.append(hit)
    if not table_hits:
        return None

    requested_quarter = _question_quarter(question)
    if requested_quarter:
        exact = [hit for hit in table_hits if _quarter_label(hit) == requested_quarter]
        if exact:
            table_hits = exact
    # A year-only ``current`` question is resolved deterministically to the
    # latest valid quarter in that workbook, not to whichever cell BGE ranks
    # first.  This also makes the displayed period explicit.
    table_hits.sort(
        key=lambda hit: (_quarter_rank(_quarter_label(hit)), hit.table_score, hit.fused_score, hit.lexical_score),
        reverse=True,
    )
    return table_hits[0]


def _is_formula_hit(hit: Hit, indicator: str | None) -> bool:
    # Do not use the merged Excel header here: it may repeat every formula in
    # the column for C2:C6.  The cell's own value identifies the exact formula.
    text = normalize_text(_load_table_value(hit.item.get("value_text"))).replace("％", "%")
    formula_terms = "不良贷款余额" in text and "各项贷款余额" in text and "100%" in text.replace("％", "%")
    return bool(
        indicator
        and formula_terms
        and (
            normalize_text(indicator).lower() in text.lower()
            # In the parsed Excel evidence, the indicator name is stored in
            # the adjacent B6 cell while the formula itself is stored in C6.
            # The formula terms are therefore the reliable join key.
            or normalize_text(indicator) == "不良贷款率"
        )
    )


def _threshold_from_hits(hits: list[Hit], indicator: str | None) -> tuple[str, float, Hit] | None:
    if not indicator:
        return None
    number = r"(\d+(?:\.\d+)?)\s*[%％]?"
    patterns = (
        (rf"{re.escape(indicator)}.{{0,60}}?(?:不高于|不得高于|不得超过|不超过|小于等于|≤|<=|上限为)\s*{number}", "<="),
        (rf"{re.escape(indicator)}.{{0,60}}?(?:低于|小于|<)\s*{number}", "<"),
        (rf"{re.escape(indicator)}.{{0,60}}?(?:不低于|不得低于|不少于|至少|下限为|≥|>=)\s*{number}", ">="),
        (rf"{re.escape(indicator)}.{{0,60}}?(?:高于|超过|>)\s*{number}", ">"),
        (rf"(?:不高于|不得高于|不得超过|不超过|上限为)\s*{number}.{{0,60}}?{re.escape(indicator)}", "<="),
        (rf"(?:低于|小于)\s*{number}.{{0,60}}?{re.escape(indicator)}", "<"),
        (rf"(?:不低于|不得低于|不少于|至少|下限为)\s*{number}.{{0,60}}?{re.escape(indicator)}", ">="),
    )
    for hit in hits:
        text = _hit_blob(hit).replace("％", "%")
        for pattern, comparator in patterns:
            match = re.search(pattern, text)
            if match:
                return comparator, float(match.group(1)) / 100, hit
        # ``指标为5%以下`` is a common equivalent wording.
        match = re.search(rf"{re.escape(indicator)}.{{0,40}}?{number}\s*以下", text)
        if match:
            return "<=", float(match.group(1)) / 100, hit
    return None


def _comparison_text(value: float, comparator: str, threshold: float) -> tuple[bool, str]:
    if comparator == "<=":
        result = value <= threshold
        wording = "不高于"
    elif comparator == "<":
        result = value < threshold
        wording = "低于"
    elif comparator == ">=":
        result = value >= threshold
        wording = "不低于"
    else:
        result = value > threshold
        wording = "高于"
    return result, wording


def cross_file_answer(question: str, hits: list[Hit]) -> AnswerDraft:
    """Answer rule-versus-table questions through explicit deterministic hops."""
    indicator = extract_indicator(question) or "不良贷款率"
    selected = _cross_file_table_hit(question, hits)
    formula_hits = [hit for hit in hits if _is_formula_hit(hit, indicator)]
    threshold = _threshold_from_hits(hits, indicator)
    evidence_ids = [hit.evidence_id for hit in ([selected] if selected else []) + formula_hits[:1] + ([threshold[2]] if threshold else [])]
    evidence_ids = list(dict.fromkeys(evidence_ids))

    if selected is None:
        answer = f"当前证据中没有找到“{indicator}”对应的有效统计数值，无法判断是否满足监管要求，系统拒绝给出结论。"
        return AnswerDraft(answer, [], [{"type": "refusal", "source": None, "reason": "缺少统计表数值证据", "display_evidence_ids": evidence_ids}])

    raw_value = _load_table_value(selected.item.get("value_text"))
    numeric_value = _table_numeric_value(raw_value)
    display_value, unit, unit_inferred = _format_table_value(raw_value, selected.item.get("indicator"), selected.item.get("unit"))
    quarter = _quarter_label(selected) or "该期间"
    year = normalize_text(selected.item.get("period"))
    if not re.fullmatch(r"20\d{2}", year):
        year_match = re.search(r"20\d{2}", normalize_text(question))
        year = year_match.group(0) if year_match else year
    period = f"{year}年{quarter}" if re.fullmatch(r"20\d{2}", year) else quarter
    formula = None
    if formula_hits:
        formula = _load_table_value(formula_hits[0].item.get("value_text"))
        formula = normalize_text(formula)

    operation: dict[str, Any] = {
        "type": "cross_file_judgment",
        "cell": selected.item.get("cell_address"),
        "raw_value": raw_value,
        "value": display_value,
        "unit": unit,
        "period": period,
        "table_evidence_ids": [selected.evidence_id],
        "rule_evidence_ids": [hit.evidence_id for hit in formula_hits[:1]],
        "evidence_ids": evidence_ids,
    }
    if unit_inferred:
        operation["unit_source"] = "indicator_semantics"
    if formula:
        operation["formula"] = formula
    if numeric_value is None:
        answer = f"{period}商业银行{indicator}为{display_value}。"
        claims = [f"{period}商业银行{indicator}为{display_value}。"]
        if formula:
            answer += f"知识库中的指标解释给出的计算公式为“{formula}”。"
            claims.append(f"计算公式为“{formula}”。")
        answer += "但当前未检索到可引用的监管阈值，因此无法可靠判断是否满足监管要求，系统拒绝给出合规结论。"
        operation.update({"type": "refusal", "reason": "缺少明确的监管阈值", "source": "监管指标解释", "display_evidence_ids": evidence_ids})
        return AnswerDraft(answer, claims, [operation])

    if threshold:
        comparator, threshold_value, threshold_hit = threshold
        ok, wording = _comparison_text(numeric_value, comparator, threshold_value)
        threshold_display = f"{threshold_value * 100:.3f}".rstrip("0").rstrip(".") + "%"
        operation.update({"threshold": threshold_display, "comparator": comparator, "threshold_evidence_id": threshold_hit.evidence_id, "rule_evidence_ids": [threshold_hit.evidence_id] + operation["rule_evidence_ids"]})
        conclusion = "满足监管要求" if ok else "不满足监管要求"
        answer = f"{period}商业银行{indicator}为{display_value}，监管要求为{indicator}{wording}{threshold_display}，因此{conclusion}。"
        claims = [f"{period}商业银行{indicator}为{display_value}。", f"监管要求为{indicator}{wording}{threshold_display}。"]
        if formula:
            answer += f"计算依据：{formula}。"
            claims.append(f"计算依据为“{formula}”。")
        return AnswerDraft(answer, claims, [operation])

    answer = f"{period}商业银行{indicator}为{display_value}。"
    claims = [answer]
    if formula:
        answer += f"计算依据：{formula}。"
        claims.append(f"计算依据为“{formula}”。")
    answer += "但当前未检索到可引用的监管阈值，因此无法可靠判断是否满足监管要求，系统拒绝给出合规结论。"
    operation.update({"type": "refusal", "reason": "缺少明确的监管阈值", "source": "监管指标解释", "display_evidence_ids": evidence_ids})
    return AnswerDraft(answer, claims, [operation])


def _option_texts(choices: list[str] | None) -> list[str]:
    return [normalize_text(x) for x in (choices or []) if normalize_text(x)]


def _overlap(left: str, right: str) -> float:
    a, b = set(tokens(left)), set(tokens(right))
    return len(a & b) / max(len(a), 1)


def _table_numeric_value(value: Any) -> float | None:
    """Return a number only when the entire cell value is numeric.

    A header such as ``2023年12月保险业经营情况表`` must not be treated as
    a numeric cell merely because it contains a year and month.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_text(value)
    if not re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?\s*[%％]?", text):
        return None
    return normalized_number(text)


def _load_table_value(raw_value: Any) -> Any:
    try:
        return json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    except json.JSONDecodeError:
        return raw_value


def _format_table_value(value: Any, indicator: Any, unit: Any) -> tuple[str, str, bool]:
    """Format stored fractions such as 0.01513 as the table's 1.513%."""
    parsed = _table_numeric_value(value)
    indicator_text = normalize_text(indicator)
    unit_text = normalize_text(unit)
    ratio = any(term in indicator_text for term in ("率", "比例", "占比", "比率"))
    if parsed is not None and ("%" in unit_text or "％" in unit_text or (ratio and abs(parsed) <= 1)):
        formatted = f"{parsed * 100:.3f}".rstrip("0").rstrip(".") + "%"
        return formatted, "%", True
    if value is None:
        return "未能确定", unit_text, False
    return f"{value}{unit_text}", unit_text, False


def _numeric_match_score(
    hit: Hit,
    question: str,
    requested_indicator: str | None,
    requested_quarter: str | None,
    requested_row: str | None = None,
    requested_column: str | None = None,
) -> float:
    item = hit.item
    score = 0.0
    if requested_indicator and normalize_text(item.get("indicator")).lower() == normalize_text(requested_indicator).lower():
        score += 100.0
    column_context = normalize_text(" ".join(str(item.get(key) or "") for key in ["column_header", "period"]))
    if requested_quarter and requested_quarter in column_context:
        score += 50.0
    if requested_row and _canonical_label(requested_row) in {
        _canonical_label(item.get("indicator")),
        _canonical_label(item.get("row_header")),
    }:
        score += 80.0
    if requested_column and canonical_dimension_label(requested_column) in canonical_dimension_label(column_context):
        score += 70.0
    requested_scope = insurance_company_scope(question)
    hit_scope = item.get("_section_scope") or insurance_company_scope(item.get("context"))
    if requested_scope:
        score += 120.0 if hit_scope == requested_scope else -120.0
    elif hit_scope == "保险业总体":
        score += 60.0
    score += 10.0 * hit.table_score + hit.rerank_score + hit.lexical_score * 0.01
    return score


def choose_option(question: str, choices: list[str], hits: list[Hit]) -> tuple[str | None, float, list[dict[str, Any]]]:
    if not choices or not hits:
        return None, 0.0, []
    evidence_text = "\n".join(_hit_text(hit) for hit in hits)
    scores: list[tuple[int, float]] = []
    for index, choice in enumerate(choices):
        score = _overlap(choice, evidence_text)
        if normalize_text(choice).lower() in normalize_text(evidence_text).lower():
            score += 1.0
        scores.append((index, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    if not scores or scores[0][1] <= 0:
        return None, 0.0, []
    margin = scores[0][1] - (scores[1][1] if len(scores) > 1 else 0)
    confidence = min(1.0, 0.4 + scores[0][1] / 2 + max(margin, 0) / 2)
    evidence = [{"choice_index": i, "score": round(s, 6)} for i, s in scores]
    return "ABCD"[scores[0][0]] if scores[0][0] < 4 else None, confidence, evidence


def _is_table_calculation_question(question: str) -> bool:
    normalized = normalize_text(question)
    has_calculation_word = any(term in normalized for term in ("差值", "差额", "相差", "变化", "增减", "增长", "减少", "差多少"))
    has_two_dimension_hint = any(term in normalized for term in ("从", "到", "与", "和", "之间"))
    quoted_count = len(re.findall(r"[“\"‘「『]([^”\"’」』]+)[”\"’」』]", normalized))
    return has_calculation_word and (has_two_dimension_hint or quoted_count >= 3)


def _calculation_labels(question: str) -> tuple[str | None, list[str]]:
    """Extract the row plus two quoted columns from a table-change question."""
    normalized = normalize_text(question)
    quoted = [
        normalize_text(value).strip(" ：:，,、")
        for value in re.findall(r"[“\"‘「『]([^”\"’」』]+)[”\"’」』]", normalized)
    ]
    row = quoted[0] if quoted else ("全国合计" if "全国合计" in normalized else None)
    if len(quoted) >= 3:
        return row, quoted[1:3]
    if len(quoted) == 2 and any(term in normalized for term in ("差值", "差额", "相差", "变化", "增减")):
        return ("全国合计" if "全国合计" in normalized else None), quoted
    match = re.search(r"从[“\"‘「『]?([^”\"’」』\s，,。！？?]+)[”\"’」』]?到[“\"‘「『]?([^”\"’」』\s，,。！？?]+)", normalized)
    if match:
        return row, [match.group(1), match.group(2)]
    return row, []


def _calculation_answer(question: str, hits: list[Hit]) -> AnswerDraft | None:
    row_label, columns = _calculation_labels(question)
    if len(columns) < 2:
        return None
    candidates = [hit for hit in hits if hit.kind == "table"]
    if row_label:
        row_key = _canonical_label(row_label)
        row_hits = [
            hit for hit in candidates
            if row_key in {
                _canonical_label(hit.item.get("indicator")),
                _canonical_label(hit.item.get("row_header")),
            }
        ]
        if row_hits:
            candidates = row_hits
    operands: list[tuple[str, Hit]] = []
    for column in columns:
        column_key = canonical_dimension_label(column)
        matches = [
            hit for hit in candidates
            if column_key in canonical_dimension_label(" ".join(
                str(hit.item.get(key) or "") for key in ["column_header", "context", "period"]
            ))
            and _table_numeric_value(_load_table_value(hit.item.get("value_text"))) is not None
        ]
        if not matches:
            return None
        matches.sort(key=lambda hit: (hit.table_score, hit.lexical_score, hit.fused_score), reverse=True)
        operands.append((column, matches[0]))
    start_label, start_hit = operands[0]
    end_label, end_hit = operands[1]
    start_value = _table_numeric_value(_load_table_value(start_hit.item.get("value_text")))
    end_value = _table_numeric_value(_load_table_value(end_hit.item.get("value_text")))
    if start_value is None or end_value is None:
        return None
    # Remove binary floating-point residue while retaining enough precision
    # for regulatory/statistical values that carry more than two decimals.
    difference = round(end_value - start_value, 10)
    unit = normalize_text(start_hit.item.get("unit") or end_hit.item.get("unit"))
    difference_display = f"{difference:.6f}".rstrip("0").rstrip(".")
    if unit:
        difference_display += unit
    period = normalize_text(start_hit.item.get("period") or end_hit.item.get("period"))
    row_display = row_label or normalize_text(start_hit.item.get("indicator")) or "该行"
    start_display = f"{start_value:.10f}".rstrip("0").rstrip(".")
    end_display = f"{end_value:.10f}".rstrip("0").rstrip(".")
    answer = f"“{row_display}”从“{start_label}”到“{end_label}”的数值变化为：{difference_display}（{end_display} - {start_display}）。"
    operation = {
        "type": "table_calculation",
        "calculation": "difference",
        "formula": f"{end_label} - {start_label}",
        "period": period,
        "row_label": row_display,
        "start_label": start_label,
        "end_label": end_label,
        "start_value": start_value,
        "end_value": end_value,
        "difference": difference,
        "result": difference,
        "unit": unit,
        "operand_evidence_ids": [start_hit.evidence_id, end_hit.evidence_id],
        "display_evidence_ids": [start_hit.evidence_id, end_hit.evidence_id],
    }
    return AnswerDraft(answer, [answer], [operation])


def table_answer(question: str, choices: list[str] | None, hits: list[Hit]) -> AnswerDraft:
    table_hits = [hit for hit in hits if hit.kind == "table"]
    if choices:
        option, confidence = choose_table_option(question, choices, table_hits)
        if option:
            selected = choices["ABCD".index(option)]
            return AnswerDraft(f"选项 {option}：{selected}", [selected], [{"type": "table_lookup", "confidence": confidence}])
    if not table_hits:
        return AnswerDraft("当前证据不足，无法可靠回答。", [], [])
    if _is_table_calculation_question(question):
        calculated = _calculation_answer(question, table_hits)
        if calculated is not None:
            return calculated
    # A table title/period alone is not an indicator. For broad questions,
    # return a useful clarification with the located source instead of picking
    # an arbitrary cell from a multi-indicator report.
    requested_indicator = extract_indicator(question)
    requested_row, requested_column = extract_dimension_labels(question)
    if requested_indicator:
        indicator_hits = [
            hit for hit in table_hits
            if normalize_text(hit.item.get("indicator")).lower() == normalize_text(requested_indicator).lower()
        ]
    elif requested_row:
        indicator_hits = [
            hit for hit in table_hits
            if _canonical_label(requested_row) in {
                _canonical_label(hit.item.get("indicator")),
                _canonical_label(hit.item.get("row_header")),
            }
        ]
    else:
        indicator_hits = [
            hit for hit in table_hits
            if hit.item.get("indicator") and _overlap(question, str(hit.item.get("indicator"))) >= 0.25
        ]
    if not indicator_hits or (not any(term in question for term in TABLE_INDICATOR_TERMS) and not (requested_row or requested_column)):
        top = table_hits[0].item
        period = _question_period(question) or top.get("period") or "未能确定"
        title = top.get("source_title") or top.get("table_name") or "该统计表"
        source = title if _has_source_hint(question) and not _is_benchmark_source(str(title)) else None
        if requested_indicator:
            answer = f"已识别指标“{requested_indicator}”，但当前证据中没有找到对应的数值单元格，无法可靠回答。"
            return AnswerDraft(answer, [answer], [{"type": "refusal", "period": period, "source": source, "reason": "指定指标没有匹配的数值证据"}])
        if source:
            answer = f"已定位到《{source}》，统计期间为 {period}。当前问题未指定具体指标名称，请补充指标后查询具体数值。"
        else:
            answer = f"目前只能识别到统计期间 {period}，但未能唯一确定统计表和具体指标。请补充统计表名称与指标名称后再查询。"
        return AnswerDraft(answer, [answer], [{"type": "clarification", "period": period, "source": source}])
    # A broad table question often retrieves title/header cells first. Never
    # present a header as a numeric answer; ask for the indicator instead.
    numeric_hits = []
    # Restrict numeric selection to cells whose indicator matches the request;
    # otherwise a nearby header or another row can contribute an unrelated number.
    numeric_candidates = indicator_hits or table_hits
    for hit in numeric_candidates:
        value = _load_table_value(hit.item.get("value_text"))
        if _table_numeric_value(value) is not None:
            numeric_hits.append(hit)
    if not numeric_hits:
        top = table_hits[0].item
        period = _question_period(question) or top.get("period") or "未能确定"
        title = top.get("source_title") or top.get("table_name") or "该统计表"
        source = title if _has_source_hint(question) and not _is_benchmark_source(str(title)) else None
        if requested_indicator:
            answer = f"已识别指标“{requested_indicator}”，但当前证据中没有找到对应的数值单元格，无法可靠回答。"
            return AnswerDraft(answer, [answer], [{"type": "refusal", "period": period, "source": source, "reason": "指定指标没有匹配的数值证据"}])
        if source:
            answer = f"已定位到《{source}》，统计期间为 {period}。当前问题未指定具体指标名称，请补充指标后查询具体数值。"
        else:
            answer = f"目前只能识别到统计期间 {period}，但未能唯一确定统计表和具体指标。请补充统计表名称与指标名称后再查询。"
        return AnswerDraft(answer, [answer], [{"type": "clarification", "period": period, "source": source}])
    requested_quarter = _question_quarter(question)
    numeric_hits.sort(
        key=lambda hit: _numeric_match_score(hit, question, requested_indicator, requested_quarter, requested_row, requested_column),
        reverse=True,
    )

    # Some regulatory workbooks store one year in ``period`` and represent the
    # four reporting quarters as repeated row blocks.  A year-only question is
    # therefore a multi-value lookup, not a request for whichever cell BGE
    # happened to rank first.  Keep one exact numeric cell per quarter and
    # expose all four cells to the answer generator and verifier.
    if requested_quarter is None:
        by_quarter: dict[str, Hit] = {}
        for hit in numeric_hits:
            quarter = _quarter_label(hit)
            if not quarter:
                continue
            previous = by_quarter.get(quarter)
            if previous is None or _numeric_match_score(hit, question, requested_indicator, None, requested_row, requested_column) > _numeric_match_score(previous, question, requested_indicator, None, requested_row, requested_column):
                by_quarter[quarter] = hit
        if len(by_quarter) >= 2:
            ordered = [by_quarter[label] for label in ("一季度", "二季度", "三季度", "四季度") if label in by_quarter]
            values: list[dict[str, Any]] = []
            display_parts: list[str] = []
            for hit in ordered:
                item = hit.item
                raw_value = _load_table_value(item.get("value_text"))
                display_value, unit, unit_inferred = _format_table_value(raw_value, item.get("indicator"), item.get("unit"))
                quarter = _quarter_label(hit) or "该季度"
                values.append({
                    "quarter": quarter,
                    "value": display_value,
                    "raw_value": raw_value,
                    "cell": item.get("cell_address"),
                    "evidence_id": hit.evidence_id,
                })
                display_parts.append(f"{quarter}{display_value}")
            year = normalize_text(ordered[0].item.get("period")) or "该年度"
            indicator_text = requested_indicator or ordered[0].item.get("indicator") or "该指标"
            dimension = f"在“{requested_column}”口径下" if requested_column else ""
            answer = f"“{indicator_text}”{dimension}，{year}年各季度数值为：" + "；".join(display_parts) + "。"
            operation: dict[str, Any] = {
                "type": "table_lookup",
                "values": values,
                "display_evidence_ids": [hit.evidence_id for hit in ordered],
                "period": year,
            }
            if requested_row:
                operation["row_label"] = requested_row
            if requested_column:
                operation["column_label"] = requested_column
            return AnswerDraft(answer, [answer], [operation])

    selected_hit = numeric_hits[0]
    top = selected_hit.item
    raw_value = _load_table_value(top.get("value_text"))
    display_value, unit, unit_inferred = _format_table_value(raw_value, top.get("indicator"), top.get("unit"))
    period = top.get("period") or "该期间"
    requested_period = _question_period(question)
    period_text = normalize_text(period)
    # Only annual workbooks use a year-level period plus a quarter column.
    # Monthly reports already carry an exact YYYY-MM period and must not be
    # rewritten as (for example) “2023年四季度”.
    if requested_quarter and requested_period and re.fullmatch(r"20\d{2}", period_text):
        match = re.search(r"(20\d{2})年", normalize_text(question))
        period = f"{match.group(1)}年{requested_quarter}" if match else requested_quarter
    elif requested_period:
        match = re.fullmatch(r"(20\d{2})-(\d{2})", requested_period)
        period = f"{match.group(1)}年{int(match.group(2))}月" if match else requested_period
    if requested_row and requested_column:
        section_scope = top.get("_section_scope") or insurance_company_scope(top.get("context"))
        scope_prefix = f"{section_scope}的" if section_scope else ""
        answer = f"{scope_prefix}“{requested_row}”在“{requested_column}”口径下、{period}的数值为 {display_value}。"
    else:
        answer = f"{top.get('indicator') or '该指标'}在{period}的值为 {display_value}。"
    operation = {
        "type": "table_lookup",
        "value": display_value,
        "raw_value": raw_value,
        "unit": unit,
        "cell": top.get("cell_address"),
    }
    if unit_inferred:
        operation["unit_source"] = "indicator_semantics"
    if requested_row:
        operation["row_label"] = requested_row
    if requested_column:
        operation["column_label"] = requested_column
    section_scope = top.get("_section_scope") or insurance_company_scope(top.get("context"))
    if section_scope:
        operation["section_scope"] = section_scope
    return AnswerDraft(answer, [answer], [operation])


def _canonical_label(value: Any) -> str:
    return canonical_table_label(value)


def choose_table_option(question: str, choices: list[str], hits: list[Hit]) -> tuple[str | None, float]:
    """Choose the option whose exact value is in the most query-relevant cell."""
    scores: list[tuple[int, float]] = []
    for index, choice in enumerate(choices[:4]):
        target = normalized_number(choice)
        best = 0.0
        for hit in hits:
            value = hit.item.get("value_text")
            try:
                value = json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError:
                pass
            parsed = _table_numeric_value(value)
            if target is not None and parsed is not None and abs(target - parsed) < max(1e-9, abs(target) * 1e-8):
                context = _hit_text(hit)
                best = max(best, 1.0 + min(1.0, _overlap(question, context)) + 0.05 * hit.lexical_score + 0.02 * hit.dense_score)
            elif target is None and normalize_text(choice) in _hit_text(hit):
                best = max(best, 1.0 + _overlap(question, _hit_text(hit)))
        scores.append((index, best))
    scores.sort(key=lambda item: item[1], reverse=True)
    if not scores or scores[0][1] <= 0:
        return None, 0.0
    margin = scores[0][1] - (scores[1][1] if len(scores) > 1 else 0)
    return "ABCD"[scores[0][0]], min(1.0, 0.65 + max(0.0, margin) / 2)


def text_answer(question: str, choices: list[str] | None, hits: list[Hit]) -> AnswerDraft:
    if _is_unbounded_latest_question(question):
        answer = "无法可靠回答“监管部门最新规定”：当前问题未指定监管部门、规定主题、文件名称或有效时间。为避免引用过时或不适用的规则，系统拒绝直接给出结论。"
        return AnswerDraft(answer, [], [{"type": "refusal", "source": None, "reason": "最新规定缺少明确范围和有效时间"}])
    if _is_absolute_prediction_question(question):
        answer = "不能根据当前问题断定某银行明年一定不会发生风险。请补充银行名称、风险类型、评估期间及相关监管指标；在缺少这些信息时，系统不会给出确定性预测。"
        return AnswerDraft(answer, [], [{"type": "clarification", "source": None, "reason": "缺少风险主体、风险类型和评估指标"}])
    if choices:
        option, confidence, _ = choose_option(question, choices, hits)
        if option:
            selected = choices["ABCD".index(option)]
            return AnswerDraft(f"选项 {option}：{selected}", [selected], [{"type": "evidence_choice", "confidence": confidence}])
    if not hits:
        return AnswerDraft("当前证据不足，无法可靠回答。", [], [])
    top = hits[0].item
    content = top.get("content") or top.get("context") or ""
    return AnswerDraft(content, [content], [])


def _hit_text(hit: Hit) -> str:
    item = hit.item
    return " ".join(str(item.get(k) or "") for k in ["content", "context", "_section_scope", "indicator", "period", "value_text", "row_header", "column_header"])


def reason(question: str, qa_type: str, choices: list[str] | None, hits: list[Hit]) -> AnswerDraft:
    if qa_type == "cross_file_judgment":
        return cross_file_answer(question, hits)
    if qa_type == "table_lookup":
        return table_answer(question, choices, hits)
    return text_answer(question, choices, hits)
