from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .query_plan import RetrievalCandidate, RetrievalResult, RetrievalTask
from .retrieval.index import Hit
from .utils import (
    canonical_dimension_label,
    canonical_table_label,
    insurance_company_scope,
    normalize_text,
)


@dataclass(frozen=True)
class RetrievalExecution:
    result: RetrievalResult
    hits: list[Hit]


class RetrievalTools:
    """Thin task-oriented facade over the existing HybridIndex."""

    def __init__(self, index: Any, top_k: int = 12):
        self.index = index
        self.top_k = max(4, int(top_k))

    def execute(self, task: RetrievalTask) -> RetrievalExecution:
        task = _normalized_task(task)
        query = _task_query(task)
        filters = _task_filters(task)
        expects_table = task.expected_value_type in {"number", "table_cell"} or any([
            task.semantic_constraints.indicator,
            task.semantic_constraints.parent_indicator,
            task.semantic_constraints.row_label,
            task.semantic_constraints.column_label,
        ])
        if expects_table:
            # The SQL-backed index returns a bounded shortlist, but wide
            # workbooks can place the numeric cell below many label/note
            # cells. Keep a larger wrapper-level candidate window, then apply
            # the task's explicit semantic constraints and numeric validation.
            hits = self.index.search_tables(query, max(self.top_k, 128), filters or None)
            hits = [hit for hit in hits if _matches_semantic_constraints(hit, task, self.index)]
            return _table_result(task, hits, self.index)
        hits = self.index.search_text(query, self.top_k, filters or None)
        hits = [hit for hit in hits if _matches_semantic_constraints(hit, task, self.index)]
        hits = _expand_structural_text_neighbours(hits, self.index)
        return _text_result(task, hits, self.index)


def _normalized_task(task: RetrievalTask) -> RetrievalTask:
    indicator = task.semantic_constraints.indicator
    canonical = _canonical_indicator_label(indicator)
    statistical_scope = task.semantic_constraints.statistical_scope
    indicator_label = canonical_table_label(indicator)
    if not statistical_scope and (
        indicator_label.startswith("保险业") or indicator_label.startswith("保险行业")
    ):
        statistical_scope = "保险业"
    if not canonical or (
        canonical == indicator and statistical_scope == task.semantic_constraints.statistical_scope
    ):
        return task
    constraints = task.semantic_constraints.model_copy(update={
        "indicator": canonical,
        "statistical_scope": statistical_scope,
    })
    return task.model_copy(update={"semantic_constraints": constraints})


def _canonical_indicator_label(value: Any) -> str:
    label = canonical_table_label(value)
    if label.endswith("总资产") or label in {"资产", "资产总额", "资产规模"}:
        return "总资产"
    return label


def _task_query(task: RetrievalTask) -> str:
    source = task.source_scope
    semantic = task.semantic_constraints
    anchors = [
        task.query,
        task.expected_information,
        source.document_title,
        f"{source.year}年" if source.year else None,
        f"{source.month}月" if source.month else None,
        f"第{source.quarter}季度" if source.quarter else None,
        semantic.indicator,
        semantic.parent_indicator,
        semantic.institution,
        semantic.region,
        semantic.period,
        semantic.statistical_scope,
        semantic.row_label,
        semantic.column_label,
    ]
    return " ".join(str(value) for value in anchors if value)


def _task_filters(task: RetrievalTask) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if task.source_scope.document_title:
        filters["title"] = [task.source_scope.document_title]
    if task.source_scope.document_type:
        filters["document_type"] = [task.source_scope.document_type]
    return filters


def _matches_semantic_constraints(hit: Hit, task: RetrievalTask, index: Any) -> bool:
    item = hit.item
    constraints = task.semantic_constraints
    source = _document_for_hit(hit, index)
    blob = normalize_text(" ".join(str(value or "") for value in [
        item.get("indicator"), item.get("row_header"), item.get("column_header"), item.get("period"),
        item.get("context"), item.get("content"), item.get("_section_scope"),
        source.get("title"), source.get("file_name"),
    ]))

    if constraints.indicator:
        requested = _canonical_indicator_label(constraints.indicator)
        labels = {
            _canonical_indicator_label(item.get("indicator")),
            _canonical_indicator_label(item.get("row_header")),
        }
        if requested not in labels and requested not in _canonical_indicator_label(blob):
            return False
    if constraints.parent_indicator and normalize_text(constraints.parent_indicator) not in blob:
        return False
    if constraints.institution and not _institution_matches(constraints.institution, blob):
        return False
    if constraints.region and not _scope_matches(constraints.region, blob):
        return False
    if constraints.statistical_scope and not _scope_matches(constraints.statistical_scope, blob):
        return False
    if constraints.row_label:
        row = canonical_table_label(constraints.row_label)
        if row not in {
            canonical_table_label(item.get("indicator")),
            canonical_table_label(item.get("row_header")),
        } and row not in canonical_table_label(blob):
            return False
    if constraints.column_label:
        column = canonical_dimension_label(constraints.column_label)
        # Do not mix row/context text into a column comparison.  A row named
        # ``全国合计`` otherwise makes every component column look like the
        # requested ``合计`` column.  Fall back to period/context only for
        # legacy records that genuinely have no column header.
        column_header = normalize_text(item.get("column_header"))
        column_blob = canonical_dimension_label(
            column_header
            or " ".join(str(item.get(key) or "") for key in ("period", "context"))
        )
        if column not in column_blob:
            return False
    requested_period = constraints.period or _scope_period(task)
    if requested_period and not _period_matches(requested_period, blob):
        return False
    if task.expected_unit and item.get("unit"):
        if normalize_text(task.expected_unit) != normalize_text(item.get("unit")):
            return False
    return True


def _institution_matches(requested: str, blob: str) -> bool:
    normalized = normalize_text(requested)
    if normalized in blob:
        return True
    requested_scope = insurance_company_scope(normalized)
    candidate_scope = insurance_company_scope(blob)
    return bool(requested_scope and candidate_scope and requested_scope == candidate_scope)


def _scope_matches(requested: str, blob: str) -> bool:
    normalized = normalize_text(requested)
    if normalized in blob:
        return True
    if canonical_table_label(normalized) in {"保险业", "保险行业"}:
        return any(value in blob for value in ("保险业", "保险行业"))
    if canonical_table_label(normalized) not in {"全国", "全国合计", "全国总体"}:
        return False
    if any(value in blob for value in ("全国", "保险业总体", "全保险业", "全部保险公司")):
        return True
    # In the corpus, an unqualified “保险业经营情况表” represents the
    # overall-industry source. Explicit life/property-company workbooks are
    # excluded so a national task cannot leak into a sub-industry value.
    return (
        insurance_company_scope(blob) is None
        and bool(re.search(r"保险业(?:经营|发展|统计).*表", blob))
    )


def _document_for_hit(hit: Hit, index: Any) -> dict[str, Any]:
    return dict(getattr(index, "doc_by_id", {}).get(str(hit.item.get("doc_id") or ""), {}) or {})


def _scope_period(task: RetrievalTask) -> str | None:
    scope = task.source_scope
    if scope.year and scope.month:
        return f"{scope.year:04d}-{scope.month:02d}"
    if scope.year and scope.quarter:
        return f"{scope.year:04d}-Q{scope.quarter}"
    if scope.year:
        return str(scope.year)
    return None


def _period_matches(requested: str, blob: str) -> bool:
    value = normalize_text(requested)
    variants = {value}
    month = re.fullmatch(r"(20\d{2})[-年/]0?(\d{1,2})(?:月)?", value)
    if month:
        year, number = month.groups()
        variants.update({f"{year}-{int(number):02d}", f"{year}年{int(number)}月"})
    quarter = re.fullmatch(r"(20\d{2})-?Q([1-4])", value, re.IGNORECASE)
    if quarter:
        year, number = quarter.groups()
        chinese = ("一", "二", "三", "四")[int(number) - 1]
        variants.update({f"{year}-Q{number}", f"{year}年第{chinese}季度", f"{year}年{chinese}季度"})
    return any(variant in blob for variant in variants)


def _table_result(task: RetrievalTask, hits: list[Hit], index: Any) -> RetrievalExecution:
    candidates: list[RetrievalCandidate] = []
    candidate_hits: list[Hit] = []
    for hit in hits:
        value = _decoded_value(hit.item.get("value_text", hit.item.get("value")))
        if task.expected_value_type in {"number", "table_cell"} and not _is_numeric_value(value):
            continue
        document = dict(getattr(index, "doc_by_id", {}).get(str(hit.item.get("doc_id") or ""), {}) or {})
        candidate = RetrievalCandidate(
            value=None if value is None else str(value),
            unit=normalize_text(hit.item.get("unit")) or task.expected_unit,
            evidence_ids=[hit.evidence_id],
            document_id=str(hit.item.get("doc_id") or "") or None,
            document_title=normalize_text(document.get("title") or hit.item.get("source_title")) or None,
            document_type=normalize_text(document.get("document_type")) or None,
            sheet_name=normalize_text(hit.item.get("sheet_name")) or None,
            cell_address=normalize_text(hit.item.get("cell_address")) or None,
            indicator=normalize_text(hit.item.get("indicator")) or None,
            row_label=normalize_text(hit.item.get("row_header")) or None,
            column_label=normalize_text(hit.item.get("column_header")) or None,
            period=normalize_text(hit.item.get("period")) or None,
            content=normalize_text(hit.item.get("context")) or None,
            score=float(max(hit.table_score, hit.fused_score, hit.rerank_score)),
        )
        candidates.append(candidate)
        candidate_hits.append(hit)
    candidates, candidate_hits = _deduplicate_candidates(candidates, candidate_hits)
    if not candidates:
        return RetrievalExecution(RetrievalResult(
            task_id=task.id,
            status="not_found",
            expected_information=task.expected_information,
        ), [])

    ambiguity = _ambiguity_reason(task, candidates)
    if ambiguity:
        return RetrievalExecution(RetrievalResult(
            task_id=task.id,
            status="ambiguous",
            expected_information=task.expected_information,
            candidates=candidates,
            evidence_ids=[item for candidate in candidates for item in candidate.evidence_ids],
            ambiguity_reason=ambiguity,
        ), candidate_hits)
    selected = candidates[0]
    return RetrievalExecution(RetrievalResult(
        task_id=task.id,
        status="resolved",
        expected_information=task.expected_information,
        selected=selected,
        candidates=candidates,
        evidence_ids=selected.evidence_ids,
    ), candidate_hits)


def _text_result(task: RetrievalTask, hits: list[Hit], index: Any) -> RetrievalExecution:
    candidates: list[RetrievalCandidate] = []
    for hit in hits:
        document = dict(getattr(index, "doc_by_id", {}).get(str(hit.item.get("doc_id") or ""), {}) or {})
        content = normalize_text(hit.item.get("content") or hit.item.get("context_window") or hit.item.get("context"))
        if not content:
            continue
        candidates.append(RetrievalCandidate(
            value=content,
            evidence_ids=[hit.evidence_id],
            document_id=str(hit.item.get("doc_id") or "") or None,
            document_title=normalize_text(document.get("title") or hit.item.get("source_title")) or None,
            document_type=normalize_text(document.get("document_type")) or None,
            content=content,
            score=float(max(hit.fused_score, hit.rerank_score, hit.lexical_score)),
        ))
    if not candidates:
        return RetrievalExecution(RetrievalResult(
            task_id=task.id,
            status="not_found",
            expected_information=task.expected_information,
        ), [])
    selected = candidates[0]
    evidence_ids = list(dict.fromkeys(item for candidate in candidates[:4] for item in candidate.evidence_ids))
    return RetrievalExecution(RetrievalResult(
        task_id=task.id,
        status="resolved",
        expected_information=task.expected_information,
        selected=selected,
        candidates=candidates[:4],
        evidence_ids=evidence_ids,
    ), hits[:4])


_STRUCTURAL_LEAD_RE = re.compile(
    r"(?:至少包括|包括如下|具体如下|下列|情景设置|主要情景指标|"
    r"^[一二三四五六七八九十]+、|^[（(][一二三四五六七八九十]+[）)])"
)


def _expand_structural_text_neighbours(
    hits: list[Hit],
    index: Any,
    *,
    following: int = 2,
) -> list[Hit]:
    """Add bounded adjacent paragraphs after headings/lead-in chunks.

    Word/PDF parsing often stores a subsection heading, its lead sentence and
    the actual enumerated list as separate paragraph records.  Expansion is
    applied after ranking, so an adjacent list cannot be discarded merely
    because the short lead-in had the stronger lexical score.  Each appended
    paragraph keeps its own evidence ID and source location.
    """
    text_rows = list(getattr(index, "text", []) or [])
    if not hits or not text_rows:
        return hits
    by_position = {
        (str(item.get("doc_id") or ""), int(item["paragraph_no"])): item
        for item in text_rows
        if item.get("paragraph_no") is not None
    }
    expanded: list[Hit] = []
    seen: set[str] = set()

    def append(candidate: Hit) -> None:
        if candidate.evidence_id not in seen:
            expanded.append(candidate)
            seen.add(candidate.evidence_id)

    for hit in hits:
        append(hit)
        content = normalize_text(hit.item.get("content"))
        paragraph = hit.item.get("paragraph_no")
        doc_id = str(hit.item.get("doc_id") or "")
        if (
            not content
            or paragraph is None
            or not doc_id
            or not _STRUCTURAL_LEAD_RE.search(content)
        ):
            continue
        center = int(paragraph)
        for offset in range(1, following + 1):
            item = by_position.get((doc_id, center + offset))
            if item is None:
                continue
            append(Hit(
                "text",
                item,
                lexical_score=hit.lexical_score,
                dense_score=hit.dense_score,
                metadata_score=hit.metadata_score,
                fused_score=hit.fused_score,
                rerank_score=hit.rerank_score,
            ))
    return expanded


def _decoded_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip('"')


def _is_numeric_value(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    return bool(re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?\s*[%％]?", normalize_text(value)))


def _deduplicate_candidates(
    candidates: list[RetrievalCandidate],
    hits: list[Hit],
) -> tuple[list[RetrievalCandidate], list[Hit]]:
    selected_candidates: list[RetrievalCandidate] = []
    selected_hits: list[Hit] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate, hit in zip(candidates, hits):
        key = (
            candidate.document_id, candidate.sheet_name, candidate.cell_address,
            candidate.value, candidate.unit,
        )
        if key in seen:
            continue
        seen.add(key)
        selected_candidates.append(candidate)
        selected_hits.append(hit)
    return selected_candidates, selected_hits


def _ambiguity_reason(task: RetrievalTask, candidates: list[RetrievalCandidate]) -> str | None:
    values = {(item.value, item.unit) for item in candidates}
    if len(values) <= 1:
        return None
    periods = {normalize_text(item.period) for item in candidates if item.period}
    columns = {normalize_text(item.column_label) for item in candidates if item.column_label}
    rows = {normalize_text(item.row_label) for item in candidates if item.row_label}
    source = task.source_scope
    semantic = task.semantic_constraints
    quarter_specified = bool(source.quarter or re.search(r"Q[1-4]|[一二三四1-4]季度", semantic.period or "", re.IGNORECASE))
    if not quarter_specified and any(re.search(r"Q[1-4]|[一二三四1-4]季度", value, re.IGNORECASE) for value in periods | columns | rows):
        return "检索到多个季度候选，但问题未指定季度"
    if not semantic.column_label and len(columns) > 1:
        return "检索到多个统计口径候选，但问题未指定口径"
    if not source.document_title and len({item.document_id for item in candidates}) > 1:
        return "检索到多个来源中的不同候选值，无法唯一确定来源"
    return "检索到多个不同数值候选，缺少可唯一定位的维度"
