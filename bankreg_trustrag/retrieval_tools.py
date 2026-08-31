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
        search_type = "table_lookup" if expects_table else "regulatory_fact"
        if expects_table:
            source = task.source_scope
            semantic = task.semantic_constraints

            structured = {
                "indicator": semantic.indicator,
                "parent_indicator": semantic.parent_indicator,
                "institution": semantic.institution,
                "region": semantic.region,
                "period": semantic.period,
                "statistical_scope": semantic.statistical_scope,
                "row_label": semantic.row_label,
                "column_label": semantic.column_label,
                "year": source.year,
                "month": source.month,
                "quarter": source.quarter,
            }

            # 删除空值，防止空字段干扰检索。
            structured = {
                key: value
                for key, value in structured.items()
                if value not in (None, "", [])
            }

            print("\n" + "=" * 100)
            print("[RETRIEVAL STRUCTURED DEBUG]")
            print("structured =", structured)
            print("=" * 100)

            hits = self.index.search_tables(
                query,
                max(self.top_k, 128),
                filters or None,
                dense=True,
                lexical=True,
                metadata=True,
                fuse=True,
                structured=structured,
            )

            hits = [
                hit
                for hit in hits
                if _matches_semantic_constraints(hit, task, self.index)
            ]

            return _table_result(task, hits, self.index)
        search = getattr(self.index, "hybrid_search", None)
        text_k = max(self.top_k, 32)
        raw_hits = (
            search(query, search_type, text_k, filters or None)
            if callable(search)
            else self.index.search_text(query, text_k, filters or None)
        )

        # The user may name an attachment/sub-document while the manifest stores
        # a longer parent title. Retry without a hard title filter and keep only
        # fuzzy title matches instead of falsely declaring not_found.
        if not raw_hits and task.source_scope.document_title:
            raw_hits = (
                search(query, search_type, text_k, None)
                if callable(search)
                else self.index.search_text(query, text_k, None)
            )
            raw_hits = [
                hit for hit in raw_hits
                if _source_title_matches(hit, task.source_scope.document_title, self.index)
            ]

        hits = [hit for hit in raw_hits if _matches_semantic_constraints(hit, task, self.index)]
        # Text constraints produced by the planner are hints, not proof. If they
        # erase every source-correct hit, relax them and let the support gate
        # below decide whether the paragraph truly answers the task.
        if not hits and raw_hits and task.expected_value_type in {"text", "string", "boolean"}:
            hits = [
                hit for hit in raw_hits
                if not task.source_scope.document_title
                or _source_title_matches(hit, task.source_scope.document_title, self.index)
            ]
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

    if not label:
        return label

    # 清理 Excel 指标名称尾部脚注
    label = re.sub(r"[*＊※]+$", "", label)
    label = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]+$", "", label)
    label = re.sub(r"(?:注|备注)\s*[0-9一二三四五六七八九十]*$", "", label)
    label = label.strip()

    if label.endswith("总资产") or label in {
        "资产",
        "资产总额",
        "资产规模",
    }:
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
        item.get("indicator"),
        item.get("row_header"),
        item.get("column_header"),
        item.get("period"),

        # structured exact 已经推断出的标准期间
        item.get("_inferred_period"),

        item.get("context"),
        item.get("content"),
        item.get("_section_scope"),
        source.get("title"),
        source.get("file_name"),
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
        row = _canonical_indicator_label(
            constraints.row_label
        )

        candidate_rows = {
            _canonical_indicator_label(
                item.get("indicator")
            ),
            _canonical_indicator_label(
                item.get("row_header")
            ),
        }

        if (
                row not in candidate_rows
                and row not in _canonical_indicator_label(blob)
        ):
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
    requested_period = _scope_period(task) or constraints.period
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
    text = normalize_text(blob)

    if not value:
        return True

    # 直接存在，最快路径
    if value in text:
        return True

    # ============================================================
    # 月度：2024-03 / 2024年3月 / 2024/03
    # ============================================================
    month = re.fullmatch(
        r"(20\d{2})(?:[-年/])0?(\d{1,2})(?:月)?",
        value,
    )

    if month:
        year = month.group(1)
        month_number = int(month.group(2))

        variants = {
            f"{year}-{month_number:02d}",
            f"{year}年{month_number}月",
            f"{year}/{month_number:02d}",
        }

        if any(item in text for item in variants):
            return True

        # Excel 中年份和月份可能分别位于 period / column_header
        return (
            year in text
            and f"{month_number}月" in text
        )

    # ============================================================
    # 季度：2024-Q1
    # ============================================================
    quarter = re.fullmatch(
        r"(20\d{2})-?Q([1-4])",
        value,
        re.IGNORECASE,
    )

    if quarter:
        year = quarter.group(1)
        q = int(quarter.group(2))
        chinese = ("一", "二", "三", "四")[q - 1]

        variants = {
            f"{year}-Q{q}",
            f"{year}年{chinese}季度",
            f"{year}年第{chinese}季度",
            f"{year}年第{q}季度",
        }

        if any(item in text for item in variants):
            return True

        # 支持：
        # period = 2024
        # column_header = 一季度
        return (
            year in text
            and any(term in text for term in {
                f"Q{q}",
                f"{chinese}季度",
                f"第{chinese}季度",
                f"{q}季度",
                f"第{q}季度",
            })
        )

    # ============================================================
    # 中文季度：2024年一季度 / 2024年第一季度 / 2024年第1季度
    # ============================================================
    chinese_quarter = re.fullmatch(
        r"(20\d{2})年(?:第)?([一二三四1234])季度",
        value,
    )

    if chinese_quarter:
        year = chinese_quarter.group(1)
        raw_q = chinese_quarter.group(2)

        mapping = {
            "一": 1,
            "二": 2,
            "三": 3,
            "四": 4,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
        }

        q = mapping[raw_q]
        chinese = ("一", "二", "三", "四")[q - 1]

        return (
            year in text
            and any(term in text for term in {
                f"{year}-Q{q}",
                f"{chinese}季度",
                f"第{chinese}季度",
                f"{q}季度",
                f"第{q}季度",
            })
        )

    # 单独年份
    if re.fullmatch(r"20\d{2}", value):
        return value in text

    return value in text


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
    """Select evidence by task support instead of ``candidates[0]``."""
    ranked: list[tuple[float, float, RetrievalCandidate, Hit]] = []
    for hit in hits:
        document = dict(getattr(index, "doc_by_id", {}).get(str(hit.item.get("doc_id") or ""), {}) or {})
        content = normalize_text(hit.item.get("content") or hit.item.get("context_window") or hit.item.get("context"))
        if not content:
            continue
        support = _task_text_support(task, hit, content)
        retrieval_strength = _retrieval_strength(hit)
        candidate = RetrievalCandidate(
            value=content,
            evidence_ids=[hit.evidence_id],
            document_id=str(hit.item.get("doc_id") or "") or None,
            document_title=normalize_text(document.get("title") or hit.item.get("source_title")) or None,
            document_type=normalize_text(document.get("document_type")) or None,
            content=content,
            score=round(support * 100.0, 6),
        )
        ranked.append((support, retrieval_strength, candidate, hit))

    if not ranked:
        return RetrievalExecution(RetrievalResult(
            task_id=task.id, status="not_found", expected_information=task.expected_information,
        ), [])

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_support = ranked[0][0]
    keep_floor = max(0.22, best_support - 0.22)
    kept = [item for item in ranked if item[0] >= keep_floor][:6]
    candidates = [item[2] for item in kept]
    candidate_hits = [item[3] for item in kept]

    if best_support < 0.36:
        return RetrievalExecution(RetrievalResult(
            task_id=task.id,
            status="not_found",
            expected_information=task.expected_information,
            candidates=[item[2] for item in ranked[:4]],
        ), [])

    selected = candidates[0]
    evidence_ids = list(dict.fromkeys(
        evidence_id for candidate in candidates[:4] for evidence_id in candidate.evidence_ids
    ))
    return RetrievalExecution(RetrievalResult(
        task_id=task.id,
        status="resolved",
        expected_information=task.expected_information,
        selected=selected,
        candidates=candidates,
        evidence_ids=evidence_ids,
    ), candidate_hits)


_CHOICE_MARKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-H])\s*[.．、)]\s*", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")


def _task_text_support(task: RetrievalTask, hit: Hit, content: str) -> float:
    claims = _inline_choice_claims(task.query)
    if not claims:
        claims = [task.expected_information, task.query]
    claim_support = max((_claim_text_support(claim, content) for claim in claims if normalize_text(claim)), default=0.0)
    retrieval = _retrieval_strength(hit)
    return max(0.0, min(1.0, 0.82 * claim_support + 0.18 * retrieval))


def _inline_choice_claims(text: str) -> list[str]:
    normalized = normalize_text(text)
    markers = list(_CHOICE_MARKER_RE.finditer(normalized))
    if len(markers) < 2:
        return []
    claims: list[str] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(normalized)
        value = normalized[marker.end():end].strip(" ；;。")
        if value:
            claims.append(value)
    return claims


def _claim_text_support(claim: str, content: str) -> float:
    claim = normalize_text(claim)
    content = normalize_text(content)
    if not claim or not content:
        return 0.0
    if claim in content:
        return 1.0
    claim_compact = _compact_match_text(claim)
    content_compact = _compact_match_text(content)
    if claim_compact and claim_compact in content_compact:
        return 0.98

    claim_grams = _char_ngrams(claim_compact, 3)
    content_grams = _char_ngrams(content_compact, 3)
    gram_coverage = len(claim_grams & content_grams) / max(len(claim_grams), 1) if claim_grams else 0.0
    claim_numbers = set(_NUMBER_RE.findall(claim))
    content_numbers = set(_NUMBER_RE.findall(content))
    number_coverage = len(claim_numbers & content_numbers) / len(claim_numbers) if claim_numbers else 0.0
    relation_match = 1.0 if _relation_signatures(claim) & _relation_signatures(content) else 0.0
    quoted = [normalize_text(value) for value in re.findall(r"《([^》]{2,120})》", claim)]
    quoted_match = 1.0 if quoted and any(value in content for value in quoted) else 0.0

    score = 0.70 * gram_coverage
    if claim_numbers:
        score += 0.15 * number_coverage
    if relation_match:
        score += 0.15
    if quoted_match:
        score += 0.10
    if claim_numbers and number_coverage == 0.0:
        score *= 0.65
    return max(0.0, min(1.0, score))


def _compact_match_text(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff%％<>≤≥=]", "", normalize_text(value))


def _char_ngrams(value: str, n: int) -> set[str]:
    if not value:
        return set()
    if len(value) <= n:
        return {value}
    return {value[index:index + n] for index in range(len(value) - n + 1)}


def _relation_signatures(value: str) -> set[str]:
    text = normalize_text(value)
    signatures: set[str] = set()
    for number in _NUMBER_RE.findall(text):
        escaped = re.escape(number)
        if re.search(rf"(?:>\s*{escaped}|{escaped}\s*年?(?:以后|以上|之外))", text):
            signatures.add(f"gt:{number}")
        if re.search(rf"(?:>=\s*{escaped}|≥\s*{escaped}|{escaped}\s*年?(?:以上|及以上))", text):
            signatures.add(f"ge:{number}")
        if re.search(rf"(?:<\s*{escaped}|{escaped}\s*年?(?:以前|以下|以内))", text):
            signatures.add(f"lt:{number}")
        if re.search(rf"(?:<=\s*{escaped}|≤\s*{escaped}|{escaped}\s*年?(?:以下|及以下|以内))", text):
            signatures.add(f"le:{number}")
    return signatures


def _retrieval_strength(hit: Hit) -> float:
    raw = max(
        float(getattr(hit, "rerank_score", 0.0) or 0.0),
        float(getattr(hit, "dense_score", 0.0) or 0.0),
        float(getattr(hit, "fused_score", 0.0) or 0.0),
    )
    if raw > 1.0:
        raw = raw / 100.0
    return max(0.0, min(1.0, raw))


def _source_title_matches(hit: Hit, requested: str, index: Any) -> bool:
    requested_key = canonical_table_label(requested)
    if not requested_key:
        return True
    document = _document_for_hit(hit, index)
    candidate = canonical_table_label(" ".join(str(value or "") for value in [
        document.get("title"), document.get("file_name"),
        hit.item.get("source_title"), hit.item.get("source_file_name"),
    ]))
    if not candidate:
        return False
    if requested_key in candidate or candidate in requested_key:
        return True
    request_grams = _char_ngrams(requested_key, 3)
    candidate_grams = _char_ngrams(candidate, 3)
    coverage = len(request_grams & candidate_grams) / max(len(request_grams), 1)
    return coverage >= 0.72


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
