from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
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
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RetrievalTools:
    """Thin task-oriented facade over the existing HybridIndex."""

    def __init__(self, index: Any, top_k: int = 12):
        self.index = index
        self.top_k = max(4, int(top_k))
        # Built lazily from the already-ingested corpus.  It avoids repeatedly
        # scanning every paragraph when several files share the same logical
        # title (for example one regulation plus dozens of attachments).
        self._document_route_profile_cache: dict[str, list[str]] | None = None

    def execute(self, task: RetrievalTask) -> RetrievalExecution:
        """Execute one retrieval action.

        ``precise`` preserves the planner constraints. ``broad`` keeps those
        terms in the semantic query but does not let uncertain metadata or
        structured fields eliminate every candidate. The adaptive agent can
        therefore react to a failed precise search by explicitly choosing a
        broader next action instead of receiving an immediate refusal.
        """
        task = _normalized_task(task)
        query = _task_query(task)
        precise = getattr(task, "search_mode", "precise") != "broad"

        # Source resolution is intentionally independent from semantic
        # precision.  A broad retry may relax row/indicator constraints, but it
        # should not forget that the user asked about a specific regulation or
        # workbook.  When many physical files belong to one logical title, the
        # resolver routes the task to the most relevant concrete files.
        source_filters, source_diagnostics = self._resolve_source_filters(task)
        filters = dict(source_filters)
        if precise and task.source_scope.document_type:
            filters["document_type"] = [task.source_scope.document_type]

        retrieval_mode = _task_retrieval_mode(task, source_diagnostics)

        if retrieval_mode == "table":
            raw_hits = _search_table_only(
                self.index,
                query,
                max(self.top_k, 128),
                filters or None,
                task=task,
            )
            hits = list(raw_hits)
            if precise:
                hits = [hit for hit in hits if _matches_semantic_constraints(hit, task, self.index)]
            hits = [hit for hit in hits if not _hit_matches_excluded_row(hit, task)]
            execution = _table_result(task, hits, self.index)
            return RetrievalExecution(
                execution.result,
                execution.hits,
                {
                    "search_mode": "precise" if precise else "broad",
                    "search_type": "table_lookup",
                    "retrieval_mode": retrieval_mode,
                    "raw_hit_count": len(raw_hits),
                    "post_constraint_hit_count": len(hits),
                    "filters": filters,
                    "selection": getattr(task, "selection", "single"),
                    "exclude_row_labels": list(getattr(task, "exclude_row_labels", []) or []),
                    "source_resolution": source_diagnostics,
                },
            )

        if retrieval_mode == "text":
            raw_hits = _search_text_only(
                self.index,
                query,
                max(self.top_k, 24),
                filters or None,
            )
            hits = list(raw_hits)
            if precise:
                hits = [hit for hit in hits if _matches_semantic_constraints(hit, task, self.index)]
            hits = _expand_structural_text_neighbours(hits, self.index)
            execution = _text_result(task, hits, self.index)
            return RetrievalExecution(
                execution.result,
                execution.hits,
                {
                    "search_mode": "precise" if precise else "broad",
                    "search_type": "regulatory_fact",
                    "retrieval_mode": retrieval_mode,
                    "raw_hit_count": len(raw_hits),
                    "post_constraint_hit_count": len(hits),
                    "filters": filters,
                    "source_resolution": source_diagnostics,
                },
            )

        # Unknown/mixed source modality: search both arms and let each parser
        # handle its native evidence type. A numeric answer is not proof that
        # the source is a table; Word/PDF paragraphs often contain percentages,
        # amounts and dates.
        text_raw = _search_text_only(
            self.index,
            query,
            max(self.top_k, 24),
            filters or None,
        )
        table_raw = _search_table_only(
            self.index,
            query,
            max(self.top_k, 64),
            filters or None,
            task=task,
        )

        text_hits = list(text_raw)
        table_hits = list(table_raw)
        if precise:
            text_hits = [
                hit for hit in text_hits
                if _matches_semantic_constraints(hit, task, self.index)
            ]
            table_hits = [
                hit for hit in table_hits
                if _matches_semantic_constraints(hit, task, self.index)
            ]
        table_hits = [
            hit for hit in table_hits
            if not _hit_matches_excluded_row(hit, task)
        ]
        text_hits = _expand_structural_text_neighbours(text_hits, self.index)

        text_execution = _text_result(task, text_hits, self.index)
        table_execution = _table_result(task, table_hits, self.index)
        execution = _choose_mixed_execution(
            task,
            text_execution,
            table_execution,
        )
        return RetrievalExecution(
            execution.result,
            execution.hits,
            {
                "search_mode": "precise" if precise else "broad",
                "search_type": "mixed_text_table",
                "retrieval_mode": retrieval_mode,
                "raw_hit_count": len(text_raw) + len(table_raw),
                "text_hit_count": len(text_hits),
                "table_hit_count": len(table_hits),
                "filters": filters,
                "source_resolution": source_diagnostics,
            },
        )


    def _resolve_source_filters(
        self,
        task: RetrievalTask,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        requested = normalize_text(task.source_scope.document_title)
        if not requested:
            return {}, {
                "mode": "unscoped",
                "requested_title": None,
                "family_candidate_count": 0,
                "selected_documents": [],
            }

        documents = _index_documents(self.index)
        family = [
            document
            for document in documents
            if _document_belongs_to_requested_family(document, requested)
        ]
        if not family:
            # Preserve the legacy title constraint when corpus metadata cannot
            # resolve the title.  The index may have its own alias handling.
            return {"title": [requested]}, {
                "mode": "title_fallback",
                "requested_title": requested,
                "family_candidate_count": 0,
                "selected_documents": [],
            }

        query_text = _document_routing_query(task)
        selected = self._rank_document_family(family, requested, query_text)

        file_names = [
            normalize_text(document.get("file_name"))
            for document in selected
            if normalize_text(document.get("file_name"))
        ]
        doc_ids = [
            str(document.get("doc_id") or document.get("id") or "")
            for document in selected
            if str(document.get("doc_id") or document.get("id") or "")
        ]

        if file_names:
            filters = {"file_name": list(dict.fromkeys(file_names))}
            mode = "document_family_file_route"
        else:
            # Some fixtures only expose titles. Keep every selected title
            # instead of collapsing duplicate logical titles to one document.
            titles = [
                normalize_text(document.get("title"))
                for document in selected
                if normalize_text(document.get("title"))
            ]
            filters = {"title": list(dict.fromkeys(titles or [requested]))}
            mode = "document_family_title_route"

        return filters, {
            "mode": mode,
            "requested_title": requested,
            "family_candidate_count": len(family),
            "selected_document_count": len(selected),
            "selected_documents": [
                {
                    "doc_id": str(document.get("doc_id") or document.get("id") or ""),
                    "title": normalize_text(document.get("title")),
                    "file_name": normalize_text(document.get("file_name")),
                    "document_type": normalize_text(document.get("document_type")),
                    "route_score": round(float(document.get("_route_score") or 0.0), 6),
                }
                for document in selected
            ],
            "selected_doc_ids": doc_ids,
        }

    def _rank_document_family(
        self,
        family: list[dict[str, Any]],
        requested_title: str,
        query_text: str,
        *,
        max_documents: int = 6,
    ) -> list[dict[str, Any]]:
        profiles = self._document_route_profiles()
        ranked: list[dict[str, Any]] = []
        best_content_score = 0.0

        for document in family:
            doc_id = str(document.get("doc_id") or document.get("id") or "")
            title = normalize_text(document.get("title"))
            file_name = normalize_text(document.get("file_name"))
            route_score = _document_title_route_score(requested_title, title, file_name)
            route_score += 4.0 * _routing_text_score(query_text, file_name)

            paragraph_scores = sorted(
                (
                    _routing_text_score(query_text, text)
                    for text in profiles.get(doc_id, [])
                    if text
                ),
                reverse=True,
            )
            content_score = paragraph_scores[0] if paragraph_scores else 0.0
            second_score = paragraph_scores[1] if len(paragraph_scores) > 1 else 0.0
            best_content_score = max(best_content_score, content_score)

            # Content relevance dominates the final routing.  Title/family
            # similarity only keeps the search inside the correct logical
            # regulation.  This is what disambiguates dozens of attachments
            # sharing the same base title.
            score = route_score + 8.0 * content_score + 2.0 * second_score
            enriched = dict(document)
            enriched["_route_score"] = score
            enriched["_content_route_score"] = content_score
            ranked.append(enriched)

        ranked.sort(
            key=lambda item: (
                float(item.get("_route_score") or 0.0),
                normalize_text(item.get("file_name")),
            ),
            reverse=True,
        )

        if len(ranked) <= max_documents:
            return ranked

        # If lexical routing is confident, keep a small set so similarly named
        # attachments cannot crowd out the correct file during chunk retrieval.
        # If no document contains meaningful query terms, retain a wider family
        # window rather than making a brittle early choice.
        limit = max_documents if best_content_score >= 0.10 else min(12, len(ranked))
        cutoff = float(ranked[min(limit - 1, len(ranked) - 1)].get("_route_score") or 0.0)
        selected = ranked[:limit]

        # Keep near-ties at the boundary (bounded) so a split heading/body pair
        # in two related files is not accidentally discarded.
        boundary = float(selected[-1].get("_route_score") or 0.0) if selected else 0.0
        for item in ranked[limit:]:
            if len(selected) >= min(12, len(ranked)):
                break
            if boundary and float(item.get("_route_score") or 0.0) >= boundary * 0.97:
                selected.append(item)
            else:
                break
        return selected

    def _document_route_profiles(self) -> dict[str, list[str]]:
        if self._document_route_profile_cache is not None:
            return self._document_route_profile_cache

        profiles: dict[str, list[str]] = {}
        for item in list(getattr(self.index, "text", []) or []):
            doc_id = str(item.get("doc_id") or "")
            if not doc_id:
                continue
            text = normalize_text(" ".join(str(item.get(key) or "") for key in (
                "content",
                "context",
                "context_window",
                "_section_scope",
                "section_title",
                "heading",
            )))
            if text:
                profiles.setdefault(doc_id, []).append(text)

        # Keep each document profile bounded.  Route scoring uses the best few
        # matching chunks, so storing thousands of near-duplicate paragraphs is
        # unnecessary.
        for doc_id, values in list(profiles.items()):
            profiles[doc_id] = values[:800]

        self._document_route_profile_cache = profiles
        return profiles


def _task_retrieval_mode(
    task: RetrievalTask,
    source_diagnostics: dict[str, Any],
) -> str:
    """Choose evidence modality independently from answer value type.

    A requested value can be numeric while the authoritative evidence is a
    sentence in Word/PDF. Only explicit table structure or a resolved
    spreadsheet source should force table retrieval.
    """
    semantic = task.semantic_constraints

    if task.expected_value_type == "table_cell":
        return "table"
    if semantic.row_label or semantic.column_label:
        return "table"

    selected = list(source_diagnostics.get("selected_documents") or [])
    if selected:
        modes = {
            _document_modality(
                item.get("document_type"),
                item.get("file_name"),
                item.get("title"),
            )
            for item in selected
        }
        modes.discard("unknown")
        if modes == {"table"}:
            return "table"
        if modes == {"text"}:
            return "text"
        if len(modes) > 1:
            return "mixed"

    # A plain numeric fact, percentage, threshold or amount is not a table
    # signal. Keep both retrieval arms available when source modality is
    # unresolved.
    if task.expected_value_type == "number":
        return "mixed"

    # Structured statistical dimensions strongly suggest a table; an indicator
    # by itself does not, because regulations also contain named ratios.
    if semantic.period and (semantic.indicator or semantic.statistical_scope):
        return "mixed"
    return "text"


def _document_modality(
    document_type: Any,
    file_name: Any,
    title: Any,
) -> str:
    blob = normalize_text(" ".join(str(value or "") for value in [
        document_type,
        file_name,
        title,
    ])).lower()

    if any(token in blob for token in (
        ".xlsx", ".xls", ".csv", "spreadsheet", "excel", "工作簿",
    )):
        return "table"
    if any(token in blob for token in (
        ".doc", ".docx", ".pdf", ".txt", "word", "pdf", "文本",
    )):
        return "text"
    return "unknown"


def _search_text_only(
    index: Any,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
) -> list[Hit]:
    search = getattr(index, "search_text", None)
    if callable(search):
        return list(search(query, top_k, filters))

    # Compatibility with test doubles that only implement hybrid_search.
    hybrid = getattr(index, "hybrid_search", None)
    if callable(hybrid):
        return [
            hit
            for hit in hybrid(query, "regulatory_fact", top_k, filters)
            if getattr(hit, "kind", None) == "text"
        ]
    return []


def _search_table_only(
    index: Any,
    query: str,
    top_k: int,
    filters: dict[str, Any] | None,
    *,
    task: RetrievalTask | None = None,
) -> list[Hit]:
    """Use the planner's structured task instead of re-parsing the query."""
    search = getattr(index, "search_tables", None)
    if callable(search):
        structured = _table_structured_hints(task) if task is not None else None
        try:
            return list(search(query, top_k, filters, structured=structured))
        except TypeError:
            return list(search(query, top_k, filters))

    hybrid = getattr(index, "hybrid_search", None)
    if callable(hybrid):
        return [
            hit for hit in hybrid(query, "table_lookup", top_k, filters)
            if getattr(hit, "kind", None) == "table"
        ]
    return []


def _table_structured_hints(task: RetrievalTask) -> dict[str, Any]:
    source = task.source_scope
    semantic = task.semantic_constraints
    return {
        "indicator": normalize_text(semantic.indicator) or None,
        "parent_indicator": normalize_text(semantic.parent_indicator) or None,
        "institution": normalize_text(semantic.institution) or None,
        "region": normalize_text(semantic.region) or None,
        "period": normalize_text(semantic.period) or None,
        "statistical_scope": normalize_text(semantic.statistical_scope) or None,
        "row_label": normalize_text(semantic.row_label) or None,
        "column_label": normalize_text(semantic.column_label) or None,
        "year": source.year,
        "month": source.month,
        "quarter": source.quarter,
    }


def _choose_mixed_execution(
    task: RetrievalTask,
    text_execution: RetrievalExecution,
    table_execution: RetrievalExecution,
) -> RetrievalExecution:
    text_ok = text_execution.result.status == "resolved"
    table_ok = table_execution.result.status == "resolved"

    if text_ok and not table_ok:
        return text_execution
    if table_ok and not text_ok:
        return table_execution
    if not text_ok and not table_ok:
        # Prefer an ambiguity result over a pure miss because it gives the Agent
        # a useful observation to refine.
        if table_execution.result.status == "ambiguous":
            return table_execution
        return text_execution

    # Both arms found evidence. Explicit statistical/table structure wins;
    # otherwise compare the selected evidence scores.
    semantic = task.semantic_constraints
    if task.expected_value_type == "table_cell" or semantic.row_label or semantic.column_label:
        return table_execution

    text_score = (
        float(text_execution.result.selected.score)
        if text_execution.result.selected is not None else 0.0
    )
    table_score = (
        float(table_execution.result.selected.score)
        if table_execution.result.selected is not None else 0.0
    )
    return table_execution if table_score > text_score * 1.15 else text_execution




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


def _index_documents(index: Any) -> list[dict[str, Any]]:
    documents = list(getattr(index, "documents", []) or [])
    if documents:
        return [dict(item) for item in documents]
    return [
        dict(value)
        for value in dict(getattr(index, "doc_by_id", {}) or {}).values()
    ]


def _document_routing_query(task: RetrievalTask) -> str:
    constraints = task.semantic_constraints
    return normalize_text(" ".join(str(value or "") for value in [
        task.query,
        task.expected_information,
        constraints.indicator,
        constraints.parent_indicator,
        constraints.institution,
        constraints.region,
        constraints.period,
        constraints.statistical_scope,
        constraints.row_label,
        constraints.column_label,
    ]))


def _document_belongs_to_requested_family(
    document: dict[str, Any],
    requested_title: str,
) -> bool:
    requested = _document_family_text(requested_title)
    if not requested:
        return False
    title = _document_family_text(document.get("title"))
    file_name = _document_family_text(document.get("file_name"))
    return any(
        requested in candidate or candidate in requested
        for candidate in (title, file_name)
        if candidate
    )


def _document_title_route_score(
    requested_title: str,
    title: str,
    file_name: str,
) -> float:
    requested = _document_family_text(requested_title)
    title_key = _document_family_text(title)
    file_key = _document_family_text(file_name)

    score = 0.0
    if title_key == requested:
        score += 3.0
    elif requested and requested in title_key:
        score += 1.6
    if file_key == requested:
        score += 2.5
    elif requested and requested in file_key:
        score += 1.4

    # “附件” is only a soft penalty when the user cites the parent regulation.
    # It must never be a hard exclusion because the requested fact may genuinely
    # live in an attachment.
    requested_mentions_attachment = "附件" in normalize_text(requested_title)
    candidate_mentions_attachment = "附件" in normalize_text(f"{title} {file_name}")
    if candidate_mentions_attachment and not requested_mentions_attachment:
        score -= 0.15
    return score


def _document_family_text(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\.(?:docx?|pdf|xlsx?|xls|csv|txt)$", "", text, flags=re.I)
    text = re.sub(r"^\s*\d+\s*[_\-—–.．、 ]+", "", text)
    text = text.replace("《", "").replace("》", "")
    return re.sub(r"[\s_：:，,。；;（）()【】\[\]\-—–]+", "", text)


def _routing_text_score(query: str, text: str) -> float:
    query_key = _routing_key(query)
    text_key = _routing_key(text)
    if not query_key or not text_key:
        return 0.0
    if query_key in text_key:
        return 1.0

    query_bigrams = {
        query_key[index:index + 2]
        for index in range(max(0, len(query_key) - 1))
    }
    text_bigrams = {
        text_key[index:index + 2]
        for index in range(max(0, len(text_key) - 1))
    }
    if not query_bigrams:
        return 0.0

    coverage = len(query_bigrams & text_bigrams) / len(query_bigrams)

    # Reward distinctive longer phrases from the query.  This helps route
    # “核心一级资本充足率阈值” to the right physical file even when dozens of
    # attachments repeat the same regulation title.
    phrases = [
        phrase
        for phrase in re.findall(r"[\u4e00-\u9fffA-Za-z0-9%％]{4,}", normalize_text(query))
        if len(phrase) >= 4
    ]
    phrase_bonus = max(
        (min(1.0, len(_routing_key(phrase)) / 18) for phrase in phrases if _routing_key(phrase) in text_key),
        default=0.0,
    )
    return min(1.0, 0.82 * coverage + 0.18 * phrase_bonus)


def _routing_key(value: Any) -> str:
    text = normalize_text(value)
    # Remove common request scaffolding that does not distinguish one attachment
    # from another.
    for token in (
        "根据", "综合", "分别说明", "请说明", "是多少", "哪些", "对应的",
        "中的", "中", "附件", "工作表", "口径", "数值",
    ):
        text = text.replace(token, "")
    return re.sub(r"[\s《》“”\"'：:，,。；;（）()【】\[\]\-—–]+", "", text)




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
        item.get("_inferred_period"),
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



def _hit_matches_excluded_row(hit: Hit, task: RetrievalTask) -> bool:
    """Apply negative row constraints after retrieval and before max/min selection."""
    excluded = {
        _normalized_row_identity(value)
        for value in (getattr(task, "exclude_row_labels", []) or [])
        if normalize_text(value)
    }
    if not excluded:
        return False

    item = hit.item
    candidate_labels = {
        _normalized_row_identity(item.get("row_header")),
        _normalized_row_identity(item.get("indicator")),
    }
    candidate_labels.discard("")
    return bool(excluded.intersection(candidate_labels))


def _normalized_row_identity(value: Any) -> str:
    return canonical_table_label(value).replace(" ", "").replace("　", "")


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
    """Allow year and quarter to live in different metadata fields."""
    value = normalize_text(requested)
    haystack = normalize_text(blob)
    if not value:
        return True
    if value in haystack:
        return True

    year_match = re.search(r"(20\d{2})", value)
    year = year_match.group(1) if year_match else None

    quarter_no = None
    q_match = re.search(r"Q([1-4])", value, re.IGNORECASE)
    if q_match:
        quarter_no = int(q_match.group(1))
    else:
        cn = re.search(r"(?:第)?([一二三四1-4])季度", value)
        if cn:
            raw = cn.group(1)
            quarter_no = int(raw) if raw.isdigit() else {"一":1,"二":2,"三":3,"四":4}[raw]
    if quarter_no is not None:
        chinese = ("一","二","三","四")[quarter_no-1]
        quarter_ok = any(v in haystack for v in (
            f"Q{quarter_no}", f"{quarter_no}季度", f"第{quarter_no}季度",
            f"{chinese}季度", f"第{chinese}季度",
        ))
        return (not year or year in haystack) and quarter_ok

    month = re.search(r"(?:20\d{2}[-年/]?)?0?([1-9]|1[0-2])月", value)
    if month:
        m=int(month.group(1))
        month_ok=any(v in haystack for v in (f"{m}月",f"{m:02d}月",f"-{m:02d}"))
        return (not year or year in haystack) and month_ok

    return not year or year in haystack


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
            period=normalize_text(
                hit.item.get("period") or hit.item.get("_inferred_period")
            ) or None,
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

    selection = getattr(task, "selection", "single")
    if selection in {"max", "min"}:
        selected_pair = _select_extreme_candidate(candidates, candidate_hits, selection)
        if selected_pair is None:
            return RetrievalExecution(RetrievalResult(
                task_id=task.id,
                status="not_found",
                expected_information=task.expected_information,
                candidates=candidates,
                evidence_ids=[item for candidate in candidates for item in candidate.evidence_ids],
                ambiguity_reason="候选集合中没有可比较的数值",
            ), candidate_hits)
        selected, selected_hit = selected_pair
        # Keep the comparison set as evidence, but place the deterministic
        # argmax/argmin candidate in ``selected`` so the Answer Agent can state
        # both the winning item and its value without performing arithmetic.
        return RetrievalExecution(RetrievalResult(
            task_id=task.id,
            status="resolved",
            expected_information=task.expected_information,
            selected=selected,
            candidates=candidates,
            evidence_ids=selected.evidence_ids,
        ), [selected_hit, *[hit for hit in candidate_hits if hit.evidence_id != selected_hit.evidence_id]])

    if selection == "all":
        selected = candidates[0]
        return RetrievalExecution(RetrievalResult(
            task_id=task.id,
            status="resolved",
            expected_information=task.expected_information,
            selected=selected,
            candidates=candidates,
            evidence_ids=[item for candidate in candidates for item in candidate.evidence_ids],
        ), candidate_hits)

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



def _select_extreme_candidate(
    candidates: list[RetrievalCandidate],
    hits: list[Hit],
    selection: str,
) -> tuple[RetrievalCandidate, Hit] | None:
    """Deterministically select argmax/argmin from one retrieved table slice."""
    numeric: list[tuple[Decimal, RetrievalCandidate, Hit]] = []
    for candidate, hit in zip(candidates, hits):
        value = _numeric_decimal(candidate.value)
        if value is not None:
            numeric.append((value, candidate, hit))
    if not numeric:
        return None
    chooser = max if selection == "max" else min
    _, candidate, hit = chooser(numeric, key=lambda item: item[0])
    return candidate, hit


def _numeric_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    text = normalize_text(value).replace(",", "")
    percentage = text.endswith(("%", "％"))
    text = text.rstrip("%％")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return number / Decimal("100") if percentage else number



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
