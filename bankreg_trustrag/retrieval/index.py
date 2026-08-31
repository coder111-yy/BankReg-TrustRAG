from __future__ import annotations

import heapq
import threading
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..schemas import TableCellEvidence, TextEvidence
from ..query import extract_dimension_labels, extract_indicator
from ..utils import (
    canonical_dimension_label,
    canonical_table_label,
    char_ngrams,
    insurance_company_scope,
    is_insurance_fund_table,
    normalize_text,
    normalized_reporting_period,
    normalized_number,
    reporting_period_details,
    tokens,
)
from .bge import BGEPipeline, PersistentVectorIndex


@dataclass
class Hit:
    kind: str
    item: dict[str, Any]
    lexical_score: float = 0.0
    dense_score: float = 0.0
    metadata_score: float = 0.0
    table_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0

    @property
    def evidence_id(self) -> str:
        return str(self.item["evidence_id"])

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.item)
        result.update({
            "kind": self.kind,
            "score": round(self.fused_score, 6),
            "fusion_score": round(self.fused_score, 6),
            "lexical_score": round(self.lexical_score, 6),
            "dense_score": round(self.dense_score, 6),
            "rerank_score": round(self.rerank_score, 6),
        })
        return result


def _table_period_candidates(
    query: str,
    structured: dict[str, Any] | None = None,
) -> list[str]:
    """Return exact-period variants suitable for SQLite IN predicates."""
    structured = structured or {}
    values: list[str] = []

    def add(value: Any) -> None:
        text = normalize_text(value)
        if text and text not in values:
            values.append(text)

    year = structured.get("year")
    month = structured.get("month")
    quarter = structured.get("quarter")
    period = normalize_text(structured.get("period"))

    if period:
        add(period)
    if year:
        add(str(int(year)))
    if year and month:
        add(f"{int(year):04d}-{int(month):02d}")
        add(f"{int(year)}年{int(month)}月")
    if year and quarter:
        q = int(quarter)
        cn = ("一", "二", "三", "四")[q - 1]
        add(f"{int(year):04d}-Q{q}")
        add(f"{int(year)}年{cn}季度")
        add(f"{int(year)}年第{cn}季度")
        add(f"{cn}季度")
    elif quarter:
        q = int(quarter)
        add(("一季度", "二季度", "三季度", "四季度")[q - 1])

    if not values:
        text = normalize_text(query)
        month_match = re.search(r"(20\d{2})年\s*0?(\d{1,2})月", text)
        if month_match:
            add(f"{month_match.group(1)}-{int(month_match.group(2)):02d}")
            add(month_match.group(1))
        else:
            _, normalized_period, q_label = reporting_period_details(text)
            add(normalized_period)
            add(q_label)
            year_match = re.search(r"(20\d{2})年", text)
            if year_match:
                add(year_match.group(1))
    return values



def _structured_year_number(structured: dict[str, Any]) -> int | None:
    value = structured.get("year")
    if value:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"(20\d{2})", normalize_text(structured.get("period")))
    return int(match.group(1)) if match else None


def _structured_quarter_number(structured: dict[str, Any]) -> int | None:
    value = structured.get("quarter")
    if value:
        try:
            number = int(value)
            return number if 1 <= number <= 4 else None
        except (TypeError, ValueError):
            pass
    text = normalize_text(structured.get("period"))
    match = re.search(r"Q([1-4])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:第)?([一二三四1-4])季度", text)
    if not match:
        return None
    raw = match.group(1)
    return int(raw) if raw.isdigit() else {"一": 1, "二": 2, "三": 3, "四": 4}[raw]


def _structured_month_number(structured: dict[str, Any]) -> int | None:
    value = structured.get("month")
    if value:
        try:
            number = int(value)
            return number if 1 <= number <= 12 else None
        except (TypeError, ValueError):
            pass
    text = normalize_text(structured.get("period"))
    match = re.search(r"(?:20\d{2}[-年/]?)?0?([1-9]|1[0-2])月", text)
    return int(match.group(1)) if match else None


def _table_row_number(item: dict[str, Any]) -> int | None:
    for key in ("row_no", "row_number", "excel_row", "row_index"):
        value = item.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    address = normalize_text(item.get("cell_address"))
    match = re.search(r"[A-Za-z]+(\d+)", address)
    return int(match.group(1)) if match else None


def _table_sheet_key(item: dict[str, Any]) -> tuple[str, str]:
    return (
        str(item.get("doc_id") or ""),
        normalize_text(item.get("sheet_name") or item.get("table_name")),
    )


def _quarter_from_text(value: Any) -> int | None:
    text = normalize_text(value)
    if not text:
        return None
    match = re.search(r"Q([1-4])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:第)?([一二三四1-4])季度", text)
    if not match:
        return None
    raw = match.group(1)
    return int(raw) if raw.isdigit() else {"一": 1, "二": 2, "三": 3, "四": 4}[raw]


def _month_from_text(value: Any) -> int | None:
    text = normalize_text(value)
    if not text:
        return None
    match = re.search(r"(?:20\d{2}[-年/]?)?0?([1-9]|1[0-2])月", text)
    return int(match.group(1)) if match else None


def _build_period_anchors(
    items: list[dict[str, Any]],
    *,
    kind: str,
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Build sparse period anchors from merged/heading cells in an Excel sheet."""
    anchors: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    extractor = _quarter_from_text if kind == "quarter" else _month_from_text

    for item in items:
        row = _table_row_number(item)
        if row is None:
            continue
        text = " ".join(str(item.get(key) or "") for key in (
            "period",
            "indicator",
            "row_header",
            "column_header",
            "value_text",
        ))
        value = extractor(text)
        if value is None:
            continue
        anchors[_table_sheet_key(item)].append((row, value))

    for key in list(anchors):
        anchors[key] = sorted(set(anchors[key]))
    return anchors


def _nearest_period_anchor(
    item: dict[str, Any],
    anchors: dict[tuple[str, str], list[tuple[int, int]]],
) -> int | None:
    row = _table_row_number(item)
    if row is None:
        return None
    values = anchors.get(_table_sheet_key(item), [])
    preceding = [value for anchor_row, value in values if anchor_row <= row]
    return preceding[-1] if preceding else None


def _structured_item_period_match(
    item: dict[str, Any],
    *,
    requested_year: int | None,
    requested_quarter: int | None,
    requested_month: int | None,
    quarter_anchors: dict[tuple[str, str], list[tuple[int, int]]],
    month_anchors: dict[tuple[str, str], list[tuple[int, int]]],
    documents: dict[str, dict[str, Any]],
) -> tuple[bool, str | None]:
    """Match period fields, including sparse merged-cell period labels."""
    direct_text = " ".join(str(item.get(key) or "") for key in (
        "period",
        "row_header",
        "column_header",
        "context",
    ))
    document = documents.get(str(item.get("doc_id") or ""), {})
    source_text = " ".join(str(document.get(key) or "") for key in (
        "title", "file_name", "local_path",
    ))

    if requested_year and str(requested_year) not in normalize_text(
        f"{direct_text} {source_text}"
    ):
        return False, None

    if requested_quarter:
        direct_quarter = _quarter_from_text(direct_text)
        actual_quarter = direct_quarter or _nearest_period_anchor(item, quarter_anchors)
        if actual_quarter != requested_quarter:
            return False, None
        chinese = ("一", "二", "三", "四")[requested_quarter - 1]
        inferred = (
            f"{requested_year}-Q{requested_quarter}"
            if requested_year
            else f"{chinese}季度"
        )
        return True, inferred

    if requested_month:
        direct_month = _month_from_text(direct_text)
        actual_month = direct_month or _nearest_period_anchor(item, month_anchors)
        if actual_month != requested_month:
            return False, None
        inferred = (
            f"{requested_year:04d}-{requested_month:02d}"
            if requested_year
            else f"{requested_month}月"
        )
        return True, inferred

    return True, str(requested_year) if requested_year else None





def _structured_scan_terms(*values: Any) -> list[str]:
    """Normalize a multi-level table header into required semantic tokens."""
    terms: list[str] = []
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        # Accept planner forms such as "截至当期-账面余额",
        # "截至当期 / 账面余额" and "截至当期→账面余额".
        for part in re.split(r"\s*(?:/|／|→|->|>|-)\s*", text):
            canonical = canonical_dimension_label(part)
            if canonical and canonical not in terms:
                terms.append(canonical)
    return terms


def _structured_scan_header_match(terms: list[str], header_blob: str) -> bool:
    """All requested header levels must occur in the candidate header path."""
    normalized = canonical_dimension_label(header_blob)
    return bool(terms) and all(term in normalized for term in terms)


def _dominant_comparable_unit(hits: list[Hit]) -> list[Hit]:
    """Keep a single comparable unit before deterministic max/min selection.

    If all values have one unit, keep them all. If multiple units are present,
    use the uniquely most frequent non-empty unit. A tie is ambiguous and
    returns an empty list so the normal agent can refine rather than comparing
    money with percentages.
    """
    units = [
        normalize_text(hit.item.get("unit"))
        for hit in hits
        if normalize_text(hit.item.get("unit"))
    ]
    if not units:
        return hits

    counts = Counter(units)
    if len(counts) == 1:
        return hits

    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return []

    dominant = ranked[0][0]
    return [
        hit for hit in hits
        if not normalize_text(hit.item.get("unit"))
        or normalize_text(hit.item.get("unit")) == dominant
    ]


class HybridIndex:
    """Hybrid lexical/structured index with optional real local BGE retrieval.

    The character route is intentionally model-free so the contest corpus never leaves
    the machine and the demo is runnable without downloading an embedding model.
    """

    def __init__(
        self,
        documents: Iterable[dict[str, Any]],
        text: Iterable[dict[str, Any]],
        tables: Iterable[dict[str, Any]],
        semantic: BGEPipeline | None = None,
        vector_dir: Any | None = None,
        table_provider: Callable[..., list[Any]] | None = None,
        table_count: int | None = None,
    ):
        self.documents = list(documents)
        self.text = list(text)
        self.tables = list(tables)
        self._text_vectors_ready = False
        self._text_vector_lock = threading.Lock()
        # Production corpora contain more than one million table cells.  Keep
        # the SQLite-backed provider instead of materialising that corpus in
        # Python; small in-memory fixtures continue to use ``tables``.
        self._table_provider = table_provider
        self._table_count = table_count
        self.semantic = semantic
        self._text_vector_index = (
            PersistentVectorIndex(semantic, vector_dir, "text")
            if semantic is not None and vector_dir is not None
            else None
        )
        self._text_vectors_ready = False
        # Small cache for source-scoped table rows used by deterministic
        # structured lookup. It is intentionally bounded and only caches
        # relatively small workbooks.
        self._structured_table_cache: dict[tuple[str, ...], list[dict[str, Any]]] = {}

        self._runtime_usage: dict[str, bool] = {
            "bm25_used": False,
            "metadata_filter_used": False,
            "structured_table_used": False,
            "bge_vector_used": False,
            "bge_reranker_used": False,
            "char_ngram_fallback_used": False,
            "rrf_used": False,
        }
        self.doc_by_id = {str(d["doc_id"]): d for d in self.documents}
        # Text evidence is only ~tens of thousands of rows, so keeping its BM25
        # token/DF structures in memory is reasonable.
        self._text_tokens = [tokens(self._text_blob(item)) for item in self.text]
        self._text_df = self._df(self._text_tokens)
        self._text_avg_len = sum(map(len, self._text_tokens)) / max(len(self._text_tokens), 1)

        # Do NOT pre-tokenize the full table corpus here.  The competition corpus
        # contains more than one million table-cell evidence rows; materializing
        # _table_tokens + _table_df duplicates a very large amount of text and can
        # exhaust RAM before the service reaches BGE initialization.  Table lexical
        # scores are computed lazily on a bounded shortlist in search_tables().
        self._text_indices_by_doc: dict[str, list[int]] = defaultdict(list)
        self._table_indices_by_doc: dict[str, list[int]] = defaultdict(list)
        self._table_indices_by_period: dict[str, list[int]] = defaultdict(list)
        self._formula_indices: list[int] = []
        for index, item in enumerate(self.text):
            self._text_indices_by_doc[str(item.get("doc_id"))].append(index)
        active_insurance_scopes: dict[tuple[str, str], str] = {}
        for index, item in enumerate(self.tables):
            doc_id = str(item.get("doc_id"))
            sheet_name = str(item.get("sheet_name") or item.get("table_name") or "")
            document = self.doc_by_id.get(doc_id, {})
            document_period = normalized_reporting_period(" ".join(
                str(document.get(key) or "")
                for key in ("title", "file_name", "local_path")
            ))
            if document_period:
                item["_document_period"] = document_period
            if is_insurance_fund_table(
                document.get("title"),
                document.get("file_name"),
                sheet_name,
                item.get("table_name"),
            ):
                scope_key = (doc_id, sheet_name)
                heading_scope = insurance_company_scope(" ".join(
                    str(item.get(key) or "") for key in ("indicator", "row_header", "value_text")
                ))
                if heading_scope:
                    active_insurance_scopes[scope_key] = heading_scope
                section_scope = active_insurance_scopes.get(scope_key, "保险业总体")
                item["_section_scope"] = section_scope
                context = normalize_text(item.get("context"))
                if section_scope not in context:
                    item["context"] = f"{section_scope} | {context}" if context else section_scope
            self._table_indices_by_doc[str(item.get("doc_id"))].append(index)
            period = str(item.get("period") or "")
            if period:
                self._table_indices_by_period[period].append(index)
            formula_blob = normalize_text(item.get("value_text")).replace("％", "%")
            if "不良贷款余额" in formula_blob and "各项贷款余额" in formula_blob and "100%" in formula_blob:
                self._formula_indices.append(index)

    @classmethod
    def from_store(cls, store: Any, semantic: BGEPipeline | None = None, vector_dir: Any | None = None) -> "HybridIndex":
        return cls(
            [dict(x) for x in store.all_documents()],
            [dict(x) for x in store.all_text()],
            [],
            semantic=semantic,
            vector_dir=vector_dir,
            table_provider=lambda **kwargs: [dict(x) for x in store.table_candidates(**kwargs)],
            table_count=store.table_count(),
        )

    @property
    def model_status(self) -> dict[str, Any]:
        status = self.semantic.status if self.semantic else {"mode": "disabled"}
        status = dict(status)
        if self._text_vector_index and self._text_vector_index.available:
            manifest = self._text_vector_index.manifest
            status["text_vector_index"] = {
                key: manifest.get(key)
                for key in ["model", "dimension", "count", "corpus_fingerprint"]
            }
        else:
            status["text_vector_index"] = None
        return status

    def begin_query(self) -> None:
        """Reset per-request retrieval usage flags before a new QA run."""
        for key in self._runtime_usage:
            self._runtime_usage[key] = False

    @property
    def runtime_status(self) -> dict[str, Any]:
        """Return actual capabilities used by the most recent request.

        Configuration is deliberately kept separate from usage.  In ``auto``
        mode a missing model may trigger the character fallback; that must not
        be reported as a successful BGE route.
        """
        status = self.model_status
        status.update({
            "vector_index_available": bool(self._text_vector_index and self._text_vector_index.available),
            "table_vector_strategy": (
                "structured_prefilter_then_bge_similarity"
                if self.semantic is not None
                else "structured_prefilter_then_character_similarity"
            ),
            **self._runtime_usage,
        })
        return status

    def _ensure_text_vectors(self) -> bool:
        if self._text_vectors_ready:
            return bool(
                self._text_vector_index
                and self._text_vector_index.available
            )

        with self._text_vector_lock:

            # 防止另一个线程已经完成初始化
            if self._text_vectors_ready:
                return bool(
                    self._text_vector_index
                    and self._text_vector_index.available
                )

            if (
                    not self._text_vector_index
                    or not self.semantic
                    or not self.text
            ):
                return False

            try:
                success = bool(
                    self._text_vector_index.build_or_load(
                        self.text,
                        self._text_blob,
                    )
                )
            except Exception:
                # 初始化失败不能污染后续请求
                self._text_vectors_ready = False
                return False

            # 只有真正成功才标 ready
            self._text_vectors_ready = success

            return success

    @staticmethod
    def _df(rows: list[list[str]]) -> Counter:
        df: Counter = Counter()
        for row in rows:
            df.update(set(row))
        return df

    @staticmethod
    def _text_blob(item: dict[str, Any]) -> str:
        return " ".join(str(item.get(k) or "") for k in ["content", "chapter", "article_no", "section"])

    @staticmethod
    def _table_blob(item: dict[str, Any]) -> str:
        return " ".join(str(item.get(k) or "") for k in ["table_name", "_section_scope", "indicator", "period", "unit", "row_header", "column_header", "context", "value_text"])

    @staticmethod
    def _table_lexical(query_tokens: list[str], item: dict[str, Any]) -> float:
        """Memory-light lexical score for table cells.

        Global BM25 statistics over a million-cell table corpus are intentionally
        avoided.  This score is only used as a cheap pre-filter before the bounded
        BGE/character-similarity shortlist.
        """
        if not query_tokens:
            return 0.0
        row_tokens = tokens(HybridIndex._table_blob(item))
        if not row_tokens:
            return 0.0
        query_terms = set(query_tokens)
        counts = Counter(row_tokens)
        matched = sum(1 for term in query_terms if counts.get(term, 0))
        if not matched:
            return 0.0
        coverage = matched / max(len(query_terms), 1)
        tf_bonus = sum(min(counts.get(term, 0), 3) for term in query_terms) / max(len(row_tokens), 1)
        return coverage + 0.1 * tf_bonus

    def _bm25(self, query_tokens: list[str], rows: list[list[str]], df: Counter, avg_len: float, index: int, k1: float = 1.5, b: float = 0.75) -> float:
        if not query_tokens or index >= len(rows):
            return 0.0
        row = rows[index]
        counts = Counter(row)
        score = 0.0
        n = len(rows)
        for term in set(query_tokens):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            denominator = frequency + k1 * (1 - b + b * len(row) / max(avg_len, 1))
            score += idf * frequency * (k1 + 1) / denominator
        return score

    @staticmethod
    def _dense(query: str, document: str) -> float:
        a, b = char_ngrams(query), char_ngrams(document)
        if not a or not b:
            return 0.0
        return len(a & b) / math.sqrt(len(a) * len(b))

    def _metadata(self, item: dict[str, Any], query: str, filters: dict[str, Any] | None) -> float:
        if not filters:
            return 0.0
        doc = self.doc_by_id.get(str(item.get("doc_id")), {})
        score = 0.0
        haystack = _metadata_key(" ".join(str(doc.get(k) or "") for k in ["title", "file_name", "local_path", "authority", "document_type"]))
        for key, expected in filters.items():
            if expected in (None, "", []):
                continue
            values = expected if isinstance(expected, list) else [expected]
            if any(_metadata_key(v) in haystack for v in values):
                score += 1.0
        return score / max(len(filters), 1)

    def _candidate_indices(self, kind: str, filters: dict[str, Any] | None) -> list[int]:
        source = self.documents
        if not filters:
            evidence_by_doc = self._text_indices_by_doc if kind == "text" else self._table_indices_by_doc
            # QA.xlsx is an evaluation set, not a source document. It may be
            # queried explicitly, but must not become default production evidence.
            return [
                index
                for doc in source
                if not self._is_benchmark_document(doc)
                for index in evidence_by_doc.get(str(doc["doc_id"]), [])
            ]
        matching_docs: list[str] = []
        for doc in source:
            haystack = _metadata_key(" ".join(str(doc.get(k) or "") for k in ["title", "file_name", "local_path", "authority", "document_type"]))
            matched = True
            for expected in filters.values():
                if expected in (None, "", []):
                    continue
                values = expected if isinstance(expected, list) else [expected]
                if not any(_metadata_key(value) in haystack for value in values):
                    matched = False
                    break
            if matched:
                matching_docs.append(str(doc["doc_id"]))
        indices_by_doc = self._text_indices_by_doc if kind == "text" else self._table_indices_by_doc
        if matching_docs:
            return [index for doc_id in matching_docs for index in indices_by_doc.get(doc_id, [])]
        # A strict metadata filter with no match should return no candidates rather
        # than silently falling back to unrelated evidence.
        return []

    @staticmethod
    def _is_benchmark_document(document: dict[str, Any]) -> bool:
        title = normalize_text(str(document.get("title") or "")).lower()
        file_name = normalize_text(str(document.get("file_name") or "")).lower()
        return title in {"qa数据", "qa数据集", "qa data"} or file_name in {"qa数据.xlsx", "qa数据集.xlsx"}

    @staticmethod
    def _numeric_terms(value: str) -> tuple[set[str], set[str]]:
        """提取文本中的百分比和普通数字，用于精确数字检索加权。"""
        normalized = normalize_text(value).replace("％", "%")

        matches = re.findall(
            r"[-+]?\d[\d,]*(?:\.\d+)?\s*%?",
            normalized,
        )

        cleaned = {
            match.replace(",", "").replace(" ", "")
            for match in matches
        }

        percentages = {
            value for value in cleaned if value.endswith("%")
        }

        # 去掉百分号后也保留数值，兼容“35”和“35%”两种写法
        numbers = {
            value.rstrip("%")
            for value in cleaned
        }

        return percentages, numbers

    def search_text(
        self,
        query: str,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
        *,
        dense: bool = True,
        lexical: bool = True,
        metadata: bool = True,
        fuse: bool = True,
    ) -> list[Hit]:
        q_tokens = tokens(query) if lexical else []
        query_percentages, query_numbers = self._numeric_terms(query)
        candidate_indices = self._candidate_indices("text", filters)
        allowed_ids = {str(self.text[index].get("evidence_id")) for index in candidate_indices}
        vector_scores: dict[str, float] = {}
        using_persisted_bge = False
        if dense and self.semantic and self._ensure_text_vectors() and self._text_vector_index:
            using_persisted_bge = self._text_vector_index.available
            vector_scores = dict(self._text_vector_index.search(query, max(top_k * 4, 32), allowed_ids))
            if vector_scores:
                self._runtime_usage["bge_vector_used"] = True
        elif dense and self.semantic and self.semantic.enabled:
            self._runtime_usage["char_ngram_fallback_used"] = True
        elif dense:
            self._runtime_usage["char_ngram_fallback_used"] = True
        if lexical:
            self._runtime_usage["bm25_used"] = True
        if metadata and filters:
            self._runtime_usage["metadata_filter_used"] = True
        hits: list[Hit] = []
        for index in candidate_indices:
            item = self.text[index]
            lexical_score = self._bm25(q_tokens, self._text_tokens, self._text_df, self._text_avg_len, index) if lexical else 0.0
            # 提取当前证据中的百分比和数字
            item_text = self._text_blob(item)
            item_percentages, item_numbers = self._numeric_terms(item_text)

            # 完整百分比匹配，例如问题和证据中都出现“35%”
            percentage_matches = (
                query_percentages & item_percentages
            )

            # 普通数字匹配，包括“35”和“35%”之间的匹配
            number_matches = query_numbers & item_numbers

            # 已经按照百分比计算过的数值，不重复计算普通数字分
            percentage_values = {
                value.rstrip("%")
                for value in percentage_matches
            }

            plain_number_matches = (
                number_matches - percentage_values
            )

            # 百分比辨识度较高，权重设为2；普通数字权重设为0.5
            numeric_bonus = (
                2.0 * len(percentage_matches)
                + 0.5 * len(plain_number_matches)
            )

            if lexical:
                lexical_score += numeric_bonus
            if dense:
                if using_persisted_bge:
                    # Do not silently substitute character similarity for
                    # corpus rows that were not in the persisted BGE top-k.
                    # The vector route is either BGE or the explicit fallback.
                    dense_score = vector_scores.get(str(item.get("evidence_id")), 0.0)
                else:
                    dense_score = self._dense(query, self._text_blob(item))
            else:
                dense_score = 0.0
            metadata_score = self._metadata(item, query, filters) if metadata else 0.0
            if lexical_score or dense_score or metadata_score:
                hits.append(Hit("text", item, lexical_score, dense_score, metadata_score=metadata_score))
        if fuse:
            self._rrf(hits)
            self._runtime_usage["rrf_used"] = True
        else:
            hits.sort(key=lambda hit: (hit.lexical_score, hit.dense_score, hit.metadata_score, hit.evidence_id), reverse=True)
        selected = sorted(hits, key=lambda hit: (hit.fused_score, hit.lexical_score, hit.dense_score, hit.evidence_id), reverse=True)[:top_k] if fuse else hits[:top_k]
        for hit in selected:
            self._attach_text_context_window(hit)
        return selected

    def _attach_text_context_window(self, hit: Hit) -> None:
        """Join nearby paragraphs so a clause split by parsing remains auditable."""
        if hit.kind != "text" or hit.item.get("context_window"):
            return
        paragraph = hit.item.get("paragraph_no")
        doc_id = str(hit.item.get("doc_id") or "")
        if paragraph is None or not doc_id:
            return
        try:
            center = int(paragraph)
        except (TypeError, ValueError):
            return
        neighbours = [
            item for item in self.text
            if str(item.get("doc_id") or "") == doc_id
            and item.get("paragraph_no") is not None
            # Long regulatory bullets are frequently split across several
            # paragraph records by DOC/PDF conversion.  A bounded window of
            # neighbouring records preserves the sentence without loading
            # unrelated parts of the document.
            and center - 6 <= int(item["paragraph_no"]) <= center + 6
        ]
        neighbours.sort(key=lambda item: int(item.get("paragraph_no") or 0))
        if neighbours:
            hit.item["context_window"] = " ".join(
                str(item.get("content") or "") for item in neighbours if item.get("content")
            )

    def search_tables(
        self,
        query: str,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
        *,
        dense: bool = True,
        lexical: bool = True,
        metadata: bool = True,
        fuse: bool = True,
        structured: dict[str, Any] | None = None,
    ) -> list[Hit]:
        structured = structured or {}
        if self._table_provider is not None:
            # A max/min/all request is a set operation, not a top-k semantic
            # retrieval problem. Scan the complete structured table slice first
            # so the true extreme cannot be hidden outside BM25/BGE top-k.
            scan_hits = self._search_tables_structured_scan(
                query,
                filters,
                structured=structured,
            )
            if scan_hits is not None:
                self._runtime_usage["structured_table_used"] = True
                if metadata and filters:
                    self._runtime_usage["metadata_filter_used"] = True
                return scan_hits

            exact_hits = self._search_tables_structured_exact(
                query,
                top_k,
                filters,
                structured=structured,
            )
            if exact_hits is not None:
                self._runtime_usage["structured_table_used"] = True
                if metadata and filters:
                    self._runtime_usage["metadata_filter_used"] = True
                return exact_hits

            return self._search_tables_lazy(
                query,
                top_k,
                filters,
                dense=dense,
                lexical=lexical,
                metadata=metadata,
                fuse=fuse,
                structured=structured,
            )
        q_tokens = tokens(query) if lexical else []
        self._runtime_usage["structured_table_used"] = True
        if metadata and filters:
            self._runtime_usage["metadata_filter_used"] = True
        candidate_indices = self._candidate_indices("table", filters)
        # Prefer the LLM planner's structured RetrievalTask. Natural-language
        # parsing remains only as a backward-compatible fallback.
        parsed_row, parsed_column = extract_dimension_labels(query)
        requested_indicator = normalize_text(structured.get("indicator")) or extract_indicator(query)
        requested_row = normalize_text(structured.get("row_label")) or (
            None if structured else parsed_row
        )
        requested_column = normalize_text(structured.get("column_label")) or (
            None if structured else parsed_column
        )
        requested_institution = normalize_text(structured.get("institution")) or None
        normalized_indicator = normalize_text(requested_indicator).lower() if requested_indicator else None
        normalized_row = _canonical_label(requested_row)
        normalized_column = canonical_dimension_label(requested_column)
        normalized_institution = canonical_dimension_label(requested_institution)
        requested_scope = (
            normalize_text(structured.get("statistical_scope"))
            or insurance_company_scope(query)
        )
        calculation_columns = _calculation_columns(query)
        is_calculation = len(calculation_columns) >= 2

        # Period can be split across metadata: e.g. year in the document title
        # and “一季度” in the row/period field. Build it from structured hints
        # first, then fall back to the query parser.
        structured_year = structured.get("year")
        structured_month = structured.get("month")
        structured_quarter = structured.get("quarter")
        structured_period = normalize_text(structured.get("period"))
        period_query = structured_period or normalize_text(query)
        period_match = re.search(r"(20\d{2})年\s*0?(\d{1,2})月", period_query)
        _, requested_document_period, requested_quarter = reporting_period_details(period_query)
        if structured_year and structured_month:
            requested_document_period = f"{int(structured_year):04d}-{int(structured_month):02d}"
        elif structured_year and structured_quarter:
            requested_document_period = f"{int(structured_year):04d}-Q{int(structured_quarter)}"
            requested_quarter = ("一季度", "二季度", "三季度", "四季度")[int(structured_quarter)-1]
        elif structured_quarter and not requested_quarter:
            requested_quarter = ("一季度", "二季度", "三季度", "四季度")[int(structured_quarter)-1]
        if requested_document_period and re.fullmatch(r"20\d{2}-Q[1-4]", requested_document_period):
            # Separate quarterly workbooks often persist only the year in each
            # cell's ``period`` field.  Their document title retains the actual
            # quarter and is authoritative even when the worksheet tab is stale.
            quarter_document_indices = [
                index for index in candidate_indices
                if re.fullmatch(r"20\d{2}-Q[1-4]", str(self.tables[index].get("_document_period") or ""))
            ]
            if quarter_document_indices:
                candidate_indices = [
                    index for index in quarter_document_indices
                    if self.tables[index].get("_document_period") == requested_document_period
                ]
        elif period_match:
            year = int(period_match.group(1))
            month = int(period_match.group(2))
            requested_period = f"{year:04d}-{month:02d}"
            candidate_set = set(candidate_indices)
            period_indices = candidate_set.intersection(self._table_indices_by_period.get(requested_period, []))
            if period_indices:
                candidate_indices = [index for index in candidate_indices if index in period_indices]
            else:
                # Annual regulatory workbooks store the year in ``period`` and
                # the reporting quarter in ``column_header``.
                year_indices = {
                    index
                    for index in candidate_indices
                    if normalize_text(self.tables[index].get("period")) == str(year)
                }
                if year_indices:
                    candidate_indices = [index for index in candidate_indices if index in year_indices]
                if 1 <= month <= 12:
                    requested_quarter = ("一季度", "二季度", "三季度", "四季度")[(month - 1) // 3]
        else:
            year_match = re.search(r"(20\d{2})年", normalize_text(query))
            if year_match:
                requested_year = year_match.group(1)
                year_indices = [
                    index for index in candidate_indices
                    if normalize_text(self.tables[index].get("period")) == requested_year
                    or normalize_text(self.tables[index].get("period")).startswith(f"{requested_year}-")
                ]
                if year_indices:
                    candidate_indices = year_indices

        if normalized_row:
            row_indices = [
                index for index in candidate_indices
                if normalized_row in {
                    _canonical_label(self.tables[index].get("indicator")),
                    _canonical_label(self.tables[index].get("row_header")),
                }
            ]
            if row_indices:
                candidate_indices = row_indices

        scoped_indices = [
            index for index in candidate_indices
            if self.tables[index].get("_section_scope")
        ]
        if scoped_indices and (requested_scope or normalized_row or normalized_indicator):
            target_scope = requested_scope or "保险业总体"
            candidate_indices = [
                index for index in scoped_indices
                if self.tables[index].get("_section_scope") == target_scope
            ]

        # For a difference/change question, the two operands must survive the
        # shortlist together.  Applying the normal single-column filter here
        # keeps only the first column and makes the reasoning stage fall back
        # to a plain value lookup.
        if normalized_column and not is_calculation:
            column_indices = [
                index for index in candidate_indices
                if normalized_column in canonical_dimension_label(" ".join(
                    str(self.tables[index].get(key) or "")
                    for key in ["column_header", "period"]
                ))
            ]
            if column_indices:
                candidate_indices = column_indices

        query_numbers = [
            normalized_number(x)
            for x in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", normalize_text(query))
        ]
        query_numbers = [x for x in query_numbers if x is not None]

        # Keep only a bounded number of table candidates in memory.  The old code
        # created lexical/dense structures for the entire 1M+ cell corpus.
        prefilter_k = max(top_k * 64, 512)
        heap: list[tuple[float, int, Hit]] = []

        for index in candidate_indices:
            item = self.tables[index]
            lexical_score = self._table_lexical(q_tokens, item) if lexical else 0.0
            metadata_score = self._metadata(item, query, filters) if metadata else 0.0
            value = normalized_number(item.get("value_text"))

            table_score = (
                0.5
                if query_numbers and value is not None
                and any(abs(value - q) < 1e-9 for q in query_numbers)
                else 0.0
            )

            item_indicator = normalize_text(item.get("indicator")).lower()
            if normalized_indicator and item_indicator == normalized_indicator:
                table_score += 2.0

            if normalized_row and normalized_row in {
                _canonical_label(item.get("indicator")),
                _canonical_label(item.get("row_header")),
            }:
                table_score += 3.0

            if normalized_column and normalized_column in canonical_dimension_label(" ".join(
                str(item.get(key) or "") for key in ["column_header", "period"]
            )):
                table_score += 3.0
            if is_calculation:
                column_context = canonical_dimension_label(" ".join(
                    str(item.get(key) or "") for key in ["column_header", "context", "period"]
                ))
                if any(canonical_dimension_label(column) in column_context for column in calculation_columns):
                    table_score += 4.0

            if requested_quarter:
                quarter_context = normalize_text(" ".join(
                    str(item.get(key) or "")
                    for key in ["period", "row_header", "column_header", "context"]
                ))
                if requested_quarter in quarter_context:
                    table_score += 2.0

            if normalized_institution:
                dimension_context = canonical_dimension_label(" ".join(
                    str(item.get(key) or "")
                    for key in ["row_header", "column_header", "context"]
                ))
                if normalized_institution in dimension_context:
                    table_score += 4.0

            # A cheap pre-score is enough to choose the shortlist.  BGE or the
            # model-free character score is applied only after this stage.
            pre_score = 4.0 * table_score + lexical_score + 0.5 * metadata_score
            if pre_score <= 0:
                continue

            hit = Hit(
                "table",
                item,
                lexical_score=lexical_score,
                dense_score=0.0,
                metadata_score=metadata_score,
                table_score=table_score,
            )
            entry = (pre_score, index, hit)
            if len(heap) < prefilter_k:
                heapq.heappush(heap, entry)
            elif pre_score > heap[0][0]:
                heapq.heapreplace(heap, entry)

        hits = [entry[2] for entry in heap]

        # For a small already-filtered candidate set, keep a fallback path for
        # questions that had no exact lexical/structured hit.
        if not hits and dense and len(candidate_indices) <= 5000:
            fallback_k = min(prefilter_k, len(candidate_indices))
            fallback_heap: list[tuple[float, int, Hit]] = []
            for index in candidate_indices:
                item = self.tables[index]
                fallback_score = self._dense(query, self._table_blob(item))
                if fallback_score <= 0:
                    continue
                hit = Hit("table", item, dense_score=fallback_score)
                entry = (fallback_score, index, hit)
                if len(fallback_heap) < fallback_k:
                    heapq.heappush(fallback_heap, entry)
                elif fallback_score > fallback_heap[0][0]:
                    heapq.heapreplace(fallback_heap, entry)
            hits = [entry[2] for entry in fallback_heap]

        if not hits:
            return []

        # BGE only scores a bounded shortlist.  If BGE is unavailable, use the
        # model-free character score on the same bounded shortlist.
        shortlist = sorted(
            hits,
            key=lambda hit: (hit.table_score, hit.lexical_score, hit.metadata_score),
            reverse=True,
        )[: max(top_k * 8, 64)]

        if dense and self.semantic:
            scores = self.semantic.similarity(
                query, [self._table_blob(hit.item) for hit in shortlist]
            )
            if scores is not None:
                self._runtime_usage["bge_vector_used"] = True
                for hit, score in zip(shortlist, scores):
                    hit.dense_score = score
            else:
                self._runtime_usage["char_ngram_fallback_used"] = True
                for hit in shortlist:
                    hit.dense_score = self._dense(query, self._table_blob(hit.item))
        elif dense:
            self._runtime_usage["char_ngram_fallback_used"] = True
            for hit in shortlist:
                hit.dense_score = self._dense(query, self._table_blob(hit.item))
        else:
            for hit in shortlist:
                hit.dense_score = 0.0

        if fuse:
            self._rrf(shortlist)
            self._runtime_usage["rrf_used"] = True
            return sorted(
                shortlist,
                key=lambda hit: (hit.fused_score, hit.table_score, hit.lexical_score, hit.dense_score, hit.evidence_id),
                reverse=True,
            )[:top_k]
        shortlist.sort(
            key=lambda hit: (hit.table_score, hit.lexical_score, hit.dense_score, hit.metadata_score, hit.evidence_id),
            reverse=True,
        )
        return shortlist[:top_k]

    def search_formula_evidence(self, indicator: str, top_k: int = 8, filters: dict[str, Any] | None = None, year: str | None = None) -> list[Hit]:
        """Retrieve formula cells that ordinary table top-k ranking can omit."""
        if self._table_provider is not None:
            items = self._lazy_table_candidates(indicator, filters, formula=True)
            hits: list[Hit] = []
            for item in items:
                if year:
                    source = normalize_text(" ".join(
                        str(self.doc_by_id.get(str(item.get("doc_id")), {}).get(key) or "")
                        for key in ("title", "file_name", "local_path")
                    ))
                    period = normalize_text(item.get("period"))
                    if year not in source and not period.startswith(year):
                        continue
                blob = normalize_text(item.get("value_text")).replace("％", "%")
                if "不良贷款余额" in blob and "各项贷款余额" in blob and "100%" in blob:
                    hits.append(Hit("table", item, lexical_score=1.0, table_score=5.0))
            self._rrf(hits)
            return sorted(hits, key=lambda hit: (hit.table_score, hit.fused_score), reverse=True)[:top_k]
        allowed = set(self._candidate_indices("table", filters)) if filters else None
        hits: list[Hit] = []
        for index in self._formula_indices:
            if allowed is not None and index not in allowed:
                continue
            item = self.tables[index]
            if year:
                doc = self.doc_by_id.get(str(item.get("doc_id")), {})
                source = normalize_text(" ".join(str(doc.get(key) or "") for key in ["title", "file_name", "local_path"]))
                period = normalize_text(item.get("period"))
                if year not in source and not period.startswith(year):
                    continue
            blob = normalize_text(item.get("value_text")).replace("％", "%")
            # Excel files often store the indicator label in B6 and the
            # formula in C6.  For the known NPL metric the formula vocabulary
            # is the join key even when the C6 record has indicator=4.0.
            if indicator and normalize_text(indicator).lower() not in blob.lower() and normalize_text(indicator) != "不良贷款率":
                continue
            metadata = self._metadata(item, indicator, filters)
            hits.append(Hit("table", item, lexical_score=1.0, metadata_score=metadata, table_score=5.0))
        self._rrf(hits)
        return sorted(hits, key=lambda hit: (hit.table_score, hit.metadata_score, hit.fused_score), reverse=True)[:top_k]

    def _matching_doc_ids(self, filters: dict[str, Any] | None) -> list[str] | None:
        """Resolve document filters without touching table evidence rows."""
        if not filters:
            return [
                str(doc["doc_id"])
                for doc in self.documents
                if not self._is_benchmark_document(doc)
            ]
        matching: list[str] = []
        for doc in self.documents:
            haystack = _metadata_key(" ".join(
                str(doc.get(key) or "")
                for key in ("title", "file_name", "local_path", "authority", "document_type")
            ))
            if all(
                expected in (None, "", [])
                or any(_metadata_key(value) in haystack for value in (expected if isinstance(expected, list) else [expected]))
                for expected in filters.values()
            ):
                matching.append(str(doc["doc_id"]))
        return matching



    def _search_tables_structured_scan(
        self,
        query: str,
        filters: dict[str, Any] | None,
        *,
        structured: dict[str, Any] | None = None,
    ) -> list[Hit] | None:
        """Scan one complete structured table slice for max/min/all queries.

        This is intentionally different from normal Top-K retrieval.  When the
        LLM has already identified a source workbook and a column/statistical
        scope, a question such as "哪项最高" requires *all* comparable numeric
        cells in that slice.  Ranking 8/64/128 semantically similar cells can
        miss the true extreme and causes the Agent to retry the same query many
        times.

        Returning None means the task is not sufficiently structured and the
        normal exact/hybrid retrieval path should continue.
        """
        if self._table_provider is None:
            return None

        structured = structured or {}
        selection = normalize_text(structured.get("selection")).lower()
        if selection not in {"max", "min", "all"}:
            return None

        # Collection scans need a source-scoped table.  The source resolver in
        # RetrievalTools already maps logical titles/document families to
        # concrete doc_ids; scanning an unbounded corpus would be unsafe.
        doc_ids = self._matching_doc_ids(filters)
        if not doc_ids or len(doc_ids) > 8:
            return None

        # The column can be represented as a full multi-level path
        # ("截至当期 / 账面余额") or split between statistical_scope and
        # column_label.  Treat every non-empty part as a required header token.
        column_terms = _structured_scan_terms(
            structured.get("statistical_scope"),
            structured.get("column_label"),
        )
        if not column_terms:
            return None

        items = self._structured_source_rows(doc_ids)
        if not items:
            return None

        indicator = _canonical_label(structured.get("indicator"))
        row_label = _canonical_label(structured.get("row_label"))
        institution = canonical_dimension_label(structured.get("institution"))
        region = canonical_dimension_label(structured.get("region"))

        requested_year = _structured_year_number(structured)
        requested_quarter = _structured_quarter_number(structured)
        requested_month = _structured_month_number(structured)
        quarter_anchors = _build_period_anchors(items, kind="quarter")
        month_anchors = _build_period_anchors(items, kind="month")

        selected: list[Hit] = []
        for raw_item in items:
            value = raw_item.get("numeric_value")
            if value is None:
                value = raw_item.get("value_text", raw_item.get("value"))
            if normalized_number(value) is None:
                continue

            # Production table_evidence contains data/text_data/note.  If
            # cell_type is available, only numeric business data can
            # participate in an extremum comparison.
            cell_type = normalize_text(raw_item.get("cell_type")).lower()
            if cell_type and cell_type != "data":
                continue

            item_indicator = _canonical_label(raw_item.get("indicator"))
            item_row = _canonical_label(raw_item.get("row_header"))
            if indicator and indicator not in {item_indicator, item_row}:
                continue
            if row_label and row_label not in {item_indicator, item_row}:
                continue

            dimension_blob = canonical_dimension_label(" ".join(
                str(raw_item.get(key) or "")
                for key in ("column_header", "context", "statistical_scope")
            ))
            if not _structured_scan_header_match(column_terms, dimension_blob):
                continue

            if institution:
                institution_blob = canonical_dimension_label(" ".join(
                    str(raw_item.get(key) or "")
                    for key in ("institution", "column_header", "row_header", "context")
                ))
                if institution not in institution_blob:
                    continue

            if region:
                region_blob = canonical_dimension_label(" ".join(
                    str(raw_item.get(key) or "")
                    for key in ("region", "row_header", "column_header", "context")
                ))
                if region not in region_blob:
                    continue

            # If the plan supplied an actual period coordinate, enforce it.
            if requested_year or requested_quarter or requested_month:
                period_match, inferred_period = _structured_item_period_match(
                    raw_item,
                    requested_year=requested_year,
                    requested_quarter=requested_quarter,
                    requested_month=requested_month,
                    quarter_anchors=quarter_anchors,
                    month_anchors=month_anchors,
                    documents=self.doc_by_id,
                )
                if not period_match:
                    continue
            else:
                inferred_period = None

            item = dict(raw_item)
            if inferred_period:
                item["_inferred_period"] = inferred_period
            item["_structured_collection_scan"] = True

            # Structured scan owns recall; scores are only for deterministic
            # display ordering. Selection itself happens in RetrievalTools.
            score = 30.0 + float(len(column_terms) * 5)
            selected.append(Hit(
                "table",
                item,
                lexical_score=0.0,
                dense_score=0.0,
                metadata_score=self._metadata(item, query, filters) if filters else 0.0,
                table_score=score,
            ))

        if not selected:
            return None

        # Never compare unlike units when the parser exposes them. This matters
        # for merged rows: an annual return (%) may physically occupy column B
        # under a "账面余额" header although it is not a monetary balance.
        selected = _dominant_comparable_unit(selected)
        if not selected:
            return None

        # Collection operations need the complete set, not top_k. Keep a hard
        # safety cap so malformed giant workbooks cannot explode one request.
        if len(selected) > 5000:
            return None

        selected.sort(
            key=lambda hit: (
                normalize_text(hit.item.get("row_header") or hit.item.get("indicator")),
                hit.evidence_id,
            )
        )
        return selected


    def _search_tables_structured_exact(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        *,
        structured: dict[str, Any] | None = None,
    ) -> list[Hit] | None:
        """Resolve a strongly-structured Excel cell without BGE/agent retries.

        The LLM planner has already identified the semantic coordinates.  For a
        source-scoped workbook task such as:
            indicator=可疑类贷款余额
            institution=大型商业银行
            quarter=1
        it is wasteful and less reliable to re-rank dozens of cells with BGE.

        This fast path scans only the already-selected physical workbook(s),
        matches row/column semantics deterministically, and reconstructs sparse
        quarter/month labels from the nearest preceding header row when Excel
        merged cells left the target row itself blank.

        Returning ``None`` means "not confident; use the normal hybrid path".
        Returning a list means the exact path was confident enough to own the
        result.
        """
        if self._table_provider is None:
            return None

        structured = structured or {}
        indicator = normalize_text(structured.get("indicator"))
        row_label = normalize_text(structured.get("row_label"))
        institution = normalize_text(structured.get("institution"))
        column_label = normalize_text(structured.get("column_label"))
        region = normalize_text(structured.get("region"))

        # Strong coordinates only.  Generic/free-text table queries still use
        # BM25/BGE/RRF.
        row_target = row_label or institution or indicator
        dimension_target = column_label or institution or region
        requested_quarter = _structured_quarter_number(structured)
        requested_month = _structured_month_number(structured)
        requested_year = _structured_year_number(structured)

        # A source-scoped row plus an explicit quarter/month already identifies
        # one scalar cell.  Requiring a second column_label made valid plans
        # such as indicator=流动性覆盖率, year=2024, quarter=4 fall back to a
        # broad semantic search and often miss the requested workbook.
        if not row_target:
            return None
        if not dimension_target and not (requested_quarter or requested_month):
            return None
        if not (requested_quarter or requested_month or normalize_text(structured.get("period"))):
            return None

        doc_ids = self._matching_doc_ids(filters)
        if not doc_ids or len(doc_ids) > 8:
            return None

        items = self._structured_source_rows(doc_ids)
        if not items:
            return None

        quarter_anchors = _build_period_anchors(items, kind="quarter")
        month_anchors = _build_period_anchors(items, kind="month")

        normalized_row_target = _canonical_label(row_target)
        normalized_dimension = canonical_dimension_label(dimension_target)
        selected: list[Hit] = []

        for raw_item in items:
            debug_blob = normalize_text(
                " ".join(
                    str(raw_item.get(k) or "")
                    for k in (
                        "indicator",
                        "row_header",
                        "column_header",
                        "context",
                    )
                )
            )

            # Table lookup tasks in this fast path are scalar-cell lookups.
            if normalized_number(raw_item.get("value_text")) is None:
                continue

            item_indicator = _canonical_label(raw_item.get("indicator"))
            item_row = _canonical_label(raw_item.get("row_header"))
            row_ok = normalized_row_target in {item_indicator, item_row}

            if not row_ok:
                continue

            dimension_blob = canonical_dimension_label(" ".join(
                str(raw_item.get(key) or "")
                for key in ("column_header", "row_header", "context")
            ))
            if normalized_dimension and normalized_dimension not in dimension_blob:
                continue

            period_match, inferred_period = _structured_item_period_match(
                raw_item,
                requested_year=requested_year,
                requested_quarter=requested_quarter,
                requested_month=requested_month,
                quarter_anchors=quarter_anchors,
                month_anchors=month_anchors,
                documents=self.doc_by_id,
            )
            if not period_match:
                continue

            item = dict(raw_item)
            if inferred_period:
                item["_inferred_period"] = inferred_period
            item["_structured_exact_match"] = True

            score = 20.0
            column = canonical_dimension_label(item.get("column_header"))
            if normalized_dimension and column and normalized_dimension == column:
                score += 8.0
            elif normalized_dimension and normalized_dimension in dimension_blob:
                score += 4.0

            if item_indicator == normalized_row_target:
                score += 5.0
            if item_row == normalized_row_target:
                score += 5.0

            selected.append(Hit(
                "table",
                item,
                lexical_score=0.0,
                dense_score=0.0,
                metadata_score=self._metadata(item, query, filters) if filters else 0.0,
                table_score=score,
            ))

        if not selected:
            return None

        selected.sort(
            key=lambda hit: (
                hit.table_score,
                hit.metadata_score,
                hit.evidence_id,
            ),
            reverse=True,
        )

        # If the strongest semantic coordinates still map to many different
        # values, let RetrievalTools perform its normal ambiguity check rather
        # than inventing a winner.
        return selected[:max(top_k, 8)]

    def _structured_source_rows(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch and lightly cache all table rows for a few selected workbooks."""
        key = tuple(sorted(str(value) for value in doc_ids))
        cached = self._structured_table_cache.get(key)
        if cached is not None:
            return [dict(item) for item in cached]

        rows = [
            dict(item)
            for item in self._table_provider(doc_ids=list(key), limit=20000)
        ]
        # A 20k result may be truncated, so never cache or exact-resolve from a
        # potentially incomplete workbook slice.
        if len(rows) >= 20000:
            return []

        if len(rows) <= 5000:
            if len(self._structured_table_cache) >= 4:
                self._structured_table_cache.clear()
            self._structured_table_cache[key] = [dict(item) for item in rows]
        return rows


    def _lazy_table_candidates(
        self,
        query: str,
        filters: dict[str, Any] | None,
        *,
        formula: bool = False,
        structured: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch a bounded, high-recall candidate set from the SQLite store."""
        if self._table_provider is None:
            return [dict(item) for item in self.tables]
        doc_ids = self._matching_doc_ids(filters)
        if not doc_ids:
            return []
        structured = structured or {}
        indicator = normalize_text(structured.get("indicator")) or extract_indicator(query)

        # When the planner supplied structured dimensions, do not run the old
        # row parser as a second semantic authority. In institution-by-column
        # sheets it can mistake “大型商业银行” for a row label and make SQLite
        # return zero candidates.
        parsed_row, _ = extract_dimension_labels(query)
        row_label = normalize_text(structured.get("row_label")) or (
            None if structured else parsed_row
        )

        periods = _table_period_candidates(query, structured)
        collected: dict[str, dict[str, Any]] = {}

        def fetch(**kwargs: Any) -> None:
            for item in self._table_provider(doc_ids=doc_ids, limit=20000, **kwargs):
                collected[str(item.get("evidence_id"))] = item

        # High-recall progressive retrieval. Start with the strongest explicit
        # structured keys, but never let one uncertain dimension zero out the
        # whole task. The final semantic filter in RetrievalTools verifies the
        # exact indicator/institution/period.
        if indicator and periods:
            fetch(indicator=indicator, periods=periods)
        if indicator:
            fetch(indicator=indicator)
        if row_label and periods:
            fetch(row_label=row_label, periods=periods)
        if row_label:
            fetch(row_label=row_label)
        if periods and not collected:
            fetch(periods=periods)
        if not (indicator or row_label or periods):
            fetch()

        if formula:
            fetch(text_terms=["不良贷款余额", "各项贷款余额"])

        if not collected:
            terms = [
                term
                for term in re.findall(r"[\u4e00-\u9fffA-Za-z]{2,}", normalize_text(query))
                if term not in {"数值", "查询", "数据"}
            ][:8]
            if terms:
                fetch(text_terms=terms)

        # Last scoped fallback: if the user explicitly selected a document and
        # structured passes still found nothing, scan a bounded slice of that
        # document rather than failing because of parser/schema drift.
        if not collected and doc_ids and len(doc_ids) <= 8:
            fetch()

        return list(collected.values())



    def _search_tables_lazy(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        *,
        dense: bool,
        lexical: bool,
        metadata: bool,
        fuse: bool,
        structured: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Run the existing table scorer over only SQL-selected candidates."""
        items = self._lazy_table_candidates(
            query,
            filters,
            structured=structured,
        )
        if not items:
            return []
        # Reuse the thoroughly tested in-memory scorer without rebuilding the
        # million-cell corpus.  This temporary index contains only this query's
        # bounded candidates and shares the already loaded semantic pipeline.
        bounded = HybridIndex(
            self.documents,
            [],
            items,
            semantic=self.semantic,
            vector_dir=None,
        )
        result = bounded.search_tables(
            query,
            top_k,
            filters,
            dense=dense,
            lexical=lexical,
            metadata=metadata,
            fuse=fuse,
            structured=structured,
        )
        for key, value in bounded._runtime_usage.items():
            self._runtime_usage[key] = self._runtime_usage[key] or value
        return result

    @staticmethod
    def _rrf(hits: list[Hit], k: int = 60) -> None:
        """Fuse the four retrieval routes into one deterministic RRF score.

        ``table_score`` is the structured-table route and
        ``metadata_score`` is the metadata-filter route.  All routes use the
        same rank-fusion scale; raw BM25, cosine and heuristic table scores
        are never compared directly.
        """
        def rank_by(attribute: str) -> dict[int, int]:
            ranked = sorted(
                hits,
                key=lambda item: (getattr(item, attribute), item.evidence_id),
                reverse=True,
            )
            return {
                id(hit): rank
                for rank, hit in enumerate(ranked, 1)
                if getattr(hit, attribute) > 0
            }

        route_ranks = {
            attribute: rank_by(attribute)
            for attribute in ("lexical_score", "dense_score", "metadata_score", "table_score")
        }
        # Structured table evidence is a more selective route than generic
        # lexical/semantic similarity. A modest weight keeps an exact
        # indicator+period+column match ahead of a merely related text cell
        # while remaining rank-based (not raw-score based).
        route_weights = {
            "lexical_score": 1.0,
            "dense_score": 1.0,
            "metadata_score": 0.5,
            "table_score": 3.0,
        }
        for hit in hits:
            hit.fused_score = sum(
                route_weights[attribute] / (k + ranks[id(hit)])
                for attribute, ranks in route_ranks.items()
                if id(hit) in ranks
            )

    @staticmethod
    def _sort_without_fusion(hits: list[Hit], top_k: int) -> list[Hit]:
        return sorted(
            hits,
            key=lambda hit: (
                hit.lexical_score,
                hit.dense_score,
                hit.metadata_score,
                hit.table_score,
                hit.evidence_id,
            ),
            reverse=True,
        )[:top_k]

    def hybrid_search(
        self,
        query: str,
        qa_type: str,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
        *,
        rerank: bool = True,
        dense: bool = True,
        lexical: bool = True,
        metadata: bool = True,
        structured: bool = True,
    ) -> list[Hit]:
        """Run lexical/dense/metadata/structured retrieval and unified RRF.

        ``rerank`` and ``dense`` are explicit controls so specialised agents
        can avoid duplicate CrossEncoder work or run a genuine ablation.
        """
        route_top_k = max(top_k * 4, 32)
        effective_filters = filters if metadata else None
        text_hits = self.search_text(
            query,
            route_top_k,
            effective_filters,
            dense=dense,
            lexical=lexical,
            metadata=metadata,
            fuse=False,
        )
        table_hits = self.search_tables(
            query,
            route_top_k,
            effective_filters,
            dense=dense,
            lexical=lexical,
            metadata=metadata,
            fuse=False,
        ) if structured and (qa_type in {"table_lookup", "cross_file_judgment"} or not self.text) else []
        if table_hits:
            self._runtime_usage["structured_table_used"] = True
        combined: dict[str, Hit] = {}
        for hit in text_hits + table_hits:
            previous = combined.get(hit.evidence_id)
            if not previous or hit.fused_score > previous.fused_score:
                combined[hit.evidence_id] = hit
        candidates = list(combined.values())
        self._rrf(candidates)
        self._runtime_usage["rrf_used"] = True
        candidates.sort(key=lambda hit: (hit.fused_score, hit.evidence_id), reverse=True)
        if rerank and self.semantic and candidates:
            rerank_limit = min(len(candidates), max(top_k * 2, self.semantic.config.rerank_top_k))
            rerank_candidates = candidates[:rerank_limit]
            scores = self.semantic.rerank(query, [self._hit_blob(hit) for hit in rerank_candidates])
            if scores is not None:
                self._runtime_usage["bge_reranker_used"] = True
                base_max = max((hit.fused_score for hit in rerank_candidates), default=1.0) or 1.0
                for hit, score in zip(rerank_candidates, scores):
                    hit.rerank_score = score
                    hit.fused_score = 0.35 * (hit.fused_score / base_max) + 0.65 * score
                candidates.sort(key=lambda hit: hit.fused_score, reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _hit_blob(hit: Hit) -> str:
        item = hit.item
        return " ".join(str(item.get(k) or "") for k in [
            "content", "context", "_section_scope", "indicator", "period", "unit",
            "row_header", "column_header", "value_text", "chapter", "article_no",
        ])


def _canonical_label(value: Any) -> str:
    text = canonical_table_label(value)

    # 清理 Excel 指标名称中的脚注标记：
    # 流动性覆盖率**
    # 流动性覆盖率*
    # 流动性覆盖率①
    # 流动性覆盖率注1
    text = re.sub(r"[*＊※]+$", "", text)
    text = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]+$", "", text)
    text = re.sub(r"(?:注|备注)\s*\d*$", "", text)

    return text.strip()


def _metadata_key(value: Any) -> str:
    """Normalize harmless source-title and filename variations."""
    normalized = normalize_text(value).lower()
    normalized = re.sub(r"[^\w\u4e00-\u9fff]|_", "", normalized)
    # 文件标题中常见的非关键差异
    normalized = normalized.replace("的", "")
    normalized = normalized.replace("版", "")
    return normalized

def _calculation_columns(query: str) -> list[str]:
    """Return both quoted operands for table difference/change queries."""
    normalized = normalize_text(query)
    quoted = [
        normalize_text(value).strip(" ：:，,、")
        for value in re.findall(r"[“\"‘「『]([^”\"’」』]+)[”\"’」』]", normalized)
    ]
    if len(quoted) >= 3:
        return quoted[1:3]
    if len(quoted) == 2 and any(term in normalized for term in ("差值", "差额", "相差", "变化", "增减")):
        return quoted
    match = re.search(r"从[“\"‘「『]?([^”\"’」』\s，,。！？?]+)[”\"’」』]?到[“\"‘「『]?([^”\"’」』\s，,。！？?]+)", normalized)
    return [match.group(1), match.group(2)] if match else []
