from __future__ import annotations

import heapq
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

    def _ensure_text_vectors(self) -> bool:
        if self._text_vectors_ready:
            return bool(self._text_vector_index and self._text_vector_index.available)
        self._text_vectors_ready = True
        if not self._text_vector_index or not self.semantic or not self.text:
            return False
        return self._text_vector_index.build_or_load(self.text, self._text_blob)

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

    def search_text(self, query: str, top_k: int = 8, filters: dict[str, Any] | None = None) -> list[Hit]:
        q_tokens = tokens(query)
        candidate_indices = self._candidate_indices("text", filters)
        allowed_ids = {str(self.text[index].get("evidence_id")) for index in candidate_indices}
        vector_scores: dict[str, float] = {}
        if self.semantic and self._ensure_text_vectors() and self._text_vector_index:
            vector_scores = dict(self._text_vector_index.search(query, max(top_k * 4, 32), allowed_ids))
        hits: list[Hit] = []
        for index in candidate_indices:
            item = self.text[index]
            lexical = self._bm25(q_tokens, self._text_tokens, self._text_df, self._text_avg_len, index)
            dense = vector_scores.get(str(item.get("evidence_id")), self._dense(query, self._text_blob(item))) if self.semantic else self._dense(query, self._text_blob(item))
            metadata = self._metadata(item, query, filters)
            if lexical or dense or metadata:
                hits.append(Hit("text", item, lexical, dense, metadata_score=metadata))
        self._rrf(hits)
        selected = sorted(hits, key=lambda hit: hit.fused_score, reverse=True)[:top_k]
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
            and abs(int(item["paragraph_no"]) - center) <= 3
        ]
        neighbours.sort(key=lambda item: int(item.get("paragraph_no") or 0))
        if neighbours:
            hit.item["context_window"] = " ".join(
                str(item.get("content") or "") for item in neighbours if item.get("content")
            )

    def search_tables(self, query: str, top_k: int = 8, filters: dict[str, Any] | None = None) -> list[Hit]:
        if self._table_provider is not None:
            return self._search_tables_lazy(query, top_k, filters)
        q_tokens = tokens(query)
        candidate_indices = self._candidate_indices("table", filters)
        requested_indicator = extract_indicator(query)
        normalized_indicator = normalize_text(requested_indicator).lower() if requested_indicator else None
        requested_row, requested_column = extract_dimension_labels(query)
        normalized_row = _canonical_label(requested_row)
        normalized_column = canonical_dimension_label(requested_column)
        requested_scope = insurance_company_scope(query)
        calculation_columns = _calculation_columns(query)
        is_calculation = len(calculation_columns) >= 2

        # Period is a high-selectivity structured key. Restricting by it before
        # scoring avoids repeatedly computing similarity over the full million-cell
        # corpus for ordinary period-specific questions.
        period_match = re.search(r"(20\d{2})年\s*0?(\d{1,2})月", normalize_text(query))
        _, requested_document_period, requested_quarter = reporting_period_details(query)
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
            lexical = self._table_lexical(q_tokens, item)
            metadata = self._metadata(item, query, filters)
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
                column_context = normalize_text(" ".join(
                    str(item.get(key) or "") for key in ["column_header", "context"]
                ))
                if requested_quarter in column_context:
                    table_score += 2.0

            # A cheap pre-score is enough to choose the shortlist.  BGE or the
            # model-free character score is applied only after this stage.
            pre_score = 4.0 * table_score + lexical + 0.5 * metadata
            if pre_score <= 0:
                continue

            hit = Hit(
                "table",
                item,
                lexical_score=lexical,
                dense_score=0.0,
                metadata_score=metadata,
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
        if not hits and len(candidate_indices) <= 5000:
            fallback_k = min(prefilter_k, len(candidate_indices))
            fallback_heap: list[tuple[float, int, Hit]] = []
            for index in candidate_indices:
                item = self.tables[index]
                dense = self._dense(query, self._table_blob(item))
                if dense <= 0:
                    continue
                hit = Hit("table", item, dense_score=dense)
                entry = (dense, index, hit)
                if len(fallback_heap) < fallback_k:
                    heapq.heappush(fallback_heap, entry)
                elif dense > fallback_heap[0][0]:
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

        if self.semantic:
            scores = self.semantic.similarity(
                query, [self._table_blob(hit.item) for hit in shortlist]
            )
            if scores is not None:
                for hit, score in zip(shortlist, scores):
                    hit.dense_score = score
            else:
                for hit in shortlist:
                    hit.dense_score = self._dense(query, self._table_blob(hit.item))
        else:
            for hit in shortlist:
                hit.dense_score = self._dense(query, self._table_blob(hit.item))

        self._rrf(shortlist)
        return sorted(
            shortlist,
            key=lambda hit: (
                hit.table_score,
                hit.fused_score,
                hit.lexical_score,
                hit.dense_score,
            ),
            reverse=True,
        )[:top_k]

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

    def _lazy_table_candidates(
        self,
        query: str,
        filters: dict[str, Any] | None,
        *,
        formula: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch a bounded, high-recall candidate set from the SQLite store."""
        if self._table_provider is None:
            return [dict(item) for item in self.tables]
        doc_ids = self._matching_doc_ids(filters)
        if not doc_ids:
            return []
        indicator = extract_indicator(query)
        row_label, _ = extract_dimension_labels(query)
        periods: list[str] = []
        month = re.search(r"(20\d{2})年\s*0?(\d{1,2})月", normalize_text(query))
        if month:
            periods.append(f"{month.group(1)}-{int(month.group(2)):02d}")
        else:
            year = re.search(r"(20\d{2})年", normalize_text(query))
            if year:
                periods.append(year.group(1))
        collected: dict[str, dict[str, Any]] = {}

        def fetch(**kwargs: Any) -> None:
            for item in self._table_provider(doc_ids=doc_ids, limit=20000, **kwargs):
                collected[str(item.get("evidence_id"))] = item

        # Exact indicator/row and period predicates use SQLite indexes and
        # cover normal table lookups.  Add broader scoped passes for tables
        # whose indicator is stored in a neighbouring/header cell.
        if indicator or row_label or periods:
            fetch(indicator=indicator, periods=periods, row_label=row_label)
            if indicator or row_label:
                fetch(indicator=indicator, row_label=row_label)
        else:
            fetch()
        if formula:
            fetch(text_terms=["不良贷款余额", "各项贷款余额"])
        if not collected:
            fetch(text_terms=[term for term in re.findall(r"[\u4e00-\u9fffA-Za-z]{2,}", normalize_text(query))[:8]])
        return list(collected.values())

    def _search_tables_lazy(self, query: str, top_k: int, filters: dict[str, Any] | None) -> list[Hit]:
        """Run the existing table scorer over only SQL-selected candidates."""
        items = self._lazy_table_candidates(query, filters)
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
        return bounded.search_tables(query, top_k, filters)

    @staticmethod
    def _rrf(hits: list[Hit], k: int = 60) -> None:
        # RRF-like fusion over lexical and dense rank lists; metadata/table bonuses
        # remain visible in the score for auditability.
        lexical_rank = {id(hit): rank for rank, hit in enumerate(sorted(hits, key=lambda x: x.lexical_score, reverse=True), 1) if hit.lexical_score > 0}
        dense_rank = {id(hit): rank for rank, hit in enumerate(sorted(hits, key=lambda x: x.dense_score, reverse=True), 1) if hit.dense_score > 0}
        for hit in hits:
            hit.fused_score = (1 / (k + lexical_rank.get(id(hit), len(hits) + 1)) if id(hit) in lexical_rank else 0) + (1 / (k + dense_rank.get(id(hit), len(hits) + 1)) if id(hit) in dense_rank else 0) + 0.15 * hit.metadata_score + 0.25 * hit.table_score

    def hybrid_search(self, query: str, qa_type: str, top_k: int = 8, filters: dict[str, Any] | None = None) -> list[Hit]:
        text_hits = self.search_text(query, top_k * 2, filters)
        table_hits = self.search_tables(query, top_k * 2, filters) if qa_type in {"table_lookup", "cross_file_judgment"} or not self.text else []
        combined: dict[str, Hit] = {}
        for hit in text_hits + table_hits:
            previous = combined.get(hit.evidence_id)
            if not previous or hit.fused_score > previous.fused_score:
                combined[hit.evidence_id] = hit
        candidates = sorted(combined.values(), key=lambda hit: hit.fused_score, reverse=True)
        if self.semantic and candidates:
            rerank_limit = min(len(candidates), max(top_k * 2, self.semantic.config.rerank_top_k))
            rerank_candidates = candidates[:rerank_limit]
            scores = self.semantic.rerank(query, [self._hit_blob(hit) for hit in rerank_candidates])
            if scores is not None:
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
    """Normalize labels whose Excel cells contain layout decoration."""
    return canonical_table_label(value)


def _metadata_key(value: Any) -> str:
    """Normalize harmless punctuation/layout differences in source hints."""
    return re.sub(r"[^\w\u4e00-\u9fff]|_", "", normalize_text(value).lower())


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
