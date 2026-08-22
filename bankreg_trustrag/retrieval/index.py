from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from ..schemas import TableCellEvidence, TextEvidence
from ..query import extract_dimension_labels, extract_indicator
from ..utils import char_ngrams, normalize_text, normalized_number, tokens
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
        table_store: Any | None = None,
    ):
        self.documents = list(documents)
        self.text = list(text)
        self.tables = list(tables)
        self.table_store = table_store
        self.semantic = semantic
        self._text_vector_index = (
            PersistentVectorIndex(semantic, vector_dir, "text")
            if semantic is not None and vector_dir is not None
            else None
        )
        self._text_vectors_ready = False
        self.doc_by_id = {str(d["doc_id"]): d for d in self.documents}
        self._text_tokens = [tokens(self._text_blob(item)) for item in self.text]
        self._table_tokens = [tokens(self._table_blob(item)) for item in self.tables]
        self._text_df = self._df(self._text_tokens)
        self._table_df = self._df(self._table_tokens)
        self._text_avg_len = sum(map(len, self._text_tokens)) / max(len(self._text_tokens), 1)
        self._table_avg_len = sum(map(len, self._table_tokens)) / max(len(self._table_tokens), 1)
        self._text_indices_by_doc: dict[str, list[int]] = defaultdict(list)
        self._table_indices_by_doc: dict[str, list[int]] = defaultdict(list)
        self._table_indices_by_period: dict[str, list[int]] = defaultdict(list)
        self._formula_indices: list[int] = []
        for index, item in enumerate(self.text):
            self._text_indices_by_doc[str(item.get("doc_id"))].append(index)
        for index, item in enumerate(self.tables):
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
            # The corpus contains over one million table cells.  Keep table
            # evidence in SQLite and retrieve a bounded candidate set per
            # query instead of materialising the whole table in RAM.
            [],
            semantic=semantic,
            vector_dir=vector_dir,
            table_store=store,
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
        # Keep the persisted vector corpus stable. ``context_window`` is added
        # only after ranking for claim verification and must not invalidate a
        # previously built embedding index.
        fields = ["content", "chapter", "article_no", "section"]
        if item.get("context_window"):
            fields.insert(1, "context_window")
        return " ".join(str(item.get(key) or "") for key in fields)

    @staticmethod
    def _metadata_value(value: Any) -> str:
        """Normalise generated filenames and human-written document titles alike."""
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", normalize_text(value).lower())

    @classmethod
    def _metadata_contains(cls, expected: Any, haystack: str) -> bool:
        value = cls._metadata_value(expected)
        return bool(value and value in cls._metadata_value(haystack))

    @staticmethod
    def _table_blob(item: dict[str, Any]) -> str:
        return " ".join(str(item.get(k) or "") for k in ["table_name", "indicator", "period", "unit", "row_header", "column_header", "context", "value_text"])

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
        haystack = normalize_text(" ".join(str(doc.get(k) or "") for k in ["title", "file_name", "local_path", "authority", "document_type"])).lower()
        for key, expected in filters.items():
            if expected in (None, "", []):
                continue
            values = expected if isinstance(expected, list) else [expected]
            if any(self._metadata_contains(v, haystack) for v in values):
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
            haystack = normalize_text(" ".join(str(doc.get(k) or "") for k in ["title", "file_name", "local_path", "authority", "document_type"])).lower()
            matched = True
            for expected in filters.values():
                if expected in (None, "", []):
                    continue
                values = expected if isinstance(expected, list) else [expected]
                if not any(self._metadata_contains(value, haystack) for value in values):
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

    def search_text(
        self,
        query: str,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
        *,
        dense: bool = True,
    ) -> list[Hit]:
        q_tokens = tokens(query)
        candidate_indices = self._candidate_indices("text", filters)
        allowed_ids = {str(self.text[index].get("evidence_id")) for index in candidate_indices}
        vector_scores: dict[str, float] = {}
        if dense and self.semantic and self._ensure_text_vectors() and self._text_vector_index:
            vector_scores = dict(self._text_vector_index.search(query, max(top_k * 4, 32), allowed_ids))
        hits: list[Hit] = []
        for index in candidate_indices:
            item = self.text[index]
            lexical = self._bm25(q_tokens, self._text_tokens, self._text_df, self._text_avg_len, index)
            dense_score = vector_scores.get(str(item.get("evidence_id")), self._dense(query, self._text_blob(item))) if dense and self.semantic else self._dense(query, self._text_blob(item))
            metadata = self._metadata(item, query, filters)
            if lexical or dense_score or metadata:
                hits.append(Hit("text", item, lexical, dense_score, metadata_score=metadata))
        self._rrf(hits)
        ranked = sorted(hits, key=lambda hit: hit.fused_score, reverse=True)[:top_k]
        return [self._with_text_context(hit) for hit in ranked]

    def _with_text_context(self, hit: Hit) -> Hit:
        """Attach neighbouring clauses without changing the cited evidence unit.

        Regulatory lists commonly put a heading in one paragraph and the
        predicate in the next.  The original paragraph remains the evidence
        unit, while the bounded context window lets support checks read that
        syntactic unit as a whole.
        """
        if hit.kind != "text":
            return hit
        item = hit.item
        paragraph_no = item.get("paragraph_no")
        doc_id = str(item.get("doc_id") or "")
        if not isinstance(paragraph_no, int) or not doc_id:
            return hit
        nearby = [
            candidate for candidate in self.text
            if str(candidate.get("doc_id") or "") == doc_id
            and isinstance(candidate.get("paragraph_no"), int)
            and abs(candidate["paragraph_no"] - paragraph_no) <= 1
        ]
        if len(nearby) <= 1:
            return hit
        enriched = dict(item)
        enriched["context_window"] = " ".join(str(candidate.get("content") or "") for candidate in sorted(nearby, key=lambda candidate: candidate["paragraph_no"]))
        return Hit(hit.kind, enriched, hit.lexical_score, hit.dense_score, hit.metadata_score, hit.table_score, hit.fused_score, hit.rerank_score)

    def search_tables(self, query: str, top_k: int = 8, filters: dict[str, Any] | None = None) -> list[Hit]:
        if self.table_store is not None:
            return self._search_tables_lazy(query, top_k, filters)
        q_tokens = tokens(query)
        candidate_indices = self._candidate_indices("table", filters)
        requested_indicator = extract_indicator(query)
        normalized_indicator = normalize_text(requested_indicator).lower() if requested_indicator else None
        requested_row, requested_column = extract_dimension_labels(query)
        normalized_row = _canonical_label(requested_row)
        normalized_column = _canonical_label(requested_column)
        # Period is a high-selectivity structured key. Restricting by it before
        # scoring avoids repeatedly computing text similarity over the full
        # million-cell corpus for ordinary period-specific questions.
        period_match = re.search(r"(20\d{2})年\s*0?(\d{1,2})月", normalize_text(query))
        requested_quarter: str | None = None
        if period_match:
            year = int(period_match.group(1))
            month = int(period_match.group(2))
            requested_period = f"{year:04d}-{month:02d}"
            candidate_set = set(candidate_indices)
            period_indices = candidate_set.intersection(self._table_indices_by_period.get(requested_period, []))
            if period_indices:
                candidate_indices = [index for index in candidate_indices if index in period_indices]
            else:
                # Annual regulatory workbooks store the year in ``period`` and
                # the reporting quarter in ``column_header``.  March/June/
                # September/December therefore need a structured month-to-
                # quarter mapping instead of being treated as an unmatched
                # monthly report.
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
            # Annual regulatory workbooks use ``period=YYYY`` while monthly
            # workbooks use ``period=YYYY-MM``.  A year-only question must be
            # narrowed before ranking; otherwise a same-indicator row from a
            # different year can win on lexical/BGE similarity.
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
        if normalized_column:
            column_indices = [
                index for index in candidate_indices
                if normalized_column in _canonical_label(" ".join(
                    str(self.tables[index].get(key) or "")
                    for key in ["column_header", "period"]
                ))
            ]
            if column_indices:
                candidate_indices = column_indices
        query_numbers = [normalized_number(x) for x in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", normalize_text(query))]
        query_numbers = [x for x in query_numbers if x is not None]
        hits: list[Hit] = []
        for index in candidate_indices:
            item = self.tables[index]
            lexical = self._bm25(q_tokens, self._table_tokens, self._table_df, self._table_avg_len, index)
            dense = self._dense(query, self._table_blob(item))
            metadata = self._metadata(item, query, filters)
            value = normalized_number(item.get("value_text"))
            table_score = 0.5 if query_numbers and value is not None and any(abs(value - q) < 1e-9 for q in query_numbers) else 0.0
            item_indicator = normalize_text(item.get("indicator")).lower()
            if normalized_indicator and item_indicator == normalized_indicator:
                # Exact indicator matching is stronger than a semantic match
                # to a neighbouring row such as ``正常类贷款占比``.
                table_score += 2.0
            if normalized_row and normalized_row in {
                _canonical_label(item.get("indicator")),
                _canonical_label(item.get("row_header")),
            }:
                table_score += 3.0
            if normalized_column and normalized_column in _canonical_label(" ".join(
                str(item.get(key) or "") for key in ["column_header", "period"]
            )):
                table_score += 3.0
            if requested_quarter:
                column_context = normalize_text(" ".join(str(item.get(key) or "") for key in ["column_header", "context"]))
                if requested_quarter in column_context:
                    table_score += 2.0
            if lexical or dense or metadata or table_score:
                hits.append(Hit("table", item, lexical, dense, metadata_score=metadata, table_score=table_score))
        # Tables are first narrowed by exact indicator/period structure. BGE
        # then scores a bounded shortlist so a million-cell corpus does not
        # trigger a million model encodings for every request.
        if self.semantic and hits:
            shortlist = sorted(
                hits,
                key=lambda hit: (hit.table_score, hit.lexical_score, hit.metadata_score, hit.dense_score),
                reverse=True,
            )[: max(top_k * 8, 64)]
            scores = self.semantic.similarity(query, [self._table_blob(hit.item) for hit in shortlist])
            if scores is not None:
                for hit, score in zip(shortlist, scores):
                    hit.dense_score = score
        self._rrf(hits)
        return sorted(hits, key=lambda hit: (hit.table_score, hit.fused_score, hit.lexical_score, hit.dense_score), reverse=True)[:top_k]

    def _matching_doc_ids(self, filters: dict[str, Any] | None) -> list[str]:
        """Apply the same all-filter semantics as the in-memory index."""
        result: list[str] = []
        for document in self.documents:
            if not filters and self._is_benchmark_document(document):
                continue
            haystack = normalize_text(" ".join(str(document.get(key) or "") for key in [
                "title", "file_name", "local_path", "authority", "document_type",
            ])).lower()
            matched = True
            for expected in (filters or {}).values():
                if expected in (None, "", []):
                    continue
                values = expected if isinstance(expected, list) else [expected]
                if not any(self._metadata_contains(value, haystack) for value in values):
                    matched = False
                    break
            if matched:
                result.append(str(document["doc_id"]))
        return result

    @staticmethod
    def _query_phrases(query: str) -> list[str]:
        # Single Chinese characters are too common for SQL LIKE recall.  Use
        # contiguous phrases and let the bounded Python scorer rank them.
        phrases = re.findall(r"[\u4e00-\u9fffA-Za-z0-9%]{2,}", normalize_text(query))
        return list(dict.fromkeys(phrases))[:12]

    def _search_tables_lazy(self, query: str, top_k: int, filters: dict[str, Any] | None) -> list[Hit]:
        """Search SQLite-backed table evidence without loading the corpus."""
        requested_indicator = extract_indicator(query)
        requested_row, requested_column = extract_dimension_labels(query)
        normalized_indicator = normalize_text(requested_indicator).lower() if requested_indicator else None
        normalized_row = _canonical_label(requested_row)
        normalized_column = _canonical_label(requested_column)
        normalized_query = normalize_text(query)
        periods: list[str] = []
        period_match = re.search(r"(20\d{2})年\s*0?(\d{1,2})月", normalized_query)
        requested_quarter: str | None = None
        if period_match:
            year, month = int(period_match.group(1)), int(period_match.group(2))
            periods.append(f"{year:04d}-{month:02d}")
            periods.append(str(year))
            if 1 <= month <= 12:
                requested_quarter = ("一季度", "二季度", "三季度", "四季度")[(month - 1) // 3]
        else:
            year_match = re.search(r"(20\d{2})年", normalized_query)
            if year_match:
                year = year_match.group(1)
                periods.extend([year, f"{year}-%"])
        doc_ids = self._matching_doc_ids(filters)
        # Exact indicator/period constraints are the primary path.  Formula
        # and free-form table questions use a bounded phrase fallback.
        rows = self.table_store.table_candidates(
            doc_ids,
            indicator=requested_indicator,
            periods=[period for period in periods if "%" not in period],
            row_label=requested_row,
            column_label=requested_column,
            text_terms=self._query_phrases(normalized_query) if not requested_indicator else None,
            limit=max(top_k * 16, 256),
        )
        if not rows and requested_indicator:
            rows = self.table_store.table_candidates(
                doc_ids,
                value_terms=[requested_indicator],
                text_terms=self._query_phrases(normalized_query),
                limit=max(top_k * 16, 256),
            )
        q_tokens = tokens(query)
        query_numbers = [normalized_number(x) for x in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", normalized_query)]
        query_numbers = [value for value in query_numbers if value is not None]
        hits: list[Hit] = []
        for row in rows:
            item = dict(row)
            blob = self._table_blob(item)
            item_tokens = tokens(blob)
            overlap = len(set(q_tokens).intersection(item_tokens)) / max(len(set(q_tokens)), 1)
            lexical = overlap
            dense = self._dense(query, blob)
            table_score = 0.0
            item_indicator = normalize_text(item.get("indicator")).lower()
            if normalized_indicator and item_indicator == normalized_indicator:
                table_score += 2.0
            if normalized_row and normalized_row in {
                _canonical_label(item.get("indicator")),
                _canonical_label(item.get("row_header")),
            }:
                table_score += 3.0
            if normalized_column and normalized_column in _canonical_label(" ".join(
                str(item.get(key) or "") for key in ["column_header", "period"]
            )):
                table_score += 3.0
            if requested_quarter and requested_quarter in normalize_text(" ".join(
                str(item.get(key) or "") for key in ["column_header", "context"]
            )):
                table_score += 2.0
            value = normalized_number(item.get("value_text"))
            if query_numbers and value is not None and any(abs(value - number) < 1e-9 for number in query_numbers):
                table_score += 0.5
            if lexical or dense or table_score:
                hits.append(Hit("table", item, lexical, dense, table_score=table_score))
        if self.semantic and hits:
            shortlist = sorted(hits, key=lambda hit: (hit.table_score, hit.lexical_score, hit.dense_score), reverse=True)[:max(top_k * 8, 64)]
            scores = self.semantic.similarity(query, [self._table_blob(hit.item) for hit in shortlist])
            if scores is not None:
                for hit, score in zip(shortlist, scores):
                    hit.dense_score = score
        self._rrf(hits)
        return sorted(hits, key=lambda hit: (hit.table_score, hit.fused_score, hit.lexical_score, hit.dense_score), reverse=True)[:top_k]

    def search_formula_evidence(self, indicator: str, top_k: int = 8, filters: dict[str, Any] | None = None, year: str | None = None) -> list[Hit]:
        """Retrieve formula cells that ordinary table top-k ranking can omit."""
        if self.table_store is not None:
            doc_ids = self._matching_doc_ids(filters)
            rows = self.table_store.table_candidates(
                doc_ids,
                value_terms=["不良贷款余额", "各项贷款余额", "100%"],
                limit=max(top_k * 8, 64),
            )
            hits: list[Hit] = []
            for row in rows:
                item = dict(row)
                source = normalize_text(" ".join(str(self.doc_by_id.get(str(item.get("doc_id")), {}).get(key) or "") for key in ["title", "file_name", "local_path"]))
                if year and year not in source and not normalize_text(item.get("period")).startswith(year):
                    continue
                blob = normalize_text(item.get("value_text")).replace("％", "%")
                if indicator and normalize_text(indicator) != "不良贷款率" and normalize_text(indicator).lower() not in blob.lower():
                    continue
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

    @staticmethod
    def _rrf(hits: list[Hit], k: int = 60) -> None:
        # RRF-like fusion over lexical and dense rank lists; metadata/table bonuses
        # remain visible in the score for auditability.
        lexical_rank = {id(hit): rank for rank, hit in enumerate(sorted(hits, key=lambda x: x.lexical_score, reverse=True), 1) if hit.lexical_score > 0}
        dense_rank = {id(hit): rank for rank, hit in enumerate(sorted(hits, key=lambda x: x.dense_score, reverse=True), 1) if hit.dense_score > 0}
        for hit in hits:
            hit.fused_score = (1 / (k + lexical_rank.get(id(hit), len(hits) + 1)) if id(hit) in lexical_rank else 0) + (1 / (k + dense_rank.get(id(hit), len(hits) + 1)) if id(hit) in dense_rank else 0) + 0.15 * hit.metadata_score + 0.25 * hit.table_score

    def hybrid_search(
        self,
        query: str,
        qa_type: str,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
        *,
        rerank: bool = True,
        dense: bool = True,
    ) -> list[Hit]:
        text_hits = self.search_text(query, top_k * 2, filters, dense=dense)
        table_hits = self.search_tables(query, top_k * 2, filters) if qa_type in {"table_lookup", "cross_file_judgment"} or not self.text else []
        combined: dict[str, Hit] = {}
        for hit in text_hits + table_hits:
            previous = combined.get(hit.evidence_id)
            if not previous or hit.fused_score > previous.fused_score:
                combined[hit.evidence_id] = hit
        candidates = sorted(combined.values(), key=lambda hit: hit.fused_score, reverse=True)
        if rerank and self.semantic and candidates:
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
            "content", "context", "indicator", "period", "unit",
            "row_header", "column_header", "value_text", "chapter", "article_no",
        ])


def _canonical_label(value: Any) -> str:
    """Normalize labels whose Excel cells contain layout spaces."""
    return re.sub(r"\s+", "", normalize_text(value)).lower()
