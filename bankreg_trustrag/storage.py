from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, authority TEXT, document_no TEXT,
  publish_date TEXT, effective_date TEXT, expire_date TEXT, document_type TEXT,
  topic_json TEXT, version TEXT, status TEXT, source_url TEXT, local_path TEXT,
  sha256 TEXT NOT NULL, file_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS text_evidence (
  evidence_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, content TEXT NOT NULL,
  page INTEGER, chapter TEXT, article_no TEXT, paragraph_no INTEGER, section TEXT,
  source_url TEXT, source_location TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);
CREATE TABLE IF NOT EXISTS table_evidence (
  evidence_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, sheet_name TEXT NOT NULL,
  table_name TEXT, indicator TEXT, period TEXT, value_text TEXT, unit TEXT,
  row_header TEXT, column_header TEXT, cell_address TEXT NOT NULL, context TEXT,
  source_url TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);
CREATE TABLE IF NOT EXISTS qa_records (
  trace_id TEXT PRIMARY KEY, question TEXT NOT NULL, qa_type TEXT NOT NULL,
  query_plan_json TEXT, evidence_ids_json TEXT, answer TEXT, confidence REAL,
  verification_json TEXT, decision TEXT, latency_ms INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS document_relations (
  source_doc_id TEXT NOT NULL, target_doc_id TEXT NOT NULL, relation_type TEXT NOT NULL,
  confidence REAL NOT NULL, rationale TEXT, PRIMARY KEY(source_doc_id, target_doc_id, relation_type),
  FOREIGN KEY(source_doc_id) REFERENCES documents(doc_id),
  FOREIGN KEY(target_doc_id) REFERENCES documents(doc_id)
);
CREATE INDEX IF NOT EXISTS idx_text_doc ON text_evidence(doc_id);
CREATE INDEX IF NOT EXISTS idx_table_doc ON table_evidence(doc_id);
CREATE INDEX IF NOT EXISTS idx_table_indicator ON table_evidence(indicator);
CREATE INDEX IF NOT EXISTS idx_table_period ON table_evidence(period);
CREATE INDEX IF NOT EXISTS idx_relation_source ON document_relations(source_doc_id);
CREATE INDEX IF NOT EXISTS idx_relation_target ON document_relations(target_doc_id);
"""


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def reset_content(self) -> None:
        self.connection.executescript("DELETE FROM document_relations; DELETE FROM text_evidence; DELETE FROM table_evidence; DELETE FROM documents;")
        self.connection.commit()

    def load_jsonl(self, artifact_dir: Path) -> None:
        self.reset_content()
        with self.connection:
            for record in _read_jsonl(artifact_dir / "documents.jsonl"):
                self.connection.execute(
                    "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record["doc_id"], record["title"], record.get("authority"), record.get("document_no"),
                        record.get("publish_date"), record.get("effective_date"), record.get("expire_date"),
                        record.get("document_type"), json.dumps(record.get("topic", []), ensure_ascii=False),
                        record.get("version"), record.get("status"), record.get("source_url"), record.get("local_path"),
                        record["sha256"], record["file_name"],
                    ),
                )
            for record in _read_jsonl(artifact_dir / "text_evidence.jsonl"):
                self.connection.execute(
                    "INSERT OR REPLACE INTO text_evidence VALUES (?,?,?,?,?,?,?,?,?,?)",
                    tuple(record.get(key) for key in ["evidence_id", "doc_id", "content", "page", "chapter", "article_no", "paragraph_no", "section", "source_url", "source_location"]),
                )
            for record in _read_jsonl(artifact_dir / "table_evidence.jsonl"):
                self.connection.execute(
                    "INSERT OR REPLACE INTO table_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record.get("evidence_id"), record.get("doc_id"), record.get("sheet_name"), record.get("table_name"),
                        record.get("indicator"), record.get("period"), json.dumps(record.get("value"), ensure_ascii=False),
                        record.get("unit"), record.get("row_header"), record.get("column_header"), record.get("cell_address"),
                        record.get("context"), record.get("source_url"),
                    ),
                )
            for record in _read_jsonl(artifact_dir / "document_relations.jsonl"):
                self.connection.execute(
                    "INSERT OR REPLACE INTO document_relations(source_doc_id,target_doc_id,relation_type,confidence,rationale) VALUES (?,?,?,?,?)",
                    (record["source_doc_id"], record["target_doc_id"], record["relation_type"], record.get("confidence", 1.0), record.get("rationale")),
                )

    def document_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def text_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM text_evidence").fetchone()[0])

    def table_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM table_evidence").fetchone()[0])

    def all_documents(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM documents"))

    def all_text(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM text_evidence"))

    def all_tables(self) -> list[sqlite3.Row]:
        """Return all table rows for small/offline callers.

        The production retrieval path must not call this method for the
        contest corpus: it contains more than one million cells.  It remains
        available for compatibility with small in-memory-style fixtures and
        export tooling.
        """
        return list(self.connection.execute("SELECT * FROM table_evidence"))

    def table_candidates(
        self,
        doc_ids: list[str] | None = None,
        *,
        indicator: str | None = None,
        periods: list[str] | None = None,
        row_label: str | None = None,
        column_label: str | None = None,
        value_terms: list[str] | None = None,
        text_terms: list[str] | None = None,
        limit: int = 8000,
    ) -> list[sqlite3.Row]:
        """Fetch a bounded table-evidence candidate set from SQLite.

        Table evidence is intentionally queried lazily.  Exact indicator and
        period predicates use the existing SQLite indexes; text/value terms
        are only a recall fallback and are still bounded before Python/BGE
        scoring.  This keeps startup memory proportional to the query rather
        than to the entire corpus.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if doc_ids is not None:
            if not doc_ids:
                return []
            clauses.append("doc_id IN (" + ",".join("?" for _ in doc_ids) + ")")
            params.extend(str(value) for value in doc_ids)
        if indicator:
            clauses.append("(indicator = ? OR row_header = ?)")
            params.extend([indicator, indicator])
        if periods:
            period_values = [str(value) for value in periods if value]
            if period_values:
                clauses.append("period IN (" + ",".join("?" for _ in period_values) + ")")
                params.extend(period_values)
        if row_label:
            clauses.append("(row_header = ? OR indicator = ?)")
            params.extend([row_label, row_label])
        if column_label:
            clauses.append("(column_header LIKE ? OR context LIKE ?)")
            wildcard = f"%{column_label}%"
            params.extend([wildcard, wildcard])
        for field, terms in (("value_text", value_terms), ("context", text_terms)):
            clean_terms = [str(term).strip() for term in (terms or []) if str(term).strip()]
            if clean_terms:
                clauses.append("(" + " OR ".join(f"{field} LIKE ?" for _ in clean_terms) + ")")
                params.extend(f"%{term}%" for term in clean_terms)
        where = " AND ".join(clauses) if clauses else "1=1"
        safe_limit = max(1, min(int(limit), 20000))
        return list(self.connection.execute(
            f"SELECT * FROM table_evidence WHERE {where} LIMIT ?",
            (*params, safe_limit),
        ))

    def get_document(self, doc_id: str) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()

    def list_documents(
        self,
        *,
        query: str | None = None,
        document_type: str | None = None,
        status: str | None = None,
        limit: int = 30,
        offset: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Return a bounded, public-safe document catalogue page."""
        clauses: list[str] = []
        params: list[Any] = []
        if query and query.strip():
            wildcard = f"%{query.strip()}%"
            clauses.append("(title LIKE ? OR file_name LIKE ? OR authority LIKE ? OR document_no LIKE ?)")
            params.extend([wildcard, wildcard, wildcard, wildcard])
        if document_type:
            clauses.append("document_type = ?")
            params.append(document_type)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = " AND ".join(clauses) if clauses else "1=1"
        total = int(self.connection.execute(f"SELECT COUNT(*) FROM documents WHERE {where}", params).fetchone()[0])
        safe_limit = max(1, min(int(limit), 100))
        safe_offset = max(0, int(offset))
        rows = self.connection.execute(
            f"""SELECT doc_id, title, authority, document_no, publish_date, effective_date,
                       expire_date, document_type, version, status, file_name
                FROM documents WHERE {where}
                ORDER BY COALESCE(publish_date, '') DESC, title, file_name
                LIMIT ? OFFSET ?""",
            (*params, safe_limit, safe_offset),
        ).fetchall()
        return total, [dict(row) for row in rows]

    def document_relations(self, doc_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT relation_type, confidence, rationale, source_doc_id, target_doc_id,
                      CASE WHEN source_doc_id=? THEN target_doc_id ELSE source_doc_id END AS related_doc_id,
                      d.title AS related_title, d.file_name AS related_file_name, d.status AS related_status
               FROM document_relations r
               JOIN documents d ON d.doc_id = CASE WHEN r.source_doc_id=? THEN r.target_doc_id ELSE r.source_doc_id END
               WHERE r.source_doc_id=? OR r.target_doc_id=?
               ORDER BY confidence DESC, relation_type, related_title""",
            (doc_id, doc_id, doc_id, doc_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def load_relations(self, records: Iterable[dict[str, Any]]) -> int:
        rows = list(records)
        with self.connection:
            self.connection.execute("DELETE FROM document_relations")
            self.connection.executemany(
                "INSERT OR REPLACE INTO document_relations(source_doc_id,target_doc_id,relation_type,confidence,rationale) VALUES (?,?,?,?,?)",
                [
                    (row["source_doc_id"], row["target_doc_id"], row["relation_type"], row.get("confidence", 1.0), row.get("rationale"))
                    for row in rows
                ],
            )
        return len(rows)

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT e.*, d.title AS source_title, d.file_name AS source_file_name, d.local_path AS source_local_path, d.status AS document_status FROM text_evidence e JOIN documents d ON d.doc_id=e.doc_id WHERE e.evidence_id=?",
            (evidence_id,),
        ).fetchone()
        if row:
            return {"kind": "text", **dict(row)}
        row = self.connection.execute(
            "SELECT e.*, d.title AS source_title, d.file_name AS source_file_name, d.local_path AS source_local_path, d.status AS document_status FROM table_evidence e JOIN documents d ON d.doc_id=e.doc_id WHERE e.evidence_id=?",
            (evidence_id,),
        ).fetchone()
        if row:
            result = {"kind": "table", **dict(row)}
            try:
                result["value"] = json.loads(result.pop("value_text"))
            except (TypeError, json.JSONDecodeError):
                result["value"] = result.pop("value_text", None)
            return result
        return None

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM qa_records ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def save_qa(self, trace_id: str, question: str, qa_type: str, query_plan: dict[str, Any], evidence_ids: list[str], answer: str, confidence: float, verification: dict[str, Any], decision: str, latency_ms: int) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO qa_records(trace_id,question,qa_type,query_plan_json,evidence_ids_json,answer,confidence,verification_json,decision,latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (trace_id, question, qa_type, json.dumps(query_plan, ensure_ascii=False), json.dumps(evidence_ids, ensure_ascii=False), answer, confidence, json.dumps(verification, ensure_ascii=False), decision, latency_ms),
            )


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    def generator():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    return generator()
