import sqlite3
from pathlib import Path


DB = Path("artifacts/bankreg.sqlite3")

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

print("=" * 100)
print("1. 查包含“2024 + 商业银行 + 监管指标”的文档")
print("=" * 100)

docs = conn.execute(
    """
    SELECT doc_id, title, file_name, document_type
    FROM documents
    WHERE title LIKE '%2024%'
       OR file_name LIKE '%2024%'
    """
).fetchall()

for row in docs:
    blob = " | ".join(str(row[k] or "") for k in row.keys())
    if "商业银行" in blob or "监管指标" in blob:
        print(dict(row))


print("\n" + "=" * 100)
print("2. 完全不经过 RAG，直接找 2024 / 一季度 / 流动性覆盖率")
print("=" * 100)

rows = conn.execute(
    """
    SELECT
        t.evidence_id,
        t.doc_id,
        d.title,
        t.indicator,
        t.period,
        t.row_header,
        t.column_header,
        t.value_text,
        t.unit,
        t.context
    FROM table_evidence t
    LEFT JOIN documents d
        ON d.doc_id = t.doc_id
    WHERE
        (
            t.indicator LIKE '%流动性覆盖率%'
            OR t.row_header LIKE '%流动性覆盖率%'
            OR t.context LIKE '%流动性覆盖率%'
        )
        AND
        (
            t.period LIKE '%2024%'
            OR d.title LIKE '%2024%'
            OR d.file_name LIKE '%2024%'
        )
    LIMIT 100
    """
).fetchall()

print("候选数量:", len(rows))

for i, row in enumerate(rows, 1):
    print("\n---", i, "---")
    print(dict(row))

conn.close()