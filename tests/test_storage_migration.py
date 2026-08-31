import sqlite3

from bankreg_trustrag.storage import Store


def test_store_migrates_legacy_table_evidence_before_creating_indexes(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE table_evidence (
          evidence_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,
          sheet_name TEXT NOT NULL, table_name TEXT, indicator TEXT,
          period TEXT, value_text TEXT, unit TEXT, row_header TEXT,
          column_header TEXT, cell_address TEXT NOT NULL, context TEXT,
          source_url TEXT
        );
        CREATE INDEX idx_table_period ON table_evidence(period);
        """
    )
    connection.close()

    store = Store(db_path)
    try:
        columns = {
            row[1]
            for row in store.connection.execute("PRAGMA table_info(table_evidence)")
        }
        assert "cell_type" in columns
        assert "numeric_value" in columns
        assert store.connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_table_cell_type'"
        ).fetchone()
    finally:
        store.close()
