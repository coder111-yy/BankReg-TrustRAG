from fastapi.testclient import TestClient

from bankreg_trustrag.api import create_app
from bankreg_trustrag.config import Settings
from bankreg_trustrag.storage import Store


def test_qa_endpoint_accepts_json_request_body(tmp_path):
    app = create_app()
    response = TestClient(app).post("/api/qa", json={"question": "2023年10月的指标是多少"})

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["trust"]["score"], (int, float))
    assert "answer" in payload


def test_evidence_endpoint_includes_original_source_location(tmp_path):
    store = Store(tmp_path / "bankreg.sqlite3")
    with store.connection:
        store.connection.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("d1", "监管办法", None, None, None, None, None, "word", "[]", None, "effective", None, "source/rule.docx", "hash", "rule.docx"))
        store.connection.execute("INSERT INTO text_evidence VALUES (?,?,?,?,?,?,?,?,?,?)", ("text:d1:p1", "d1", "监管要求。", None, None, None, 1, None, None, "rule.docx:paragraph:1"))
    evidence = store.get_evidence("text:d1:p1")

    assert evidence["source_file_name"] == "rule.docx"
    assert evidence["source_local_path"] == "source/rule.docx"


def test_document_relation_api_returns_not_found_for_unknown_document():
    response = TestClient(create_app()).get("/api/documents/unknown/relations")

    assert response.status_code == 404


def _source_app(tmp_path, local_path: str):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings = Settings(
        data_dir=data_dir,
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "bankreg.sqlite3",
        bge_mode="disabled",
        bge_vector_dir=tmp_path / "vectors",
    )
    store = Store(settings.db_path)
    with store.connection:
        store.connection.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("d1", "监管办法", None, None, None, None, None, "word", "[]", None, "effective", None, local_path, "hash", "rule.docx"),
        )
    store.close()
    return create_app(settings), data_dir


def test_document_source_api_returns_stored_file_inside_dataset(tmp_path):
    app, data_dir = _source_app(tmp_path, "source/rule.docx")
    source_file = data_dir / "source" / "rule.docx"
    source_file.parent.mkdir()
    source_file.write_bytes(b"local regulatory source")

    response = TestClient(app).get("/api/documents/d1/source")

    assert response.status_code == 200
    assert response.content == b"local regulatory source"
    assert 'filename="rule.docx"' in response.headers["content-disposition"]


def test_document_source_api_returns_not_found_for_missing_document(tmp_path):
    app, _ = _source_app(tmp_path, "source/rule.docx")

    response = TestClient(app).get("/api/documents/unknown/source")

    assert response.status_code == 404


def test_document_source_api_rejects_path_outside_dataset(tmp_path):
    app, _ = _source_app(tmp_path, "../outside.docx")
    (tmp_path / "outside.docx").write_bytes(b"must not be served")

    response = TestClient(app).get("/api/documents/d1/source")

    assert response.status_code == 404


def test_document_catalog_api_filters_and_pages_results(tmp_path):
    app, _ = _source_app(tmp_path, "source/rule.docx")
    store = Store(tmp_path / "bankreg.sqlite3")
    with store.connection:
        store.connection.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("d2", "统计报表", "国家金融监督管理总局", None, "2025-01-01", None, None, "excel", "[]", None, "effective", None, "source/report.xlsx", "hash2", "report.xlsx"),
        )
    store.close()

    response = TestClient(app).get("/api/documents", params={"query": "报表", "document_type": "excel", "limit": 1})
    payload = response.json()

    assert response.status_code == 200
    assert payload["total"] == 1
    assert payload["items"] == [{
        "doc_id": "d2", "title": "统计报表", "authority": "国家金融监督管理总局", "document_no": None,
        "publish_date": "2025-01-01", "effective_date": None, "expire_date": None, "document_type": "excel",
        "version": None, "status": "effective", "file_name": "report.xlsx",
    }]
