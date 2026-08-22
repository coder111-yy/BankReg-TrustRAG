from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import Settings
from .service import TrustRAGService


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    choices: list[str] | None = None
    qa_type: str | None = None
    filters: dict[str, Any] | None = None


def create_app(settings: Settings | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse

    settings = settings or Settings.from_env(Path.cwd())
    service = TrustRAGService(settings)
    app = FastAPI(title="BankReg-TrustRAG", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "documents": service.store.document_count(), "text_evidence": service.store.text_count(), "table_evidence": service.store.table_count(), "bge": service.index.model_status}

    @app.post("/api/qa")
    def qa(request: QARequest) -> dict[str, Any]:
        return service.ask(request.question, request.choices, request.qa_type, request.filters).to_dict()

    @app.post("/api/documents/ingest")
    def ingest():
        from .ingestion.manifest import build_manifest
        summary = build_manifest(settings.data_dir, settings.artifact_dir)
        service.reload()
        return summary

    @app.get("/api/documents")
    def documents(
        query: str | None = None,
        document_type: str | None = None,
        status: str | None = None,
        limit: int = 30,
        offset: int = 0,
    ) -> dict[str, Any]:
        safe_limit = max(1, min(limit, 100))
        safe_offset = max(0, offset)
        total, items = service.store.list_documents(
            query=query,
            document_type=document_type,
            status=status,
            limit=safe_limit,
            offset=safe_offset,
        )
        return {"total": total, "limit": safe_limit, "offset": safe_offset, "items": items}

    @app.get("/api/documents/{doc_id}")
    def document(doc_id: str):
        row = service.store.get_document(doc_id)
        if not row:
            raise HTTPException(status_code=404, detail="document not found")
        return dict(row)

    @app.get("/api/documents/{doc_id}/source")
    def document_source(doc_id: str) -> FileResponse:
        """Return an original source file only when its stored path is safe.

        The endpoint deliberately accepts a document identifier rather than a
        filesystem path.  Imported manifests contain paths relative to the
        configured dataset directory; resolving and re-checking that boundary
        prevents malformed database rows from exposing arbitrary local files.
        """
        row = service.store.get_document(doc_id)
        if not row:
            raise HTTPException(status_code=404, detail="document not found")
        local_path = row["local_path"]
        if not local_path:
            raise HTTPException(status_code=404, detail="source file unavailable")
        source_root = settings.data_dir.resolve()
        stored_path = Path(str(local_path))
        if stored_path.is_absolute():
            raise HTTPException(status_code=404, detail="source file unavailable")
        source_path = (source_root / stored_path).resolve()
        try:
            source_path.relative_to(source_root)
        except ValueError:
            raise HTTPException(status_code=404, detail="source file unavailable") from None
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="source file unavailable")
        return FileResponse(source_path, filename=str(row["file_name"]))

    @app.get("/api/documents/{doc_id}/relations")
    def document_relations(doc_id: str) -> list[dict[str, Any]]:
        if not service.store.get_document(doc_id):
            raise HTTPException(status_code=404, detail="document not found")
        return service.store.document_relations(doc_id)

    @app.get("/api/evidence/{evidence_id:path}")
    def evidence(evidence_id: str):
        row = service.store.get_evidence(evidence_id)
        if not row:
            raise HTTPException(status_code=404, detail="evidence not found")
        return row

    @app.get("/api/history")
    def history(limit: int = 50):
        return service.store.history(max(1, min(limit, 200)))

    frontend_dir = Path.cwd() / "frontend"
    if frontend_dir.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app
