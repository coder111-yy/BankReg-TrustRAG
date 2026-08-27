from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import Query
from pydantic import BaseModel, Field

from .config import Settings
from .service import TrustRAGService


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    choices: list[str] | None = None
    qa_type: str | None = None
    filters: dict[str, Any] | None = None


class ConversationCreateRequest(BaseModel):
    memory_scope_id: str = Field(min_length=8, max_length=128)
    title: str | None = Field(default=None, max_length=80)


class ChatRequest(QARequest):
    memory_scope_id: str = Field(min_length=8, max_length=128)
    conversation_id: str | None = Field(default=None, min_length=8, max_length=80)


def create_app(settings: Settings | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse

    settings = settings or Settings.from_env(Path.cwd())
    service = TrustRAGService(settings)
    app = FastAPI(title="BankReg-TrustRAG", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "documents": service.store.document_count(), "text_evidence": service.store.text_count(), "table_evidence": service.store.table_count(), "bge": service.index.model_status}

    @app.post("/api/qa")
    def qa(request: QARequest) -> dict[str, Any]:
        return service.ask(request.question, request.choices, request.qa_type, request.filters).to_dict()

    @app.post("/api/conversations")
    def create_conversation(request: ConversationCreateRequest) -> dict[str, Any]:
        scope = _safe_memory_scope(request.memory_scope_id)
        conversation = service.store.create_conversation(
            "conv_" + uuid.uuid4().hex,
            scope,
            request.title or "新对话",
        )
        return conversation

    @app.get("/api/conversations")
    def conversations(memory_scope_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return service.store.list_conversations(_safe_memory_scope(memory_scope_id), limit)

    @app.get("/api/conversations/{conversation_id}/messages")
    def conversation_messages(conversation_id: str, memory_scope_id: str, limit: int = 100) -> list[dict[str, Any]]:
        scope = _safe_memory_scope(memory_scope_id)
        if not service.store.get_conversation(conversation_id, scope):
            raise HTTPException(status_code=404, detail="conversation not found")
        return service.store.conversation_messages(conversation_id, scope, limit)

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    def delete_conversation(
        conversation_id: str,
        memory_scope_id: Annotated[str, Query(min_length=8, max_length=128)],
    ) -> None:
        scope = _safe_memory_scope(memory_scope_id)
        if not service.store.delete_conversation(conversation_id, scope):
            raise HTTPException(status_code=404, detail="conversation not found")

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest):
        """Stream public workflow status plus a verified answer to the chat UI.

        Retrieval and verification run in a worker thread because the existing
        RAG pipeline is intentionally synchronous.  The queue bridges real
        pipeline milestones to SSE without exposing hidden chain-of-thought.
        """
        scope = _safe_memory_scope(request.memory_scope_id)
        conversation_id = request.conversation_id
        if conversation_id and not service.store.get_conversation(conversation_id, scope):
            raise HTTPException(status_code=404, detail="conversation not found")
        if not conversation_id:
            conversation_id = "conv_" + uuid.uuid4().hex
            service.store.create_conversation(conversation_id, scope)
        short_term = service.store.recent_conversation_messages(conversation_id, scope, limit=8)
        recalled = service.store.recall_memories(scope, request.question, exclude_conversation_id=conversation_id, limit=4)
        context_messages = [*short_term]
        if _looks_like_follow_up(request.question):
            context_messages.extend({"role": "user", "content": item["question"], "memory_type": "long_term"} for item in recalled)
        service.store.add_conversation_message(
            "msg_" + uuid.uuid4().hex,
            conversation_id,
            scope,
            "user",
            request.question,
        )
        conversation = service.store.get_conversation(conversation_id, scope) or {}
        if conversation.get("title") == "新对话":
            service.store.update_conversation_title(conversation_id, scope, _conversation_title(request.question))

        async def event_stream():
            loop = asyncio.get_running_loop()
            progress: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

            def observe(stage: str, details: dict[str, Any]) -> None:
                loop.call_soon_threadsafe(progress.put_nowait, {"stage": stage, **details})

            yield _sse_event("conversation", {"conversation_id": conversation_id})
            yield _sse_event(
                "memory",
                {
                    "short_term_messages": len(short_term),
                    "long_term_matches": len(recalled),
                    "long_term_questions": [item["question"][:120] for item in recalled],
                    "note": "记忆仅用于理解上下文，不能替代本地监管资料证据。",
                },
            )
            task = asyncio.create_task(asyncio.to_thread(
                service.ask,
                request.question,
                request.choices,
                request.qa_type,
                request.filters,
                context_messages,
                observe,
            ))
            while not task.done():
                try:
                    update = await asyncio.wait_for(progress.get(), timeout=0.12)
                    yield _sse_event("status", update)
                except TimeoutError:
                    continue
            while not progress.empty():
                yield _sse_event("status", progress.get_nowait())
            try:
                response = await task
            except Exception:
                yield _sse_event("error", {"message": "本次问答未完成，请稍后重试。"})
                return
            response_data = response.to_dict()
            service.store.add_conversation_message(
                "msg_" + uuid.uuid4().hex,
                conversation_id,
                scope,
                "assistant",
                response.answer,
                trace_id=response.trace_id,
                metadata={
                    "qa_type": response.qa_type,
                    "decision": response.trust.get("decision"),
                    "trust_score": response.trust.get("score"),
                    "latency_ms": response.latency_ms,
                    "evidence_ids": [item.get("evidence_id") for item in response.evidence],
                },
            )
            service.store.remember_answer(
                "mem_" + uuid.uuid4().hex,
                scope,
                conversation_id,
                request.question,
                response.answer,
                response.qa_type,
                str(response.trust.get("decision") or ""),
                [str(item.get("evidence_id")) for item in response.evidence if item.get("evidence_id")],
            )
            # The upstream model response is fully verified before it reaches
            # this loop.  Chunking the accepted text keeps the UI responsive
            # without showing unverified model tokens.
            for chunk in _answer_chunks(response.answer):
                yield _sse_event("answer_delta", {"text": chunk})
                await asyncio.sleep(0.008)
            yield _sse_event("complete", {"response": response_data})

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

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


def _safe_memory_scope(value: str) -> str:
    scope = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", scope):
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="invalid memory scope")
    return scope


def _looks_like_follow_up(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return len(compact) <= 80 and any(token in compact for token in ("这个", "那个", "上述", "刚才", "前面", "继续", "再", "那", "它"))


def _conversation_title(question: str) -> str:
    compact = re.sub(r"\s+", " ", question).strip()
    return (compact[:38] + "…") if len(compact) > 39 else compact


def _answer_chunks(answer: str, chunk_size: int = 24):
    text = str(answer or "")
    for index in range(0, len(text), chunk_size):
        yield text[index:index + chunk_size]


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Encode a compact SSE frame without requiring a server-specific add-on."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
