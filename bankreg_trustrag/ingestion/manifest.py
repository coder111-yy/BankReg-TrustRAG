from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .parsers import ParseResult, parse_file


SUPPORTED = {".doc", ".docx", ".pdf", ".xls", ".xlsx"}


def iter_source_files(data_dir: Path) -> Iterator[Path]:
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def build_document_relations(documents: list[dict]) -> list[dict]:
    """Emit only relationships directly evidenced by local filenames/metadata."""
    relations: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    by_sha: dict[str, list[dict]] = {}
    by_title = {str(item.get("title") or ""): item for item in documents}
    for document in documents:
        by_sha.setdefault(str(document.get("sha256") or ""), []).append(document)
    for group in by_sha.values():
        if len(group) < 2:
            continue
        primary = group[0]
        for duplicate in group[1:]:
            key = (str(duplicate["doc_id"]), str(primary["doc_id"]), "duplicate_of")
            if key not in seen and key[0] != key[1]:
                seen.add(key)
                relations.append({"source_doc_id": key[0], "target_doc_id": key[1], "relation_type": key[2], "confidence": 1.0, "rationale": "identical_sha256"})
    for document in documents:
        title = str(document.get("title") or "")
        marker = "_附件"
        if marker not in title:
            continue
        parent_title = title.split(marker, 1)[0]
        parent = by_title.get(parent_title)
        if parent and parent["doc_id"] != document["doc_id"]:
            key = (str(document["doc_id"]), str(parent["doc_id"]), "attachment_of")
            if key not in seen:
                seen.add(key)
                relations.append({"source_doc_id": key[0], "target_doc_id": key[1], "relation_type": key[2], "confidence": 1.0, "rationale": "filename_attachment_marker"})
    return relations


def build_manifest(data_dir: Path, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    documents: list[dict] = []
    text_evidence: list[dict] = []
    table_evidence: list[dict] = []
    errors: list[dict] = []
    for path in iter_source_files(data_dir):
        try:
            result: ParseResult = parse_file(path, data_dir)
            documents.append(result.document.to_dict())
            text_evidence.extend(item.to_dict() for item in result.text_evidence)
            table_evidence.extend(item.to_dict() for item in result.table_evidence)
            errors.extend({"local_path": result.document.local_path, "warning": warning} for warning in result.warnings)
        except Exception as exc:
            errors.append({"local_path": str(path), "warning": f"unhandled parse failure: {type(exc).__name__}: {exc}"})
    write_jsonl(artifact_dir / "documents.jsonl", documents)
    write_jsonl(artifact_dir / "text_evidence.jsonl", text_evidence)
    write_jsonl(artifact_dir / "table_evidence.jsonl", table_evidence)
    relations = build_document_relations(documents)
    write_jsonl(artifact_dir / "document_relations.jsonl", relations)
    write_jsonl(artifact_dir / "ingestion_errors.jsonl", errors)
    summary = {
        "data_dir": str(data_dir.resolve()),
        "documents": len(documents),
        "unique_documents": len({item["doc_id"] for item in documents}),
        "duplicate_documents": len(documents) - len({item["doc_id"] for item in documents}),
        "text_evidence": len(text_evidence),
        "table_evidence": len(table_evidence),
        "document_relations": len(relations),
        "errors": len(errors),
        "error_samples": errors[:20],
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
