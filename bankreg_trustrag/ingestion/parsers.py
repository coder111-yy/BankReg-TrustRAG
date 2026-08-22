from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas import Document, TableCellEvidence, TextEvidence
from ..utils import compact_path, first_nonempty, normalize_text, sha256_file, stable_id


@dataclass
class ParseResult:
    document: Document
    text_evidence: list[TextEvidence] = field(default_factory=list)
    table_evidence: list[TableCellEvidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _document_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {".docx": "word", ".doc": "word", ".pdf": "pdf", ".xlsx": "excel", ".xls": "excel"}.get(suffix, "other")


def _title(path: Path) -> str:
    stem = path.stem
    # Attachment names often repeat the title after an ordinal prefix and underscore.
    stem = re.sub(r"^\d+_", "", stem)
    parts = stem.split("_")
    if len(parts) > 1 and parts[-1] == parts[-2]:
        parts = parts[:-1]
    return normalize_text("_".join(parts))


def _date_from_name(text: str) -> str | None:
    m = re.search(r"(20\d{2})年(?:0?(\d{1,2})月)?(?:0?(\d{1,2})日)?", text)
    if not m:
        m = re.search(r"(20\d{2})[-年](\d{1,2})[-月](\d{1,2})?", text)
    if not m:
        return None
    year, month, day = m.group(1), m.group(2) or "01", m.group(3) or "01"
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _metadata(path: Path, root: Path) -> Document:
    title = _title(path)
    digest = sha256_file(path)
    doc_id = "DOC_" + digest[:16]
    date = _date_from_name(path.name)
    status = "effective" if date and date >= "2025-01-01" else "unknown"
    return Document(
        doc_id=doc_id,
        title=title,
        authority="国家金融监督管理总局" if any(x in title for x in ["监管", "银行业", "保险业", "金融"]) else None,
        document_no=None,
        publish_date=date,
        effective_date=None,
        expire_date=None,
        document_type=_document_type(path),
        topic=[],
        version=None,
        status=status,
        source_url=None,
        local_path=compact_path(path, root),
        sha256=digest,
        file_name=path.name,
    )


def _paragraph_evidence(doc: Document, paragraphs: list[tuple[str, int | None, str | None, str | None]]) -> list[TextEvidence]:
    evidence: list[TextEvidence] = []
    for index, (content, page, section, article_no) in enumerate(paragraphs, 1):
        content = normalize_text(content)
        if not content:
            continue
        evidence.append(
            TextEvidence(
                evidence_id=f"text:{doc.doc_id}:p{index}",
                doc_id=doc.doc_id,
                content=content,
                page=page,
                chapter=section,
                article_no=article_no,
                paragraph_no=index,
                section=section,
                source_url=doc.source_url,
                source_location=f"{doc.file_name}:paragraph:{index}",
            )
        )
    return evidence


def parse_docx(path: Path, doc: Document) -> list[TextEvidence]:
    from docx import Document as WordDocument

    source: list[tuple[str, int | None, str | None, str | None]] = []
    current_section: str | None = None
    document = WordDocument(str(path))
    for paragraph in document.paragraphs:
        content = paragraph.text
        if paragraph.style and paragraph.style.name.lower().startswith("heading"):
            current_section = normalize_text(content) or current_section
        article = None
        match = re.match(r"(第[一二三四五六七八九十百零0-9]+[章节条款])", normalize_text(content))
        if match:
            article = match.group(1)
        source.append((content, None, current_section, article))
    for table_index, table in enumerate(document.tables, 1):
        for row_index, row in enumerate(table.rows, 1):
            text = " | ".join(normalize_text(cell.text) for cell in row.cells)
            source.append((f"表{table_index} 行{row_index}: {text}", None, current_section, None))
    return _paragraph_evidence(doc, source)


def parse_legacy_doc(path: Path, doc: Document) -> tuple[list[TextEvidence], list[str]]:
    warnings: list[str] = []
    try:
        completed = subprocess.run(["antiword", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        if completed.returncode == 0 and completed.stdout.strip():
            return _paragraph_evidence(doc, [(p, None, None, None) for p in completed.stdout.splitlines()]), []
        warnings.append(f"antiword returned {completed.returncode}: {completed.stderr[:300]}")
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        warnings.append(f"legacy .doc parser unavailable: {exc}")
    fallback = _parse_wps_utf16_doc(path)
    if fallback:
        return _paragraph_evidence(doc, [(p, None, None, None) for p in fallback.splitlines()]), warnings + ["used local OLE/UTF-16 fallback for legacy .doc"]
    return [], warnings


def _parse_wps_utf16_doc(path: Path) -> str:
    """Recover text from WPS-generated OLE .doc files.

    Several contest attachments have a valid WordDocument stream but report
    zero words and are rejected by antiword.  WPS stores the visible body as
    UTF-16LE runs in that stream.  This conservative fallback extracts only
    printable CJK/Latin runs and leaves the original file untouched.
    """
    try:
        import olefile

        with olefile.OleFileIO(str(path)) as ole:
            if not ole.exists("WordDocument"):
                return ""
            raw = ole.openstream("WordDocument").read()
        decoded = raw.decode("utf-16le", errors="ignore")
        runs: list[str] = []
        current: list[str] = []

        def flush() -> None:
            if not current:
                return
            value = normalize_text("".join(current))
            if value and value not in {"PAGE", "MERGEFORMAT"}:
                runs.append(value)
            current.clear()

        for char in decoded:
            code = ord(char)
            if char in "\r\n\t" or 0x20 <= code <= 0x7E or 0x3000 <= code <= 0x9FFF:
                current.append(char)
            else:
                flush()
        flush()
        # Metadata and field names are short; the body contains several CJK
        # characters.  This also prevents returning an apparently successful
        # evidence record for an empty or unsupported .doc file.
        body = [run for run in runs if sum(0x4E00 <= ord(char) <= 0x9FFF for char in run) >= 2]
        return "\n".join(body)
    except (ImportError, OSError, ValueError):
        return ""


def parse_pdf(path: Path, doc: Document) -> list[TextEvidence]:
    from pypdf import PdfReader

    paragraphs: list[tuple[str, int | None, str | None, str | None]] = []
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for block in re.split(r"\n\s*\n|\n", text):
            if normalize_text(block):
                paragraphs.append((block, page_number, None, None))
    return _paragraph_evidence(doc, paragraphs)


def _as_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return normalize_text(value)


def _period_from_header(header: str | None) -> str | None:
    if not header:
        return None
    value = normalize_text(header)
    m = re.search(r"(20\d{2})年\s*0?(\d{1,2})月", value)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.search(r"(20\d{2})[-/]0?(\d{1,2})", value)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.search(r"(20\d{2})年", value)
    return m.group(1) if m else value if re.search(r"\d", value) else None


def _cell_records(doc: Document, sheet_name: str, rows: list[list[Any]], table_name: str | None = None) -> list[TableCellEvidence]:
    records: list[TableCellEvidence] = []
    if not rows:
        return records
    width = max((len(r) for r in rows), default=0)
    header_rows = _header_row_count(rows)
    headers: list[str] = []
    for column in range(width):
        pieces = [normalize_text(rows[row][column]) for row in range(header_rows) if column < len(rows[row]) and normalize_text(rows[row][column])]
        headers.append(" / ".join(dict.fromkeys(pieces)))
    for row_index, row in enumerate(rows, 1):
        row_header = first_nonempty(row[:2])
        indicator = row_header
        for column_index, value in enumerate(row, 1):
            if value is None or normalize_text(value) == "":
                continue
            column_header = headers[column_index - 1] or None
            period = _period_from_header(column_header) or _period_from_header(doc.title)
            unit = None
            unit_match = re.search(r"(%)|(%|‰|亿元|万元|百万元|元)", f"{column_header or ''} {row_header or ''}")
            if unit_match:
                unit = unit_match.group(1)
            address = _excel_address(row_index, column_index)
            context = f"{indicator or ''} | {column_header or ''} | {normalize_text(value)}"
            records.append(
                TableCellEvidence(
                    evidence_id=f"cell:{doc.doc_id}:{sheet_name}:{address}",
                    doc_id=doc.doc_id,
                    sheet_name=sheet_name,
                    table_name=table_name,
                    indicator=indicator,
                    period=period,
                    value=_as_value(value),
                    unit=unit,
                    row_header=row_header,
                    column_header=column_header,
                    cell_address=address,
                    context=normalize_text(context),
                    source_url=doc.source_url,
                )
            )
    return records


def _header_row_count(rows: list[list[Any]]) -> int:
    """Find the first data row instead of assuming every workbook has 4 headers.

    The old fixed four-row window included the first data row in column
    headers for sheets whose data starts on row 4.  That polluted headers and
    periods (for example ``合计 / 45167.98``), making exact row/column lookup
    impossible even though the source cell was present.
    """
    for row_index, row in enumerate(rows[:12]):
        if not row or not first_nonempty(row[:2]):
            continue
        numeric_values = [
            value for value in row[1:]
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numeric_values:
            return max(1, row_index)
    return min(4, len(rows))


def _excel_address(row: int, column: int) -> str:
    result = ""
    number = column
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return f"{result}{row}"


def parse_excel(path: Path, doc: Document) -> tuple[list[TableCellEvidence], list[str]]:
    warnings: list[str] = []
    records: list[TableCellEvidence] = []
    try:
        if path.suffix.lower() == ".xlsx":
            import openpyxl

            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            for sheet in workbook.worksheets:
                rows = [list(row) for row in sheet.iter_rows(values_only=True)]
                records.extend(_cell_records(doc, sheet.title, rows, sheet.title))
        else:
            import xlrd

            workbook = xlrd.open_workbook(str(path), on_demand=True)
            for sheet in workbook.sheets():
                rows = [sheet.row_values(i) for i in range(sheet.nrows)]
                records.extend(_cell_records(doc, sheet.name, rows, sheet.name))
    except ImportError as exc:
        warnings.append(f"missing spreadsheet dependency for {path.suffix}: {exc}")
    except Exception as exc:  # malformed attachments must be recorded, not abort a corpus build
        warnings.append(f"spreadsheet parse failed: {type(exc).__name__}: {exc}")
    return records, warnings


def parse_file(path: Path, root: Path) -> ParseResult:
    doc = _metadata(path, root)
    if path.suffix.lower() == ".docx":
        return ParseResult(doc, text_evidence=parse_docx(path, doc))
    if path.suffix.lower() == ".doc":
        text, warnings = parse_legacy_doc(path, doc)
        return ParseResult(doc, text_evidence=text, warnings=warnings)
    if path.suffix.lower() == ".pdf":
        try:
            return ParseResult(doc, text_evidence=parse_pdf(path, doc))
        except Exception as exc:
            return ParseResult(doc, warnings=[f"pdf parse failed: {type(exc).__name__}: {exc}"])
    if path.suffix.lower() in {".xlsx", ".xls"}:
        tables, warnings = parse_excel(path, doc)
        return ParseResult(doc, table_evidence=tables, warnings=warnings)
    return ParseResult(doc, warnings=[f"unsupported file type: {path.suffix}"])
