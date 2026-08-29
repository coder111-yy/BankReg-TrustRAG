from __future__ import annotations

import re
import subprocess
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas import Document, TableCellEvidence, TextEvidence
from ..utils import (
    compact_path,
    first_nonempty,
    insurance_company_scope,
    is_insurance_fund_table,
    normalize_text,
    normalized_reporting_period,
    sha256_file,
    stable_id,
)


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


def parse_legacy_doc(
    path: Path,
    doc: Document
) -> tuple[list[TextEvidence], list[str]]:
    """
    使用 LibreOffice 将旧版 .doc 转换为 .docx，
    然后复用现有的 parse_docx() 解析正文和表格。
    """

    # 1. 优先从 PATH 查找 soffice
    soffice = shutil.which("soffice")

    # 2. Windows 常见 LibreOffice 安装位置
    if not soffice:
        candidates = [
            Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
            Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
        ]

        for candidate in candidates:
            if candidate.exists():
                soffice = str(candidate)
                break

    if not soffice:
        return [], [
            "LibreOffice not found. "
            "Please install LibreOffice or add soffice.exe to PATH."
        ]

    try:
        # 使用临时目录，避免转换后的 docx 污染原始数据目录
        with tempfile.TemporaryDirectory(prefix="bankreg_doc_") as temp_dir:
            temp_path = Path(temp_dir)

            completed = subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    str(temp_path),
                    str(path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )

            if completed.returncode != 0:
                return [], [
                    f"LibreOffice conversion failed "
                    f"(returncode={completed.returncode}): "
                    f"{completed.stderr[:500]}"
                ]

            # LibreOffice 转换后的文件
            converted_path = temp_path / f"{path.stem}.docx"

            if not converted_path.exists():
                return [], [
                    f"LibreOffice conversion produced no DOCX file: {path}"
                ]

            # 直接复用现有 DOCX 解析逻辑
            evidence = parse_docx(converted_path, doc)

            return evidence, []

    except subprocess.TimeoutExpired:
        return [], [
            f"LibreOffice conversion timeout: {path}"
        ]

    except Exception as exc:
        return [], [
            f"LibreOffice .doc parser failed: {exc}"
        ]


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
    parsed_period = normalized_reporting_period(value)
    if parsed_period:
        return parsed_period
    m = re.search(r"(20\d{2})[-/]0?(\d{1,2})", value)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    return value if re.search(r"\d", value) else None


_QUARTER_LABELS = {"一季度", "二季度", "三季度", "四季度"}


def _quarter_label(value: Any) -> str | None:
    """Normalize the quarter marker used by block-form statistical sheets."""
    text = normalize_text(value)
    if text in _QUARTER_LABELS:
        return text
    match = re.fullmatch(r"第?([一二三四1-4])季度", text)
    if not match:
        return None
    return {
        "一": "一季度", "1": "一季度",
        "二": "二季度", "2": "二季度",
        "三": "三季度", "3": "三季度",
        "四": "四季度", "4": "四季度",
    }[match.group(1)]


def _infer_unit(*values: Any) -> str | None:
    match = re.search(r"(百分比|%|％|‰|亿元|万元|百万元|元)", " ".join(normalize_text(value) for value in values if value))
    return match.group(1) if match else None


def _infer_sheet_unit(rows: list[list[Any]], header_rows: int) -> str | None:
    """Return a sheet-wide unit only when the header is unambiguous.

    Some statistical workbooks place ``单位：亿元`` in a visually merged
    cell above just one physical column.  Per-column header extraction then
    loses the unit for neighbouring numeric cells.  Do not propagate when a
    header also advertises count-like units because those sheets are mixed.
    """
    header = " ".join(
        normalize_text(value)
        for row in rows[:header_rows]
        for value in row
        if normalize_text(value)
    )
    if re.search(r"(?:万件|件|万户|户|家|个)", header):
        return None
    units = {
        "%" if value in {"百分比", "%", "％"} else value
        for value in re.findall(r"百分比|%|％|‰|万亿元|亿元|百万元|万元|元", header)
    }
    return next(iter(units)) if len(units) == 1 else None


def _looks_like_quarter_matrix(rows: list[list[Any]], header_rows: int) -> bool:
    """Detect quarter blocks x indicator rows x institution columns."""
    if header_rows < 2 or max((len(row) for row in rows), default=0) < 3:
        return False
    active_quarter: str | None = None
    quarter_count = 0
    metric_rows = 0
    for row in rows[header_rows:header_rows + 60]:
        if not row:
            continue
        quarter = _quarter_label(row[0] if len(row) > 0 else None)
        if quarter:
            active_quarter = quarter
            quarter_count += 1
        metric = normalize_text(row[1] if len(row) > 1 else None)
        values = [value for value in row[2:] if value is not None and normalize_text(value)]
        if active_quarter and metric and values:
            metric_rows += 1
    return quarter_count >= 2 and metric_rows >= 4


def _quarter_matrix_records(
    doc: Document,
    sheet_name: str,
    rows: list[list[Any]],
    header_rows: int,
    headers: list[str],
    table_name: str | None,
) -> list[TableCellEvidence]:
    records: list[TableCellEvidence] = []
    period = _period_from_header(doc.title)
    sheet_unit = _infer_unit(" ".join(headers), " ".join(normalize_text(value) for row in rows[:header_rows] for value in row if value))
    active_quarter: str | None = None
    for row_index, row in enumerate(rows[header_rows:], header_rows + 1):
        quarter = _quarter_label(row[0] if len(row) > 0 else None)
        if quarter:
            active_quarter = quarter
        indicator = normalize_text(row[1] if len(row) > 1 else None) or None
        if not active_quarter or not indicator:
            continue
        for column_index, value in enumerate(row[2:], 3):
            if value is None or normalize_text(value) == "":
                continue
            column_header = headers[column_index - 1] if column_index - 1 < len(headers) else None
            column_header = column_header or None
            normalized_value = normalize_text(value)
            context = f"{indicator} | {active_quarter} | {column_header or ''} | {normalized_value}"
            address = _excel_address(row_index, column_index)
            records.append(
                TableCellEvidence(
                    evidence_id=f"cell:{doc.doc_id}:{sheet_name}:{address}",
                    doc_id=doc.doc_id,
                    sheet_name=sheet_name,
                    table_name=table_name,
                    indicator=indicator,
                    period=period,
                    value=_as_value(value),
                    unit=_infer_unit(column_header, indicator) or sheet_unit,
                    row_header=active_quarter,
                    column_header=column_header,
                    cell_address=address,
                    context=normalize_text(context),
                    source_url=doc.source_url,
                )
            )
    return records


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
    if _looks_like_quarter_matrix(rows, header_rows):
        return _quarter_matrix_records(doc, sheet_name, rows, header_rows, headers, table_name)
    scoped_insurance_report = is_insurance_fund_table(doc.title, sheet_name, table_name)
    sheet_unit = _infer_sheet_unit(rows, header_rows)
    active_scope = "保险业总体" if scoped_insurance_report else None
    for row_index, row in enumerate(rows, 1):
        row_header = first_nonempty(row[:2])
        indicator = row_header
        if scoped_insurance_report:
            active_scope = insurance_company_scope(row_header) or active_scope
        for column_index, value in enumerate(row, 1):
            if value is None or normalize_text(value) == "":
                continue
            column_header = headers[column_index - 1] or None
            period = _period_from_header(column_header) or _period_from_header(doc.title)
            unit = None
            unit = _infer_unit(column_header, row_header)
            if unit is None and isinstance(value, (int, float)) and not isinstance(value, bool):
                unit = sheet_unit
            address = _excel_address(row_index, column_index)
            context_parts = [active_scope, indicator, column_header, normalize_text(value)]
            context = " | ".join(str(part) for part in context_parts if part)
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
