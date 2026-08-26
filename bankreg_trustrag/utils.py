from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


_TABLE_OUTLINE_PREFIX_RE = re.compile(
    r"^(?:\((?:\d+|[一二三四五六七八九十百]+)\)[、.．:：]?|(?:\d+|[一二三四五六七八九十百]+)[)、.．:：](?!\d))"
)
_TABLE_HIERARCHY_PREFIX_RE = re.compile(
    r"^(?:(?:其中(?:包括|含)?|包括)[、,:：，]?)+",
    re.IGNORECASE,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}:{hashlib.sha1(raw.encode('utf-8', 'ignore')).hexdigest()[:16]}"


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def canonical_table_label(value: Any) -> str:
    """Normalize a user/table label without erasing meaningful digits.

    Spreadsheet row labels often contain visual outline prefixes such as
    ``1、财产险`` or ``(二) 人身险`` while users naturally ask for
    ``财产险`` and ``人身险``.  A separator is required before a numeric
    prefix is removed, so labels such as ``1年期贷款`` remain intact.
    """
    label = re.sub(r"\s+", "", normalize_text(value)).lower()
    label = _TABLE_OUTLINE_PREFIX_RE.sub("", label, count=1)
    label = _TABLE_HIERARCHY_PREFIX_RE.sub("", label, count=1)
    return label.replace("总计", "合计")


def canonical_dimension_label(value: Any) -> str:
    """Normalize equivalent separators in hierarchical column labels."""
    return re.sub(r"[/／\\|｜_\-—–]+", "", canonical_table_label(value))


def insurance_company_scope(value: Any) -> str | None:
    """Return the explicit insurance-company section named by text."""
    text = re.sub(r"\s+", "", normalize_text(value))
    if re.search(r"(?:财产保险|财产险|产险)公司", text):
        return "财产保险公司"
    if re.search(r"(?:人身保险|人身险|寿险)公司", text):
        return "人身险公司"
    if any(term in text for term in ("保险业总体", "保险业整体", "保险行业总体", "全保险业", "全部保险公司", "保险业合计")):
        return "保险业总体"
    return None


def is_insurance_fund_table(*values: Any) -> bool:
    """Identify insurance-fund-use reports that contain repeated sections."""
    text = normalize_text(" ".join(str(value or "") for value in values))
    return "保险" in text and "资金运用" in text


_QUARTER_NUMBER = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
}


def reporting_period_details(value: Any) -> tuple[str | None, str | None, str | None]:
    """Parse a reporting month/quarter/year from natural-language metadata.

    Returns the matched display text, a sortable normalized value such as
    ``2023-Q4`` or ``2023-09``, and a Chinese quarter label when applicable.
    Quarter metadata is intentionally read from document titles as well as
    questions because some source workbooks contain stale worksheet names.
    """
    text = normalize_text(value)
    month_match = re.search(r"(20\d{2})年\s*0?(1[0-2]|[1-9])月", text)
    if month_match:
        year = int(month_match.group(1))
        month = int(month_match.group(2))
        quarter = ("一季度", "二季度", "三季度", "四季度")[(month - 1) // 3]
        return month_match.group(0), f"{year:04d}-{month:02d}", quarter

    quarter_match = re.search(
        r"(20\d{2})年\s*(?:第\s*)?([一二三四1-4])\s*季度",
        text,
    )
    if not quarter_match:
        quarter_match = re.search(r"(20\d{2})\s*[-_/]?\s*[Qq]([1-4])", text)
    if quarter_match:
        year = int(quarter_match.group(1))
        quarter_number = _QUARTER_NUMBER[quarter_match.group(2)]
        quarter_label = ("一季度", "二季度", "三季度", "四季度")[quarter_number - 1]
        return quarter_match.group(0), f"{year:04d}-Q{quarter_number}", quarter_label

    year_match = re.search(r"(20\d{2})年", text)
    if year_match:
        return year_match.group(0), year_match.group(1), None
    return None, None, None


def normalized_reporting_period(value: Any) -> str | None:
    """Return only the canonical reporting-period key for ``value``."""
    return reporting_period_details(value)[1]


def tokens(text: str) -> list[str]:
    text = normalize_text(text).lower()
    # Preserve Chinese characters as single terms and group latin/numeric terms.
    return re.findall(r"[\u4e00-\u9fff]|[a-z]+|\d+(?:\.\d+)?%?", text)


def char_ngrams(text: str, n: int = 2) -> set[str]:
    clean = re.sub(r"\s+", "", normalize_text(text).lower())
    if len(clean) <= n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def normalized_number(value: object) -> float | None:
    if value is None:
        return None
    text = normalize_text(value).replace(",", "")
    percent = "%" in text or "％" in text
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {".", "-", "-."}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 100 if percent else number


def numbers_in(text: str) -> list[float]:
    values: list[float] = []
    for match in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?\s*%?", normalize_text(text)):
        value = normalized_number(match)
        if value is not None:
            values.append(value)
    return values


def compact_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def first_nonempty(values: Iterable[object]) -> str | None:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return None

